"""
AlphaGenome K562 Genome-Wide Track Pipeline
============================================

Tiles the hg38 reference genome with overlapping 1 MB windows, queries the
AlphaGenome API for K562 ATAC-seq, ChIP-histone, TF-ChIP-seq, and contact-map
predictions, stitches 1D tracks back into per-chromosome arrays, and saves
results as compressed NPZ files.

Usage
-----
    export ALPHAGENOME_API_KEY="your_key_here"

    # Full genome, default settings
    python run_pipeline.py --output-dir output/

    # Single chromosome smoke-test
    python run_pipeline.py --chroms chr22 --output-dir output/ --batch-size 5

    # Custom stride
    python run_pipeline.py --chroms chr21 chr22 --stride 262144 --output-dir output/

Output layout
-------------
    output/
    ├── atac/
    │   ├── chr1.npz          (float32 array: genome_bins × num_tracks)
    │   ├── chr1.json         (track metadata sidecar)
    │   └── ...
    ├── chip_histone/
    ├── chip_tf/
    └── contact_maps/
        ├── chr1_0_1048576.npz
        └── ...               (one file per prediction window)

Checkpointing
-------------
    If an output file already exists for a given (chrom, output_type) it is
    skipped automatically, allowing interrupted runs to resume cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from alphagenome.models import dna_client, dna_output
from alphagenome.data import genome

import config
from sliding_window import generate_windows
from stitching import stitch_1d_tracks, save_contact_map_window
from output_writers import save_chrom_npz

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1D output types that are stitched into a genome-wide array
# ---------------------------------------------------------------------------
_1D_OUTPUT_TYPES = [
    dna_output.OutputType.ATAC,
    dna_output.OutputType.CHIP_HISTONE,
    dna_output.OutputType.CHIP_TF,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genome-wide AlphaGenome K562 track prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Root output directory.  Sub-directories per track type are created automatically.",
    )
    p.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        metavar="CHROM",
        help="Chromosomes to process (e.g. chr1 chr22).  Defaults to all hg38 standard chromosomes.",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=config.WINDOW_SIZE,
        help="Prediction window width in bp.  Must be a length supported by AlphaGenome (16384, 114688, 524288, or 1048576).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=config.DEFAULT_STRIDE,
        help="Window stride in bp.  Must be ≤ window-size.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="Number of windows to send to predict_intervals in one call.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=config.MAX_WORKERS,
        help="Parallel workers used by predict_intervals.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="AlphaGenome API key.  Falls back to $ALPHAGENOME_API_KEY.",
    )
    p.add_argument(
        "--no-contact-maps",
        action="store_true",
        help="Skip contact-map predictions (saves time and disk space).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip chromosomes whose output files already exist (default: True).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Overwrite existing output files.",
    )
    return p.parse_args(argv)


def resolve_api_key(cli_key: str | None) -> str:
    key = cli_key or config.API_KEY
    if not key:
        log.error(
            "No API key provided.  Set $ALPHAGENOME_API_KEY or pass --api-key."
        )
        sys.exit(1)
    return key


def build_requested_outputs(
    no_contact_maps: bool,
) -> list[dna_output.OutputType]:
    outputs = list(_1D_OUTPUT_TYPES)
    if not no_contact_maps:
        outputs.append(dna_output.OutputType.CONTACT_MAPS)
    return outputs


def _chrom_is_complete(
    chrom: str,
    output_dir: Path,
    requested_outputs: list[dna_output.OutputType],
    no_contact_maps: bool,
) -> bool:
    """Return True if all expected output files already exist for *chrom*."""
    for ot in _1D_OUTPUT_TYPES:
        if ot not in requested_outputs:
            continue
        subdir = output_dir / config.OUTPUT_TYPE_DIRS[ot]
        if not (subdir / f"{chrom}.npz").exists():
            return False
    return True


def run_chrom(
    chrom: str,
    chrom_size: int,
    client: dna_client.DnaClient,
    window_size: int,
    stride: int,
    batch_size: int,
    max_workers: int,
    ontology_terms: list,
    requested_outputs: list[dna_output.OutputType],
    output_dir: Path,
    no_contact_maps: bool,
) -> None:
    log.info(f"[{chrom}]  size={chrom_size:,} bp")

    windows = generate_windows(chrom, chrom_size, window_size=window_size, stride=stride)
    log.info(f"[{chrom}]  {len(windows)} windows  (window={window_size:,}  stride={stride:,})")

    # Resize any short windows (e.g. chromosome < window_size) to the nearest
    # supported model length before calling the API.
    resized_windows: list[genome.Interval] = []
    for w in windows:
        if w.width != window_size:
            try:
                dna_client.validate_sequence_length(window_size)
                resized_windows.append(w.resize(window_size))
            except Exception:
                # If resize goes out of bounds, use the original truncated interval
                # and let the API handle it (the model accepts other lengths too).
                resized_windows.append(w)
        else:
            resized_windows.append(w)

    # ------------------------------------------------------------------ #
    # Predict in batches
    # ------------------------------------------------------------------ #
    all_outputs: list[dna_output.Output] = []
    n_batches = (len(resized_windows) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(n_batches),
        desc=f"{chrom} batches",
        unit="batch",
        leave=False,
    ):
        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, len(resized_windows))
        batch = resized_windows[batch_start:batch_end]

        batch_outputs = client.predict_intervals(
            batch,
            organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
            progress_bar=False,
            max_workers=max_workers,
        )
        all_outputs.extend(batch_outputs)

    log.info(f"[{chrom}]  received {len(all_outputs)} outputs")

    # ------------------------------------------------------------------ #
    # Stitch and save 1D tracks
    # ------------------------------------------------------------------ #
    for ot in _1D_OUTPUT_TYPES:
        if ot not in requested_outputs:
            continue

        subdir = output_dir / config.OUTPUT_TYPE_DIRS[ot]
        npz_path = subdir / f"{chrom}.npz"
        if npz_path.exists():
            log.info(f"[{chrom}]  {ot.name} — already exists, skipping.")
            continue

        log.info(f"[{chrom}]  stitching {ot.name} …")
        try:
            values, metadata, resolution = stitch_1d_tracks(
                outputs=all_outputs,
                windows=windows,   # original (not resized) windows for position math
                stride=stride,
                output_type=ot,
                chrom_size=chrom_size,
            )
        except ValueError as exc:
            log.warning(f"[{chrom}]  {ot.name} — skipped: {exc}")
            continue

        npz_out, json_out = save_chrom_npz(
            values=values,
            metadata=metadata,
            resolution=resolution,
            chrom=chrom,
            chrom_size=chrom_size,
            out_dir=subdir,
        )
        log.info(
            f"[{chrom}]  {ot.name} — saved  {npz_out.name}  "
            f"shape={values.shape}  res={resolution} bp"
        )

    # ------------------------------------------------------------------ #
    # Save contact maps (one NPZ per window)
    # ------------------------------------------------------------------ #
    if not no_contact_maps and dna_output.OutputType.CONTACT_MAPS in requested_outputs:
        cm_dir = output_dir / config.OUTPUT_TYPE_DIRS[dna_output.OutputType.CONTACT_MAPS]
        cm_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"[{chrom}]  saving contact maps …")
        for output, window in zip(all_outputs, windows):
            out_file = cm_dir / f"{window.chromosome}_{window.start}_{window.end}.npz"
            if out_file.exists():
                continue
            try:
                save_contact_map_window(output, window, cm_dir)
            except ValueError as exc:
                log.warning(f"[{chrom}]  contact map window {window} skipped: {exc}")

        log.info(f"[{chrom}]  contact maps saved to {cm_dir}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    api_key = resolve_api_key(args.api_key)
    client  = dna_client.create(api_key)

    chroms = args.chroms or config.ALL_CHROMS
    # Validate chromosome names
    for c in chroms:
        if c not in config.HG38_CHROM_SIZES:
            log.error(
                f"Unknown chromosome '{c}'.  "
                "Use standard hg38 names like chr1 … chr22, chrX, chrY."
            )
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_outputs = build_requested_outputs(args.no_contact_maps)

    # The API accepts CURIE strings directly in ontology_terms
    ontology_terms = [config.K562_ONTOLOGY_TERM]

    log.info(
        f"Pipeline settings:\n"
        f"  chromosomes  : {chroms}\n"
        f"  window size  : {args.window_size:,} bp\n"
        f"  stride       : {args.stride:,} bp\n"
        f"  batch size   : {args.batch_size}\n"
        f"  max workers  : {args.max_workers}\n"
        f"  output types : {[ot.name for ot in requested_outputs]}\n"
        f"  ontology     : {config.K562_ONTOLOGY_TERM}\n"
        f"  output dir   : {output_dir}\n"
        f"  resume       : {args.resume}"
    )

    for chrom in tqdm(chroms, desc="Chromosomes", unit="chrom"):
        chrom_size = config.HG38_CHROM_SIZES[chrom]

        if args.resume and _chrom_is_complete(
            chrom, output_dir, requested_outputs, args.no_contact_maps
        ):
            log.info(f"[{chrom}]  all outputs present — skipping (--resume).")
            continue

        run_chrom(
            chrom=chrom,
            chrom_size=chrom_size,
            client=client,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            ontology_terms=ontology_terms,
            requested_outputs=requested_outputs,
            output_dir=output_dir,
            no_contact_maps=args.no_contact_maps,
        )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
