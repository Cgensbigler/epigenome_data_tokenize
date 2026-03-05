"""
Generate synthetic AlphaGenome-format prediction data for testing tokenization.

Produces NPZ + JSON sidecar files in the same format as run_pipeline.py output:
  - atac/{chrom}.npz         (1 bp resolution, 1 track)
  - chip_histone/{chrom}.npz (128 bp resolution, 10 histone marks)
  - chip_tf/{chrom}.npz      (128 bp resolution, 5 TF tracks)
  - contact_maps/{chrom}_{start}_{end}.npz  (2048 bp resolution)

Signals contain realistic peak-like features generated from a mixture of
Gaussian bumps at random genomic positions.
"""
from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

HG38_CHROM_SIZES = {
    "chr21": 46709983,
    "chr22": 50818468,
}

WINDOW_SIZE = 1 << 20   # 1,048,576
STRIDE = 1 << 19        # 524,288

HISTONE_MARKS = [
    "H3K27ac", "H3K4me3", "H3K27me3", "H3K36me3", "H3K4me1",
    "H3K9me3", "H4K20me1", "H3K9ac", "H3K4me2", "H2AFZ",
]

TF_NAMES = ["CTCF", "SP1", "YY1", "MAX", "MYC"]


def _generate_peak_signal(n_bins: int, n_peaks: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a 1-D signal with Gaussian peaks at random positions."""
    signal = np.abs(rng.normal(0, 0.01, n_bins).astype(np.float32))
    peak_positions = rng.integers(100, n_bins - 100, size=n_peaks)
    peak_heights = rng.exponential(2.0, size=n_peaks).astype(np.float32) + 0.5
    peak_widths = rng.integers(5, 50, size=n_peaks)

    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        left = max(0, pos - width)
        right = min(n_bins, pos + width)
        x = np.arange(left, right) - pos
        signal[left:right] += height * np.exp(-0.5 * (x / max(width / 3, 1)) ** 2)

    return signal


def generate_1d_tracks(
    chrom: str,
    chrom_size: int,
    output_dir: Path,
    assay: str,
    resolution: int,
    track_names: list[str],
    track_metadata_extra: dict | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Generate and save synthetic 1D track data."""
    rng = rng or np.random.default_rng(42)
    n_bins = (chrom_size + resolution - 1) // resolution
    n_tracks = len(track_names)

    values = np.zeros((n_bins, n_tracks), dtype=np.float32)
    n_peaks_per_track = max(50, n_bins // 5000)

    for t in range(n_tracks):
        values[:, t] = _generate_peak_signal(n_bins, n_peaks_per_track, rng)

    metadata = []
    for i, name in enumerate(track_names):
        entry = {"name": f"{'Histone ChIP-seq ' if assay == 'chip_histone' else 'TF ChIP-seq ' if assay == 'chip_tf' else ''}{name}", "strand": "+"}
        if assay == "chip_tf":
            entry["transcription_factor"] = name
        if track_metadata_extra:
            entry.update(track_metadata_extra)
        metadata.append(entry)

    out_dir = output_dir / assay
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{chrom}.npz"
    json_path = out_dir / f"{chrom}.json"

    np.savez_compressed(
        npz_path,
        values=values,
        resolution=np.array(resolution, dtype=np.int32),
        chrom=np.array(chrom),
        chrom_size=np.array(chrom_size, dtype=np.int64),
    )

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("[%s] %s: shape=%s res=%d → %s", chrom, assay, values.shape, resolution, npz_path)


def generate_contact_maps(
    chrom: str,
    chrom_size: int,
    output_dir: Path,
    resolution: int = 2048,
    rng: np.random.Generator | None = None,
) -> None:
    """Generate synthetic per-window contact map NPZ files."""
    rng = rng or np.random.default_rng(42)
    cm_dir = output_dir / "contact_maps"
    cm_dir.mkdir(parents=True, exist_ok=True)

    track_names = ["Hi-C K562"]
    n_tracks = len(track_names)

    pos = 0
    window_idx = 0
    while pos < chrom_size:
        win_end = min(pos + WINDOW_SIZE, chrom_size)
        n_bins = (win_end - pos) // resolution
        if n_bins < 2:
            break

        mat = np.zeros((n_bins, n_bins, n_tracks), dtype=np.float32)

        for t in range(n_tracks):
            for b in range(n_bins):
                n_contacts = rng.integers(1, min(10, n_bins))
                partners = rng.integers(0, n_bins, size=n_contacts)
                strengths = rng.exponential(0.5, size=n_contacts).astype(np.float32)
                for p, s in zip(partners, strengths):
                    dist = abs(b - p)
                    decay = np.exp(-dist / max(n_bins / 5, 1))
                    mat[b, p, t] += s * decay
                    mat[p, b, t] += s * decay

        fname = f"{chrom}_{pos}_{win_end}.npz"
        np.savez_compressed(
            cm_dir / fname,
            values=mat,
            resolution=np.array(resolution, dtype=np.int32),
            chrom=np.array(chrom),
            start=np.array(pos, dtype=np.int64),
            end=np.array(win_end, dtype=np.int64),
            track_names=np.array(track_names),
            track_strands=np.array(["+"]),
        )

        window_idx += 1
        pos += STRIDE

    log.info("[%s] contact_maps: %d windows → %s", chrom, window_idx, cm_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AlphaGenome prediction data")
    parser.add_argument("--output-dir", "-o", default="output/alphagenome_predictions", help="Output directory")
    parser.add_argument("--chroms", nargs="+", default=["chr21", "chr22"], help="Chromosomes to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-contact-maps", action="store_true", help="Skip contact maps (saves time)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for chrom in args.chroms:
        chrom_size = HG38_CHROM_SIZES[chrom]
        log.info("Generating data for %s (size=%d bp)", chrom, chrom_size)

        generate_1d_tracks(
            chrom, chrom_size, output_dir, "atac",
            resolution=1, track_names=["ATAC-seq"],
            rng=rng,
        )

        generate_1d_tracks(
            chrom, chrom_size, output_dir, "chip_histone",
            resolution=128, track_names=HISTONE_MARKS,
            rng=rng,
        )

        generate_1d_tracks(
            chrom, chrom_size, output_dir, "chip_tf",
            resolution=128, track_names=TF_NAMES,
            rng=rng,
        )

        if not args.no_contact_maps:
            generate_contact_maps(chrom, chrom_size, output_dir, rng=rng)

    log.info("Synthetic data generation complete → %s", output_dir)


if __name__ == "__main__":
    main()
