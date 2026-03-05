"""
ENCODE Tracks Tokenizer — main CLI entry point.

Usage examples
--------------
# K562, all three assays, hg38, 10 kb Hi-C bins
python run.py \\
    --assay TF-ChIP-seq ATAC-seq Hi-C \\
    --biosample K562 \\
    --assembly GRCh38 \\
    --hic-resolution 10000 \\
    --output-dir ./output

# Only download ATAC-seq for multiple cell types
python run.py \\
    --assay ATAC-seq \\
    --biosample "K562" "GM12878" "HepG2" \\
    --output-dir ./output

# Hi-C only, specific chromosomes, include trans contacts
python run.py \\
    --assay Hi-C \\
    --biosample K562 \\
    --chroms chr1 chr2 chr3 \\
    --hic-include-trans \\
    --output-dir ./output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import (
    ASSAY_TITLES,
    DEFAULT_ASSEMBLY,
    DEFAULT_HIC_RESOLUTION,
    DEFAULT_WORKERS,
    PEAK_ASSAYS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_peak_assay(
    assay: str,
    biosamples: list[str],
    assembly: str,
    output_dir: Path,
    workers: int,
    max_experiments: int | None = None,
) -> list:
    """Download + process one peak assay; return list of token DataFrames."""
    from downloader import batch_download_peak_experiments
    from encode_api import iter_peak_experiments
    from peak_processor import process_peak_experiments
    from tokenizer import write_peak_tokens

    all_frames = []

    for biosample in biosamples:
        logger.info("=== %s | %s | %s ===", assay, biosample, assembly)

        experiments = list(iter_peak_experiments(assay, biosample, assembly))
        if not experiments:
            logger.warning("No experiments found for %s / %s", assay, biosample)
            continue

        if max_experiments is not None and len(experiments) > max_experiments:
            logger.info("Limiting from %d to %d experiments (--max-experiments)", len(experiments), max_experiments)
            experiments = experiments[:max_experiments]

        logger.info("Found %d experiments to download", len(experiments))
        downloaded = batch_download_peak_experiments(experiments, output_dir, assay, workers)

        for acc, df in process_peak_experiments(downloaded, assay):
            token_df = write_peak_tokens(acc, df, assay, output_dir)
            all_frames.append(token_df)

    return all_frames


def run_hic(
    biosamples: list[str],
    assembly: str,
    output_dir: Path,
    workers: int,
    resolution: int,
    chromosomes: list[str] | None,
    include_trans: bool,
    max_experiments: int | None = None,
) -> list:
    """Download + process Hi-C; return list of token DataFrames."""
    from downloader import batch_download_hic_experiments
    from encode_api import iter_hic_experiments
    from hic_processor import process_hic_experiments
    from tokenizer import write_hic_tokens

    all_frames = []

    for biosample in biosamples:
        logger.info("=== Hi-C | %s | %s ===", biosample, assembly)

        experiments = list(iter_hic_experiments(biosample, assembly))
        if not experiments:
            logger.warning("No Hi-C experiments found for %s", biosample)
            continue

        if max_experiments is not None and len(experiments) > max_experiments:
            logger.info("Limiting from %d to %d Hi-C experiments (--max-experiments)", len(experiments), max_experiments)
            experiments = experiments[:max_experiments]

        logger.info("Found %d Hi-C experiments to download", len(experiments))
        downloaded = batch_download_hic_experiments(experiments, output_dir, workers)

        for acc, df in process_hic_experiments(
            downloaded,
            resolution=resolution,
            chromosomes=chromosomes,
            include_trans=include_trans,
        ):
            token_df = write_hic_tokens(acc, df, output_dir)
            all_frames.append(token_df)

    return all_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Download ENCODE experiments and tokenize them into peak / "
            "contact-frequency records for use in genomic models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--assay",
        nargs="+",
        choices=list(ASSAY_TITLES.keys()),
        default=list(ASSAY_TITLES.keys()),
        metavar="ASSAY",
        help=(
            "One or more assay types to process. "
            f"Choices: {', '.join(ASSAY_TITLES.keys())}."
        ),
    )
    parser.add_argument(
        "--biosample",
        nargs="+",
        default=None,
        metavar="TERM",
        help=(
            "ENCODE biosample term name(s) to filter experiments "
            "(e.g. K562, GM12878). Omit to download all available."
        ),
    )
    parser.add_argument(
        "--assembly",
        default=DEFAULT_ASSEMBLY,
        help="Genome assembly (ENCODE value, e.g. GRCh38 or hg19).",
    )
    parser.add_argument(
        "--hic-resolution",
        type=int,
        default=DEFAULT_HIC_RESOLUTION,
        metavar="BP",
        help="Hi-C bin size in base pairs.",
    )
    parser.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        metavar="CHROM",
        help=(
            "Restrict Hi-C extraction to these chromosomes "
            "(e.g. chr1 chr2). Defaults to all standard chromosomes."
        ),
    )
    parser.add_argument(
        "--hic-include-trans",
        action="store_true",
        default=False,
        help="Also extract inter-chromosomal (trans) Hi-C contacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel download threads.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of experiments per assay (useful for development / disk constraints).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.getLogger().setLevel(args.log_level)
    logger.info("ENCODE Tracks Tokenizer starting up")
    logger.info("Assays : %s", args.assay)
    logger.info("Biosamples: %s", args.biosample or "(all)")
    logger.info("Assembly  : %s", args.assembly)
    logger.info("Output dir: %s", args.output_dir.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    biosamples = args.biosample or [None]  # None → no biosample filter

    # Collect frames for merged output
    peak_frames: dict[str, list] = {}
    hic_frames: list = []

    for assay in args.assay:
        if assay in PEAK_ASSAYS:
            frames = run_peak_assay(
                assay,
                biosamples,
                args.assembly,
                args.output_dir,
                args.workers,
                max_experiments=args.max_experiments,
            )
            peak_frames.setdefault(assay, []).extend(frames)
        elif assay == "Hi-C":
            frames = run_hic(
                biosamples,
                args.assembly,
                args.output_dir,
                args.workers,
                resolution=args.hic_resolution,
                chromosomes=args.chroms,
                include_trans=args.hic_include_trans,
                max_experiments=args.max_experiments,
            )
            hic_frames.extend(frames)

    # Write merged / summary files
    from tokenizer import merge_and_write_all

    merge_and_write_all(peak_frames, hic_frames, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
