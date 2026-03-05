"""
Output writers for tokenized peak and contact data.

Each chromosome is written as a pair of files:
    {out_dir}/{assay}/{chrom}.parquet   — compressed columnar store (snappy)
    {out_dir}/{assay}/{chrom}.tsv       — human-readable summary (≤ TSV_MAX_ROWS rows)

A per-assay merged Parquet is also produced after all chromosomes are written:
    {out_dir}/{assay}/all_chroms.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import TSV_MAX_ROWS

log = logging.getLogger(__name__)


def write_tokens(
    df: pd.DataFrame,
    out_dir: Path,
    chrom: str,
) -> tuple[Path, Path]:
    """
    Write a token DataFrame to ``{out_dir}/{chrom}.parquet`` and
    ``{out_dir}/{chrom}.tsv``.

    Args:
        df:      Token DataFrame (peak or contact).
        out_dir: Directory for this assay (created if absent).
        chrom:   Chromosome name (used as the base filename).

    Returns:
        Tuple of ``(parquet_path, tsv_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{chrom}.parquet"
    tsv_path     = out_dir / f"{chrom}.tsv"

    df.to_parquet(parquet_path, index=False, compression="snappy")
    df.head(TSV_MAX_ROWS).to_csv(tsv_path, sep="\t", index=False)

    log.info(
        "Wrote %d tokens → %s  (%.1f MB)",
        len(df),
        parquet_path,
        parquet_path.stat().st_size / 1e6,
    )
    return parquet_path, tsv_path


def merge_chroms(
    out_dir: Path,
    chroms: list[str],
) -> Path | None:
    """
    Concatenate per-chromosome Parquet files into a single
    ``all_chroms.parquet`` in *out_dir*.

    Skips missing chromosome files without raising.  Returns the merged path
    or None if no files were found.
    """
    frames: list[pd.DataFrame] = []
    for chrom in chroms:
        p = out_dir / f"{chrom}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            log.debug("merge_chroms: %s not found, skipping.", p)

    if not frames:
        log.warning("merge_chroms: no chromosome files found in %s", out_dir)
        return None

    merged = pd.concat(frames, ignore_index=True)
    out_path = out_dir / "all_chroms.parquet"
    merged.to_parquet(out_path, index=False, compression="snappy")

    log.info(
        "Merged %d chroms → %s  (%d tokens, %.1f MB)",
        len(frames),
        out_path,
        len(merged),
        out_path.stat().st_size / 1e6,
    )
    return out_path
