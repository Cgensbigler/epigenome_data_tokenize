"""
Tokenizer — writes processed peaks and Hi-C contacts to disk.

Output layout inside <output_dir>/processed/:
    <assay_slug>/<experiment_accession>/tokens.parquet   (primary)
    <assay_slug>/<experiment_accession>/tokens.tsv       (human-readable summary)
    <assay_slug>/all_experiments.parquet                 (merged, assay-level)

Each Parquet file is compressed with snappy for fast columnar access.
TSV files contain the same data but capped at 10 000 rows for readability.

Design notes
------------
- All coordinate columns are stored as int64 (positions) or float32 (scores).
- String columns (chrom, assay, biosample, target) are stored as Parquet
  dictionary-encoded for space efficiency.
- A `token_id` column is added: a stable string key formatted as
  `{assay}:{accession}:{chrom}:{start}:{end}` for peak tokens, and
  `{assay}:{accession}:{chrom1}:{start1}:{chrom2}:{start2}` for Hi-C.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import HIC_TOKEN_COLUMNS, PEAK_TOKEN_COLUMNS, PROCESSED_SUBDIR

logger = logging.getLogger(__name__)

TSV_MAX_ROWS = 10_000


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _assay_slug(assay: str) -> str:
    return assay.lower().replace(" ", "_").replace("-", "_")


def _experiment_out_dir(output_dir: Path, assay: str, accession: str) -> Path:
    return output_dir / PROCESSED_SUBDIR / _assay_slug(assay) / accession


# ---------------------------------------------------------------------------
# token_id generation
# ---------------------------------------------------------------------------

def _add_peak_token_ids(df: pd.DataFrame, assay: str) -> pd.DataFrame:
    """Add a stable token_id column to a peak DataFrame."""
    df = df.copy()
    df.insert(
        0,
        "token_id",
        (
            _assay_slug(assay)
            + ":"
            + df["experiment_accession"]
            + ":"
            + df["chrom"]
            + ":"
            + df["start"].astype(str)
            + ":"
            + df["end"].astype(str)
        ),
    )
    return df


def _add_hic_token_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable token_id column to a Hi-C DataFrame."""
    df = df.copy()
    df.insert(
        0,
        "token_id",
        (
            "hic:"
            + df["experiment_accession"]
            + ":"
            + df["chrom1"]
            + ":"
            + df["start1"].astype(str)
            + ":"
            + df["chrom2"]
            + ":"
            + df["start2"].astype(str)
        ),
    )
    return df


# ---------------------------------------------------------------------------
# Type coercion to stable dtypes
# ---------------------------------------------------------------------------

_PEAK_INT_COLS = ["start", "end", "summit"]
_PEAK_FLOAT_COLS = ["auc", "signal_value", "p_value", "q_value"]
_HIC_INT_COLS = ["start1", "end1", "start2", "end2", "resolution"]
_HIC_FLOAT_COLS = ["contact_freq"]


def _coerce_peak_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _PEAK_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
    for col in _PEAK_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df


def _coerce_hic_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _HIC_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
    for col in _HIC_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_experiment(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "tokens.parquet"
    tsv_path = out_dir / "tokens.tsv"

    df.to_parquet(parquet_path, index=False, compression="snappy")
    df.head(TSV_MAX_ROWS).to_csv(tsv_path, sep="\t", index=False)

    logger.info(
        "Wrote %d tokens → %s (parquet: %.1f MB)",
        len(df),
        out_dir,
        parquet_path.stat().st_size / 1e6,
    )


def _write_merged(frames: list[pd.DataFrame], out_dir: Path, name: str) -> None:
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{name}.parquet"
    merged.to_parquet(parquet_path, index=False, compression="snappy")
    logger.info(
        "Wrote merged %s: %d tokens → %s (%.1f MB)",
        name,
        len(merged),
        parquet_path,
        parquet_path.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_peak_tokens(
    accession: str,
    df: pd.DataFrame,
    assay: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Finalise and write peak tokens for one experiment.

    Returns the token DataFrame (with token_id prepended) for downstream
    merging.
    """
    df = _coerce_peak_dtypes(df)
    df = _add_peak_token_ids(df, assay)

    # Ensure all schema columns are present
    for col in ["token_id"] + PEAK_TOKEN_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    out_dir = _experiment_out_dir(output_dir, assay, accession)
    _write_experiment(df, out_dir)
    return df


def write_hic_tokens(
    accession: str,
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Finalise and write Hi-C contact tokens for one experiment.

    Returns the token DataFrame (with token_id prepended).
    """
    df = _coerce_hic_dtypes(df)
    df = _add_hic_token_ids(df)

    for col in ["token_id"] + HIC_TOKEN_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    out_dir = _experiment_out_dir(output_dir, "Hi-C", accession)
    _write_experiment(df, out_dir)
    return df


def merge_and_write_all(
    peak_frames: dict[str, list[pd.DataFrame]],  # {assay: [dfs]}
    hic_frames: list[pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Write assay-level merged Parquet files and a global summary TSV.

    peak_frames keys are assay names (e.g. "TF-ChIP-seq", "ATAC-seq").
    """
    processed_dir = output_dir / PROCESSED_SUBDIR

    # Per-assay merged peaks
    for assay, frames in peak_frames.items():
        assay_dir = processed_dir / _assay_slug(assay)
        _write_merged(frames, assay_dir, "all_experiments")

    # Merged Hi-C
    if hic_frames:
        hic_dir = processed_dir / "hi_c"
        _write_merged(hic_frames, hic_dir, "all_experiments")

    # Global summary TSV (one row per experiment)
    summary_rows: list[dict] = []

    for assay, frames in peak_frames.items():
        for df in frames:
            if df.empty:
                continue
            row = {
                "assay": assay,
                "experiment_accession": df["experiment_accession"].iloc[0],
                "biosample": df["biosample"].iloc[0],
                "target": df["target"].iloc[0] if "target" in df.columns else "",
                "n_tokens": len(df),
                "n_chroms": df["chrom"].nunique(),
                "median_auc": float(df["auc"].median()) if "auc" in df.columns else float("nan"),
            }
            summary_rows.append(row)

    for df in hic_frames:
        if df.empty:
            continue
        row = {
            "assay": "Hi-C",
            "experiment_accession": df["experiment_accession"].iloc[0],
            "biosample": df["biosample"].iloc[0],
            "target": "",
            "n_tokens": len(df),
            "n_chroms": df["chrom1"].nunique(),
            "median_auc": float("nan"),
        }
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = processed_dir / "summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        logger.info("Summary written to %s (%d experiments)", summary_path, len(summary_df))
