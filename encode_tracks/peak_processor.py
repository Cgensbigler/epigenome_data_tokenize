"""
Peak processing for TF-ChIP-seq and ATAC-seq experiments.

For each experiment this module:
  1. Parses the ENCODE IDR-optimal narrowPeak file.
  2. Computes the absolute summit position (chromStart + peak_offset).
  3. Integrates the fold-change BigWig over the peak interval to produce AUC.
  4. Returns a pandas DataFrame following the PEAK_TOKEN_COLUMNS schema.

narrowPeak format (0-indexed columns):
  0  chrom
  1  chromStart   (0-based)
  2  chromEnd
  3  name
  4  score        (0–1000 scaled)
  5  strand       (. = unstranded)
  6  signalValue  (fold-change at summit; -1 if not available)
  7  pValue       (-log10 p-value; -1 if not available)
  8  qValue       (-log10 q-value; -1 if not available)
  9  peak         (0-based offset from chromStart to summit; -1 if not available)
"""

from __future__ import annotations

import gzip
import logging
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import PEAK_TOKEN_COLUMNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# narrowPeak parser
# ---------------------------------------------------------------------------

_NARROWPEAK_DTYPES = {
    "chrom": str,
    "start": np.int64,
    "end": np.int64,
    "name": str,
    "score": np.int32,
    "strand": str,
    "signal_value": np.float32,
    "p_value": np.float32,
    "q_value": np.float32,
    "peak_offset": np.int32,
}

_NARROWPEAK_COLS = list(_NARROWPEAK_DTYPES.keys())


def _open_maybe_gz(path: Path):
    """Open a plain or gzip-compressed file transparently."""
    if path.suffix == ".gz" or str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_narrowpeak(path: Path) -> pd.DataFrame:
    """
    Parse a narrowPeak file into a DataFrame.

    The returned DataFrame uses the first 10 columns; any extra columns
    in the file are ignored.
    """
    rows: list[list] = []
    with _open_maybe_gz(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                # Tolerate files that omit the peak_offset column
                parts += ["-1"] * (10 - len(parts))
            rows.append(parts[:10])

    if not rows:
        logger.warning("No peaks found in %s", path)
        return pd.DataFrame(columns=_NARROWPEAK_COLS)

    df = pd.DataFrame(rows, columns=_NARROWPEAK_COLS)

    # Cast numeric columns
    for col, dtype in _NARROWPEAK_DTYPES.items():
        if dtype in (np.int64, np.int32, np.float32):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    return df


# ---------------------------------------------------------------------------
# BigWig AUC integration
# ---------------------------------------------------------------------------

def _compute_auc_pyBigWig(bw, chrom: str, start: int, end: int) -> float:
    """
    Integrate the BigWig signal over [start, end) using pyBigWig.

    Strategy: fetch the mean signal over the interval and multiply by its
    length.  pyBigWig returns None when the region has no data (e.g. the
    chromosome is absent); we treat that as 0.
    """
    try:
        mean_val = bw.stats(chrom, start, end, type="mean", exact=True)[0]
    except RuntimeError:
        return 0.0
    if mean_val is None or math.isnan(mean_val):
        return 0.0
    return float(mean_val) * (end - start)


def compute_auc_batch(
    peaks: pd.DataFrame,
    signal_path: Path,
) -> np.ndarray:
    """
    Return a float32 array of AUC values, one per row in *peaks*.

    Requires pyBigWig.  If the signal file is absent or unreadable the
    array is filled with NaN, which the tokenizer treats as missing.
    """
    try:
        import pyBigWig  # noqa: PLC0415
    except ImportError:
        logger.warning("pyBigWig not installed; AUC will be NaN")
        return np.full(len(peaks), np.nan, dtype=np.float32)

    if signal_path is None or not Path(signal_path).exists():
        logger.warning("Signal file missing; AUC will be NaN")
        return np.full(len(peaks), np.nan, dtype=np.float32)

    auc = np.empty(len(peaks), dtype=np.float32)
    bw = pyBigWig.open(str(signal_path))
    try:
        for i, (_, row) in enumerate(peaks.iterrows()):
            auc[i] = _compute_auc_pyBigWig(bw, row["chrom"], int(row["start"]), int(row["end"]))
    finally:
        bw.close()

    return auc


# ---------------------------------------------------------------------------
# Summit computation
# ---------------------------------------------------------------------------

def compute_summits(peaks: pd.DataFrame) -> np.ndarray:
    """
    Return absolute 0-based summit positions.

    If peak_offset == -1 (not available), the midpoint of the peak is used
    as a best estimate.
    """
    offsets = peaks["peak_offset"].to_numpy(dtype=np.int64)
    starts = peaks["start"].to_numpy(dtype=np.int64)
    ends = peaks["end"].to_numpy(dtype=np.int64)

    summits = np.where(offsets >= 0, starts + offsets, (starts + ends) // 2)
    return summits.astype(np.int64)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_peak_experiment(
    info: dict,
    assay: str,
) -> pd.DataFrame | None:
    """
    Process a single peak experiment into a tokenization-ready DataFrame.

    *info* must have keys: accession, target, biosample, peak_path,
    signal_path (may be None).

    Returns None on failure.
    """
    acc = info["accession"]
    peak_path = info.get("peak_path")

    if peak_path is None or not Path(peak_path).exists():
        logger.warning("Skipping %s — peak file not found", acc)
        return None

    logger.info("Processing peaks for %s (%s)", acc, assay)

    peaks = parse_narrowpeak(Path(peak_path))
    if peaks.empty:
        logger.warning("No peaks in %s", acc)
        return None

    summits = compute_summits(peaks)
    auc = compute_auc_batch(peaks, info.get("signal_path"))

    out = pd.DataFrame({
        "chrom": peaks["chrom"],
        "start": peaks["start"],
        "end": peaks["end"],
        "summit": summits,
        "auc": auc,
        "signal_value": peaks["signal_value"],
        "p_value": peaks["p_value"],
        "q_value": peaks["q_value"],
        "strand": peaks["strand"],
        "experiment_accession": acc,
        "target": info.get("target", ""),
        "biosample": info.get("biosample", ""),
        "assay": assay,
    })

    # Ensure column order matches schema
    missing = [c for c in PEAK_TOKEN_COLUMNS if c not in out.columns]
    for col in missing:
        out[col] = ""

    out = out[PEAK_TOKEN_COLUMNS]

    logger.info(
        "%s: %d peaks processed (AUC missing: %d)",
        acc,
        len(out),
        int(np.isnan(auc).sum()),
    )
    return out


def process_peak_experiments(
    experiments: list[dict],
    assay: str,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Yield (experiment_accession, DataFrame) for each successfully processed
    experiment.
    """
    for info in experiments:
        df = process_peak_experiment(info, assay)
        if df is not None:
            yield info["accession"], df
