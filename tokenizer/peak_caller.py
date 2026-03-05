"""
ENCODE-equivalent peak calling on AlphaGenome predicted 1D tracks.

Strategy mirrors MACS2 `--call-summits` behaviour:
  1. (ATAC only) Mean-pool from 1 bp to ATAC_DOWNSAMPLE_RESOLUTION bp.
  2. Gaussian-smooth the signal (σ = SMOOTH_SIGMA_BINS).
  3. scipy.signal.find_peaks with prominence + width + distance filters.
  4. scipy.signal.peak_widths at half-prominence to define [start, end).
  5. AUC = numpy.trapz(signal[start:end]) × resolution.

All tracks in a chromosome are processed independently and the results are
concatenated into a single DataFrame written as one Parquet file per chrom.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from .config import (
    ATAC_DOWNSAMPLE_RESOLUTION,
    MIN_PEAK_DISTANCE_BP,
    MIN_PEAK_WIDTH_BINS,
    MIN_PROMINENCE_ABSOLUTE,
    PEAK_TOKEN_COLUMNS,
    PEAK_WIDTH_REL_HEIGHT,
    PROMINENCE_FRACTION,
    SMOOTH_SIGMA_BINS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal pre-processing
# ---------------------------------------------------------------------------

def _mean_pool(signal: np.ndarray, factor: int) -> np.ndarray:
    """Down-sample *signal* by *factor* using non-overlapping mean pooling."""
    n = len(signal)
    trimmed = n - (n % factor)          # drop last partial bin
    return signal[:trimmed].reshape(-1, factor).mean(axis=1).astype(np.float32)


def _smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 1-D Gaussian filter.  Returns float32."""
    if sigma <= 0:
        return signal.astype(np.float32)
    return gaussian_filter1d(signal.astype(np.float64), sigma=sigma).astype(np.float32)


def _preprocess(
    raw: np.ndarray,
    native_resolution: int,
    target_resolution: int,
) -> tuple[np.ndarray, int]:
    """
    Down-sample (if needed) and smooth the raw signal.

    Returns ``(processed_signal, effective_resolution)`` where
    ``effective_resolution`` is the bp-per-bin after any down-sampling.
    """
    if target_resolution > native_resolution:
        factor = target_resolution // native_resolution
        signal = _mean_pool(raw, factor)
        res = target_resolution
    else:
        signal = raw.astype(np.float32)
        res = native_resolution

    signal = _smooth(signal, SMOOTH_SIGMA_BINS)
    return signal, res


# ---------------------------------------------------------------------------
# Peak calling on a single 1-D signal vector
# ---------------------------------------------------------------------------

def _call_peaks_1d(
    signal: np.ndarray,
    resolution: int,
    chrom: str,
    chrom_offset: int,
) -> pd.DataFrame:
    """
    Call peaks on a 1-D float32 array and return a DataFrame of peak tokens.

    Args:
        signal:       Preprocessed 1-D signal array (one bin per element).
        resolution:   Bin size in bp.
        chrom:        Chromosome name (stored in output).
        chrom_offset: Genomic position of bin 0 (bp).  Always 0 for stitched
                      whole-chrom arrays, but exposed for future partial use.

    Returns:
        DataFrame with columns matching ``PEAK_TOKEN_COLUMNS`` (minus
        ``token_id``, ``track_name``, ``assay`` — filled by the caller).
    """
    if len(signal) == 0 or np.all(signal <= 0):
        return pd.DataFrame(columns=["chrom", "start", "end", "summit",
                                     "auc", "signal_value", "prominence"])

    nonzero = signal[signal > 0]
    if len(nonzero) == 0:
        return pd.DataFrame(columns=["chrom", "start", "end", "summit",
                                     "auc", "signal_value", "prominence"])

    median_nonzero = float(np.median(nonzero))
    min_prominence = max(
        PROMINENCE_FRACTION * median_nonzero,
        MIN_PROMINENCE_ABSOLUTE,
    )
    min_distance_bins = max(1, math.ceil(MIN_PEAK_DISTANCE_BP / resolution))

    peaks_idx, properties = find_peaks(
        signal,
        prominence=min_prominence,
        width=MIN_PEAK_WIDTH_BINS,
        distance=min_distance_bins,
    )

    if len(peaks_idx) == 0:
        return pd.DataFrame(columns=["chrom", "start", "end", "summit",
                                     "auc", "signal_value", "prominence"])

    # Width at half-prominence → peak boundaries
    widths, _, left_ips, right_ips = peak_widths(
        signal, peaks_idx, rel_height=PEAK_WIDTH_REL_HEIGHT
    )

    # Convert fractional bin indices to integer bin indices, then to bp
    left_bins  = np.floor(left_ips).astype(np.int64)
    right_bins = np.ceil(right_ips).astype(np.int64)
    right_bins = np.clip(right_bins, left_bins + 1, len(signal))

    starts   = chrom_offset + left_bins  * resolution
    ends     = chrom_offset + right_bins * resolution
    summits  = chrom_offset + peaks_idx  * resolution

    # AUC: trapezoid integral (signal × bin_width) over [left_bin, right_bin)
    auc = np.array([
        float(np.trapz(signal[l:r]) * resolution)
        for l, r in zip(left_bins, right_bins)
    ], dtype=np.float32)

    signal_at_summit = signal[peaks_idx].astype(np.float32)
    prominences      = properties["prominences"].astype(np.float32)

    return pd.DataFrame({
        "chrom":        chrom,
        "start":        starts.astype(np.int64),
        "end":          ends.astype(np.int64),
        "summit":       summits.astype(np.int64),
        "auc":          auc,
        "signal_value": signal_at_summit,
        "prominence":   prominences,
    })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def call_peaks_chrom(
    npz_path: Path,
    json_path: Path,
    assay: str,
    atac_downsample_res: int = ATAC_DOWNSAMPLE_RESOLUTION,
) -> pd.DataFrame:
    """
    Call peaks on all tracks for one chromosome and return a combined DataFrame.

    Args:
        npz_path:           Path to ``{chrom}.npz`` (from output_writers.save_chrom_npz).
        json_path:          Path to ``{chrom}.json`` metadata sidecar.
        assay:              One of ``"atac"``, ``"chip_histone"``, ``"chip_tf"``.
        atac_downsample_res: Target resolution for ATAC down-sampling (bp).

    Returns:
        DataFrame with all columns from ``PEAK_TOKEN_COLUMNS``.  Empty if no
        peaks found.
    """
    # ---- load data ---------------------------------------------------------
    data       = np.load(npz_path, allow_pickle=False)
    values     = data["values"]          # (genome_bins, num_tracks)
    resolution = int(data["resolution"])
    chrom      = str(data["chrom"])
    chrom_size = int(data["chrom_size"])

    with open(json_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)        # list of dicts, one per track

    num_tracks = values.shape[1] if values.ndim == 2 else 1
    if values.ndim == 1:
        values = values[:, np.newaxis]

    if num_tracks != len(metadata):
        log.warning(
            "%s: track count mismatch — values has %d tracks, json has %d entries. "
            "Using min of the two.",
            npz_path.name, num_tracks, len(metadata),
        )
        num_tracks = min(num_tracks, len(metadata))

    # ---- decide effective resolution after any down-sampling ---------------
    if assay == "atac" and atac_downsample_res > resolution:
        target_res = atac_downsample_res
    else:
        target_res = resolution

    log.info(
        "[%s] %s — %d tracks at native %d bp → calling at %d bp",
        chrom, assay, num_tracks, resolution, target_res,
    )

    # ---- process each track ------------------------------------------------
    frames: list[pd.DataFrame] = []

    for t_idx in range(num_tracks):
        track_name = metadata[t_idx].get("name", f"track_{t_idx}")
        raw_signal = values[:, t_idx]

        processed, eff_res = _preprocess(raw_signal, resolution, target_res)

        df = _call_peaks_1d(
            signal=processed,
            resolution=eff_res,
            chrom=chrom,
            chrom_offset=0,
        )

        if df.empty:
            continue

        df["track_name"] = track_name
        df["assay"]      = assay
        df["token_id"]   = (
            assay + ":" + track_name + ":" + chrom
            + ":" + df["start"].astype(str)
            + ":" + df["end"].astype(str)
        )

        frames.append(df)

    if not frames:
        log.warning("[%s] %s — no peaks found in any track.", chrom, assay)
        return pd.DataFrame(columns=PEAK_TOKEN_COLUMNS)

    result = pd.concat(frames, ignore_index=True)

    # Ensure column order matches schema
    for col in PEAK_TOKEN_COLUMNS:
        if col not in result.columns:
            result[col] = ""
    result = result[PEAK_TOKEN_COLUMNS]

    log.info(
        "[%s] %s — %d peaks across %d tracks",
        chrom, assay, len(result), num_tracks,
    )
    return result
