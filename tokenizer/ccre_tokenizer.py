"""
ENCODE cCRE-style regulatory element tokenizer.

Uses the ATAC-seq peak tokens as the universe of candidate cis-regulatory
elements (cCREs) and classifies each peak according to ENCODE Registry v3
rules:

  PLS            H3K4me3_z ≥ 1.64 AND tss_distance ≤ 200 bp
  pELS           H3K4me3_z < 1.64 AND H3K27ac_z ≥ 1.64 AND tss_distance ≤ 2000 bp
  dELS           H3K4me3_z < 1.64 AND H3K27ac_z ≥ 1.64 AND tss_distance > 2000 bp
  CTCF-only      H3K4me3_z < 1.64 AND H3K27ac_z < 1.64 AND CTCF_z ≥ 1.64
  DNase-H3K4me3  H3K4me3_z ≥ 1.64 AND tss_distance > 200 bp
  low-DNase      all signals below threshold (fall-through)

Z-scores are computed per-chromosome (zero-mean, unit-variance across all
ATAC peaks for H3K4me3, H3K27ac, CTCF).

Signal extraction
-----------------
For each ATAC peak [start, end), the mean of the ChIP signal within that
interval is computed using the native 128 bp ChIP resolution.

Output schema
-------------
  token_id          string   — unique ID per peak
  chrom             string
  start, end        int64    — ATAC peak boundaries (0-based half-open)
  summit            int64    — ATAC summit position
  auc               float32  — ATAC peak AUC
  signal_value      float32  — ATAC signal at summit
  ccre_class        string   — one of PLS/pELS/dELS/CTCF-only/DNase-H3K4me3/low-DNase
  tss_distance      int64    — distance to nearest TSS (bp)
  nearest_gene      string   — name of nearest gene
  atac_signal       float32  — mean ATAC signal over peak interval
  h3k4me3_z         float32  — Z-score (classification input)
  h3k27ac_z         float32
  ctcf_z            float32
  <histone mark>    float32  — raw mean ChIP signal for all 10 histone marks
  <TF name>         float32  — raw mean ChIP signal, one col per unique TF
                                (TF experiments are mean-aggregated)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ASSAY_ATAC,
    ASSAY_CHIP_HISTONE,
    ASSAY_CHIP_TF,
    CCRE_Z_THRESHOLD,
    TSS_ELS_DIST,
    TSS_PLS_DIST,
)
from .gtf_utils import TSSIndex

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _bp_range_to_bins(
    start_bp: int,
    end_bp:   int,
    resolution: int,
) -> tuple[int, int]:
    """Convert a [start, end) bp range to [bin_start, bin_end) bin indices."""
    b_start = start_bp // resolution
    b_end   = max(b_start + 1, (end_bp + resolution - 1) // resolution)
    return b_start, b_end


def _mean_signal_over_peaks(
    values:     np.ndarray,      # (genome_bins, num_tracks)
    resolution: int,             # bp per bin
    peaks:      pd.DataFrame,    # columns: start, end (bp, 0-based)
) -> np.ndarray:
    """
    Return an (n_peaks, num_tracks) float32 array of mean ChIP signal over
    each peak interval.
    """
    n_peaks    = len(peaks)
    num_tracks = values.shape[1] if values.ndim == 2 else 1
    if values.ndim == 1:
        values = values[:, np.newaxis]

    result = np.zeros((n_peaks, num_tracks), dtype=np.float32)
    n_bins = values.shape[0]

    for i, (_, row) in enumerate(peaks.iterrows()):
        b_start, b_end = _bp_range_to_bins(int(row["start"]), int(row["end"]), resolution)
        b_start = max(b_start, 0)
        b_end   = min(b_end, n_bins)
        if b_start >= b_end:
            continue
        result[i] = values[b_start:b_end, :].mean(axis=0)

    return result


def _mean_atac_over_peaks(
    values:     np.ndarray,      # (genome_bins, 1)  at 1 bp
    resolution: int,             # always 1 for ATAC
    peaks:      pd.DataFrame,
) -> np.ndarray:
    """Return (n_peaks,) float32 of mean ATAC signal over each peak."""
    n_bins  = values.shape[0]
    n_peaks = len(peaks)
    result  = np.zeros(n_peaks, dtype=np.float32)
    for i, (_, row) in enumerate(peaks.iterrows()):
        s = max(int(row["start"]), 0)
        e = min(int(row["end"]), n_bins)
        if s >= e:
            continue
        track_col = values[:, 0] if values.ndim == 2 else values
        result[i] = float(track_col[s:e].mean())
    return result


# ---------------------------------------------------------------------------
# Column name helpers (shared logic with fixed_bin_tokenizer)
# ---------------------------------------------------------------------------

_HISTONE_RE = re.compile(r"histone\s+chip[-_]?seq\s+", re.IGNORECASE)


def _histone_col_name(meta: dict) -> str:
    name    = meta.get("name", "") or ""
    cleaned = _HISTONE_RE.sub("", name).strip()
    for token in cleaned.split():
        if re.match(r"[Hh]\d+[A-Za-z]\d*", token):
            return token.lower()
    return cleaned.lower().replace(" ", "_") or f"histone_{meta.get('index', 0)}"


def _tf_col_name(meta: dict) -> str:
    tf = meta.get("transcription_factor") or ""
    if tf:
        return tf.lower().replace(" ", "_")
    name = meta.get("name", "") or ""
    for prefix in ("TF ChIP-seq ", "ChIP-seq "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip().lower().replace(" ", "_") or f"tf_{meta.get('index', 0)}"


# ---------------------------------------------------------------------------
# Z-scoring
# ---------------------------------------------------------------------------

def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score a 1-D float array; returns zeros if std ≈ 0."""
    std = float(np.std(arr))
    if std < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - np.mean(arr)) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# ENCODE classification
# ---------------------------------------------------------------------------

def _classify(
    h3k4me3_z:    np.ndarray,
    h3k27ac_z:    np.ndarray,
    ctcf_z:       np.ndarray,
    tss_distance: np.ndarray,
    z_thresh:     float = CCRE_Z_THRESHOLD,
    pls_dist:     int   = TSS_PLS_DIST,
    els_dist:     int   = TSS_ELS_DIST,
) -> np.ndarray:
    """
    Apply ENCODE classification rules to arrays of per-peak Z-scores.

    Returns a string array of cCRE class labels.
    """
    n = len(h3k4me3_z)
    labels = np.full(n, "low-DNase", dtype=object)

    hi_k4me3 = h3k4me3_z >= z_thresh
    hi_k27ac  = h3k27ac_z >= z_thresh
    hi_ctcf   = ctcf_z    >= z_thresh
    near_tss  = tss_distance <= pls_dist
    prox_tss  = tss_distance <= els_dist

    # Rules applied in ascending priority order — later masks overwrite earlier.
    # CTCF-only (no active histone marks, but CTCF enriched)
    mask = (~hi_k4me3) & (~hi_k27ac) & hi_ctcf
    labels[mask] = "CTCF-only"

    # DNase-H3K4me3 (H3K4me3 high, but NOT at a TSS)
    mask = hi_k4me3 & (~near_tss)
    labels[mask] = "DNase-H3K4me3"

    # dELS (H3K27ac high, distal — > 2 kb from TSS)
    mask = (~hi_k4me3) & hi_k27ac & (~prox_tss)
    labels[mask] = "dELS"

    # pELS (H3K27ac high, proximal — ≤ 2 kb from TSS)
    mask = (~hi_k4me3) & hi_k27ac & prox_tss
    labels[mask] = "pELS"

    # PLS (H3K4me3 high, at a TSS — overrides DNase-H3K4me3)
    mask = hi_k4me3 & near_tss
    labels[mask] = "PLS"

    return labels


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_ccre_tokens(
    chrom:          str,
    input_dir:      str | Path,
    peak_tokens_dir: str | Path,
    gtf_cache_dir:  str | Path,
) -> pd.DataFrame:
    """
    Build ENCODE cCRE-style tokens for one chromosome.

    Args:
        chrom:           Chromosome name (e.g. ``"chr22"``).
        input_dir:       Root output directory containing ``atac/``,
                         ``chip_histone/``, ``chip_tf/`` subdirectories.
        peak_tokens_dir: Directory containing per-assay peak token Parquet
                         files (output of the ``peak`` tokenizer scheme).
                         Expected path: ``<peak_tokens_dir>/atac/{chrom}.parquet``.
        gtf_cache_dir:   Directory for caching the GENCODE GTF file.

    Returns:
        DataFrame with the cCRE output schema described in the module docstring.
        Empty DataFrame if no ATAC peaks are found for this chromosome.
    """
    input_dir       = Path(input_dir)
    peak_tokens_dir = Path(peak_tokens_dir)
    gtf_cache_dir   = Path(gtf_cache_dir)

    # ---- Load ATAC peaks (universe of cCREs) --------------------------------
    atac_parquet = peak_tokens_dir / ASSAY_ATAC / f"{chrom}.parquet"
    if not atac_parquet.exists():
        log.warning("[%s] ATAC peak tokens not found: %s — skipping", chrom, atac_parquet)
        return pd.DataFrame()

    atac_peaks = pd.read_parquet(atac_parquet)
    atac_peaks = atac_peaks[atac_peaks["assay"] == ASSAY_ATAC].reset_index(drop=True)

    if atac_peaks.empty:
        log.warning("[%s] no ATAC peaks found — skipping cCRE tokenization", chrom)
        return pd.DataFrame()

    n_peaks = len(atac_peaks)
    log.info("[%s] cCRE — %d ATAC peaks (cCRE universe)", chrom, n_peaks)

    # ---- Load ATAC signal values for mean-signal-over-peak ------------------
    atac_npz  = input_dir / ASSAY_ATAC / f"{chrom}.npz"
    atac_json = input_dir / ASSAY_ATAC / f"{chrom}.json"
    atac_signal_col = np.zeros(n_peaks, dtype=np.float32)

    if atac_npz.exists():
        atac_data = np.load(atac_npz, allow_pickle=False)
        atac_vals = atac_data["values"].astype(np.float32)
        if atac_vals.ndim == 1:
            atac_vals = atac_vals[:, np.newaxis]
        atac_res  = int(atac_data["resolution"])
        atac_signal_col = _mean_atac_over_peaks(atac_vals, atac_res, atac_peaks)
    else:
        log.warning("[%s] ATAC NPZ not found — atac_signal column will be zeros", chrom)

    # ---- Load ChIP histone ---------------------------------------------------
    histone_feature_cols: dict[str, np.ndarray] = {}    # col_name → (n_peaks,)
    h3k4me3_raw  = np.zeros(n_peaks, dtype=np.float32)
    h3k27ac_raw  = np.zeros(n_peaks, dtype=np.float32)

    histone_npz  = input_dir / ASSAY_CHIP_HISTONE / f"{chrom}.npz"
    histone_json = input_dir / ASSAY_CHIP_HISTONE / f"{chrom}.json"

    if histone_npz.exists():
        hdata    = np.load(histone_npz, allow_pickle=False)
        h_values = hdata["values"].astype(np.float32)
        h_res    = int(hdata["resolution"])
        if h_values.ndim == 1:
            h_values = h_values[:, np.newaxis]

        with open(histone_json, "r", encoding="utf-8") as fh:
            h_meta: list[dict] = json.load(fh)

        # Extract mean signal over each ATAC peak for all histone tracks
        h_peak_signals = _mean_signal_over_peaks(h_values, h_res, atac_peaks)
        # (n_peaks, num_histone_tracks)

        # Aggregate per-column (same histone mark may have multiple experiments)
        h_accum: dict[str, list[np.ndarray]] = {}
        for t_idx, meta in enumerate(h_meta):
            col = _histone_col_name(meta)
            h_accum.setdefault(col, []).append(h_peak_signals[:, t_idx])

        for col, arrays in h_accum.items():
            agg = np.stack(arrays).mean(axis=0).astype(np.float32)
            histone_feature_cols[col] = agg

        # Extract key marks for Z-scoring
        if "h3k4me3" in histone_feature_cols:
            h3k4me3_raw = histone_feature_cols["h3k4me3"]
        if "h3k27ac" in histone_feature_cols:
            h3k27ac_raw = histone_feature_cols["h3k27ac"]
    else:
        log.warning("[%s] ChIP-histone NPZ not found — histone columns will be zeros", chrom)

    # ---- Load ChIP TF --------------------------------------------------------
    tf_feature_cols: dict[str, np.ndarray] = {}
    ctcf_raw = np.zeros(n_peaks, dtype=np.float32)

    tf_npz  = input_dir / ASSAY_CHIP_TF / f"{chrom}.npz"
    tf_json = input_dir / ASSAY_CHIP_TF / f"{chrom}.json"

    if tf_npz.exists():
        tdata    = np.load(tf_npz, allow_pickle=False)
        t_values = tdata["values"].astype(np.float32)
        t_res    = int(tdata["resolution"])
        if t_values.ndim == 1:
            t_values = t_values[:, np.newaxis]

        with open(tf_json, "r", encoding="utf-8") as fh:
            t_meta: list[dict] = json.load(fh)

        t_peak_signals = _mean_signal_over_peaks(t_values, t_res, atac_peaks)

        t_accum: dict[str, list[np.ndarray]] = {}
        for t_idx, meta in enumerate(t_meta):
            col = _tf_col_name(meta)
            t_accum.setdefault(col, []).append(t_peak_signals[:, t_idx])

        for col, arrays in t_accum.items():
            agg = np.stack(arrays).mean(axis=0).astype(np.float32)
            tf_feature_cols[col] = agg

        if "ctcf" in tf_feature_cols:
            ctcf_raw = tf_feature_cols["ctcf"]
    else:
        log.warning("[%s] ChIP-TF NPZ not found — TF columns will be zeros", chrom)

    # ---- Z-score classification marks ----------------------------------------
    h3k4me3_z = _zscore(h3k4me3_raw)
    h3k27ac_z = _zscore(h3k27ac_raw)
    ctcf_z    = _zscore(ctcf_raw)

    # ---- TSS distance -------------------------------------------------------
    from .gtf_utils import ensure_gtf  # lazy import to avoid circular deps

    gtf_path = ensure_gtf(gtf_cache_dir)
    tss_index = TSSIndex(gtf_path)

    summit_positions = atac_peaks["summit"].values.astype(np.int64)
    tss_distances, nearest_genes = tss_index.nearest_tss(chrom, summit_positions)

    # ---- Classify ------------------------------------------------------------
    ccre_labels = _classify(h3k4me3_z, h3k27ac_z, ctcf_z, tss_distances)

    # ---- Assemble output DataFrame ------------------------------------------
    df = pd.DataFrame({
        "token_id":    (
            "ccre:" + chrom
            + ":" + atac_peaks["start"].astype(str)
            + ":" + atac_peaks["end"].astype(str)
        ),
        "chrom":        chrom,
        "start":        atac_peaks["start"].values.astype(np.int64),
        "end":          atac_peaks["end"].values.astype(np.int64),
        "summit":       atac_peaks["summit"].values.astype(np.int64),
        "auc":          atac_peaks["auc"].values.astype(np.float32),
        "signal_value": atac_peaks["signal_value"].values.astype(np.float32),
        "ccre_class":   ccre_labels,
        "tss_distance": tss_distances,
        "nearest_gene": nearest_genes,
        "atac_signal":  atac_signal_col,
        "h3k4me3_z":    h3k4me3_z,
        "h3k27ac_z":    h3k27ac_z,
        "ctcf_z":       ctcf_z,
    })

    # Add raw histone signal columns
    for col, arr in histone_feature_cols.items():
        df[col] = arr

    # Add raw TF signal columns
    for col, arr in tf_feature_cols.items():
        df[col] = arr

    # ---- Class distribution summary -----------------------------------------
    dist = df["ccre_class"].value_counts().to_dict()
    log.info(
        "[%s] cCRE classification complete — %d elements: %s",
        chrom, n_peaks, dist,
    )

    return df
