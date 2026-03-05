"""
Contact-map tokenizer for AlphaGenome predicted DNA contact frequencies.

Each per-window NPZ (output/contact_maps/{chrom}_{start}_{end}.npz) holds a
2-D predicted contact matrix of shape (bins, bins, num_tracks).  Windows
overlap by (window_size - stride) bp, so naïve concatenation would
double-count border regions.

De-duplication strategy
-----------------------
The same center-stride logic used in stitching.py is applied: for each
window only contacts where **both** anchor bins fall inside the "kept" genomic
slice are retained.

  Window i at [win_start, win_end):
    overlap = win_end - win_start - stride
    first window  → keep bins whose genomic start ∈ [0, stride)
    middle window → keep bins ∈ [win_start + overlap//2, win_start + overlap//2 + stride)
    last window   → keep bins ∈ [win_end - stride, win_end)

After de-duplication contacts below CONTACT_MIN_FREQ are dropped and the
surviving entries are written as a long-format sparse DataFrame.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONTACT_MIN_FREQ, CONTACT_TOKEN_COLUMNS

log = logging.getLogger(__name__)

# Regex to parse filenames like "chr1_0_1048576.npz"
_WINDOW_RE = re.compile(r"^(?P<chrom>[^_]+(?:_[^_]+)?)_(?P<start>\d+)_(?P<end>\d+)\.npz$")


def _parse_window_filename(path: Path) -> tuple[str, int, int] | None:
    """Return (chrom, start, end) parsed from a contact-map NPZ filename."""
    m = _WINDOW_RE.match(path.name)
    if m is None:
        return None
    return m.group("chrom"), int(m.group("start")), int(m.group("end"))


def _kept_genomic_range(
    win_start: int,
    win_end: int,
    stride: int,
    is_first: bool,
    is_last: bool,
) -> tuple[int, int]:
    """
    Return the [keep_start, keep_end) genomic bp range for this window.

    Mirrors the logic in stitching.stitch_1d_tracks.
    """
    win_len = win_end - win_start
    if is_first and is_last:
        return win_start, win_end
    if is_first:
        return win_start, min(win_start + stride, win_end)
    if is_last:
        return max(win_end - stride, win_start), win_end
    overlap = win_len - stride
    keep_start = win_start + overlap // 2
    return keep_start, keep_start + stride


def build_contact_tokens(
    contact_map_dir: Path,
    chrom: str,
    stride: int,
) -> pd.DataFrame:
    """
    Build a sparse long-format contact token DataFrame for one chromosome.

    Args:
        contact_map_dir: Directory containing per-window NPZ files.
        chrom:           Chromosome name (e.g. ``"chr22"``).
        stride:          The stride used when generating prediction windows (bp).

    Returns:
        DataFrame with columns from ``CONTACT_TOKEN_COLUMNS``.  Empty if no
        windows found or all contacts are below threshold.
    """
    # ---- discover and sort windows for this chrom --------------------------
    window_files = sorted(
        [p for p in contact_map_dir.glob(f"{chrom}_*.npz")
         if _parse_window_filename(p) is not None
         and _parse_window_filename(p)[0] == chrom],
        key=lambda p: _parse_window_filename(p)[1],
    )

    if not window_files:
        log.warning("[%s] no contact-map window files found in %s", chrom, contact_map_dir)
        return pd.DataFrame(columns=CONTACT_TOKEN_COLUMNS)

    log.info("[%s] contact maps — %d windows to process", chrom, len(window_files))

    n = len(window_files)
    all_frames: list[pd.DataFrame] = []

    for i, npz_path in enumerate(window_files):
        parsed = _parse_window_filename(npz_path)
        if parsed is None:
            continue
        _, win_start, win_end = parsed

        is_first = (i == 0)
        is_last  = (i == n - 1)

        keep_start, keep_end = _kept_genomic_range(
            win_start, win_end, stride, is_first, is_last
        )

        # ---- load window NPZ -----------------------------------------------
        data = np.load(npz_path, allow_pickle=False)
        values       = data["values"]         # (bins, bins, num_tracks) float32
        resolution   = int(data["resolution"])
        track_names  = data["track_names"].tolist()   # list[str]

        if values.ndim != 3:
            log.warning("%s: unexpected shape %s, skipping", npz_path.name, values.shape)
            continue

        n_bins, _, num_tracks = values.shape

        if num_tracks == 0:
            # Contact maps were filtered to 0 tracks — this happens when
            # ontology_terms was restricted to a cell line (e.g. K562/EFO:0002067)
            # that has no contact-map tracks in the model.  Re-run the prediction
            # pipeline with ontology_terms=None to obtain contact map predictions.
            log.debug(
                "%s: 0 contact-map tracks (ontology filter likely excluded all tracks).",
                npz_path.name,
            )
            continue

        # ---- compute which bins to keep ------------------------------------
        # Bin i covers [win_start + i*resolution, win_start + (i+1)*resolution)
        bin_starts = win_start + np.arange(n_bins, dtype=np.int64) * resolution
        keep_mask  = (bin_starts >= keep_start) & (bin_starts < keep_end)
        kept_indices = np.where(keep_mask)[0]

        if len(kept_indices) == 0:
            continue

        # ---- extract contacts where BOTH anchors are in kept slice ---------
        # Build coordinate grids for the sub-matrix
        sub = values[np.ix_(kept_indices, kept_indices)]   # (k, k, num_tracks)
        k   = len(kept_indices)

        # Only keep upper-triangle (bin1 ≤ bin2) to avoid duplicates
        row_idx, col_idx = np.triu_indices(k)

        for t_idx, track_name in enumerate(track_names):
            freq_flat = sub[row_idx, col_idx, t_idx]

            # Filter noise
            nonzero_mask = freq_flat >= CONTACT_MIN_FREQ
            if not np.any(nonzero_mask):
                continue

            r = row_idx[nonzero_mask]
            c = col_idx[nonzero_mask]
            f = freq_flat[nonzero_mask].astype(np.float32)

            bin1_starts = bin_starts[kept_indices[r]]
            bin2_starts = bin_starts[kept_indices[c]]
            bin1_ends   = bin1_starts + resolution
            bin2_ends   = bin2_starts + resolution

            token_ids = (
                "contact:" + track_name + ":" + chrom
                + ":" + pd.array(bin1_starts, dtype="int64").astype(str)
                + ":" + pd.array(bin2_starts, dtype="int64").astype(str)
            )

            frame = pd.DataFrame({
                "token_id":    token_ids,
                "chrom":       chrom,
                "bin1_start":  bin1_starts.astype(np.int64),
                "bin1_end":    bin1_ends.astype(np.int64),
                "bin2_start":  bin2_starts.astype(np.int64),
                "bin2_end":    bin2_ends.astype(np.int64),
                "contact_freq": f,
                "track_name":  track_name,
                "resolution":  np.int64(resolution),
            })
            all_frames.append(frame)

        log.debug(
            "[%s] window %d/%d (%d–%d) — kept %d bins",
            chrom, i + 1, n, win_start, win_end, len(kept_indices),
        )

    if not all_frames:
        log.warning(
            "[%s] contact maps — no contacts found.  "
            "If all windows have 0 tracks, re-run the prediction pipeline "
            "without an ontology filter (ontology_terms=None) to get contact "
            "map predictions for all available tissues.",
            chrom,
        )
        return pd.DataFrame(columns=CONTACT_TOKEN_COLUMNS)

    result = pd.concat(all_frames, ignore_index=True)

    # Enforce column order
    for col in CONTACT_TOKEN_COLUMNS:
        if col not in result.columns:
            result[col] = ""
    result = result[CONTACT_TOKEN_COLUMNS]

    log.info(
        "[%s] contact maps — %d contact tokens across %d tracks",
        chrom, len(result), result["track_name"].nunique(),
    )
    return result
