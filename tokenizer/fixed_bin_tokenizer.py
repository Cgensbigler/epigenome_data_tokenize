"""
Fixed-bin feature matrix tokenizer.

Produces a dense feature matrix where every row is one genomic bin and every
column is one predicted signal track.  The four supported resolutions are:

  128 bp  — ATAC pooled 128×,  ChIP mean-pooled to 128 bp (identity)
  256 bp  — ATAC pooled 256×,  ChIP mean-pooled 2×
  512 bp  — ATAC pooled 512×,  ChIP mean-pooled 4×
  2048 bp — ATAC pooled 2048×, ChIP mean-pooled 16×, + contact map columns

Contact maps are only included at the 2048 bp resolution (their native
resolution in the AlphaGenome predictions).  Two contact-summary modes are
supported via the ``contact_mode`` argument:

  "summary" (default)
    One column ``contact_total`` per contact-map track: the sum of contact
    frequencies for each bin with all other bins in the chromosome.  This
    keeps the column count fixed regardless of chromosome size.

  "full"
    Stores the complete contact vector per bin as a nested numpy array.
    Useful for downstream models that need pairwise relationships; note that
    downstream Parquet readers must support nested arrays.

Column naming
-------------
  ATAC         → "atac"
  Histone ChIP → normalised histone mark name, e.g. "h3k27ac", "h3k4me3"
  TF ChIP      → ``transcription_factor`` field lowercased, e.g. "ctcf", "sp1"
                 Multiple tracks for the same TF are mean-aggregated into one col.
  Contact      → "contact_total" (summary) or "contact_vec" (full) per track,
                 prefixed with track name.
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
    ASSAY_CONTACT_MAPS,
    CONTACT_MAP_RESOLUTION,
    CONTACT_MIN_FREQ,
    FIXED_BIN_RESOLUTIONS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def _mean_pool(signal: np.ndarray, factor: int) -> np.ndarray:
    """Down-sample a 1-D signal by *factor* via non-overlapping mean pooling."""
    if factor <= 1:
        return signal.astype(np.float32)
    n = len(signal)
    trimmed = n - (n % factor)
    return signal[:trimmed].reshape(-1, factor).mean(axis=1).astype(np.float32)


def _align_bins(
    values: np.ndarray,        # (genome_bins_src, num_tracks)
    src_res: int,
    tgt_res: int,
    chrom_size: int,
) -> np.ndarray:
    """
    Return a (n_bins_tgt, num_tracks) float32 array aligned to *tgt_res*.

    If src_res == tgt_res: identity.
    If src_res < tgt_res:  mean-pool by factor = tgt_res // src_res.
    """
    if src_res == tgt_res:
        return values.astype(np.float32)

    if src_res > tgt_res:
        raise ValueError(
            f"Source resolution {src_res} bp > target {tgt_res} bp; "
            "upsampling is not supported in fixed-bin tokenizer."
        )

    factor = tgt_res // src_res
    num_tracks = values.shape[1] if values.ndim == 2 else 1
    if values.ndim == 1:
        values = values[:, np.newaxis]

    n_bins_tgt = len(range(0, chrom_size, tgt_res))
    out = np.zeros((n_bins_tgt, num_tracks), dtype=np.float32)

    for t in range(num_tracks):
        pooled = _mean_pool(values[:, t], factor)
        copy_len = min(len(pooled), n_bins_tgt)
        out[:copy_len, t] = pooled[:copy_len]

    return out


# ---------------------------------------------------------------------------
# Column name extraction from JSON sidecars
# ---------------------------------------------------------------------------

_HISTONE_RE = re.compile(r"histone\s+chip[-_]?seq\s+", re.IGNORECASE)


def _histone_col_name(track_meta: dict) -> str:
    """
    Derive a short histone mark column name from the track metadata dict.

    Examples:
      "Histone ChIP-seq H3K27ac"      → "h3k27ac"
      "Histone ChIP-seq H3K4me3"      → "h3k4me3"
      "H4K20me1 Histone ChIP-seq ..."  → "h4k20me1"
    """
    name = track_meta.get("name", "") or ""
    # Strip "Histone ChIP-seq" prefix/suffix and take remaining token
    cleaned = _HISTONE_RE.sub("", name).strip()
    # Use the first whitespace-delimited word that looks like a histone mark
    for token in cleaned.split():
        if re.match(r"[Hh]\d+[A-Za-z]\d*", token):
            return token.lower()
    # Fallback: use the whole cleaned name lowercased
    return cleaned.lower().replace(" ", "_") or f"histone_{track_meta.get('index', 0)}"


def _tf_col_name(track_meta: dict) -> str:
    """
    Derive a TF column name from the track metadata dict.

    Prefers the ``transcription_factor`` field; falls back to parsing the
    track name.
    """
    tf = track_meta.get("transcription_factor") or ""
    if tf:
        return tf.lower().replace(" ", "_")
    # Fallback: parse from name
    name = track_meta.get("name", "") or ""
    # Strip common prefixes
    for prefix in ("TF ChIP-seq ", "ChIP-seq "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip().lower().replace(" ", "_") or f"tf_{track_meta.get('index', 0)}"


def _build_1d_columns(
    values: np.ndarray,           # (n_bins_tgt, num_tracks)
    metadata: list[dict],
    assay: str,
) -> dict[str, np.ndarray]:
    """
    Return a {col_name: 1-D array} dict for one assay's values.

    For TF ChIP-seq, tracks with the same TF name are mean-aggregated into
    one column.
    """
    if assay == ASSAY_ATAC:
        if values.ndim == 2:
            # ATAC is always a single track; average if multiple exist
            return {"atac": values.mean(axis=1)}
        return {"atac": values}

    if assay == ASSAY_CHIP_HISTONE:
        cols: dict[str, np.ndarray] = {}
        for t_idx, meta in enumerate(metadata):
            col = _histone_col_name(meta)
            arr = values[:, t_idx] if values.ndim == 2 else values
            if col in cols:
                cols[col] = (cols[col] + arr) / 2.0  # incremental mean
            else:
                cols[col] = arr.copy()
        return cols

    if assay == ASSAY_CHIP_TF:
        # Aggregate duplicate TFs
        accum: dict[str, list[np.ndarray]] = {}
        for t_idx, meta in enumerate(metadata):
            col = _tf_col_name(meta)
            arr = values[:, t_idx] if values.ndim == 2 else values
            accum.setdefault(col, []).append(arr)
        return {col: np.stack(arrs).mean(axis=0) for col, arrs in accum.items()}

    raise ValueError(f"Unknown assay: {assay!r}")


# ---------------------------------------------------------------------------
# Contact map loading and summarisation (2048 bp only)
# ---------------------------------------------------------------------------

_CONTACT_WINDOW_RE = re.compile(
    r"^(?P<chrom>[^_]+(?:_[^_]+)?)_(?P<start>\d+)_(?P<end>\d+)\.npz$"
)


def _load_contact_summary(
    contact_map_dir: Path,
    chrom: str,
    n_bins: int,
    resolution: int = CONTACT_MAP_RESOLUTION,
    mode: str = "summary",
) -> dict[str, np.ndarray]:
    """
    Build contact-summary columns for the feature matrix.

    For each contact-map track, sums all contact frequencies for each bin
    (``mode="summary"``) or stores the full contact vector (``mode="full"``).

    Returns dict of {col_name: array(n_bins, ...)} or empty dict if no
    contact map files are found for this chromosome.
    """
    files = sorted(
        [p for p in contact_map_dir.glob(f"{chrom}_*.npz")
         if _CONTACT_WINDOW_RE.match(p.name)
         and _CONTACT_WINDOW_RE.match(p.name).group("chrom") == chrom],
        key=lambda p: int(_CONTACT_WINDOW_RE.match(p.name).group("start")),
    )

    if not files:
        log.warning(
            "[%s] no contact-map NPZ files found in %s — contact columns omitted",
            chrom, contact_map_dir,
        )
        return {}

    # Determine number of tracks from first file
    first_data = np.load(files[0], allow_pickle=False)
    first_values = first_data["values"]   # (bins, bins, num_tracks)
    if first_values.ndim != 3:
        log.warning("[%s] unexpected contact map shape %s — skipping", chrom, first_values.shape)
        return {}

    num_tracks = first_values.shape[2]
    if num_tracks == 0:
        log.warning(
            "[%s] contact maps have 0 tracks (ontology filter excluded all tracks).",
            chrom,
        )
        return {}

    try:
        track_names = first_data["track_names"].tolist()
    except KeyError:
        track_names = [f"contact_track_{i}" for i in range(num_tracks)]

    if mode == "summary":
        contact_sums = np.zeros((n_bins, num_tracks), dtype=np.float64)
    else:
        contact_vecs: list[list[np.ndarray]] = [[] for _ in range(num_tracks)]

    for npz_path in files:
        m = _CONTACT_WINDOW_RE.match(npz_path.name)
        win_start = int(m.group("start"))

        data = np.load(npz_path, allow_pickle=False)
        mat = data["values"]   # (bins, bins, num_tracks)
        if mat.ndim != 3 or mat.shape[2] != num_tracks:
            continue

        n_win_bins = mat.shape[0]
        for b in range(n_win_bins):
            genome_bin = win_start // resolution + b
            if genome_bin >= n_bins:
                continue
            for t in range(num_tracks):
                row = mat[b, :, t]
                if mode == "summary":
                    contact_sums[genome_bin, t] += float(
                        np.sum(row[row >= CONTACT_MIN_FREQ])
                    )

    if mode == "summary":
        result: dict[str, np.ndarray] = {}
        for t, tname in enumerate(track_names):
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", tname).lower()
            col = f"contact_total_{safe_name}" if len(track_names) > 1 else "contact_total"
            result[col] = contact_sums[:, t].astype(np.float32)
        return result

    # mode == "full" — currently not implemented; fall back to summary
    log.warning("contact_mode='full' not yet implemented; using summary")
    return _load_contact_summary(
        contact_map_dir, chrom, n_bins, resolution, mode="summary"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_fixed_bin_tokens(
    chrom: str,
    input_dir: str | Path,
    resolution: int,
    contact_map_dir: str | Path | None = None,
    contact_mode: str = "summary",
) -> pd.DataFrame:
    """
    Build a fixed-bin feature matrix for one chromosome at *resolution* bp.

    Args:
        chrom:           Chromosome name (e.g. ``"chr22"``).
        input_dir:       Root output directory containing ``atac/``,
                         ``chip_histone/``, ``chip_tf/`` subdirectories.
        resolution:      Bin size in bp.  Must be in FIXED_BIN_RESOLUTIONS.
        contact_map_dir: Directory containing contact-map window NPZ files.
                         Required (and used) only when *resolution* ==
                         CONTACT_MAP_RESOLUTION.  Ignored otherwise.
        contact_mode:    ``"summary"`` (default) or ``"full"``.

    Returns:
        DataFrame with columns: ``chrom``, ``start``, ``end``, and one column
        per signal track.  ``start`` and ``end`` are 0-based half-open
        genomic coordinates.
    """
    if resolution not in FIXED_BIN_RESOLUTIONS:
        raise ValueError(
            f"Resolution {resolution} not in FIXED_BIN_RESOLUTIONS "
            f"({FIXED_BIN_RESOLUTIONS})"
        )

    input_dir = Path(input_dir)

    assay_dirs = {
        ASSAY_ATAC:         input_dir / ASSAY_ATAC,
        ASSAY_CHIP_HISTONE: input_dir / ASSAY_CHIP_HISTONE,
        ASSAY_CHIP_TF:      input_dir / ASSAY_CHIP_TF,
    }

    # ---- Load all 1D assays -------------------------------------------------
    feature_cols: dict[str, np.ndarray] = {}
    chrom_size: int | None = None
    n_bins_tgt: int | None = None

    for assay, assay_dir in assay_dirs.items():
        npz_path  = assay_dir / f"{chrom}.npz"
        json_path = assay_dir / f"{chrom}.json"

        if not npz_path.exists():
            log.warning("[%s] %s not found: %s — skipping assay", chrom, assay, npz_path)
            continue

        data       = np.load(npz_path, allow_pickle=False)
        values     = data["values"].astype(np.float32)   # (genome_bins, num_tracks)
        src_res    = int(data["resolution"])
        c_size     = int(data["chrom_size"])

        if values.ndim == 1:
            values = values[:, np.newaxis]

        with open(json_path, "r", encoding="utf-8") as fh:
            metadata: list[dict] = json.load(fh)

        if chrom_size is None:
            chrom_size = c_size

        expected_tgt_bins = len(range(0, c_size, resolution))
        if n_bins_tgt is None:
            n_bins_tgt = expected_tgt_bins

        # Pool to target resolution
        aligned = _align_bins(values, src_res, resolution, c_size)

        # Derive column names and aggregate duplicates
        cols = _build_1d_columns(aligned, metadata, assay)
        feature_cols.update(cols)

        log.info(
            "[%s] %s: %d tracks → %d columns at %d bp",
            chrom, assay, values.shape[1], len(cols), resolution,
        )

    if n_bins_tgt is None or chrom_size is None:
        log.error("[%s] no data loaded — returning empty DataFrame", chrom)
        return pd.DataFrame()

    # ---- Contact maps at 2048 bp only ---------------------------------------
    if resolution == CONTACT_MAP_RESOLUTION and contact_map_dir is not None:
        contact_dir = Path(contact_map_dir)
        contact_cols = _load_contact_summary(
            contact_dir, chrom, n_bins_tgt, resolution, mode=contact_mode,
        )
        feature_cols.update(contact_cols)
        log.info(
            "[%s] contact maps: %d summary columns added", chrom, len(contact_cols)
        )

    # ---- Assemble DataFrame --------------------------------------------------
    bin_starts = np.arange(n_bins_tgt, dtype=np.int64) * resolution
    bin_ends   = np.minimum(bin_starts + resolution, chrom_size)

    df_data: dict[str, np.ndarray] = {
        "chrom": np.full(n_bins_tgt, chrom, dtype=object),
        "start": bin_starts,
        "end":   bin_ends,
    }

    for col, arr in feature_cols.items():
        # Ensure length matches — truncate or pad with zeros
        if len(arr) < n_bins_tgt:
            padded = np.zeros(n_bins_tgt, dtype=np.float32)
            padded[:len(arr)] = arr
            arr = padded
        else:
            arr = arr[:n_bins_tgt]
        df_data[col] = arr.astype(np.float32)

    df = pd.DataFrame(df_data)

    log.info(
        "[%s] fixed_bin %d bp — %d bins × %d feature columns",
        chrom, resolution, len(df), len(df.columns) - 3,
    )
    return df
