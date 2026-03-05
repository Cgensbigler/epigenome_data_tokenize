"""
Output writers for genome-wide track arrays.

Each stitched chromosome is saved as a pair of files:
  {output_dir}/{track_type}/{chrom}.npz   — compressed numpy arrays
  {output_dir}/{track_type}/{chrom}.json  — track metadata sidecar

NPZ layout (1D tracks)
-----------------------
  values      float32 (genome_bins, num_tracks)  — stitched prediction values
  resolution  int32 scalar                        — bin size in bp
  chrom       str scalar                          — chromosome name
  chrom_size  int64 scalar                        — chromosome length in bp

JSON sidecar
------------
A list of objects, one per track, each containing all columns from the
``TrackData.metadata`` DataFrame (at minimum: "name" and "strand").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_chrom_npz(
    values: np.ndarray,
    metadata: pd.DataFrame,
    resolution: int,
    chrom: str,
    chrom_size: int,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write a stitched chromosome track array to disk.

    Args:
        values:     Float32 array of shape ``(genome_bins, num_tracks)``.
        metadata:   Pandas DataFrame with track metadata (name, strand, …).
        resolution: Bin size in bp.
        chrom:      Chromosome name (e.g. ``"chr1"``).
        chrom_size: Full chromosome length in bp.
        out_dir:    Directory to write into (created if absent).

    Returns:
        Tuple of ``(npz_path, json_path)`` for the two written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = out_dir / f"{chrom}.npz"
    json_path = out_dir / f"{chrom}.json"

    # ---- NPZ ---------------------------------------------------------------
    np.savez_compressed(
        npz_path,
        values=values.astype(np.float32),
        resolution=np.array(resolution, dtype=np.int32),
        chrom=np.array(chrom),
        chrom_size=np.array(chrom_size, dtype=np.int64),
    )

    # ---- JSON sidecar ------------------------------------------------------
    # Convert DataFrame to list-of-dicts; handle non-serialisable types.
    records = metadata.reset_index(drop=True).to_dict(orient="records")
    # Ensure every value is JSON-serialisable (some columns may be numpy scalars)
    clean_records = [
        {k: _to_json_safe(v) for k, v in row.items()} for row in records
    ]
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(clean_records, fh, indent=2)

    return npz_path, json_path


def load_chrom_npz(
    chrom: str,
    out_dir: str | Path,
) -> tuple[np.ndarray, pd.DataFrame, int, int]:
    """Load a previously saved chromosome track file.

    Args:
        chrom:   Chromosome name.
        out_dir: Directory that contains ``{chrom}.npz`` and ``{chrom}.json``.

    Returns:
        Tuple of ``(values, metadata, resolution, chrom_size)``.
    """
    out_dir   = Path(out_dir)
    npz_path  = out_dir / f"{chrom}.npz"
    json_path = out_dir / f"{chrom}.json"

    data       = np.load(npz_path, allow_pickle=False)
    values     = data["values"]
    resolution = int(data["resolution"])
    chrom_size = int(data["chrom_size"])

    with open(json_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    metadata = pd.DataFrame(records)

    return values, metadata, resolution, chrom_size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_json_safe(value: object) -> object:
    """Convert numpy scalars / non-serialisable objects to plain Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
