"""
Track stitching logic for genome-wide predictions.

1D tracks (ATAC, ChIP-histone, ChIP-TF)
----------------------------------------
Each prediction window overlaps its neighbours by ``window_size - stride`` bp.
To avoid boundary artifacts only the *central stride* portion of each window
is retained:

  Window i starts at  s_i = i * stride
  Its center slice covers  [s_i + overlap/2,  s_i + overlap/2 + stride)
                         = [s_i + (L - S) / 2,  s_i + (L + S) / 2)
  where  L = window_size,  S = stride,  overlap = L - S

Edge windows are handled specially:
  - First window  → use [0, overlap/2 + stride) = [0, (L+S)/2)
  - Last window   → use [overlap/2, end_of_window)

All position arithmetic is done in genomic bp and converted to bin indices
using the track's native resolution (``TrackData.resolution``).

Contact maps (2D)
-----------------
Contact maps are 2D matrices of shape (bins, bins, num_tracks) and cannot
realistically be stitched into a genome-wide matrix in memory.  Instead,
``save_contact_map_window`` writes each window's predictions directly to a
per-window NPZ file tagged with its coordinates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_output


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_down(pos: int, res: int) -> int:
    """Round *pos* down to the nearest multiple of *res*."""
    return (pos // res) * res


def _align_up(pos: int, res: int) -> int:
    """Round *pos* up to the nearest multiple of *res*."""
    return _ceil_div(pos, res) * res


def stitch_1d_tracks(
    outputs: Sequence[dna_output.Output],
    windows: Sequence[genome.Interval],
    stride: int,
    output_type: dna_output.OutputType,
    chrom_size: int,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Stitch per-window 1D ``TrackData`` objects into a single chromosome array.

    Args:
        outputs:     One ``Output`` per window, in order.
        windows:     Corresponding ``genome.Interval`` objects.
        stride:      The stride used when generating windows (bp).
        output_type: Which output type to extract (ATAC, CHIP_HISTONE, CHIP_TF).
        chrom_size:  Total chromosome length in bp (used for final array size).

    Returns:
        A 3-tuple ``(values, metadata, resolution)`` where:
        - ``values``     — numpy float32 array of shape ``(genome_bins, num_tracks)``
        - ``metadata``   — pandas DataFrame with track metadata (name, strand, …)
        - ``resolution`` — the bin size in bp
    """
    if len(outputs) != len(windows):
        raise ValueError(
            f"outputs ({len(outputs)}) and windows ({len(windows)}) must have "
            "the same length."
        )
    if len(outputs) == 0:
        raise ValueError("outputs list is empty.")

    # ------------------------------------------------------------------ #
    # Extract TrackData for the requested output type from the first window
    # to learn resolution and metadata.
    # ------------------------------------------------------------------ #
    track_attr = _output_type_to_attr(output_type)
    first_track = getattr(outputs[0], track_attr)
    if first_track is None:
        raise ValueError(
            f"OutputType {output_type} not present in model output. "
            "Ensure it was included in requested_outputs."
        )

    resolution: int = first_track.resolution
    num_tracks: int = first_track.num_tracks
    metadata: pd.DataFrame = first_track.metadata.copy()

    genome_bins = _ceil_div(chrom_size, resolution)
    stitched = np.zeros((genome_bins, num_tracks), dtype=np.float32)
    # Track how many windows contributed to each bin (for potential averaging
    # at seams — currently we write non-overlapping slices so this stays 0/1).
    counts = np.zeros(genome_bins, dtype=np.int32)

    n = len(outputs)
    for i, (output, window) in enumerate(zip(outputs, windows)):
        track = getattr(output, track_attr)
        if track is None:
            raise ValueError(
                f"Window {i} ({window}) is missing output type {output_type}."
            )

        win_start = window.start   # genomic start of this window
        win_end   = window.end     # genomic end   (may be < win_start + L for last win)
        win_len   = win_end - win_start

        # ---- decide which genomic slice to keep from this window ----------
        if n == 1:
            # Single window covers the whole chromosome.
            keep_start_bp = win_start
            keep_end_bp   = win_end
        elif i == 0:
            # First window: keep [win_start, win_start + stride) but capped at win_end
            keep_start_bp = win_start
            keep_end_bp   = min(win_start + stride, win_end)
        elif i == n - 1:
            # Last window: keep [win_end - stride, win_end) but floored at win_start
            keep_start_bp = max(win_end - stride, win_start)
            keep_end_bp   = win_end
        else:
            # Middle windows: keep the central stride-wide slice
            overlap = win_len - stride
            keep_start_bp = win_start + overlap // 2
            keep_end_bp   = keep_start_bp + stride

        # ---- convert to bin indices relative to the window ----------------
        # Relative to the window's start, aligned to resolution boundaries.
        rel_start = _align_down(keep_start_bp - win_start, resolution)
        rel_end   = _align_up(min(keep_end_bp - win_start, win_len), resolution)

        # Clip rel_end to actual available bins in this window
        max_rel_end = (track.values.shape[0]) * resolution
        rel_end = min(rel_end, max_rel_end)

        if rel_end <= rel_start:
            continue

        # ---- extract slice from TrackData ---------------------------------
        slice_bins = track.slice_by_positions(rel_start, rel_end)
        slice_values = slice_bins.values  # (bins, num_tracks)

        # ---- place into stitched array ------------------------------------
        dst_bin_start = (win_start + rel_start) // resolution
        dst_bin_end   = dst_bin_start + slice_values.shape[0]

        # Guard against minor off-by-one at the chromosome boundary
        if dst_bin_end > genome_bins:
            trim = dst_bin_end - genome_bins
            slice_values = slice_values[:-trim]
            dst_bin_end  = genome_bins

        stitched[dst_bin_start:dst_bin_end] = slice_values
        counts[dst_bin_start:dst_bin_end] += 1

    return stitched, metadata, resolution


def save_contact_map_window(
    output: dna_output.Output,
    window: genome.Interval,
    out_dir: str | Path,
) -> Path:
    """Save a single window's contact-map predictions to a compressed NPZ file.

    The file is named ``{chrom}_{start}_{end}.npz`` and contains:
    - ``values``     — float32 array of shape (bins, bins, num_tracks)
    - ``resolution`` — scalar int
    - ``chrom``      — chromosome name string
    - ``start``      — window start (bp)
    - ``end``        — window end   (bp)
    - ``track_names``  — 1-D array of track name strings
    - ``track_strands`` — 1-D array of strand strings

    Args:
        output:  Model output for this window.
        window:  Genomic interval this window covers.
        out_dir: Directory to write the NPZ file into (created if absent).

    Returns:
        Path to the written NPZ file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contact_track = output.contact_maps
    if contact_track is None:
        raise ValueError(
            "contact_maps not present in model output. "
            "Ensure OutputType.CONTACT_MAPS was included in requested_outputs."
        )

    fname = f"{window.chromosome}_{window.start}_{window.end}.npz"
    out_path = out_dir / fname

    np.savez_compressed(
        out_path,
        values=contact_track.values.astype(np.float32),
        resolution=np.array(contact_track.resolution, dtype=np.int32),
        chrom=np.array(window.chromosome),
        start=np.array(window.start, dtype=np.int64),
        end=np.array(window.end, dtype=np.int64),
        track_names=np.array(contact_track.metadata["name"].tolist()),
        track_strands=np.array(contact_track.metadata["strand"].tolist()),
    )
    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _output_type_to_attr(output_type: dna_output.OutputType) -> str:
    """Map an ``OutputType`` enum to the corresponding ``Output`` attribute name."""
    _map = {
        dna_output.OutputType.ATAC:         "atac",
        dna_output.OutputType.CHIP_HISTONE: "chip_histone",
        dna_output.OutputType.CHIP_TF:      "chip_tf",
        dna_output.OutputType.CAGE:         "cage",
        dna_output.OutputType.DNASE:        "dnase",
        dna_output.OutputType.RNA_SEQ:      "rna_seq",
    }
    if output_type not in _map:
        raise ValueError(
            f"OutputType {output_type} is not a stitchable 1D output. "
            "Use save_contact_map_window for CONTACT_MAPS."
        )
    return _map[output_type]
