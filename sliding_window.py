"""
Sliding-window interval generation for genome-wide tiling.

`generate_windows` produces a list of `genome.Interval` objects that tile a
chromosome at the configured window size and stride.  Each window is exactly
`window_size` bp wide, except for the very last window on a chromosome which
is truncated to the chromosome boundary (and may therefore be shorter).  The
caller is responsible for padding/resizing if the model requires a fixed input
length.
"""

from __future__ import annotations

import math
from typing import Iterator

from alphagenome.data import genome

from config import HG38_CHROM_SIZES, WINDOW_SIZE, DEFAULT_STRIDE


def generate_windows(
    chrom: str,
    chrom_size: int,
    window_size: int = WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> list[genome.Interval]:
    """Return a list of tiling intervals for *chrom*.

    Windows are placed at positions ``0, stride, 2*stride, ...``.  Each window
    has width ``window_size``, except the last one which is clamped to the
    chromosome boundary.

    Special case: when the chromosome is shorter than ``window_size`` a single
    window covering ``[0, chrom_size)`` is returned and the caller should
    resize it to the nearest supported model length before calling the API.

    Args:
        chrom:       Chromosome name, e.g. ``"chr1"``.
        chrom_size:  Length of the chromosome in base pairs.
        window_size: Width of each prediction window (default: 1 MB).
        stride:      Step between consecutive window starts (default: 512 KB).

    Returns:
        Ordered list of `genome.Interval` objects covering the chromosome.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride > window_size:
        raise ValueError(
            f"stride ({stride}) must not exceed window_size ({window_size})"
        )

    intervals: list[genome.Interval] = []

    if chrom_size <= window_size:
        # Entire chromosome fits in one window; return as-is (may be narrower
        # than window_size — caller must resize before API call).
        intervals.append(genome.Interval(chrom, 0, chrom_size))
        return intervals

    start = 0
    while start < chrom_size:
        end = min(start + window_size, chrom_size)
        intervals.append(genome.Interval(chrom, start, end))
        if end == chrom_size:
            break
        start += stride

    return intervals


def generate_all_windows(
    chroms: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    chrom_sizes: dict[str, int] | None = None,
) -> dict[str, list[genome.Interval]]:
    """Generate tiling windows for a set of chromosomes.

    Args:
        chroms:      Chromosomes to tile.  Defaults to all hg38 standard chroms.
        window_size: Window width in bp.
        stride:      Stride in bp.
        chrom_sizes: Mapping of chrom → size.  Defaults to ``HG38_CHROM_SIZES``.

    Returns:
        Dict mapping chromosome name to its list of `genome.Interval` windows.
    """
    if chrom_sizes is None:
        chrom_sizes = HG38_CHROM_SIZES
    if chroms is None:
        chroms = list(chrom_sizes.keys())

    result: dict[str, list[genome.Interval]] = {}
    for chrom in chroms:
        if chrom not in chrom_sizes:
            raise KeyError(
                f"Chromosome '{chrom}' not found in chrom_sizes. "
                "Pass a custom chrom_sizes dict if using non-standard chromosomes."
            )
        result[chrom] = generate_windows(
            chrom, chrom_sizes[chrom], window_size=window_size, stride=stride
        )
    return result


def window_count(
    chrom_size: int,
    window_size: int = WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> int:
    """Return the number of windows that ``generate_windows`` will produce."""
    if chrom_size <= window_size:
        return 1
    return math.ceil((chrom_size - window_size) / stride) + 1
