"""
Hi-C contact matrix processing.

Uses hic-straw to read ENCODE .hic files and extract KR-normalised
contact frequencies at a configurable bin resolution.

For each non-zero bin pair the output token contains:
    chrom1, start1, end1, chrom2, start2, end2,
    contact_freq, experiment_accession, biosample, resolution

Design notes
------------
- Only *cis* (same-chromosome) and *trans* (inter-chromosome) contacts
  above a minimum frequency threshold are kept to keep the output sparse
  and model-friendly.
- KR normalisation is preferred; VC_SQRT is used as fallback when KR
  vectors are absent (common for low-coverage experiments).
- hic-straw returns data as three parallel arrays (binX, binY, counts).
  We expand these into absolute genomic coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import (
    DEFAULT_HIC_NORMALIZATION,
    DEFAULT_HIC_RESOLUTION,
    HIC_CHROMOSOMES,
    HIC_TOKEN_COLUMNS,
)

logger = logging.getLogger(__name__)

# Minimum normalised contact frequency to include (filters near-zero noise)
MIN_CONTACT_FREQ = 1e-4


# ---------------------------------------------------------------------------
# hic-straw helpers
# ---------------------------------------------------------------------------

def _open_hic(hic_path: Path):
    """Return an open hicstraw.HiCFile handle."""
    try:
        import hicstraw  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "hic-straw is required for Hi-C processing. "
            "Install it with: pip install hic-straw"
        ) from exc
    return hicstraw.HiCFile(str(hic_path))


def _available_chromosomes(hic_file) -> list[str]:
    """Return chromosome names present in the .hic file."""
    return [c.name for c in hic_file.getChromosomes() if c.name.lower() != "all"]


def _available_resolutions(hic_file) -> list[int]:
    """Return the BP resolutions present in the .hic file, sorted ascending."""
    try:
        return sorted(hic_file.getResolutions())
    except Exception:
        return []


def _resolve_resolution(hic_file, requested: int) -> int:
    """
    Return the resolution that will actually be used.

    If *requested* is present in the file, return it unchanged.
    Otherwise pick the nearest available resolution and emit a warning so
    the caller is never silently given the wrong bin size.
    """
    available = _available_resolutions(hic_file)
    if not available:
        logger.warning(
            "Could not query available resolutions; attempting %d bp as requested.",
            requested,
        )
        return requested

    if requested in available:
        return requested

    nearest = min(available, key=lambda r: abs(r - requested))
    logger.warning(
        "Requested resolution %d bp is not in this .hic file. "
        "Available: %s. Using nearest: %d bp.",
        requested,
        available,
        nearest,
    )
    return nearest


def _normalisation_to_use(hic_file, chrom: str, resolution: int) -> str:
    """
    Determine the best available normalisation for a given chromosome.

    Prefers KR; falls back to VC_SQRT, then NONE.
    """
    preferred = [DEFAULT_HIC_NORMALIZATION, "VC_SQRT", "NONE"]
    try:
        available_norms = hic_file.getNormalizationTypes()
        for norm in preferred:
            if norm in available_norms:
                return norm
    except Exception:
        pass
    return "NONE"


def _extract_cis_contacts(
    hic_file,
    chrom: str,
    resolution: int,
    normalization: str,
) -> pd.DataFrame:
    """
    Extract intra-chromosomal (cis) contacts for *chrom* at *resolution*.

    Returns a DataFrame with columns matching HIC_TOKEN_COLUMNS (minus
    experiment_accession, biosample, resolution which are added upstream).
    """
    try:
        import hicstraw  # noqa: PLC0415
        mzd = hic_file.getMatrixZoomData(chrom, chrom, "observed", normalization, "BP", resolution)
        records = mzd.getRecords(0, int(1e12), 0, int(1e12))
    except Exception as exc:
        logger.debug("Could not extract cis contacts for %s: %s", chrom, exc)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    bin1 = np.array([r.binX for r in records], dtype=np.int64)
    bin2 = np.array([r.binY for r in records], dtype=np.int64)
    counts = np.array([r.counts for r in records], dtype=np.float32)

    mask = (counts > MIN_CONTACT_FREQ) & np.isfinite(counts)
    bin1, bin2, counts = bin1[mask], bin2[mask], counts[mask]

    if len(bin1) == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        "chrom1": chrom,
        "start1": bin1,
        "end1": bin1 + resolution,
        "chrom2": chrom,
        "start2": bin2,
        "end2": bin2 + resolution,
        "contact_freq": counts,
    })


def _extract_trans_contacts(
    hic_file,
    chrom1: str,
    chrom2: str,
    resolution: int,
    normalization: str,
) -> pd.DataFrame:
    """
    Extract inter-chromosomal (trans) contacts between chrom1 and chrom2.
    """
    try:
        mzd = hic_file.getMatrixZoomData(chrom1, chrom2, "observed", normalization, "BP", resolution)
        records = mzd.getRecords(0, int(1e12), 0, int(1e12))
    except Exception as exc:
        logger.debug("Could not extract trans contacts %s x %s: %s", chrom1, chrom2, exc)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    bin1 = np.array([r.binX for r in records], dtype=np.int64)
    bin2 = np.array([r.binY for r in records], dtype=np.int64)
    counts = np.array([r.counts for r in records], dtype=np.float32)

    mask = (counts > MIN_CONTACT_FREQ) & np.isfinite(counts)
    bin1, bin2, counts = bin1[mask], bin2[mask], counts[mask]

    if len(bin1) == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        "chrom1": chrom1,
        "start1": bin1,
        "end1": bin1 + resolution,
        "chrom2": chrom2,
        "start2": bin2,
        "end2": bin2 + resolution,
        "contact_freq": counts,
    })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_hic_experiment(
    info: dict,
    resolution: int = DEFAULT_HIC_RESOLUTION,
    chromosomes: list[str] | None = None,
    include_trans: bool = False,
) -> pd.DataFrame | None:
    """
    Process a single Hi-C experiment into a tokenization-ready DataFrame.

    Parameters
    ----------
    info : dict
        Must contain keys: accession, biosample, hic_path.
    resolution : int
        Bin size in bp (e.g. 10_000 for 10 kb).
    chromosomes : list[str] | None
        Chromosomes to extract.  Defaults to HIC_CHROMOSOMES (hg38 autosomes
        + chrX/chrY).  Pass a subset for faster development runs.
    include_trans : bool
        If True, also extract inter-chromosomal contacts (much larger output).

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns from HIC_TOKEN_COLUMNS, or None on failure.
    """
    acc = info["accession"]
    hic_path = info.get("hic_path")

    if hic_path is None or not Path(hic_path).exists():
        logger.warning("Skipping %s — .hic file not found", acc)
        return None

    logger.info("Processing Hi-C for %s at %d bp resolution (requested)", acc, resolution)

    hic_file = _open_hic(Path(hic_path))

    # Validate and snap to the nearest resolution that actually exists in the file
    resolution = _resolve_resolution(hic_file, resolution)
    logger.info("%s: using resolution=%d bp", acc, resolution)

    file_chroms = set(_available_chromosomes(hic_file))
    target_chroms = chromosomes if chromosomes is not None else HIC_CHROMOSOMES
    target_chroms = [c for c in target_chroms if c in file_chroms]

    if not target_chroms:
        logger.warning("%s: no matching chromosomes in .hic file", acc)
        return None

    normalization = _normalisation_to_use(hic_file, target_chroms[0], resolution)
    logger.info("%s: using normalisation=%s", acc, normalization)

    frames: list[pd.DataFrame] = []

    # Cis contacts
    for chrom in target_chroms:
        df = _extract_cis_contacts(hic_file, chrom, resolution, normalization)
        if not df.empty:
            frames.append(df)

    # Trans contacts (optional — large)
    if include_trans:
        for i, c1 in enumerate(target_chroms):
            for c2 in target_chroms[i + 1:]:
                df = _extract_trans_contacts(hic_file, c1, c2, resolution, normalization)
                if not df.empty:
                    frames.append(df)

    if not frames:
        logger.warning("%s: no contacts extracted", acc)
        return None

    result = pd.concat(frames, ignore_index=True)
    result["experiment_accession"] = acc
    result["biosample"] = info.get("biosample", "")
    result["resolution"] = resolution

    result = result[HIC_TOKEN_COLUMNS]

    logger.info(
        "%s: %d contact tokens (norm=%s, cis-only=%s)",
        acc,
        len(result),
        normalization,
        not include_trans,
    )
    return result


def process_hic_experiments(
    experiments: list[dict],
    resolution: int = DEFAULT_HIC_RESOLUTION,
    chromosomes: list[str] | None = None,
    include_trans: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (accession, DataFrame) for each successfully processed Hi-C experiment."""
    for info in experiments:
        df = process_hic_experiment(
            info,
            resolution=resolution,
            chromosomes=chromosomes,
            include_trans=include_trans,
        )
        if df is not None:
            yield info["accession"], df
