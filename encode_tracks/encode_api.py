"""
ENCODE REST API client.

Queries the ENCODE portal to discover experiments and select the appropriate
file types for download.  Returns structured metadata dicts rather than raw
JSON so downstream modules remain decoupled from the API shape.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator
from urllib.parse import urlencode

import requests

from config import (
    ASSAY_TITLES,
    BIGWIG_FILE_TYPE,
    DEFAULT_ASSEMBLY,
    ENCODE_BASE_URL,
    ENCODE_HEADERS,
    ENCODE_PAGE_LIMIT,
    ENCODE_SEARCH_URL,
    HIC_FILE_TYPE,
    NARROWPEAK_FILE_TYPE,
    PEAK_OUTPUT_TYPES,
    SIGNAL_OUTPUT_TYPE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = 3) -> Any:
    """GET with exponential back-off retries."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=ENCODE_HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("Request failed (%s); retrying in %ds…", exc, wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Experiment search
# ---------------------------------------------------------------------------

def search_experiments(
    assay: str,
    biosample: str | None = None,
    assembly: str = DEFAULT_ASSEMBLY,
) -> list[dict]:
    """
    Return a list of released ENCODE experiments for the given assay.

    Each dict contains the minimal fields needed for downstream file
    selection: accession, assay_title, biosample_term_name, and the
    raw ENCODE files list.
    """
    encode_assay = ASSAY_TITLES.get(assay)
    if encode_assay is None:
        raise ValueError(f"Unknown assay '{assay}'. Valid choices: {list(ASSAY_TITLES)}")

    params: dict[str, Any] = {
        "type": "Experiment",
        "assay_title": encode_assay,
        "status": "released",
        "format": "json",
        "limit": ENCODE_PAGE_LIMIT,
        "frame": "embedded",
    }
    # Hi-C experiments are not indexed by assembly at the experiment level in
    # ENCODE — the assembly lives on individual files.  Adding assembly= to
    # the Hi-C search returns a 404.  For peak assays the filter is safe and
    # useful to narrow results early.
    if assay != "Hi-C" and assembly:
        params["assembly"] = assembly
    if biosample:
        params["biosample_ontology.term_name"] = biosample

    logger.info("Searching ENCODE: assay=%s biosample=%s assembly=%s", assay, biosample, assembly)

    experiments: list[dict] = []
    start = 0

    while True:
        params["from"] = start
        data = _get(ENCODE_SEARCH_URL, params=params)
        hits = data.get("@graph", [])
        experiments.extend(hits)
        total = data.get("total", len(hits))
        logger.debug("Fetched %d / %d experiments (offset=%d)", len(experiments), total, start)
        if len(experiments) >= total or not hits:
            break
        start += ENCODE_PAGE_LIMIT

    logger.info("Found %d %s experiments for biosample=%s", len(experiments), assay, biosample)
    return experiments


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def _file_matches_assembly(f: dict, assembly: str) -> bool:
    """
    Return True if the file's assembly field matches *assembly*.

    ENCODE returns assembly as a plain string on file objects when using
    frame=embedded (e.g. "GRCh38"), NOT as a list.  This helper handles
    both that case and the rare list form defensively.
    """
    file_assembly = f.get("assembly")
    if isinstance(file_assembly, str):
        return file_assembly == assembly
    if isinstance(file_assembly, list):
        return any(
            (a.get("ncbi_version_name", a) if isinstance(a, dict) else a) == assembly
            for a in file_assembly
        )
    return False


def _best_peak_file(files: list[dict], assembly: str) -> dict | None:
    """
    Choose the highest-priority IDR-optimal narrowPeak file for the assembly.

    Priority follows PEAK_OUTPUT_TYPES order.
    """
    candidates = [
        f for f in files
        if f.get("file_type") == NARROWPEAK_FILE_TYPE
        and f.get("status") == "released"
        and _file_matches_assembly(f, assembly)
    ]

    for preferred_type in PEAK_OUTPUT_TYPES:
        for f in candidates:
            if f.get("output_type") == preferred_type:
                return f

    # Last resort: any released narrowPeak for this assembly
    return candidates[0] if candidates else None


def _best_signal_file(files: list[dict], assembly: str) -> dict | None:
    """
    Choose the fold-change-over-control BigWig for the assembly.
    Falls back to signal p-value if unavailable.
    """
    candidates = [
        f for f in files
        if f.get("file_type") == BIGWIG_FILE_TYPE
        and f.get("status") == "released"
        and _file_matches_assembly(f, assembly)
    ]

    for f in candidates:
        if f.get("output_type") == SIGNAL_OUTPUT_TYPE:
            return f

    # Fallback: signal p-value bigwig
    for f in candidates:
        if "signal" in f.get("output_type", "").lower():
            return f

    return candidates[0] if candidates else None


def _best_hic_file(files: list[dict], assembly: str | None = None) -> dict | None:
    """
    Return the best released .hic file, preferring the requested assembly.

    Because Hi-C experiments aren't filtered by assembly at search time, we
    do the filtering here on the file objects.
    """
    candidates = [
        f for f in files
        if f.get("file_type") == HIC_FILE_TYPE and f.get("status") == "released"
    ]
    if not candidates:
        return None
    if assembly:
        assembly_match = [f for f in candidates if _file_matches_assembly(f, assembly)]
        if assembly_match:
            return assembly_match[0]
        # If no file matches the requested assembly, log and fall back
        logger.debug(
            "No .hic file matching assembly=%s; falling back to first available", assembly
        )
    return candidates[0]


def _file_url(f: dict) -> str:
    href = f.get("href", "")
    if href.startswith("http"):
        return href
    return ENCODE_BASE_URL + href


def _extract_target(experiment: dict) -> str:
    """Pull the TF target gene symbol from an experiment dict."""
    target = experiment.get("target", {})
    if isinstance(target, dict):
        return target.get("gene_name", target.get("label", ""))
    return ""


def _biosample_name(experiment: dict) -> str:
    bt = experiment.get("biosample_ontology", {})
    if isinstance(bt, dict):
        return bt.get("term_name", "")
    return ""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_peak_assay_files(
    experiment: dict,
    assembly: str = DEFAULT_ASSEMBLY,
) -> dict | None:
    """
    For a TF-ChIP-seq or ATAC-seq experiment, return a dict with:
        accession, assay, target, biosample,
        peak_url, peak_accession,
        signal_url, signal_accession
    Returns None if required files are missing.
    """
    files = experiment.get("files", [])
    peak_file = _best_peak_file(files, assembly)
    signal_file = _best_signal_file(files, assembly)

    if peak_file is None:
        logger.debug(
            "Skipping %s — no IDR narrowPeak found",
            experiment.get("accession"),
        )
        return None

    result: dict[str, Any] = {
        "accession": experiment.get("accession", ""),
        "assay": experiment.get("assay_title", ""),
        "target": _extract_target(experiment),
        "biosample": _biosample_name(experiment),
        "peak_url": _file_url(peak_file),
        "peak_accession": peak_file.get("accession", ""),
        "peak_output_type": peak_file.get("output_type", ""),
        "signal_url": _file_url(signal_file) if signal_file else None,
        "signal_accession": signal_file.get("accession", "") if signal_file else "",
    }
    return result


def get_hic_files(experiment: dict, assembly: str | None = None) -> dict | None:
    """
    For a Hi-C experiment, return a dict with:
        accession, biosample, hic_url, hic_accession
    Returns None if no .hic file is found.
    """
    files = experiment.get("files", [])
    hic_file = _best_hic_file(files, assembly=assembly)

    if hic_file is None:
        logger.debug(
            "Skipping %s — no .hic file found",
            experiment.get("accession"),
        )
        return None

    return {
        "accession": experiment.get("accession", ""),
        "biosample": _biosample_name(experiment),
        "hic_url": _file_url(hic_file),
        "hic_accession": hic_file.get("accession", ""),
    }


def iter_peak_experiments(
    assay: str,
    biosample: str | None = None,
    assembly: str = DEFAULT_ASSEMBLY,
) -> Iterator[dict]:
    """Yield file-metadata dicts for every usable peak-type experiment."""
    experiments = search_experiments(assay, biosample, assembly)
    for exp in experiments:
        info = get_peak_assay_files(exp, assembly)
        if info:
            yield info


def iter_hic_experiments(
    biosample: str | None = None,
    assembly: str = DEFAULT_ASSEMBLY,
) -> Iterator[dict]:
    """Yield file-metadata dicts for every usable Hi-C experiment."""
    # assembly is intentionally NOT passed to search_experiments for Hi-C;
    # ENCODE does not index Hi-C at the experiment level by assembly.
    # File-level assembly filtering happens inside get_hic_files.
    experiments = search_experiments("Hi-C", biosample, assembly)
    for exp in experiments:
        info = get_hic_files(exp, assembly=assembly)
        if info:
            yield info
