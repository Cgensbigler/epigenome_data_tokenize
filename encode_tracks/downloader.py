"""
Parallel file downloader for ENCODE assets.

Files are organized as:
    <output_dir>/raw/<assay_slug>/<experiment_accession>/<file_accession>.<ext>

A download is skipped if the destination file already exists and its size
matches the Content-Length header (resume / idempotent re-runs).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from config import (
    DEFAULT_WORKERS,
    DOWNLOAD_CHUNK_SIZE,
    ENCODE_HEADERS,
    RAW_SUBDIR,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _assay_slug(assay: str) -> str:
    """Normalise assay name to a filesystem-safe directory component."""
    return assay.lower().replace(" ", "_").replace("-", "_")


def raw_dir(output_dir: Path, assay: str, experiment_accession: str) -> Path:
    """Return the directory where raw files for an experiment are stored."""
    return output_dir / RAW_SUBDIR / _assay_slug(assay) / experiment_accession


def expected_path(
    output_dir: Path,
    assay: str,
    experiment_accession: str,
    file_accession: str,
    url: str,
) -> Path:
    """Derive the local file path from the remote URL's extension."""
    remote_path = urlparse(url).path
    ext = "".join(Path(remote_path).suffixes)  # e.g. ".bed.gz" or ".bigWig"
    if not ext:
        ext = ".bin"
    filename = f"{file_accession}{ext}"
    return raw_dir(output_dir, assay, experiment_accession) / filename


# ---------------------------------------------------------------------------
# Single-file download
# ---------------------------------------------------------------------------

def _remote_size(url: str) -> int | None:
    """Return Content-Length from a HEAD request, or None if unavailable."""
    try:
        resp = requests.head(url, headers=ENCODE_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def download_file(
    url: str,
    dest: Path,
    *,
    desc: str = "",
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    """
    Download *url* to *dest*, skipping if already complete.

    progress_callback receives the number of bytes written in each chunk,
    allowing the caller to update an external progress bar.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume check: skip if file exists and size matches remote
    if dest.exists():
        remote_sz = _remote_size(url)
        local_sz = dest.stat().st_size
        if remote_sz is not None and local_sz == remote_sz:
            logger.debug("Skipping (already complete): %s", dest.name)
            if progress_callback:
                progress_callback(local_sz)
            return dest
        elif dest.stat().st_size > 0 and remote_sz is None:
            logger.debug("Skipping (exists, cannot verify size): %s", dest.name)
            return dest

    logger.info("Downloading %s → %s", url, dest)

    with requests.get(
        url,
        headers=ENCODE_HEADERS,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    if progress_callback:
                        progress_callback(len(chunk))

    return dest


# ---------------------------------------------------------------------------
# Batch download helpers
# ---------------------------------------------------------------------------

def _download_task(task: dict) -> tuple[str, Path | Exception]:
    """Worker function executed in a thread pool."""
    key = task["key"]
    try:
        path = download_file(task["url"], task["dest"], desc=key)
        return key, path
    except Exception as exc:
        logger.error("Failed to download %s: %s", key, exc)
        return key, exc


# ---------------------------------------------------------------------------
# Parallel batch download  (flat file-level queue)
# ---------------------------------------------------------------------------
#
# Each experiment has 2 files (narrowPeak + BigWig).  The previous design
# submitted one *experiment* per worker, making the two files download
# sequentially inside that worker slot — so only N/2 real downloads ran at
# a time.  The flat-queue design submits every *file* independently so all
# workers are saturated for the full duration of the run.


def _build_peak_file_tasks(
    experiments: Sequence[dict],
    output_dir: Path,
    assay: str,
) -> list[dict]:
    """Return a flat list of per-file download tasks for peak experiments."""
    tasks: list[dict] = []
    for exp in experiments:
        acc = exp["accession"]
        tasks.append({
            "key": f"{acc}/peak",
            "url": exp["peak_url"],
            "dest": expected_path(output_dir, assay, acc, exp["peak_accession"], exp["peak_url"]),
            "accession": acc,
            "role": "peak",
        })
        if exp.get("signal_url"):
            tasks.append({
                "key": f"{acc}/signal",
                "url": exp["signal_url"],
                "dest": expected_path(
                    output_dir, assay, acc, exp["signal_accession"], exp["signal_url"]
                ),
                "accession": acc,
                "role": "signal",
            })
    return tasks


def batch_download_peak_experiments(
    experiments: Sequence[dict],
    output_dir: Path,
    assay: str,
    workers: int = DEFAULT_WORKERS,
) -> list[dict]:
    """
    Download all narrowPeak + BigWig files in parallel using a flat file queue.

    Workers pull individual files rather than experiments, so all worker
    slots are kept busy for the full duration of the run.  Returns a list
    of experiment info dicts with peak_path and signal_path filled in.
    """
    all_tasks = _build_peak_file_tasks(experiments, output_dir, assay)
    file_results: dict[str, Path] = {}

    with tqdm(total=len(all_tasks), desc=f"Downloading {assay}", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_task, task): task for task in all_tasks}
            for fut in as_completed(futures):
                key, outcome = fut.result()
                if isinstance(outcome, Path):
                    file_results[key] = outcome
                pbar.update(1)

    # Reassemble experiment dicts with resolved paths
    out_experiments: list[dict] = []
    for exp in experiments:
        acc = exp["accession"]
        result = dict(exp)
        result["peak_path"] = file_results.get(f"{acc}/peak")
        result["signal_path"] = file_results.get(f"{acc}/signal")
        out_experiments.append(result)

    return out_experiments


def batch_download_hic_experiments(
    experiments: Sequence[dict],
    output_dir: Path,
    workers: int = DEFAULT_WORKERS,
) -> list[dict]:
    """Download all Hi-C files in parallel (one file per experiment)."""
    all_tasks = [
        {
            "key": f"{exp['accession']}/hic",
            "url": exp["hic_url"],
            "dest": expected_path(
                output_dir, "Hi-C", exp["accession"], exp["hic_accession"], exp["hic_url"]
            ),
            "accession": exp["accession"],
            "role": "hic",
        }
        for exp in experiments
    ]
    file_results: dict[str, Path] = {}

    with tqdm(total=len(all_tasks), desc="Downloading Hi-C", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_task, task): task for task in all_tasks}
            for fut in as_completed(futures):
                key, outcome = fut.result()
                if isinstance(outcome, Path):
                    file_results[key] = outcome
                pbar.update(1)

    out_experiments: list[dict] = []
    for exp in experiments:
        acc = exp["accession"]
        result = dict(exp)
        result["hic_path"] = file_results.get(f"{acc}/hic")
        out_experiments.append(result)

    return out_experiments
