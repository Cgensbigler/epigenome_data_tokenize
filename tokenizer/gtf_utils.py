"""
GENCODE GTF utilities for TSS annotation.

Downloads and caches the GENCODE v46 hg38 basic annotation GTF, parses all
transcript records, and provides fast nearest-TSS lookup via numpy.searchsorted.

TSS selection strategy (mirrors ENCODE cCRE pipeline):
  - Parse all transcripts from the GTF.
  - For each gene, keep the MANE Select transcript if present; otherwise keep
    the transcript with the longest span.
  - The TSS is the 5' end of the kept transcript:
      strand '+' → transcript start
      strand '-' → transcript end
  - Result: one TSS per gene, deduplicated, sorted by position per chromosome.
"""

from __future__ import annotations

import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np

from .config import GENCODE_GTF_FILENAME, GENCODE_GTF_URL

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

def ensure_gtf(cache_dir: str | Path) -> Path:
    """
    Return the path to the cached GENCODE GTF, downloading it if absent.

    Args:
        cache_dir: Directory to store the downloaded file (created if absent).

    Returns:
        Path to ``gencode.v46.basic.annotation.gtf.gz``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gtf_path = cache_dir / GENCODE_GTF_FILENAME

    if gtf_path.exists():
        log.info("GENCODE GTF already cached: %s", gtf_path)
        return gtf_path

    log.info("Downloading GENCODE GTF from %s …", GENCODE_GTF_URL)
    log.info("This is a ~50 MB download; it will be cached at %s", gtf_path)

    tmp_path = gtf_path.with_suffix(".gtf.gz.tmp")
    try:
        urllib.request.urlretrieve(GENCODE_GTF_URL, tmp_path)
        tmp_path.rename(gtf_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    log.info("GENCODE GTF downloaded (%.1f MB)", gtf_path.stat().st_size / 1e6)
    return gtf_path


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def _open_gtf(gtf_path: Path):
    """Open a plain or gzip-compressed GTF file."""
    if str(gtf_path).endswith(".gz"):
        return gzip.open(gtf_path, "rt", encoding="utf-8")
    return open(gtf_path, "r", encoding="utf-8")


def _parse_attribute(attrs: str, key: str) -> str:
    """Extract a value from a GTF attribute string."""
    for field in attrs.split(";"):
        field = field.strip()
        if field.startswith(key + " "):
            return field[len(key) + 1:].strip().strip('"')
    return ""


def _build_tss_table(gtf_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Parse the GTF and build per-chromosome TSS tables.

    Returns:
        Dict mapping chromosome → (positions_sorted, gene_names_sorted)
        where positions are 0-based TSS positions (int64).
    """
    # Collect: gene_id → {transcript_id: (chrom, tss, gene_name, is_mane, span)}
    transcripts: dict[str, dict] = {}  # transcript_id → record

    log.info("Parsing GENCODE GTF for TSS positions …")
    n_records = 0

    with _open_gtf(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "transcript":
                continue

            chrom  = parts[0]
            start  = int(parts[3]) - 1   # GTF is 1-based → convert to 0-based
            end    = int(parts[4])        # end is inclusive in GTF → exclusive in 0-based
            strand = parts[6]
            attrs  = parts[8]

            gene_id    = _parse_attribute(attrs, "gene_id").split(".")[0]
            tx_id      = _parse_attribute(attrs, "transcript_id").split(".")[0]
            gene_name  = _parse_attribute(attrs, "gene_name") or gene_id
            tx_support = _parse_attribute(attrs, "transcript_support_level")
            tag        = _parse_attribute(attrs, "tag")

            is_mane = "MANE_Select" in tag or "MANE Select" in tag

            tss = start if strand == "+" else end - 1

            transcripts[tx_id] = {
                "chrom":     chrom,
                "tss":       tss,
                "gene_name": gene_name,
                "gene_id":   gene_id,
                "is_mane":   is_mane,
                "span":      end - start,
                "support":   tx_support,
            }
            n_records += 1

    log.info("Parsed %d transcript records", n_records)

    # Per gene: keep MANE Select if present, else longest transcript
    gene_best: dict[str, dict] = {}
    for tx in transcripts.values():
        gid = tx["gene_id"]
        if gid not in gene_best:
            gene_best[gid] = tx
        else:
            prev = gene_best[gid]
            if tx["is_mane"] and not prev["is_mane"]:
                gene_best[gid] = tx
            elif not prev["is_mane"] and not tx["is_mane"]:
                if tx["span"] > prev["span"]:
                    gene_best[gid] = tx

    # Group by chromosome → sort by TSS position
    chrom_records: dict[str, list[tuple[int, str]]] = {}
    for tx in gene_best.values():
        chrom_records.setdefault(tx["chrom"], []).append(
            (tx["tss"], tx["gene_name"])
        )

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, records in chrom_records.items():
        records.sort(key=lambda x: x[0])
        positions  = np.array([r[0] for r in records], dtype=np.int64)
        gene_names = np.array([r[1] for r in records], dtype=object)
        result[chrom] = (positions, gene_names)

    log.info("TSS table built for %d chromosomes (%d genes)", len(result), len(gene_best))
    return result


# ---------------------------------------------------------------------------
# TSS lookup
# ---------------------------------------------------------------------------

class TSSIndex:
    """
    Fast nearest-TSS lookup for one or all chromosomes.

    Build once per run and reuse across all chromosome queries.
    """

    def __init__(self, gtf_path: Path) -> None:
        self._tables = _build_tss_table(gtf_path)

    def nearest_tss(
        self,
        chrom: str,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (distances, gene_names) for each query position.

        Uses binary search (O(log n)) per position.  Distances are always
        non-negative (absolute distance to nearest TSS).

        Args:
            chrom:     Chromosome name (e.g. ``"chr22"``).
            positions: 1-D int64 array of 0-based genomic positions.

        Returns:
            - distances:  int64 array, distance in bp to nearest TSS
            - gene_names: str array, name of nearest gene
        """
        if chrom not in self._tables:
            n = len(positions)
            return np.full(n, -1, dtype=np.int64), np.full(n, "", dtype=object)

        tss_pos, tss_names = self._tables[chrom]
        n = len(positions)
        distances  = np.empty(n, dtype=np.int64)
        gene_names = np.empty(n, dtype=object)

        for i, pos in enumerate(positions):
            idx = int(np.searchsorted(tss_pos, pos))

            # Candidates: idx-1 and idx (left and right neighbours)
            best_dist = np.int64(10**15)
            best_name = ""

            for j in (idx - 1, idx):
                if 0 <= j < len(tss_pos):
                    d = abs(int(pos) - int(tss_pos[j]))
                    if d < best_dist:
                        best_dist = d
                        best_name = str(tss_names[j])

            distances[i]  = best_dist
            gene_names[i] = best_name

        return distances, gene_names
