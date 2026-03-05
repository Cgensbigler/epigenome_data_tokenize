"""
Central configuration for the ENCODE tracks tokenizer pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# ENCODE REST API
# ---------------------------------------------------------------------------
ENCODE_BASE_URL = "https://www.encodeproject.org"
ENCODE_SEARCH_URL = f"{ENCODE_BASE_URL}/search/"

# Request headers — ENCODE recommends identifying your client
ENCODE_HEADERS = {
    "Accept": "application/json",
}

# Max results per search page (ENCODE supports up to 10000)
ENCODE_PAGE_LIMIT = 500

# ---------------------------------------------------------------------------
# Assay configuration
# ---------------------------------------------------------------------------

# Maps user-facing assay names to ENCODE assay_title query values
ASSAY_TITLES = {
    "TF-ChIP-seq": "TF ChIP-seq",
    "ATAC-seq": "ATAC-seq",
    "Hi-C": "Hi-C",
}

# Peak-calling assays (those that produce narrowPeak + BigWig)
PEAK_ASSAYS = {"TF-ChIP-seq", "ATAC-seq"}

# ---------------------------------------------------------------------------
# File type selection per assay
# ---------------------------------------------------------------------------

# ENCODE output_type values for IDR-optimal peak files
PEAK_OUTPUT_TYPES = [
    "IDR thresholded peaks",        # preferred — IDR across replicates
    "optimal IDR thresholded peaks",
    "pseudoreplicated IDR thresholded peaks",  # fallback when single replicate
]

# ENCODE output_type for signal BigWig (fold change is cross-experiment comparable)
SIGNAL_OUTPUT_TYPE = "fold change over control"

# ENCODE file_type strings
NARROWPEAK_FILE_TYPE = "bed narrowPeak"
BIGWIG_FILE_TYPE = "bigWig"
HIC_FILE_TYPE = "hic"

# ---------------------------------------------------------------------------
# Genome assembly
# ---------------------------------------------------------------------------
DEFAULT_ASSEMBLY = "GRCh38"

# ---------------------------------------------------------------------------
# Hi-C settings
# ---------------------------------------------------------------------------
DEFAULT_HIC_RESOLUTION = 10_000        # bp (10 kb)
DEFAULT_HIC_NORMALIZATION = "KR"       # KR balancing (ENCODE default)
HIC_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# ---------------------------------------------------------------------------
# Output paths (relative to the user-specified output dir)
# ---------------------------------------------------------------------------
RAW_SUBDIR = "raw"
PROCESSED_SUBDIR = "processed"

# Within raw/, files are organized as: raw/{assay}/{experiment_accession}/
RAW_LAYOUT = "{assay}/{experiment_accession}"

# ---------------------------------------------------------------------------
# Download settings
# ---------------------------------------------------------------------------
DEFAULT_WORKERS = 8
DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MB chunks
REQUEST_TIMEOUT = 60            # seconds per HTTP request

# ---------------------------------------------------------------------------
# Tokenization output schema
# ---------------------------------------------------------------------------

# Column order for peak token output (TF-ChIP-seq + ATAC-seq)
PEAK_TOKEN_COLUMNS = [
    "chrom",
    "start",
    "end",
    "summit",           # absolute genomic position of signal summit
    "auc",              # integral of fold-change BigWig over [start, end]
    "signal_value",     # fold-change at summit (narrowPeak col 6)
    "p_value",          # -log10(p-value) from narrowPeak col 7
    "q_value",          # -log10(q-value) from narrowPeak col 8
    "strand",
    "experiment_accession",
    "target",           # TF target gene symbol (empty for ATAC-seq)
    "biosample",
    "assay",
]

# Column order for Hi-C contact token output
HIC_TOKEN_COLUMNS = [
    "chrom1",
    "start1",
    "end1",
    "chrom2",
    "start2",
    "end2",
    "contact_freq",     # KR-normalized contact count
    "experiment_accession",
    "biosample",
    "resolution",
]
