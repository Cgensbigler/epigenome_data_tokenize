"""
Configuration for the predicted-track tokenizer.

Peak-calling parameters are designed to mirror ENCODE MACS2 defaults as
closely as possible when operating on continuous predicted signal tracks
(rather than raw read alignments).

MACS2 reference parameters (ENCODE pipelines):
  ATAC-seq : --nomodel --nolambda --keep-dup all --call-summits -p 0.01
  ChIP-seq  : --call-summits -q 0.05  (IDR across replicates)
"""

# ---------------------------------------------------------------------------
# Assay identifiers (match output/ subdirectory names)
# ---------------------------------------------------------------------------
ASSAY_ATAC          = "atac"
ASSAY_CHIP_HISTONE  = "chip_histone"
ASSAY_CHIP_TF       = "chip_tf"
ASSAY_CONTACT_MAPS  = "contact_maps"

ALL_1D_ASSAYS = [ASSAY_ATAC, ASSAY_CHIP_HISTONE, ASSAY_CHIP_TF]

# ---------------------------------------------------------------------------
# ATAC-seq: downsample before peak calling
# ---------------------------------------------------------------------------
# ATAC is predicted at 1 bp resolution → chr1 has ~249 M values.
# Mean-pooling to 32 bp reduces memory 32× while remaining 4× finer than
# the 128 bp ChIP resolution.
ATAC_DOWNSAMPLE_RESOLUTION: int = 32   # bp

# ---------------------------------------------------------------------------
# Peak-calling parameters (scipy.signal.find_peaks / peak_widths)
# ---------------------------------------------------------------------------

# Gaussian smoothing σ in bins applied before find_peaks.
# 3 bins ≈ MACS2's local background window at the respective resolutions:
#   ATAC (32 bp)   → 3 × 32 = 96 bp half-width
#   ChIP (128 bp)  → 3 × 128 = 384 bp half-width
SMOOTH_SIGMA_BINS: float = 3.0

# Prominence threshold as a fraction of the global median non-zero signal.
# Matches the spirit of MACS2's fold-change-over-control enrichment cutoff.
PROMINENCE_FRACTION: float = 0.5

# Hard minimum prominence — guards against zero-signal tracks where
# PROMINENCE_FRACTION × median_nonzero would also be near zero.
MIN_PROMINENCE_ABSOLUTE: float = 1e-4

# Minimum peak width in bins (avoids single-bin spikes).
MIN_PEAK_WIDTH_BINS: int = 2

# Minimum distance between adjacent peaks in bins.
# 200 bp / resolution mirrors MACS2's --bw default shift model.
#   ATAC (32 bp)  → ceil(200/32) = 7 bins
#   ChIP (128 bp) → ceil(200/128) = 2 bins
MIN_PEAK_DISTANCE_BP: int = 200   # converted to bins per-track in peak_caller.py

# Relative height for peak_widths (half-prominence, matching MACS2 summit
# width at half-maximum concept).
PEAK_WIDTH_REL_HEIGHT: float = 0.5

# ---------------------------------------------------------------------------
# Contact-map filtering
# ---------------------------------------------------------------------------
# Contacts below this threshold are treated as noise and dropped.
CONTACT_MIN_FREQ: float = 1e-4

# ---------------------------------------------------------------------------
# Token output schemas
# ---------------------------------------------------------------------------

# Column order for peak token Parquet files
PEAK_TOKEN_COLUMNS = [
    "token_id",
    "chrom",
    "start",          # 0-based peak left boundary (bp)
    "end",            # 0-based peak right boundary (bp, exclusive)
    "summit",         # absolute genomic position of signal maximum
    "auc",            # numpy.trapz(signal[start:end]) × resolution  (signal × bp)
    "signal_value",   # raw predicted value at summit bin
    "prominence",     # summit height above surrounding baseline
    "track_name",     # full name from .json sidecar
    "assay",          # atac / chip_histone / chip_tf
]

# Column order for contact token Parquet files
CONTACT_TOKEN_COLUMNS = [
    "token_id",
    "chrom",
    "bin1_start",     # 0-based left boundary of anchor 1 (bp)
    "bin1_end",
    "bin2_start",     # 0-based left boundary of anchor 2 (bp)
    "bin2_end",
    "contact_freq",   # predicted contact frequency (KR-equivalent normalised)
    "track_name",
    "resolution",     # bin size in bp
]

# Max rows written to the human-readable .tsv sidecar
TSV_MAX_ROWS: int = 10_000

# ---------------------------------------------------------------------------
# Fixed-bin tokenizer
# ---------------------------------------------------------------------------
# Resolutions to produce.  At 2048 bp, contact maps are included at their
# native resolution; below 2048 bp contact maps are excluded.
FIXED_BIN_RESOLUTIONS: list[int] = [128, 256, 512, 2048]

# Resolution at which contact maps are predicted (matches AlphaGenome output)
CONTACT_MAP_RESOLUTION: int = 2048

# ---------------------------------------------------------------------------
# cCRE tokenizer — ENCODE classification thresholds
# ---------------------------------------------------------------------------
# Z-score threshold for "high" signal (p < 0.05 one-tailed, ENCODE standard)
CCRE_Z_THRESHOLD: float = 1.64

# TSS proximity thresholds (bp) for PLS vs ELS classification
TSS_PLS_DIST: int = 200       # within 200 bp → promoter-like (PLS)
TSS_ELS_DIST: int = 2_000     # within 2 kb   → proximal enhancer-like (pELS)

# GENCODE v46 hg38 basic annotation GTF URL and local cache filename
GENCODE_GTF_URL: str = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_46/gencode.v46.basic.annotation.gtf.gz"
)
GENCODE_GTF_FILENAME: str = "gencode.v46.basic.annotation.gtf.gz"

# cCRE class labels (matches ENCODE Registry terminology)
CCRE_CLASSES = ["PLS", "pELS", "dELS", "CTCF-only", "DNase-H3K4me3", "low-DNase"]
