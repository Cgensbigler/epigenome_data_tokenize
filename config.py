"""
Pipeline configuration for AlphaGenome K562 genome-wide track prediction.

Set ALPHAGENOME_API_KEY in your environment before running.
"""

import os
from alphagenome.models import dna_output

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
API_KEY: str = os.environ.get("ALPHAGENOME_API_KEY", "")

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
# 1 MB = 2^20 bp — maximum (and recommended) context window
WINDOW_SIZE: int = 1 << 20  # 1,048,576 bp

# Default stride: half the window (512 KB).  Overlap = window - stride.
DEFAULT_STRIDE: int = 1 << 19  # 524,288 bp

# Number of windows to send in one predict_intervals call.
# The API is well-suited for 1000s of predictions; keep batches modest to
# avoid timeouts and to allow checkpointing.
DEFAULT_BATCH_SIZE: int = 10

# Parallel workers passed to predict_intervals.
MAX_WORKERS: int = 5

# ---------------------------------------------------------------------------
# Cell type / ontology
# ---------------------------------------------------------------------------
# K562 (chronic myelogenous leukemia cell line) — EFO ontology
K562_ONTOLOGY_TERM: str = "EFO:0002067"

# ---------------------------------------------------------------------------
# Output types requested from the model
# ---------------------------------------------------------------------------
REQUESTED_OUTPUTS = [
    dna_output.OutputType.ATAC,
    dna_output.OutputType.CHIP_HISTONE,
    dna_output.OutputType.CHIP_TF,
    dna_output.OutputType.CONTACT_MAPS,
]

# Human-readable names used for output subdirectories
OUTPUT_TYPE_DIRS: dict[dna_output.OutputType, str] = {
    dna_output.OutputType.ATAC: "atac",
    dna_output.OutputType.CHIP_HISTONE: "chip_histone",
    dna_output.OutputType.CHIP_TF: "chip_tf",
    dna_output.OutputType.CONTACT_MAPS: "contact_maps",
}

# ---------------------------------------------------------------------------
# hg38 chromosome sizes (GRCh38.p13)
# Source: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes
# ---------------------------------------------------------------------------
HG38_CHROM_SIZES: dict[str, int] = {
    "chr1":  248956422,
    "chr2":  242193529,
    "chr3":  198295559,
    "chr4":  190214555,
    "chr5":  181538259,
    "chr6":  170805979,
    "chr7":  159345973,
    "chr8":  145138636,
    "chr9":  138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16":  90338345,
    "chr17":  83257441,
    "chr18":  80373285,
    "chr19":  58617616,
    "chr20":  64444167,
    "chr21":  46709983,
    "chr22":  50818468,
    "chrX":  156040895,
    "chrY":   57227415,
}

# Ordered list of all standard chromosomes (convenient for iteration)
ALL_CHROMS: list[str] = list(HG38_CHROM_SIZES.keys())
