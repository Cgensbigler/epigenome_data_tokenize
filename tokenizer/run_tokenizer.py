"""
Predicted Track Tokenizer — CLI entry point.

Three tokenization schemes are supported, selected via ``--scheme``:

  peak (default)
    Call ENCODE-style peaks on ATAC, ChIP-histone, and TF-ChIP tracks, and
    build a sparse contact-frequency matrix from contact-map windows.

    Output layout:
      <output_dir>/atac/{chrom}.parquet
      <output_dir>/chip_histone/{chrom}.parquet
      <output_dir>/chip_tf/{chrom}.parquet
      <output_dir>/contact_maps/{chrom}.parquet

  fixed_bin
    Dense feature matrix at each of 128 / 256 / 512 / 2048 bp resolutions.
    At 2048 bp, contact-map summary columns are included when contact-map
    NPZ files are present.

    Output layout:
      <output_dir>/fixed_bin/128bp/{chrom}.parquet
      <output_dir>/fixed_bin/256bp/{chrom}.parquet
      <output_dir>/fixed_bin/512bp/{chrom}.parquet
      <output_dir>/fixed_bin/2048bp/{chrom}.parquet

  ccre
    ENCODE cCRE-style regulatory element tokens.  Requires the ``peak``
    scheme to have already been run (reads ATAC peak Parquet files).
    Downloads GENCODE v46 GTF on first use (~50 MB, cached).

    Output layout:
      <output_dir>/ccre/{chrom}.parquet

Usage examples
--------------
  # Peak tokens (all chroms)
  python tokenizer/run_tokenizer.py --scheme peak \\
      --input-dir output/ --output-dir output/tokens/

  # Fixed-bin feature matrices (single chrom smoke-test)
  python tokenizer/run_tokenizer.py --scheme fixed_bin --chroms chr22 \\
      --input-dir output/ --output-dir output/tokens/

  # cCRE tokens (requires peak tokens to exist)
  python tokenizer/run_tokenizer.py --scheme ccre --chroms chr22 \\
      --input-dir output/ --output-dir output/tokens/ \\
      --peak-tokens-dir output/tokens/ --gtf-cache-dir data/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap: allow running as `python tokenizer/run_tokenizer.py` from
# the project root OR as `python run_tokenizer.py` from inside tokenizer/.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Only add the project root — root config.py is found here, and the
# tokenizer package is resolved as tokenizer/ relative to the root.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as root_config                              # noqa: E402  (project root config)
from tokenizer.config import (                            # noqa: E402
    ALL_1D_ASSAYS,
    ASSAY_ATAC,
    ASSAY_CHIP_HISTONE,
    ASSAY_CHIP_TF,
    ASSAY_CONTACT_MAPS,
    ATAC_DOWNSAMPLE_RESOLUTION,
    FIXED_BIN_RESOLUTIONS,
)
from tokenizer.peak_caller import call_peaks_chrom        # noqa: E402
from tokenizer.contact_tokenizer import build_contact_tokens  # noqa: E402
from tokenizer.fixed_bin_tokenizer import build_fixed_bin_tokens  # noqa: E402
from tokenizer.ccre_tokenizer import build_ccre_tokens    # noqa: E402
from tokenizer.writers import merge_chroms, write_tokens  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Tokenize AlphaGenome predicted tracks into Parquet files.\n\n"
            "Schemes: peak (default) | fixed_bin | ccre"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Common arguments ---------------------------------------------------
    p.add_argument(
        "--scheme", default="peak",
        choices=["peak", "fixed_bin", "ccre"],
        help=(
            "Tokenization scheme: "
            "'peak' (ENCODE-style peak tokens + contact matrix), "
            "'fixed_bin' (dense feature matrix at multiple resolutions), "
            "'ccre' (ENCODE cCRE classification over ATAC peaks)."
        ),
    )
    p.add_argument(
        "--input-dir", "-i", required=True,
        help="Root directory containing atac/, chip_histone/, chip_tf/, contact_maps/ subdirs.",
    )
    p.add_argument(
        "--output-dir", "-o", required=True,
        help="Root directory for token output.  Sub-directories are created automatically.",
    )
    p.add_argument(
        "--chroms", nargs="+", default=None, metavar="CHROM",
        help="Chromosomes to process.  Defaults to all hg38 standard chromosomes.",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip chromosomes whose output Parquet already exists.",
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--no-merge", action="store_true", default=False,
        help="Skip writing all_chroms.parquet after processing.",
    )

    # ---- Peak-scheme arguments ----------------------------------------------
    peak_grp = p.add_argument_group("peak scheme options")
    peak_grp.add_argument(
        "--assays", nargs="+",
        default=ALL_1D_ASSAYS + [ASSAY_CONTACT_MAPS],
        choices=ALL_1D_ASSAYS + [ASSAY_CONTACT_MAPS],
        metavar="ASSAY",
        help="Which assays to tokenize in peak scheme.  "
             "Choices: atac chip_histone chip_tf contact_maps.",
    )
    peak_grp.add_argument(
        "--stride", type=int, default=root_config.DEFAULT_STRIDE,
        help="Stride used during prediction (needed for contact-map de-duplication).",
    )
    peak_grp.add_argument(
        "--atac-resolution", type=int, default=ATAC_DOWNSAMPLE_RESOLUTION,
        help="Down-sample ATAC from 1 bp to this resolution before peak calling.",
    )

    # ---- Fixed-bin scheme arguments -----------------------------------------
    fixedbin_grp = p.add_argument_group("fixed_bin scheme options")
    fixedbin_grp.add_argument(
        "--resolutions", nargs="+", type=int, default=FIXED_BIN_RESOLUTIONS,
        metavar="RES",
        help="Bin resolutions in bp.  Defaults to all: 128 256 512 2048.",
    )
    fixedbin_grp.add_argument(
        "--contact-map-dir", default=None,
        help="Directory containing contact-map window NPZ files.  "
             "Defaults to <input-dir>/contact_maps/.  "
             "Used only at 2048 bp resolution.",
    )
    fixedbin_grp.add_argument(
        "--contact-mode", default="summary", choices=["summary", "full"],
        help="Contact summary mode at 2048 bp: 'summary' (total per bin) or 'full' (vector).",
    )

    # ---- cCRE scheme arguments ----------------------------------------------
    ccre_grp = p.add_argument_group("ccre scheme options")
    ccre_grp.add_argument(
        "--peak-tokens-dir", default=None,
        help="Directory containing peak token Parquet files from the 'peak' scheme.  "
             "Defaults to <output-dir>/ (assumes peak scheme was run to the same dir).",
    )
    ccre_grp.add_argument(
        "--gtf-cache-dir", default="data/",
        help="Directory to cache the GENCODE v46 GTF file (~50 MB, downloaded once).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-chromosome workers — peak scheme
# ---------------------------------------------------------------------------

def _process_1d_assay(
    chrom: str,
    assay: str,
    input_dir: Path,
    output_dir: Path,
    resume: bool,
    atac_downsample_res: int,
) -> bool:
    """Call peaks for one (chrom, assay) pair and write tokens."""
    assay_in_dir  = input_dir  / assay
    assay_out_dir = output_dir / assay

    npz_path  = assay_in_dir / f"{chrom}.npz"
    json_path = assay_in_dir / f"{chrom}.json"
    out_path  = assay_out_dir / f"{chrom}.parquet"

    if resume and out_path.exists():
        log.info("[%s] %s — already tokenized, skipping.", chrom, assay)
        return True

    if not npz_path.exists():
        log.warning("[%s] %s — input NPZ not found: %s", chrom, assay, npz_path)
        return False
    if not json_path.exists():
        log.warning("[%s] %s — metadata JSON not found: %s", chrom, assay, json_path)
        return False

    log.info("[%s] %s — calling peaks …", chrom, assay)
    try:
        df = call_peaks_chrom(
            npz_path=npz_path,
            json_path=json_path,
            assay=assay,
            atac_downsample_res=atac_downsample_res,
        )
    except Exception as exc:
        log.error("[%s] %s — peak calling failed: %s", chrom, assay, exc, exc_info=True)
        return False

    if df.empty:
        log.warning("[%s] %s — no peaks; writing empty token file.", chrom, assay)

    write_tokens(df, assay_out_dir, chrom)
    return True


def _process_contact_maps(
    chrom: str,
    input_dir: Path,
    output_dir: Path,
    stride: int,
    resume: bool,
) -> bool:
    """Build contact tokens for one chromosome and write them."""
    contact_in_dir  = input_dir  / ASSAY_CONTACT_MAPS
    contact_out_dir = output_dir / ASSAY_CONTACT_MAPS
    out_path        = contact_out_dir / f"{chrom}.parquet"

    if resume and out_path.exists():
        log.info("[%s] contact_maps — already tokenized, skipping.", chrom)
        return True

    if not contact_in_dir.exists():
        log.warning("[%s] contact_maps — input dir not found: %s", chrom, contact_in_dir)
        return False

    log.info("[%s] contact_maps — building sparse contact matrix …", chrom)
    try:
        df = build_contact_tokens(
            contact_map_dir=contact_in_dir,
            chrom=chrom,
            stride=stride,
        )
    except Exception as exc:
        log.error("[%s] contact_maps — failed: %s", chrom, exc, exc_info=True)
        return False

    if df.empty:
        log.warning("[%s] contact_maps — no contacts above threshold; writing empty file.", chrom)

    write_tokens(df, contact_out_dir, chrom)
    return True


# ---------------------------------------------------------------------------
# Per-chromosome workers — fixed_bin scheme
# ---------------------------------------------------------------------------

def _process_fixed_bin(
    chrom: str,
    resolution: int,
    input_dir: Path,
    output_dir: Path,
    contact_map_dir: Path | None,
    contact_mode: str,
    resume: bool,
) -> bool:
    """Build a fixed-bin feature matrix for one (chrom, resolution) pair."""
    res_label  = f"{resolution}bp"
    res_out_dir = output_dir / "fixed_bin" / res_label
    out_path   = res_out_dir / f"{chrom}.parquet"

    if resume and out_path.exists():
        log.info("[%s] fixed_bin %s — already exists, skipping.", chrom, res_label)
        return True

    log.info("[%s] fixed_bin %s — building feature matrix …", chrom, res_label)
    try:
        df = build_fixed_bin_tokens(
            chrom=chrom,
            input_dir=input_dir,
            resolution=resolution,
            contact_map_dir=contact_map_dir,
            contact_mode=contact_mode,
        )
    except Exception as exc:
        log.error(
            "[%s] fixed_bin %s — failed: %s", chrom, res_label, exc, exc_info=True
        )
        return False

    if df.empty:
        log.warning("[%s] fixed_bin %s — empty result; skipping write.", chrom, res_label)
        return False

    write_tokens(df, res_out_dir, chrom)
    return True


# ---------------------------------------------------------------------------
# Per-chromosome workers — ccre scheme
# ---------------------------------------------------------------------------

def _process_ccre(
    chrom: str,
    input_dir: Path,
    output_dir: Path,
    peak_tokens_dir: Path,
    gtf_cache_dir: Path,
    resume: bool,
) -> bool:
    """Build cCRE tokens for one chromosome."""
    ccre_out_dir = output_dir / "ccre"
    out_path     = ccre_out_dir / f"{chrom}.parquet"

    if resume and out_path.exists():
        log.info("[%s] ccre — already tokenized, skipping.", chrom)
        return True

    log.info("[%s] ccre — classifying ATAC peaks …", chrom)
    try:
        df = build_ccre_tokens(
            chrom=chrom,
            input_dir=input_dir,
            peak_tokens_dir=peak_tokens_dir,
            gtf_cache_dir=gtf_cache_dir,
        )
    except Exception as exc:
        log.error("[%s] ccre — failed: %s", chrom, exc, exc_info=True)
        return False

    if df.empty:
        log.warning("[%s] ccre — no cCRE tokens produced; skipping write.", chrom)
        return False

    write_tokens(df, ccre_out_dir, chrom)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chroms = args.chroms or root_config.ALL_CHROMS
    for c in chroms:
        if c not in root_config.HG38_CHROM_SIZES:
            log.error("Unknown chromosome '%s'. Use standard hg38 names.", c)
            sys.exit(1)

    log.info(
        "Tokenizer settings:\n"
        "  scheme       : %s\n"
        "  chromosomes  : %s\n"
        "  input dir    : %s\n"
        "  output dir   : %s\n"
        "  resume       : %s",
        args.scheme,
        chroms,
        input_dir,
        output_dir,
        args.resume,
    )

    # =========================================================================
    if args.scheme == "peak":
    # =========================================================================
        assays = args.assays
        log.info("  assays       : %s\n  stride       : %s bp\n  ATAC res     : %s bp",
                 assays, args.stride, args.atac_resolution)

        for chrom in tqdm(chroms, desc="Chromosomes [peak]", unit="chrom"):
            for assay in assays:
                if assay == ASSAY_CONTACT_MAPS:
                    _process_contact_maps(
                        chrom=chrom,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        stride=args.stride,
                        resume=args.resume,
                    )
                else:
                    _process_1d_assay(
                        chrom=chrom,
                        assay=assay,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        resume=args.resume,
                        atac_downsample_res=args.atac_resolution,
                    )

        if not args.no_merge:
            log.info("Merging per-chromosome Parquet files …")
            for assay in assays:
                assay_out_dir = output_dir / assay
                if assay_out_dir.exists():
                    merge_chroms(assay_out_dir, chroms)

    # =========================================================================
    elif args.scheme == "fixed_bin":
    # =========================================================================
        resolutions = args.resolutions
        contact_map_dir = (
            Path(args.contact_map_dir)
            if args.contact_map_dir
            else input_dir / ASSAY_CONTACT_MAPS
        )

        log.info(
            "  resolutions  : %s bp\n"
            "  contact dir  : %s\n"
            "  contact mode : %s",
            resolutions, contact_map_dir, args.contact_mode,
        )

        for chrom in tqdm(chroms, desc="Chromosomes [fixed_bin]", unit="chrom"):
            for resolution in resolutions:
                _process_fixed_bin(
                    chrom=chrom,
                    resolution=resolution,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    contact_map_dir=contact_map_dir,
                    contact_mode=args.contact_mode,
                    resume=args.resume,
                )

        if not args.no_merge:
            log.info("Merging per-chromosome Parquet files …")
            for resolution in resolutions:
                res_label   = f"{resolution}bp"
                res_out_dir = output_dir / "fixed_bin" / res_label
                if res_out_dir.exists():
                    merge_chroms(res_out_dir, chroms)

    # =========================================================================
    elif args.scheme == "ccre":
    # =========================================================================
        peak_tokens_dir = (
            Path(args.peak_tokens_dir)
            if args.peak_tokens_dir
            else output_dir
        )
        gtf_cache_dir = Path(args.gtf_cache_dir)

        log.info(
            "  peak tokens  : %s\n  GTF cache    : %s",
            peak_tokens_dir, gtf_cache_dir,
        )

        for chrom in tqdm(chroms, desc="Chromosomes [ccre]", unit="chrom"):
            _process_ccre(
                chrom=chrom,
                input_dir=input_dir,
                output_dir=output_dir,
                peak_tokens_dir=peak_tokens_dir,
                gtf_cache_dir=gtf_cache_dir,
                resume=args.resume,
            )

        if not args.no_merge:
            log.info("Merging per-chromosome cCRE Parquet files …")
            ccre_out_dir = output_dir / "ccre"
            if ccre_out_dir.exists():
                merge_chroms(ccre_out_dir, chroms)

    else:
        log.error("Unknown scheme: %s", args.scheme)
        sys.exit(1)

    log.info("Tokenizer complete.")


if __name__ == "__main__":
    main()
