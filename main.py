#!/usr/bin/env python3
"""
Coverage per position per chromosome from BAM files.

This script computes per-base read coverage for each chromosome present in 
a reference FASTA file (or a set of user-specified regions) using a sorted 
BAM file. 

Coverage is computed using pysam.count_coverage(), summing A/C/G/T counts
per position. Results are plotted as coverage vs genomic position.

Usage
python main.py --bam sample.sorted.bam --fasta reference.fasta --out coverage.png
python main.py --bam sample.sorted.bam --fasta reference.fasta \
    --regions chr1:1000-2000 chr2:5000-8000 --out regions.png

Optional arguments allow selecting a single chromosome, selecting one or more
regions in the format 'chr:start-end' (1-based, inclusive), defining bin size,
or adjusting figure size.

Requirements
- pysam
- numpy
- matplotlib
"""

import argparse
import pysam
import numpy as np
import matplotlib.pyplot as plt
import sys


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot per-position coverage per chromosome."
    )
    parser.add_argument(
        "--bam",
        required=True,
        help="Input BAM file (must be sorted and indexed)."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Reference FASTA file used for alignment."
    )
    parser.add_argument(
        "--chrom",
        default=None,
        help="Optional: process only this chromosome/contig."
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        metavar="CHR:START-END",
        help=(
            "Optional: one or more regions to process instead of the full "
            "contig set, in the format 'chr:start-end' (1-based, inclusive). "
            "Example: --regions chr1:1000-2000 chr2:5000-8000. "
            "If supplied, --chrom is ignored."
        ),
    )
    parser.add_argument(
        "--out",
        default="coverage.png",
        help="Output image file (default: coverage.png)."
    )
    parser.add_argument(
        "--binsize",
        type=int,
        default=1,
        help="Bin size in bases for coverage aggregation (default: 1, no binning)."
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=15.0,
        help="Figure width in inches (default: 15)."
    )
    parser.add_argument(
        "--figheight",
        type=float,
        default=3.0,
        help="Figure height per chromosome in inches (default: 3)."
    )
    parser.add_argument(
        "--max-clamp",
        type=float,
        default=None,
        help="Clamp the maximum y-axis value to this number (optional)."
    )

    return parser.parse_args()


def parse_region(region_str):
    """
    Parse a region string of the form 'chr:start-end' (1-based, inclusive).

    Parameters
    ----------
    region_str : str
        Region string, e.g. 'chr1:1000-2000'.

    Returns
    -------
    tuple of (str, int, int)
        (chrom, start, end) where start and end are **1-based inclusive**
        coordinates as given by the user. Conversion to 0-based half-open
        coordinates is performed in `main`.
    """
    if ":" not in region_str:
        raise ValueError(
            f"Region '{region_str}' must be in the format 'chr:start-end'"
        )
    chrom, coords = region_str.split(":", 1)
    if "-" not in coords:
        raise ValueError(
            f"Region '{region_str}' must be in the format 'chr:start-end'"
        )
    start_s, end_s = coords.split("-", 1)
    try:
        start = int(start_s)
        end = int(end_s)
    except ValueError:
        raise ValueError(
            f"Region '{region_str}' coordinates must be integers"
        )
    if start < 1 or end < start:
        raise ValueError(
            f"Region '{region_str}' has invalid coordinates "
            f"(start must be >= 1 and end >= start)"
        )
    return chrom, start, end


def compute_coverage(bam, fasta, chrom, start, end, binsize):
    """
    Compute per-base coverage for a genomic interval, including zeros.

    Parameters
    ----------
    bam : pysam.AlignmentFile
        Open BAM file handle.
    fasta : pysam.FastaFile
        Open FASTA file handle.
    chrom : str
        Chromosome/contig name.
    start : int
        0-based start coordinate (inclusive).
    end : int
        0-based end coordinate (exclusive).
    binsize : int
        Bin size in bases for coverage aggregation.

    Returns
    -------
    np.ndarray
        1D array of coverage values (length = end - start when binsize == 1,
        otherwise the number of bins).
    """
    chrom_len = fasta.get_reference_length(chrom)
    if start < 0 or end > chrom_len or end <= start:
        raise ValueError(
            f"Invalid interval {chrom}:{start}-{end} "
            f"(chromosome length is {chrom_len})"
        )

    # count_coverage returns four arrays: A, C, G, T
    cov = bam.count_coverage(
        contig=chrom,
        start=start,
        end=end,
        quality_threshold=0,
    )

    # Sum nucleotide-specific counts to get total depth
    depth = np.sum(np.array(cov), axis=0)

    if len(depth) != (end - start):
        raise RuntimeError(
            f"Coverage length mismatch for {chrom}:{start}-{end}"
        )

    if binsize > 1:
        n_bins = int(np.ceil((end - start) / binsize))
        depth = np.array([
            depth[i * binsize : (i + 1) * binsize].mean()
            for i in range(n_bins)
        ])

    return depth


def main():
    """
    Main execution logic.
    """
    args = parse_args()

    print("Opening BAM and FASTA files...", file=sys.stderr)

    bam = pysam.AlignmentFile(args.bam, "rb")
    fasta = pysam.FastaFile(args.fasta)

    # Determine regions/intervals to process
    if args.regions:
        intervals = []
        for region_str in args.regions:
            chrom, start1, end1 = parse_region(region_str)
            if chrom not in fasta.references:
                raise ValueError(
                    f"Chromosome '{chrom}' (from region '{region_str}') "
                    f"not found in FASTA."
                )
            chrom_len = fasta.get_reference_length(chrom)
            if end1 > chrom_len:
                raise ValueError(
                    f"Region '{region_str}' exceeds chromosome length "
                    f"({chrom_len})."
                )
            intervals.append((chrom, start1 - 1, end1))
    elif args.chrom:
        if args.chrom not in fasta.references:
            raise ValueError(f"Chromosome '{args.chrom}' not found in FASTA.")
        chrom_len = fasta.get_reference_length(args.chrom)
        intervals = [(args.chrom, 0, chrom_len)]
    else:
        intervals = [
            (chrom, 0, fasta.get_reference_length(chrom))
            for chrom in fasta.references
        ]

    print(f"Processing {len(intervals)} region(s)...", file=sys.stderr)

    # Prepare figure
    fig, axes = plt.subplots(
        len(intervals),
        1,
        figsize=(args.figwidth, args.figheight * len(intervals)),
        sharex=False
    )

    # Handle single-chromosome case
    if len(intervals) == 1:
        axes = [axes]

    # Compute and plot coverage per interval
    for ax, (chrom, start, end) in zip(axes, intervals):
        chrom_len = fasta.get_reference_length(chrom)
        if start == 0 and end == chrom_len:
            label = chrom
        else:
            label = f"{chrom}:{start + 1}-{end}"

        print(f"Computing coverage for {label}...", file=sys.stderr)

        depth = compute_coverage(bam, fasta, chrom, start, end, args.binsize)

        x = (
            start + np.arange(len(depth)) * args.binsize
            if args.binsize > 1
            else start + np.arange(len(depth))
        )

        ax.plot(x, depth, linewidth=0.4)
        ax.set_ylabel(label)
        ax.set_xlim(start, end)

    axes[-1].set_xlabel("Genomic position (bp)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    bam.close()
    fasta.close()

    print(f"Done. Figure written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
