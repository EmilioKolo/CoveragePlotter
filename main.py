#!/usr/bin/env python3
"""
Coverage per position per chromosome from BAM files.

This script computes per-base read coverage for each chromosome present in a
reference FASTA file using a sorted BAM file. 

Coverage is computed using pysam.count_coverage(), summing A/C/G/T counts
per position. Results are plotted as coverage vs genomic position.

Usage
python main.py --bam sample.sorted.bam --fasta reference.fasta --out coverage.png

Optional arguments allow selecting a single chromosome, defining bin size or 
adjusting figure size.

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

    return parser.parse_args()


def compute_coverage(bam, fasta, chrom, binsize):
    """
    Compute per-base coverage for a single chromosome, including zeros.

    Parameters
    bam: pysam.AlignmentFile
        Open BAM file handle.
    fasta: pysam.FastaFile
        Open FASTA file handle.
    chrom: str
        Chromosome/contig name.

    Returns
    np.ndarray
        1D array of coverage values (length = chromosome length).
    """
    chrom_len = fasta.get_reference_length(chrom)

    # count_coverage returns four arrays: A, C, G, T
    cov = bam.count_coverage(
        contig=chrom,
        start=0,
        end=chrom_len,
        quality_threshold=0
    )

    # Sum nucleotide-specific counts to get total depth
    depth = np.sum(np.array(cov), axis=0)

    if len(depth) != chrom_len:
        raise RuntimeError(f"Coverage length mismatch for {chrom}")

    if binsize > 1:
        n_bins = int(np.ceil(chrom_len / binsize))
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

    # Determine chromosomes to process
    if args.chrom:
        if args.chrom not in fasta.references:
            raise ValueError(f"Chromosome '{args.chrom}' not found in FASTA.")
        chroms = [args.chrom]
    else:
        chroms = list(fasta.references)

    print(f"Processing {len(chroms)} chromosome(s)...", file=sys.stderr)

    # Prepare figure
    fig, axes = plt.subplots(
        len(chroms),
        1,
        figsize=(args.figwidth, args.figheight * len(chroms)),
        sharex=False
    )

    # Handle single-chromosome case
    if len(chroms) == 1:
        axes = [axes]

    # Compute and plot coverage per chromosome
    for ax, chrom in zip(axes, chroms):
        print(f"Computing coverage for {chrom}...", file=sys.stderr)

        depth = compute_coverage(bam, fasta, chrom, args.binsize)

        x = (
            np.arange(len(depth)) * args.binsize
            if args.binsize > 1
            else np.arange(len(depth))
        )

        ax.plot(x, depth, linewidth=0.4)
        ax.set_ylabel(chrom)
        ax.set_xlim(0, len(depth) * args.binsize)
        ax.set_ylim(bottom=-10)
        # Add a horizontal line through the x-axis (y=0)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='y=0')

    axes[-1].set_xlabel("Genomic position (bp)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    bam.close()
    fasta.close()

    print(f"Done. Figure written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
