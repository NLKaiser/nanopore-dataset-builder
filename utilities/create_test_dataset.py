"""
create_test_dataset.py

A utility script to randomly sample a fixed number of examples from an
existing NumPy (.npy) dataset and move them into a dedicated test split.
The selected samples are written to a test/ subdirectory, while the
remaining samples are written back as the training dataset. The script
uses memory-mapping and batch processing to efficiently handle massive
files while avoiding excessive RAM usage.
"""

import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap


def parse_args():
    """
    Handles command-line arguments for test set extraction.
    """
    p = argparse.ArgumentParser(
        description="Extract a random test set from .npy files and save the remaining samples separately"
    )

    # Input file arguments
    p.add_argument(
        "chunks_file",
        nargs="?",
        default="chunks.npy",
        help="Input chunks .npy file (default: chunks.npy)"
    )
    p.add_argument(
        "refs_file",
        nargs="?",
        default="references.npy",
        help="Input references .npy file (default: references.npy)"
    )
    p.add_argument(
        "ref_lens_file",
        nargs="?",
        default="reference_lengths.npy",
        help="Input reference lengths .npy file (default: reference_lengths.npy)"
    )

    # Output and sizing configuration
    p.add_argument(
        "-o", "--out-dir",
        default="./dataset",
        dest="out_dir",
        help="Output directory (default: ./dataset)"
    )
    p.add_argument(
        "--test-count",
        type=int,
        default=50000,
        dest="test_count",
        help="Number of test samples to extract (default: 50.000)"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        dest="batch_size",
        help="I/O batch size to manage memory usage during extraction (default: 10.000)"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Configuration and directory setup
    out_dir = args.out_dir
    test_subdir = "test"
    seed = 0  # Hardcoded for reproducibility

    # Check whether the required directories already exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, test_subdir), exist_ok=True)

    # Open source files in memory-map read-only mode to handle large files without RAM exhaustion
    chunks_src = np.load(args.chunks_file, mmap_mode="r")
    refs_src = np.load(args.refs_file, mmap_mode="r")
    ref_lens_src = np.load(args.ref_lens_file, mmap_mode="r")

    # Validate that all input files have matching row counts
    n = chunks_src.shape[0]
    if not (refs_src.shape[0] == n and ref_lens_src.shape[0] == n):
        raise ValueError("Source files must have the same number of rows.")

    # Ensure requested split doesn't exceed available data
    if args.test_count >= n:
        raise ValueError(
            f"Requested {args.test_count} test samples, but dataset only has {n} rows."
        )

    train_count = n - args.test_count

    # Generate shuffled indices for the extraction
    rng = np.random.default_rng(seed)
    print("Selecting and shuffling global indices...")

    shuffled_idx = rng.permutation(n)

    test_idx = shuffled_idx[:args.test_count]
    train_idx = shuffled_idx[args.test_count:]

    print(
        f"Dataset size: {n} | "
        f"Remaining: {train_count} | "
        f"Test: {args.test_count}"
    )

    # Determine shapes for output memmaps based on input dimensions
    chunk_shape = (
        (chunks_src.shape[1],)
        if chunks_src.ndim == 1
        else chunks_src.shape[1:]
    )

    ref_shape = (
        (refs_src.shape[1],)
        if refs_src.ndim == 1
        else refs_src.shape[1:]
    )

    # Define output file paths
    base_chunks = os.path.basename(args.chunks_file)
    base_refs = os.path.basename(args.refs_file)
    base_ref_lens = os.path.basename(args.ref_lens_file)

    # Initialize output memmaps (mode 'w+' creates the files on disk)
    train_chunks = open_memmap(
        os.path.join(out_dir, base_chunks),
        mode="w+",
        dtype=chunks_src.dtype,
        shape=(train_count, *chunk_shape),
    )

    test_chunks = open_memmap(
        os.path.join(out_dir, test_subdir, base_chunks),
        mode="w+",
        dtype=chunks_src.dtype,
        shape=(args.test_count, *chunk_shape),
    )

    train_refs = open_memmap(
        os.path.join(out_dir, base_refs),
        mode="w+",
        dtype=refs_src.dtype,
        shape=(train_count, *ref_shape),
    )

    test_refs = open_memmap(
        os.path.join(out_dir, test_subdir, base_refs),
        mode="w+",
        dtype=refs_src.dtype,
        shape=(args.test_count, *ref_shape),
    )

    train_ref_lens = open_memmap(
        os.path.join(out_dir, base_ref_lens),
        mode="w+",
        dtype=ref_lens_src.dtype,
        shape=(train_count,),
    )

    test_ref_lens = open_memmap(
        os.path.join(out_dir, test_subdir, base_ref_lens),
        mode="w+",
        dtype=ref_lens_src.dtype,
        shape=(args.test_count,),
    )

    # Process remaining training data in batches to optimize disk I/O performance
    print("Writing remaining training data...")
    for start in range(0, train_count, args.batch_size):
        end = min(train_count, start + args.batch_size)
        batch_indices = train_idx[start:end]

        train_chunks[start:end] = chunks_src[batch_indices]
        train_refs[start:end] = refs_src[batch_indices]
        train_ref_lens[start:end] = ref_lens_src[batch_indices]

    # Process test data in batches
    print("Writing test data...")
    for start in range(0, args.test_count, args.batch_size):
        end = min(args.test_count, start + args.batch_size)
        batch_indices = test_idx[start:end]

        test_chunks[start:end] = chunks_src[batch_indices]
        test_refs[start:end] = refs_src[batch_indices]
        test_ref_lens[start:end] = ref_lens_src[batch_indices]

    # Explicitly flush changes to disk to ensure data integrity
    for m in [
        train_chunks,
        test_chunks,
        train_refs,
        test_refs,
        train_ref_lens,
        test_ref_lens,
    ]:
        m.flush()

    print("\nExtraction complete.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
