"""
split_dataset.py

A utility script to randomly sample and split large NumPy (.npy) datasets 
into distinct training and validation sets. It uses memory-mapping to 
efficiently handle massive files, writing the shuffled outputs directly to 
disk in batches.
"""

import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap

def parse_args():
    """
    Handles command-line arguments for dataset splitting and extraction.
    """
    p = argparse.ArgumentParser(
        description="Extract exact amounts of random, shuffled training and validation data from .npy files"
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
        help="Output directory (default: ./data)"
    )
    p.add_argument(
        "--train-count",
        type=int,
        default=1000000,
        dest="train_count",
        help="Number of training samples to extract (default: 1.000.000)"
    )
    p.add_argument(
        "--val-count",
        type=int,
        default=50000,
        dest="val_count",
        help="Number of validation samples to extract (default: 50.000)"
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
    val_subdir = "validation"
    seed = 0  # Hardcoded for reproducibility
    
    # Check whether the required directories already exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, val_subdir), exist_ok=True)

    # Open source files in memory-map read-only mode to handle large files without RAM exhaustion
    chunks_src = np.load(args.chunks_file, mmap_mode='r')
    refs_src = np.load(args.refs_file, mmap_mode='r')
    ref_lens_src = np.load(args.ref_lens_file, mmap_mode='r')

    # Validate that all input files have matching row counts
    n = chunks_src.shape[0]
    if not (refs_src.shape[0] == n and ref_lens_src.shape[0] == n):
        raise ValueError("Source files must have the same number of rows.")

    # Ensure requested split doesn't exceed available data
    total_requested = args.train_count + args.val_count
    if total_requested > n:
        raise ValueError(f"Requested {total_requested} samples, but dataset only has {n} rows.")

    # Generate shuffled indices for the entire extraction
    rng = np.random.default_rng(seed)
    print("Selecting and shuffling global indices...")
    selected_idx = rng.choice(n, size=total_requested, replace=False)

    # Slice the shuffled pool into validation and training sets
    val_idx = selected_idx[:args.val_count]
    train_idx = selected_idx[args.val_count:]

    print(f"Dataset size: {n} | Train: {args.train_count} | Val: {args.val_count}")

    # Determine shapes for output memmaps based on input dimensions
    chunk_shape = (chunks_src.shape[1],) if chunks_src.ndim == 1 else chunks_src.shape[1:]
    ref_shape = (refs_src.shape[1],) if refs_src.ndim == 1 else refs_src.shape[1:]

    # Define output file paths
    base_chunks = os.path.basename(args.chunks_file)
    base_refs = os.path.basename(args.refs_file)
    base_ref_lens = os.path.basename(args.ref_lens_file)

    # Initialize output memmaps (mode 'w+' creates the files on disk)
    train_chunks = open_memmap(os.path.join(out_dir, base_chunks), mode='w+', 
                               dtype=chunks_src.dtype, shape=(args.train_count, *chunk_shape))
    val_chunks = open_memmap(os.path.join(out_dir, val_subdir, base_chunks), mode='w+', 
                             dtype=chunks_src.dtype, shape=(args.val_count, *chunk_shape))

    train_refs = open_memmap(os.path.join(out_dir, base_refs), mode='w+', 
                             dtype=refs_src.dtype, shape=(args.train_count, *ref_shape))
    val_refs = open_memmap(os.path.join(out_dir, val_subdir, base_refs), mode='w+', 
                           dtype=refs_src.dtype, shape=(args.val_count, *ref_shape))

    train_ref_lens = open_memmap(os.path.join(out_dir, base_ref_lens), mode='w+', 
                                 dtype=ref_lens_src.dtype, shape=(args.train_count,))
    val_ref_lens = open_memmap(os.path.join(out_dir, val_subdir, base_ref_lens), mode='w+', 
                               dtype=ref_lens_src.dtype, shape=(args.val_count,))

    # Process training data in batches to optimize disk I/O performance
    print("Writing shuffled training data...")
    for start in range(0, args.train_count, args.batch_size):
        end = min(args.train_count, start + args.batch_size)
        batch_indices = train_idx[start:end]
        
        train_chunks[start:end] = chunks_src[batch_indices]
        train_refs[start:end] = refs_src[batch_indices]
        train_ref_lens[start:end] = ref_lens_src[batch_indices]

    # Process validation data in batches
    print("Writing shuffled validation data...")
    for start in range(0, args.val_count, args.batch_size):
        end = min(args.val_count, start + args.batch_size)
        batch_indices = val_idx[start:end]
        
        val_chunks[start:end] = chunks_src[batch_indices]
        val_refs[start:end] = refs_src[batch_indices]
        val_ref_lens[start:end] = ref_lens_src[batch_indices]

    # Explicitly flush changes to disk to ensure data integrity
    for m in [train_chunks, val_chunks, train_refs, val_refs, train_ref_lens, val_ref_lens]:
        m.flush()

    print(f"\nExtraction complete.")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
