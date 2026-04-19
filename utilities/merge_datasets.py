"""
merge_datasets.py

A utility script to combine two separate NumPy (.npy) datasets into a single, 
globally shuffled dataset. It extracts a perfectly balanced (50/50) split 
of training and validation data from both sources. It uses memory-mapping and 
batch processing to safely handle massive datasets.
"""

import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap

def parse_args():
    """
    Parses command-line arguments for merging two datasets with 50/50 balanced sampling.
    """
    p = argparse.ArgumentParser(
        description="Randomly merge train and validation data from two dataset directories "
                    "with balanced 50/50 sampling from each dataset separately (full random shuffle) "
                    "and reduce to exact counts"
    )

    # Input directories
    p.add_argument(
        "dataset_dir1",
        help="First input dataset directory. Expected structure:\n"
                    "  chunks.npy\n"
                    "  references.npy\n"
                    "  reference_lengths.npy\n"
                    "  validation/\n"
                    "    chunks.npy\n"
                    "    references.npy\n"
                    "    reference_lengths.npy"
    )
    p.add_argument(
        "dataset_dir2",
        help="Second input dataset directory (same structure as dataset_dir1)"
    )

    # Output and sizing configuration
    p.add_argument(
        "-o", "--out-dir",
        default="./merged",
        dest="out_dir",
        help="Output directory (default: ./merged)"
    )
    p.add_argument(
        "--train-count",
        type=int,
        default=1000000,
        dest="train_count",
        help="Exact number of training samples to output (default: 1.000.000)"
    )
    p.add_argument(
        "--val-count",
        type=int,
        default=50000,
        dest="val_count",
        help="Exact number of validation samples to output (default: 50.000)"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        dest="batch_size",
        help="Batch size for memory-efficient disk I/O (default: 10.000)"
    )

    return p.parse_args()

def prepare_source_selections(n1, n2, target1, target2, rng):
    """
    Computes which rows to pick from each source and where they should land in the 
    final merged file to ensure a perfect global shuffle.
    """
    # Randomly select specific indices from each source's available pool
    local1 = rng.choice(n1, size=target1, replace=False)
    local2 = rng.choice(n2, size=target2, replace=False)

    # Create boolean masks for efficient filtering during the batch loop
    is1 = np.zeros(n1, dtype=bool)
    is1[local1] = True
    is2 = np.zeros(n2, dtype=bool)
    is2[local2] = True

    # Interleave indices: Shuffle all available output slots, then assign 
    # specific slots to Source 1 and the remaining to Source 2.
    all_pos = np.arange(target1 + target2)
    rng.shuffle(all_pos)
    pos1 = all_pos[:target1]
    pos2 = all_pos[target1:]

    return is1, is2, pos1, pos2

def write_selected_batches(
    src_chunks,
    src_refs,
    src_ref_lens,
    is_selected,
    out_chunks,
    out_refs,
    out_ref_lens,
    output_positions,
    batch_size,
):
    """
    Iterates through source files in batches, filtering for selected rows and 
    writing them to their pre-shuffled positions in the output memmap.
    """
    write_ptr = 0
    n = src_chunks.shape[0]

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        
        # Determine which rows in the current batch were selected during setup
        local_is = is_selected[start:end]
        if local_is.any():
            # Extract only the selected data from the batch
            selected_chunks = src_chunks[start:end][local_is]
            selected_refs = src_refs[start:end][local_is]
            selected_lens = src_ref_lens[start:end][local_is]
            n_w = selected_chunks.shape[0]

            # Map the selected batch rows to their globally shuffled output indices
            pos = output_positions[write_ptr : write_ptr + n_w]
            out_chunks[pos] = selected_chunks
            out_refs[pos] = selected_refs
            out_ref_lens[pos] = selected_lens
            
            write_ptr += n_w

    return write_ptr

def main():
    args = parse_args()

    # Configuration and directory setup
    val_subdir = "validation"
    seed = 0
    
    # Check whether the required directories already exist
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, val_subdir), exist_ok=True)

    # Initialize memory-mapped views for all input sources (Read-Only)
    # We load both Train and Val for both Source 1 and Source 2
    def get_paths(d, sub=""):
        return (os.path.join(d, sub, "chunks.npy"), 
                os.path.join(d, sub, "references.npy"), 
                os.path.join(d, sub, "reference_lengths.npy"))

    t1_p = get_paths(args.dataset_dir1)
    v1_p = get_paths(args.dataset_dir1, val_subdir)
    t2_p = get_paths(args.dataset_dir2)
    v2_p = get_paths(args.dataset_dir2, val_subdir)

    # Load sources into memmaps
    chunks_t1 = np.load(t1_p[0], mmap_mode='r'); refs_t1 = np.load(t1_p[1], mmap_mode='r'); lens_t1 = np.load(t1_p[2], mmap_mode='r')
    chunks_t2 = np.load(t2_p[0], mmap_mode='r'); refs_t2 = np.load(t2_p[1], mmap_mode='r'); lens_t2 = np.load(t2_p[2], mmap_mode='r')
    chunks_v1 = np.load(v1_p[0], mmap_mode='r'); refs_v1 = np.load(v1_p[1], mmap_mode='r'); lens_v1 = np.load(v1_p[2], mmap_mode='r')
    chunks_v2 = np.load(v2_p[0], mmap_mode='r'); refs_v2 = np.load(v2_p[1], mmap_mode='r'); lens_v2 = np.load(v2_p[2], mmap_mode='r')

    # Quick validation of source consistency
    n_t1, n_t2 = chunks_t1.shape[0], chunks_t2.shape[0]
    n_v1, n_v2 = chunks_v1.shape[0], chunks_v2.shape[0]

    rng = np.random.default_rng(seed)

    # Calculate 50/50 split targets, distributing the remainder randomly if count is odd
    def get_balanced_targets(total, count1, count2):
        h = total // 2
        r = total % 2
        t1 = h + (1 if r and rng.random() < 0.5 else 0)
        t2 = total - t1
        if count1 < t1 or count2 < t2:
            raise ValueError(f"Insufficient data for balanced split: Need {t1}/{t2}, have {count1}/{count2}")
        return t1, t2

    target_t1, target_t2 = get_balanced_targets(args.train_count, n_t1, n_t2)
    target_v1, target_v2 = get_balanced_targets(args.val_count, n_v1, n_v2)

    print(f"Merging Train: {target_t1} from Dir1, {target_t2} from Dir2")
    print(f"Merging Val:   {target_v1} from Dir1, {target_v2} from Dir2")

    # Generate the global shuffle mapping
    is_t1, is_t2, pos_t1, pos_t2 = prepare_source_selections(n_t1, n_t2, target_t1, target_t2, rng)
    is_v1, is_v2, pos_v1, pos_v2 = prepare_source_selections(n_v1, n_v2, target_v1, target_v2, rng)

    # Setup output memmaps
    c_shape = chunks_t1.shape[1:]
    r_shape = refs_t1.shape[1:]
    
    # Function to initialize output files
    def init_out(path, count, shape, dtype):
        return open_memmap(path, mode='w+', dtype=dtype, shape=(count, *shape))

    train_chunks = init_out(os.path.join(args.out_dir, "chunks.npy"), args.train_count, c_shape, chunks_t1.dtype)
    val_chunks = init_out(os.path.join(args.out_dir, val_subdir, "chunks.npy"), args.val_count, c_shape, chunks_v1.dtype)
    
    train_refs = init_out(os.path.join(args.out_dir, "references.npy"), args.train_count, r_shape, refs_t1.dtype)
    val_refs = init_out(os.path.join(args.out_dir, val_subdir, "references.npy"), args.val_count, r_shape, refs_v1.dtype)
    
    train_lens = init_out(os.path.join(args.out_dir, "reference_lengths.npy"), args.train_count, (), lens_t1.dtype)
    val_lens = init_out(os.path.join(args.out_dir, val_subdir, "reference_lengths.npy"), args.val_count, (), lens_v1.dtype)

    # Execute the writes
    t_ptr = write_selected_batches(chunks_t1, refs_t1, lens_t1, is_t1, train_chunks, train_refs, train_lens, pos_t1, args.batch_size)
    t_ptr += write_selected_batches(chunks_t2, refs_t2, lens_t2, is_t2, train_chunks, train_refs, train_lens, pos_t2, args.batch_size)

    v_ptr = write_selected_batches(chunks_v1, refs_v1, lens_v1, is_v1, val_chunks, val_refs, val_lens, pos_v1, args.batch_size)
    v_ptr += write_selected_batches(chunks_v2, refs_v2, lens_v2, is_v2, val_chunks, val_refs, val_lens, pos_v2, args.batch_size)

    # Flush all buffers to disk
    for m in [train_chunks, val_chunks, train_refs, val_refs, train_lens, val_lens]:
        m.flush()

    print(f"\nMerge complete. Processed {t_ptr} training and {v_ptr} validation rows.")

if __name__ == "__main__":
    main()
