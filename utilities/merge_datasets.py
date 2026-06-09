"""
merge_datasets.py

A utility script to combine multiple separate NumPy (.npy) datasets into a single,
globally shuffled dataset. It extracts a weighted split of training and validation
data from each source according to user-provided ratios that sum to 1. 
The script uses memory-mapping and batch processing to safely handle massive datasets.
"""

import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap

def parse_args():
    """
    Parses command-line arguments for merging multiple datasets with weighted sampling.
    """
    p = argparse.ArgumentParser(
        description="Randomly merge train and validation data from multiple dataset directories "
                    "with weighted sampling from each dataset separately (full random shuffle) "
                    "and reduce to exact output counts"
    )

    # Input directories
    p.add_argument(
        "dataset_dirs",
        nargs="+",
        help="Input dataset directories. Each directory is expected to contain:\n"
             "  chunks.npy\n"
             "  references.npy\n"
             "  reference_lengths.npy\n"
             "  validation/\n"
             "    chunks.npy\n"
             "    references.npy\n"
             "    reference_lengths.npy"
    )

    # Mixing ratios
    p.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        required=True,
        dest="ratios",
        help="Mixing ratios for each dataset directory. Must have the same length as dataset_dirs "
             "and should sum to 1."
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
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        dest="seed",
        help="Random seed (default: 0)"
    )

    return p.parse_args()


def get_paths(d, sub=""):
    """
    Returns the standard file paths for a dataset directory.
    """
    return (
        os.path.join(d, sub, "chunks.npy"),
        os.path.join(d, sub, "references.npy"),
        os.path.join(d, sub, "reference_lengths.npy"),
    )


def load_source_dataset(dataset_dir):
    """
    Loads one source dataset (train + validation) as memory-mapped arrays.
    """
    val_subdir = "validation"

    t_p = get_paths(dataset_dir)
    v_p = get_paths(dataset_dir, val_subdir)

    chunks_t = np.load(t_p[0], mmap_mode="r")
    refs_t = np.load(t_p[1], mmap_mode="r")
    lens_t = np.load(t_p[2], mmap_mode="r")

    chunks_v = np.load(v_p[0], mmap_mode="r")
    refs_v = np.load(v_p[1], mmap_mode="r")
    lens_v = np.load(v_p[2], mmap_mode="r")

    return {
        "train": {
            "chunks": chunks_t,
            "refs": refs_t,
            "lens": lens_t,
        },
        "val": {
            "chunks": chunks_v,
            "refs": refs_v,
            "lens": lens_v,
        },
    }


def check_compatible_shapes_and_dtypes(sources):
    """
    Verifies that all source datasets have matching shapes and dtypes
    for train and validation splits.
    """
    first_train = sources[0]["train"]
    first_val = sources[0]["val"]

    expected_train_chunk_shape = first_train["chunks"].shape[1:]
    expected_train_ref_shape = first_train["refs"].shape[1:]
    expected_train_lens_shape = first_train["lens"].shape[1:]

    expected_val_chunk_shape = first_val["chunks"].shape[1:]
    expected_val_ref_shape = first_val["refs"].shape[1:]
    expected_val_lens_shape = first_val["lens"].shape[1:]

    expected_train_chunk_dtype = first_train["chunks"].dtype
    expected_train_ref_dtype = first_train["refs"].dtype
    expected_train_lens_dtype = first_train["lens"].dtype

    expected_val_chunk_dtype = first_val["chunks"].dtype
    expected_val_ref_dtype = first_val["refs"].dtype
    expected_val_lens_dtype = first_val["lens"].dtype

    for i, src in enumerate(sources[1:], start=2):
        tr = src["train"]
        va = src["val"]

        if tr["chunks"].shape[1:] != expected_train_chunk_shape:
            raise ValueError(f"Train chunks shape mismatch in dataset {i}")
        if tr["refs"].shape[1:] != expected_train_ref_shape:
            raise ValueError(f"Train references shape mismatch in dataset {i}")
        if tr["lens"].shape[1:] != expected_train_lens_shape:
            raise ValueError(f"Train reference_lengths shape mismatch in dataset {i}")

        if va["chunks"].shape[1:] != expected_val_chunk_shape:
            raise ValueError(f"Validation chunks shape mismatch in dataset {i}")
        if va["refs"].shape[1:] != expected_val_ref_shape:
            raise ValueError(f"Validation references shape mismatch in dataset {i}")
        if va["lens"].shape[1:] != expected_val_lens_shape:
            raise ValueError(f"Validation reference_lengths shape mismatch in dataset {i}")

        if tr["chunks"].dtype != expected_train_chunk_dtype:
            raise ValueError(f"Train chunks dtype mismatch in dataset {i}")
        if tr["refs"].dtype != expected_train_ref_dtype:
            raise ValueError(f"Train references dtype mismatch in dataset {i}")
        if tr["lens"].dtype != expected_train_lens_dtype:
            raise ValueError(f"Train reference_lengths dtype mismatch in dataset {i}")

        if va["chunks"].dtype != expected_val_chunk_dtype:
            raise ValueError(f"Validation chunks dtype mismatch in dataset {i}")
        if va["refs"].dtype != expected_val_ref_dtype:
            raise ValueError(f"Validation references dtype mismatch in dataset {i}")
        if va["lens"].dtype != expected_val_lens_dtype:
            raise ValueError(f"Validation reference_lengths dtype mismatch in dataset {i}")


def get_proportional_targets(total, ratios, available_counts, rng):
    """
    Converts a total desired output count into exact per-source integer counts
    using the provided ratios.
    This uses a largest-remainder strategy so the final integer targets sum
    exactly to 'total'.
    """
    ratios = np.asarray(ratios, dtype=float)
    raw = ratios * total

    targets = np.floor(raw).astype(int)
    remainder = total - int(targets.sum())

    if remainder > 0:
        frac = raw - targets
        # Randomised tie-breaking for equal fractional parts
        tie_break = rng.random(len(ratios)) * 1e-12
        order = np.argsort(-(frac + tie_break))
        targets[order[:remainder]] += 1

    for i, (avail, need) in enumerate(zip(available_counts, targets), start=1):
        if avail < need:
            raise ValueError(
                f"Insufficient data in dataset {i}: need {need}, have {avail}"
            )

    return targets


def prepare_source_selections(counts, targets, rng):
    """
    Computes which rows to pick from each source and where they should land in the
    final merged file to ensure a perfect global shuffle.
    """
    total = int(sum(targets))

    # Randomly assign output slots across all sources
    all_pos = np.arange(total)
    rng.shuffle(all_pos)

    output_positions = []
    offset = 0
    for t in targets:
        output_positions.append(all_pos[offset:offset + t])
        offset += t

    # Randomly select specific indices from each source's available pool
    source_masks = []
    for n, target in zip(counts, targets):
        local = rng.choice(n, size=target, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[local] = True
        source_masks.append(mask)

    return source_masks, output_positions


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
            pos = output_positions[write_ptr:write_ptr + n_w]
            out_chunks[pos] = selected_chunks
            out_refs[pos] = selected_refs
            out_ref_lens[pos] = selected_lens

            write_ptr += n_w

    return write_ptr


def init_out(path, count, shape, dtype):
    """
    Initialises an output memory-mapped file.
    """
    return open_memmap(path, mode="w+", dtype=dtype, shape=(count, *shape))


def main():
    args = parse_args()

    if len(args.dataset_dirs) != len(args.ratios):
        raise ValueError(
            f"dataset_dirs and --ratios must have the same length "
            f"({len(args.dataset_dirs)} != {len(args.ratios)})"
        )

    ratio_sum = float(np.sum(args.ratios))
    if not np.isclose(ratio_sum, 1.0, atol=1e-8):
        raise ValueError(f"--ratios must sum to 1.0, got {ratio_sum}")

    if any(r < 0 for r in args.ratios):
        raise ValueError("All ratios must be non-negative")

    # Normalise slightly to guard against tiny floating-point error
    ratios = np.asarray(args.ratios, dtype=float)
    ratios = ratios / ratios.sum()

    # Configuration and directory setup
    val_subdir = "validation"
    rng = np.random.default_rng(args.seed)

    # Check whether the required directories already exist
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, val_subdir), exist_ok=True)

    # Load all source datasets into memmaps
    sources = []
    for d in args.dataset_dirs:
        sources.append(load_source_dataset(d))

    # Validation of source consistency
    check_compatible_shapes_and_dtypes(sources)

    # Collect source sizes
    train_counts = [src["train"]["chunks"].shape[0] for src in sources]
    val_counts = [src["val"]["chunks"].shape[0] for src in sources]

    # Calculate exact per-source counts for train and validation
    target_train = get_proportional_targets(args.train_count, ratios, train_counts, rng)
    target_val = get_proportional_targets(args.val_count, ratios, val_counts, rng)

    print("Requested train counts per source:")
    for i, t in enumerate(target_train, start=1):
        print(f"  Source {i}: {t}")
    print("Requested validation counts per source:")
    for i, t in enumerate(target_val, start=1):
        print(f"  Source {i}: {t}")

    # Generate the global shuffle mapping for train and validation
    train_masks, train_positions = prepare_source_selections(train_counts, target_train, rng)
    val_masks, val_positions = prepare_source_selections(val_counts, target_val, rng)

    # Setup output memmaps
    train_chunks_shape = sources[0]["train"]["chunks"].shape[1:]
    train_refs_shape = sources[0]["train"]["refs"].shape[1:]
    train_lens_shape = sources[0]["train"]["lens"].shape[1:]

    val_chunks_shape = sources[0]["val"]["chunks"].shape[1:]
    val_refs_shape = sources[0]["val"]["refs"].shape[1:]
    val_lens_shape = sources[0]["val"]["lens"].shape[1:]

    train_chunks = init_out(
        os.path.join(args.out_dir, "chunks.npy"),
        args.train_count,
        train_chunks_shape,
        sources[0]["train"]["chunks"].dtype,
    )
    val_chunks = init_out(
        os.path.join(args.out_dir, val_subdir, "chunks.npy"),
        args.val_count,
        val_chunks_shape,
        sources[0]["val"]["chunks"].dtype,
    )

    train_refs = init_out(
        os.path.join(args.out_dir, "references.npy"),
        args.train_count,
        train_refs_shape,
        sources[0]["train"]["refs"].dtype,
    )
    val_refs = init_out(
        os.path.join(args.out_dir, val_subdir, "references.npy"),
        args.val_count,
        val_refs_shape,
        sources[0]["val"]["refs"].dtype,
    )

    train_lens = init_out(
        os.path.join(args.out_dir, "reference_lengths.npy"),
        args.train_count,
        train_lens_shape,
        sources[0]["train"]["lens"].dtype,
    )
    val_lens = init_out(
        os.path.join(args.out_dir, val_subdir, "reference_lengths.npy"),
        args.val_count,
        val_lens_shape,
        sources[0]["val"]["lens"].dtype,
    )

    # Execute the writes for each source
    train_written = 0
    val_written = 0

    for i, src in enumerate(sources):
        t_ptr = write_selected_batches(
            src["train"]["chunks"],
            src["train"]["refs"],
            src["train"]["lens"],
            train_masks[i],
            train_chunks,
            train_refs,
            train_lens,
            train_positions[i],
            args.batch_size,
        )
        train_written += t_ptr

        v_ptr = write_selected_batches(
            src["val"]["chunks"],
            src["val"]["refs"],
            src["val"]["lens"],
            val_masks[i],
            val_chunks,
            val_refs,
            val_lens,
            val_positions[i],
            args.batch_size,
        )
        val_written += v_ptr

    # Flush all buffers to disk
    for m in [train_chunks, val_chunks, train_refs, val_refs, train_lens, val_lens]:
        m.flush()

    print(f"\nMerge complete. Processed {train_written} training and {val_written} validation rows.")

if __name__ == "__main__":
    main()
