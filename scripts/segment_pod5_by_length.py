#!/usr/bin/env python3

"""
segment_pod5.py

Segments long POD5 reads into fixed-length windows. 
Each segment is written as a standalone read with a unique, 
deterministic UUID derived from the original Read ID and its segment index.
"""

import argparse
from pathlib import Path
import sys
import uuid
import copy
import numpy as np
import pod5 as p5

def parse_args():
    """Parses CLI arguments for input/output paths and window length."""
    p = argparse.ArgumentParser(
        description="Segment POD5 signals into fixed-length windows."
    )
    p.add_argument(
        "input",
        help="Path to input .pod5 file"
    )
    p.add_argument(
        "length",
        type=int,
        help="Target segment length (N samples)"
    )
    p.add_argument(
        "-o", "--output",
        help="Optional output path (default: {input}_segmented.pod5)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_segmented.pod5")

    # Target window size (N)
    window_size = args.length
    
    # Namespace used for generating deterministic UUIDs (v5)
    SEGMENT_NAMESPACE = uuid.NAMESPACE_DNS

    with p5.Reader(str(in_path)) as reader, p5.Writer(str(out_path)) as writer:
        for record in reader.reads():
            signal = record.signal
            
            # Skip reads shorter than the target window
            if signal is None or len(signal) < window_size:
                continue

            # 1. Convert the record to a mutable Read object
            # This captures all original metadata (pore info, calibration, etc.)
            base_read = record.to_read()
            orig_id_str = str(base_read.read_id)
            n_segments = len(signal) // window_size

            for j in range(n_segments):
                # 2. Slice the signal into a non-overlapping window
                # Copy ensures memory independence; int16 is the standard POD5 storage type.
                start_idx = j * window_size
                end_idx = (j + 1) * window_size
                seg_signal = signal[start_idx : end_idx].copy().astype(np.int16)

                # 3. Generate a Deterministic Unique ID
                # Format: OriginalID + ".s" + Index (e.g., "uuid.s0")
                new_uuid = uuid.uuid5(SEGMENT_NAMESPACE, f"{orig_id_str}.s{j}")

                # 4. Create the segment by deep-copying the original read
                # Deep copy is essential to avoid modifying the same metadata object in memory.
                seg_read = copy.deepcopy(base_read)
                seg_read.read_id = new_uuid
                seg_read.signal = seg_signal

                # 5. Update temporal metadata
                # Adjusts the sample offset so the segment knows its place in the original time series.
                if hasattr(seg_read, "start_sample"):
                    seg_read.start_sample = int(base_read.start_sample + start_idx)

                # 6. Append the segment to the new POD5 file
                writer.add_read(seg_read)

    print(f"Segmentation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    main()
