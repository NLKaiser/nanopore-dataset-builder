#!/usr/bin/env python3

"""
make_training_from_pod5.py

Generates synchronized NumPy arrays from Oxford Nanopore POD5 signals 
and reference sequences for machine learning training. The signals are normalised
pA values.

Inputs:
 --tsv path  : TSV with header containing at least: QNAME, RNAME, POS, END, STRAND
 --fasta path: reference FASTA
 --pod5 path : a single POD5 file
 --padlen n  : integer, length to pad/truncate reference sequences
 --shift f   : float, shift value for the normalisation
 --scale f   : float, scale value for the normalisation

Outputs:
 - chunks.npy            (float32 array, shape (N_reads, signal_len))
 - references.npy        (uint8 array, shape (N_reads, padlen), 0-padded)
 - reference_lengths.npy (int32 array, true reference lengths)
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pod5 as _pod5

def parse_args():
    """Handles command-line configuration for dataset generation."""
    p = argparse.ArgumentParser(
        description="Create training arrays from TSV, FASTA and POD5"
    )
    p.add_argument("--tsv",
        required=True,
        help="Input TSV with QNAME, RNAME, POS, END, STRAND"
    )
    p.add_argument("--fasta",
        required=True,
        help="Reference FASTA file"
    )
    p.add_argument("--pod5",
        required=True,
        help="POD5 file containing raw signal data"
    )
    p.add_argument("--padlen",
        type=int,
        required=True,
        help="Fixed length for reference sequence padding"
    )
    p.add_argument("--shift",
        type=float,
        default="0.0",
        help="Normalisation of pA values, shift"
    )
    p.add_argument("--scale",
        type=float,
        default="1.0",
        help="Normalisation of pA values, scale"
    )
    p.add_argument("--out-dir",
        default=".",
        help="Output directory (default: current)"
    )
    p.add_argument("--skip-missing",
        action="store_true",
        help="Skip reads with errors instead of aborting"
    )
    return p.parse_args()

# Mapping nucleotides to integer labels: A=1, C=2, G=3, T/U=4. 
# 0 is reserved for padding.
NUC_MAP = {
    ord('A'): 1, ord('a'): 1,
    ord('C'): 2, ord('c'): 2,
    ord('G'): 3, ord('g'): 3,
    ord('T'): 4, ord('t'): 4,
    ord('U'): 4, ord('u'): 4,
}

def load_fasta(fasta_path: str) -> Dict[str, bytes]:
    """
    Loads an entire FASTA file into a dictionary of byte strings.
    Reads in binary mode to preserve memory and speed up processing.
    """
    seqs: Dict[str, bytearray] = {}
    cur = None
    with open(fasta_path, "rb") as fh:
        for raw in fh:
            if raw.startswith(b'>'):
                # Extract first word of header as contig name
                header = raw[1:].strip().split(None, 1)[0].decode("utf-8")
                cur = header
                seqs[cur] = bytearray()
            else:
                if cur is not None:
                    seqs[cur].extend(raw.strip())
    return {k: bytes(v) for k, v in seqs.items()}

def revcomp_bytes(bs: bytes) -> bytes:
    """Computes the reverse complement of a DNA byte string."""
    comp = {ord('A'):ord('T'), ord('a'):ord('t'),
            ord('C'):ord('G'), ord('c'):ord('g'),
            ord('G'):ord('C'), ord('g'):ord('c'),
            ord('T'):ord('A'), ord('t'):ord('a'),
            ord('U'):ord('A'), ord('u'):ord('a'),
            ord('N'):ord('N'), ord('n'):ord('n')}
    out = bytearray(len(bs))
    for i, b in enumerate(reversed(bs)):
        out[i] = comp.get(b, ord('N'))
    return bytes(out)

def encode_reference(subseq: bytes, padlen: int) -> Tuple[np.ndarray, int]:
    """
    Maps DNA characters to NUC_MAP integers and pads/truncates to padlen.
    Returns the integer array and the original sequence length.
    """
    L = len(subseq)
    enc = np.zeros((padlen,), dtype=np.int8)
    upto = min(L, padlen)
    
    if upto > 0:
        arr = np.frombuffer(subseq[:upto], dtype=np.uint8)
        mapped = np.zeros_like(arr, dtype=np.int8)
        for k, v in NUC_MAP.items():
            mapped[arr == k] = v
        enc[:upto] = mapped
        
    return enc, upto

def read_tsv_rows(tsv_path: str):
    """
    Generator that parses TSV rows into dictionaries.
    Ensures all required metadata columns are present.
    """
    with open(tsv_path, "rt", encoding="utf8") as fh:
        header = None
        for i, line in enumerate(fh):
            line = line.rstrip("\n")
            if i == 0:
                header = [h.strip() for h in line.split("\t")]
                header_u = [h.upper() for h in header]
                required = {'QNAME','RNAME','POS','END','STRAND'}
                if not required.issubset(set(header_u)):
                    raise SystemExit(f"TSV header missing columns. Required: {required}")
                yield ('__HEADER__', header, header_u)
                continue
            
            if not line: continue
            cols = line.split("\t")
            d = {header_u[j]: cols[j] if j < len(cols) else "" for j in range(len(header))}
            yield d

def get_signal_from_pod5_dataset(ds, read_id: str, shift: float, scale: float) -> np.ndarray:
    """
    Retrieves raw signal from POD5. Includes fallback logic to handle
    different POD5 API versions and UUID vs string ID formats.
    Returns normalised signal: (pA - shift) / scale.
    """
    import uuid
    read_obj = None
    
    # 1. Direct lookup attempt
    if hasattr(ds, "get_read"):
        try:
            read_obj = ds.get_read(read_id)
        except Exception:
            try:
                read_obj = ds.get_read(uuid.UUID(read_id))
            except Exception:
                read_obj = None

    # 2. Sequential fallback if index lookup fails
    if read_obj is None and hasattr(ds, "reads"):
        target_str = str(read_id).lower()
        for r in ds.reads():
            rid_obj = getattr(r, "read_id", None) or getattr(r, "id", None)
            if rid_obj and str(rid_obj).lower() == target_str:
                read_obj = r
                break

    if read_obj is None:
        raise FileNotFoundError(f"Read {read_id} not found in POD5 dataset")

    # 3. Signal extraction
    if not hasattr(read_obj, "signal_pa"):
        raise RuntimeError(f"Read {read_id} has no signal_pa")

    signal_pa = np.asarray(read_obj.signal_pa, dtype=np.float32)

    # 4. pA normalisation
    norm_signal = (signal_pa - shift) / scale

    return norm_signal.astype(np.float32).ravel()

def main():
    args = parse_args()
    
    # Check whether the required directory already exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load shared genomic context
    fasta = load_fasta(args.fasta)
    ds = _pod5.DatasetReader(args.pod5)

    signals_list, refs_enc_list, ref_lens = [], [], []
    skipped = 0

    # Main processing loop
    for row in read_tsv_rows(args.tsv):
        if isinstance(row, tuple): continue # Skip header metadata
        
        qname, rname = row['QNAME'], row['RNAME']
        try:
            # Convert 1-based genomic coords to 0-based slice
            pos, end = int(row['POS']), int(row['END'])
            start0, end0 = max(0, pos - 1), end
        except (ValueError, KeyError):
            skipped += 1
            continue

        if rname not in fasta:
            if args.skip_missing: skipped += 1; continue
            else: raise SystemExit(f"Contig {rname} missing from FASTA reference")

        # Extract sequence slice and handle strand orientation
        ref_seq = fasta[rname]
        sub = ref_seq[start0:min(len(ref_seq), end0)]
        
        if not sub:
            skipped += 1; continue

        if row.get('STRAND') == '-':
            sub = revcomp_bytes(sub)

        # Encode reference and fetch corresponding raw signal
        try:
            enc, L = encode_reference(sub, args.padlen)
            sig = get_signal_from_pod5_dataset(ds, qname, args.shift, args.scale)
            
            signals_list.append(sig)
            refs_enc_list.append(enc)
            ref_lens.append(L)
        except Exception as e:
            if args.skip_missing: 
                print(f"Skipping {qname}: {e}", file=sys.stderr)
                skipped += 1
                continue
            else: raise

    try:
        ds.close()
    except Exception:
        pass

    if not signals_list:
        raise SystemExit("No valid samples were processed. Check your TSV/POD5 inputs.")

    # Consolidate lists into final 2D NumPy arrays
    # Note: signals_list must have uniform length for np.stack to work
    try:
        chunks_arr = np.stack(signals_list).astype(np.float32)
    except ValueError:
        print("Error: Inconsistent signal lengths detected. Stacking failed.", file=sys.stderr)
        raise

    references_arr = np.stack(refs_enc_list).astype(np.int8)
    lengths_arr = np.asarray(ref_lens, dtype=np.int64)

    # Atomic save to output directory
    np.save(os.path.join(args.out_dir, "chunks.npy"), chunks_arr)
    np.save(os.path.join(args.out_dir, "references.npy"), references_arr)
    np.save(os.path.join(args.out_dir, "reference_lengths.npy"), lengths_arr)

    print(f"Dataset created: {len(signals_list)} samples saved (Skipped {skipped})", file=sys.stderr)

if __name__ == "__main__":
    main()
