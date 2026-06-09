#!/usr/bin/env python3

"""
extract_labels_from_sam.py

Read a SAM file and produce a TSV of high-confidence alignment labels.
Filtering criteria: MAPQ, identity, coverage, and soft-clipping limits.
"""

import sys
import re
import argparse
from collections import defaultdict

def parse_args():
    """
    Handles CLI arguments for filtering thresholds.
    """
    p = argparse.ArgumentParser(
        description="Extract high-confidence alignment labels from a SAM file"
    )
    p.add_argument(
        "input",
        help="Input SAM file"
    )
    p.add_argument(
        "-o", "--output",
        default="labels.tsv",
        help="Output TSV file (default: labels.tsv)"
    )
    p.add_argument(
        "--minlen",
        default=0,
        help="Minimum read length (default: 0)"
    )
    p.add_argument(
        "--maxlen",
        default=500,
        help="Maximum read length (default: 500)"
    )
    p.add_argument(
        "--mapq",
        type=int,
        default=20,
        help="Minimum MAPQ to accept (default: 20)"
    )
    p.add_argument(
        "--identity",
        type=float,
        default=0.92,
        help="Minimum alignment identity (default: 0.92)"
    )
    p.add_argument(
        "--coverage",
        type=float,
        default=0.80,
        help="Minimum aligned coverage of the read (default: 0.80)"
    )
    p.add_argument(
        "--max-soft-clip",
        type=int,
        default=2,
        dest="max_soft_clip",
        help="Max allowed soft-clipped bases at either end (default: 2)"
    )
    return p.parse_args()

# Regex for parsing CIGAR strings (e.g., 100M2D10S)
cigar_re = re.compile(r'(\d+)([MIDNSHP=XB])')

def parse_cigar(cigar):
    """
    Parses a CIGAR string to calculate reference consumption, 
    aligned bases, and clipping lengths.
    """
    if cigar == '*' or cigar == '':
        return [], 0, 0, 0, 0
    
    ops = [(int(m.group(1)), m.group(2)) for m in cigar_re.finditer(cigar)]
    ref_consumed = 0
    aln_bases = 0
    leading_soft = 0
    trailing_soft = 0

    for length, op in ops:
        # M, =, X consume both query and reference
        if op in ('M', '=', 'X'):
            ref_consumed += length
            aln_bases += length
        # D and N consume only reference
        elif op in ('D', 'N'):
            ref_consumed += length

    # Identify soft-clipping at the start and end of the read
    if ops:
        if ops[0][1] == 'S':
            leading_soft = ops[0][0]
        if ops[-1][1] == 'S':
            trailing_soft = ops[-1][0]

    return ops, ref_consumed, aln_bases, leading_soft, trailing_soft

def get_tag_value(fields, tag):
    """
    Extracts specific SAM tags (like NM or AS) from the optional fields.
    """
    prefix = tag + ':'
    for f in fields[11:]:
        if f.startswith(prefix):
            parts = f.split(':', 2)
            if len(parts) >= 3:
                return parts[2]
    return None

def main():
    args = parse_args()
    
    # PASS 1: Identify reads that are "ambiguous"
    # We flag any QNAME that has secondary (0x100) or supplementary (0x800) alignments
    ambiguous = set()
    total_alignments = 0

    print("PASS 1: scanning for secondary/supplementary alignments...")
    with open(args.input, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('@'):
                continue
            total_alignments += 1
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 11:
                continue
            
            qname = cols[0]
            try:
                flag = int(cols[1])
            except ValueError:
                flag = 0

            # Check flags: 256 (secondary) or 2048 (supplementary)
            if (flag & 0x100) != 0 or (flag & 0x800) != 0:
                ambiguous.add(qname)

    print(f"Total alignments scanned: {total_alignments}")
    print(f"Reads excluded due to multi-mapping: {len(ambiguous)}")

    # PASS 2: Filter primary alignments and write output
    print("PASS 2: evaluating primary alignments and writing labels...")
    kept = 0

    with open(args.input, 'rt', encoding='utf-8', errors='replace') as fh, \
         open(args.output, 'wt', encoding='utf-8') as out:

        # Write TSV header
        out.write("\t".join([
            "QNAME","RNAME","POS","END","STRAND","MAPQ",
            "IDENTITY","ALN_LEN","READ_LEN","CIGAR","NM","AS"
        ]) + "\n")

        for line in fh:
            if line.startswith('@'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 11:
                continue

            qname = cols[0]
            flag = int(cols[1])
            rname = cols[2]
            pos = cols[3]
            mapq = int(cols[4]) if cols[4].isdigit() else 0
            cigar = cols[5]
            seq = cols[9]

            # Primary Filter Logic:
            # 1. Skip unmapped reads (0x4)
            # 2. Skip reads flagged in PASS 1 as ambiguous
            # 3. Skip secondary/supplementary alignments (0x100 | 0x800)
            if (flag & 0x4) != 0:
                continue
            if qname in ambiguous:
                continue
            if (flag & 0x100) != 0 or (flag & 0x800) != 0:
                continue

            # Parse CIGAR and extract optional tags
            ops, ref_consumed, aln_bases, leading_soft, trailing_soft = parse_cigar(cigar)
            read_len = len(seq) if seq and seq != '*' else None
            nm_val = get_tag_value(cols, 'NM')
            as_val = get_tag_value(cols, 'AS')

            try:
                nm = int(nm_val) if nm_val is not None else None
                ascore = int(as_val) if as_val is not None else None
            except ValueError:
                nm = ascore = None

            # Skip records missing essential stats
            if read_len is None or aln_bases == 0 or nm is None:
                continue

            # Read length filtering
            if read_len < args.minlen:
                continue

            if read_len > args.maxlen:
                continue

            # Calculate metrics
            identity = 1.0 - (nm / aln_bases)
            cov = aln_bases / read_len

            # Apply quality thresholds
            if mapq < args.mapq:
                continue
            if identity < args.identity:
                continue
            if cov < args.coverage:
                continue
            if leading_soft > args.max_soft_clip or trailing_soft > args.max_soft_clip:
                continue

            # Calculate reference end position and strand
            try:
                pos_int = int(pos)
            except ValueError:
                continue

            ref_end = pos_int + ref_consumed - 1 if ref_consumed > 0 else pos_int
            strand = '-' if (flag & 16) != 0 else '+'

            # Format and write data
            out.write("\t".join([
                qname, rname, str(pos_int), str(ref_end), strand, str(mapq),
                f"{identity:.4f}", str(aln_bases), str(read_len), cigar,
                str(nm), str(ascore) if ascore is not None else ''
            ]) + "\n")

            kept += 1

    print(f"Finished. Kept {kept} reads. Results: {args.output}")

if __name__ == "__main__":
    main()
