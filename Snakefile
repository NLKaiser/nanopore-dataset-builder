import os
import glob

# --- Configuration & Parameters ---
configfile: "config/config.yaml"

IN_DIR = os.path.abspath(config.get("input_dir", "."))
OUT_DIR = config.get("output_dir", "output")
REFERENCES = config.get("references", "*.fasta")

# Logic Flags
DO_SEGMENT = config.get("segment", False)

# Numerical Parameters (Defaults maintained)
SEGMENT_LENGTH = config.get("segment_length", 5000)
MIN_LENGTH = config.get("min_length", 0)
MAX_LENGTH = config.get("max_length", 500)
MAPQ = config.get("mapq", 20)
IDENTITY = config.get("identity", 0.92)
MAX_SOFT_CLIP = config.get("max_soft_clip", 2)
COVERAGE = config.get("coverage", 0.80)
PAD_LENGTH = config.get("pad_length", 500)
SHIFT = config.get("shift", 0)
SCALE = config.get("scale", 1)

# Model Settings
MODEL = config.get("model", "dna_r10.4.1_e8.2_400bps_sup@v5.2.0")
DEVICE = config.get("device", "auto")

# --- File Discovery ---
FAST5_PATHS = glob.glob(os.path.join(IN_DIR, "**", "*.fast5"), recursive=True)
EXISTING_POD5 = glob.glob(os.path.join(IN_DIR, "**", "*.pod5"), recursive=True)

# Generate sample names from relative paths to avoid naming collisions
SAMPLES = {
    os.path.relpath(f, IN_DIR).replace(os.sep, "_").replace(".fast5", ""): f
    for f in FAST5_PATHS
}

# --- Input Helper Functions ---
def get_basecall_input(wildcards):
    """Determines if we basecall the segmented or the merged file."""
    if DO_SEGMENT:
        return os.path.join(OUT_DIR, "segment_pod5", "segmented.pod5")
    else:
        return os.path.join(OUT_DIR, "merge_pod5", "merged.pod5")

# --- Target Definitions ---
ALL_TARGETS = [
    os.path.join(OUT_DIR, "extract_labels_from_sam", "labels.tsv")
]

if DO_SEGMENT:
    ALL_TARGETS.extend([
        os.path.join(OUT_DIR, "dataset", "chunks.npy"),
        os.path.join(OUT_DIR, "dataset", "references.npy"),
        os.path.join(OUT_DIR, "dataset", "reference_lengths.npy")
    ])

# --- Rules ---

rule all:
    input:
        ALL_TARGETS

rule convert_to_pod5:
    input:
        lambda wildcards: SAMPLES[wildcards.sample]
    output:
        os.path.join(OUT_DIR, "convert_to_pod5", "{sample}.pod5")
    shell:
        """
        mkdir -p {OUT_DIR}/convert_to_pod5
        pod5 convert fast5 {input} --output {output} --force
        """

rule merge_pod5:
    input:
        converted = expand(os.path.join(OUT_DIR, "convert_to_pod5", "{sample}.pod5"), sample=SAMPLES.keys()),
        existing = EXISTING_POD5
    output:
        os.path.join(OUT_DIR, "merge_pod5", "merged.pod5")
    run:
        # Filter: ensure files exist and are not empty
        candidates = list(input.converted) + list(input.existing)
        valid_inputs = [f for f in candidates if os.path.exists(f) and os.path.getsize(f) > 0]
        
        # Safety: prevent circular merging (if output file is somehow in glob)
        out_abs = os.path.abspath(str(output))
        valid_inputs = [f for f in valid_inputs if os.path.abspath(f) != out_abs]

        if valid_inputs:
            inputs_str = " ".join(valid_inputs)
            shell("pod5 merge {inputs_str} -o {output} --force")
        else:
            print("Notice: No valid POD5 files found to merge.")
            shell("touch {output}")

rule segment_pod5:
    input:
        os.path.join(OUT_DIR, "merge_pod5", "merged.pod5")
    output:
        os.path.join(OUT_DIR, "segment_pod5", "segmented.pod5")
    log:
        os.path.join(OUT_DIR, "logs", "segment_pod5.log")
    shell:
        "python3 scripts/segment_pod5_by_length.py {input} {SEGMENT_LENGTH} -o {output} > {log} 2>&1"

rule basecall:
    input:
        get_basecall_input
    output:
        os.path.join(OUT_DIR, "basecall", "basecalled.fastq")
    shell:
        """
        if [ -s {input} ]; then
            dorado basecaller --device {DEVICE} --emit-fastq {MODEL} {input} > {output}
        else
            touch {output}
        fi
        """

rule minimap2_index:
    input:
        REFERENCES
    output:
        os.path.join(OUT_DIR, "minimap2", "references.mmi")
    shell:
        "minimap2 -d {output} {input}"

rule minimap2_map:
    input:
        seqs = os.path.join(OUT_DIR, "basecall", "basecalled.fastq"),
        idx = os.path.join(OUT_DIR, "minimap2", "references.mmi")
    output:
        os.path.join(OUT_DIR, "minimap2", "alignment.sam")
    shell:
        "minimap2 -a {input.idx} {input.seqs} > {output}"

rule extract_labels_from_sam:
    input:
        os.path.join(OUT_DIR, "minimap2", "alignment.sam")
    output:
        os.path.join(OUT_DIR, "extract_labels_from_sam", "labels.tsv")
    log:
        os.path.join(OUT_DIR, "logs", "extract_labels_from_sam.log")
    shell:
        "python3 scripts/extract_labels_from_sam.py --minlen {MIN_LENGTH} --maxlen {MAX_LENGTH} "
        "--mapq {MAPQ} --identity {IDENTITY} --coverage {COVERAGE} --max-soft-clip {MAX_SOFT_CLIP} "
        "{input} -o {output} > {log} 2>&1"

rule make_dataset_from_labels:
    input:
        labels = os.path.join(OUT_DIR, "extract_labels_from_sam", "labels.tsv"),
        references = REFERENCES,
        pod5 = os.path.join(OUT_DIR, "segment_pod5", "segmented.pod5")
    output:
        os.path.join(OUT_DIR, "dataset", "chunks.npy"),
        os.path.join(OUT_DIR, "dataset", "references.npy"),
        os.path.join(OUT_DIR, "dataset", "reference_lengths.npy")
    params:
        out_dir = os.path.join(OUT_DIR, "dataset")
    log:
        os.path.join(OUT_DIR, "logs", "make_dataset_from_labels.log")
    shell:
        "python3 scripts/make_dataset_from_labels.py --tsv {input.labels} --fasta {input.references} "
        "--pod5 {input.pod5} --padlen {PAD_LENGTH} --shift {SHIFT} --scale {SCALE} --out-dir {params.out_dir} > {log} 2>&1"
