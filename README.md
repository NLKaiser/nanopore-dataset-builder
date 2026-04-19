# Nanopore Dataset Builder

A Snakemake pipeline that automates the creation of labeled training datasets from raw nanopore sequence data.

This workflow handles the entire process from raw `FAST5` or `POD5` files to final `.npy` training chunks. It performs conversion, merging, segmentation, basecalling (via Dorado), alignment (via Minimap2), and rigorous label extraction to prepare data for machine learning applications.

## Features
* **Conversion:** Automatically converts `FAST5` to `POD5`.
* **Segmentation:** Splits long reads into manageable chunks (e.g., 5000 signals).
* **Basecalling:** Integration with Dorado for high-accuracy basecalling.
* **Labeling:** Aligns reads to a reference and filters by Soft Clipping, MapQ, Identity, and Coverage to ensure high-quality labels.
* **Dataset Creation:** Outputs numpy arrays (`chunks.npy`, `references.npy`, `reference_lengths.npy`) ready for machine learning.

## Prerequisites

1.  **Conda / Mamba:** You need a package manager to handle dependencies.
2.  **Dorado:** This pipeline requires the `dorado` executable to be installed and available in your system path.
    * [Download Dorado here](https://github.com/nanoporetech/dorado)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NLKaiser/nanopore-dataset-builder.git
    cd nanopore-dataset-builder
    ```

2.  **Create the environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate nanopore-dataset-builder
    ```

## Configuration

All parameters are managed in `config.yaml`. Edit this file to match your data locations and filtering criteria.

## Usage

Once configured, run the pipeline using Snakemake.  

    
    snakemake --configfile config.yaml --cores 4
    

## Output Structure

The pipeline organizes results into the directory defined in output_dir (default: output/).

### Explanation of Key Outputs
`dataset/chunks.npy`: A numpy array containing the raw normalized signal chunks extracted from the POD5 files.

`dataset/references.npy`: The corresponding "ground truth" sequence for each chunk, derived from the alignment to the reference genome.

`extract_labels_from_sam/labels.tsv`: A summary file containing the filtering metrics (Identity, Soft Clipping, Coverage) for every read that passed the quality checks.

## Utilities

`split_dataset.py`: A utility to randomly sample and split paired NumPy datasets into training and validation sets.
  
    python split_dataset.py \
        chunks.npy \
        references.npy \
        reference_lengths.npy \
        --out-dir ./dataset \
        --train-count 1000000 \
        --val-count 50000
  
`merge_datasets.py`: This utility combines two separate NumPy datasets into a single, globally shuffled dataset using a balanced 50/50 sampling strategy from both sources.

    python merge_datasets.py \
        ./path_to_dataset_1 \
        ./path_to_dataset_2 \
        --out-dir ./merged \
        --train-count 1000000 \
        --val-count 50000
