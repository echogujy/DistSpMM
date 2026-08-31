#!/usr/bin/env python3
"""Download and preprocess real (SuiteSparse) datasets for DistSPMM.

This is the pipeline used to prepare the real datasets in the paper: it downloads
a `.tar.gz` from the SuiteSparse Matrix Collection, extracts the `.mtx`, converts
it to DistSPMM's binary CSR format, and removes the intermediate `.mtx`.

Output (one directory per matrix under `data/`):
    data/<name>/<name>_indptr.bin   int64
    data/<name>/<name>_indices.bin  int64
    data/<name>/<name>_values.bin   float32

This matches `SparseMatrix::load_from_csr` in `src/sparse_matrix.h` and is the
format expected by `scripts/` and `run_all.sh`.

Run inside the `datapass` conda environment:
    conda activate datapass
    python3 tools/download_suitesparse.py --dataset nlpkkt160
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.io import mmread

# SuiteSparse Matrix Collection URLs (sparse.tamu.edu).
# `web-Google` is a small graph (0.9M x 0.9M, ~5 MB) used as a quick example.
DATA_DICT = {
    "nlpkkt160": "https://sparse.tamu.edu/MM/Schenk/nlpkkt160.tar.gz",
}


def download_data(data_name, save_path):
    """Download the .tar.gz for `data_name` into `save_path`. Returns its path."""
    url_name = DATA_DICT[data_name]
    file_name = url_name.split("/")[-1]   # e.g. "web-Google.tar.gz"
    dest = os.path.join(save_path, file_name)
    if not os.path.exists(dest):
        subprocess.run(["wget", url_name, "-P", save_path,
                        "--no-check-certificate"], check=True)
    else:
        print(f"{file_name} already exists.")
    return dest


def process_mtx_file(mtx_file_path, data_dir):
    """Convert a .mtx file to DistSPMM's binary format under `data_dir/<name>/`."""
    try:
        matrix = mmread(mtx_file_path).tocsr()
        name = os.path.splitext(os.path.basename(mtx_file_path))[0]
        print(f"Matrix read from {mtx_file_path} "
              f"Shape: {matrix.shape} dtype: {matrix.dtype} NNZ: {matrix.nnz}")

        out_dir = os.path.join(data_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        matrix.indptr.astype(np.int64).tofile(os.path.join(out_dir, f"{name}_indptr.bin"))
        matrix.indices.astype(np.int64).tofile(os.path.join(out_dir, f"{name}_indices.bin"))
        matrix.data.astype(np.float32).tofile(os.path.join(out_dir, f"{name}_values.bin"))

        # Remove the intermediate Matrix Market file.
        os.remove(mtx_file_path)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to process mtx file {mtx_file_path}: {e}")


def extract_and_process(file_path, data_dir):
    """Extract a downloaded archive and convert the contained .mtx file."""
    try:
        base_name = os.path.basename(file_path).replace(".tar.gz", "").replace(".tar", "")
        extracted_dir = os.path.join(os.path.dirname(file_path), base_name)
        mtx_file_path = os.path.join(extracted_dir, f"{base_name}.mtx")

        if not os.path.exists(mtx_file_path):
            print(f"Expected mtx file not found: {mtx_file_path}; extracting...", flush=True)
            subprocess.run(["tar", "-xzf", file_path, "-C", os.path.dirname(file_path)],
                           check=True)
            print(f"Successfully extracted: {file_path}", flush=True)

        process_mtx_file(mtx_file_path, data_dir)
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract and process {file_path}: {e}")


def processing_data(data_name, data_dir):
    file_path = download_data(data_name, data_dir)
    if file_path is not None:
        extract_and_process(file_path, data_dir)
        print(f"Data processing completed successfully for {data_name}.", flush=True)


def processing_data_all(data_names, data_dir, num_processes=4):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(lambda n: processing_data(n, data_dir), data_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None,
                        help="download & output root (default: <repo>/data)")
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="datasets to process (default: all in DATA_DICT)")
    parser.add_argument("--jobs", type=int, default=1, help="parallel workers")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "data")
    os.makedirs(args.data_dir, exist_ok=True)

    names = args.dataset if args.dataset else list(DATA_DICT.keys())
    processing_data_all(names, args.data_dir, num_processes=args.jobs)
