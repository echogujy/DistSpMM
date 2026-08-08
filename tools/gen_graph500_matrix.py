#!/usr/bin/env python3
"""Generate a Graph500 R-MAT matrix and export it in DistSPMM's binary CSR format.

Graph500 is the *generated* (synthetic) dataset family in DistSPMM's experiments.
For the real (downloaded) SuiteSparse datasets, use `tools/download_suitesparse.py`.

Requires: numpy, scipy, networkit.

Output (one directory per matrix under `data/`):
    data/<name>/<name>_indptr.bin   int64
    data/<name>/<name>_indices.bin  int64
    data/<name>/<name>_values.bin   float32

This matches `SparseMatrix::load_from_csr` in `src/sparse_matrix.h`.

Usage:
    python3 gen_graph500_matrix.py                     # Graph500_16, Graph500_32
    python3 gen_graph500_matrix.py --base-scale 15     # smaller, quick smoke test
    python3 gen_graph500_matrix.py --num-gpu 16 32 64  # custom GPU counts
"""

import argparse
import math
import os

import numpy as np
import networkit as nk
from networkit import algebraic


def get_csr_from_networkit(scale, edge_factor=16):
    """Generate a Graph500 R-MAT graph as a scipy CSR matrix."""
    # Graph500 standard parameters (A, B, C, D).
    generator = nk.generators.RmatGenerator(scale, edge_factor,
                                            0.57, 0.19, 0.19, 0.05)
    G = generator.generate()
    return algebraic.adjacencyMatrix(G, matrixType='sparse').tocsr()


def to_binary_files(csr_mat, data_dir, data_name):
    """Write a CSR matrix to the binary format expected by DistSPMM."""
    indptr = csr_mat.indptr.astype(np.int64)
    indices = csr_mat.indices.astype(np.int64)
    values = csr_mat.data.astype(np.float32)

    out_dir = os.path.join(data_dir, data_name)
    os.makedirs(out_dir, exist_ok=True)
    indptr.tofile(os.path.join(out_dir, f"{data_name}_indptr.bin"))
    indices.tofile(os.path.join(out_dir, f"{data_name}_indices.bin"))
    values.tofile(os.path.join(out_dir, f"{data_name}_values.bin"))
    print(f"[gen_graph500_matrix] wrote {data_name} "
          f"({csr_mat.shape[0]}x{csr_mat.shape[1]}, nnz={csr_mat.nnz}) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-scale", type=int, default=20,
                        help="base scale; actual scale = base + log2(num_gpu)")
    parser.add_argument("--num-gpu", type=int, nargs="+", default=[16, 32],
                        help="GPU counts to generate a matrix for")
    parser.add_argument("--edge-factor", type=int, default=16)
    parser.add_argument("--data-dir", default=None,
                        help="binary output root (default: <repo>/data)")
    parser.add_argument("--mtx-dir", default=None,
                        help="optional dir to also write .mtx copies")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "data")

    for num_gpu in args.num_gpu:
        scale = args.base_scale + int(math.log2(num_gpu))
        csr_mat = get_csr_from_networkit(scale, args.edge_factor)
        data_name = f"Graph500_{num_gpu}"
        to_binary_files(csr_mat, args.data_dir, data_name)
        if args.mtx_dir:
            from scipy.io import mmwrite
            os.makedirs(args.mtx_dir, exist_ok=True)
            mmwrite(os.path.join(args.mtx_dir, f"{data_name}.mtx"), csr_mat)
