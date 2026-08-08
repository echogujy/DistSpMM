# DistSpMM

**DistSpMM: Accelerating Sparse Matrix Dense Matrix Multiplication on GPUs**

This is the official implementation of the paper *DistSpMM: Accelerating Sparse
Matrix Dense Matrix Multiplication on GPUs*. DistSpMM is a distributed sparse-matrix
dense-matrix multiplication (SpMM) system built on **NVSHMEM**, targeting multi-GPU /
multi-node clusters (A800 and H100). By co-designing the data layout, the pipeline,
and the communication strategy, it significantly reduces the communication overhead
of distributed SpMM.

> This repository contains only the core implementation of DistSpMM. It does **not**
> bundle any third-party SOTA baselines. If you wish to reproduce the comparisons,
> obtain the official implementations of those methods (e.g. CoLa, TileSpMM,
> MGG-SpMM, CAGNET, SUMMA-1D/2D) yourself.

## Contributions

The paper makes three co-designed contributions, all implemented in this repository:

1. **HSDMA** (`src/h_sdma.h`): a lightweight algorithm that reduces communication by
   optimizing the dense-matrix task allocation. It first refines node-level placement
   and then GPU-level placement to minimize total communication.
2. **Topology-aware two-stage pipeline** (`src/dist_spmm.cu` + `src/comm_man.h`):
   manages the InfiniBand / NVLink bandwidth disparity to maximize the overlap of
   computation and communication via hierarchical double buffers.
3. **Adaptive communication selector** (`src/comm_man.h`): a performance model that
   dynamically chooses the optimal communication granularity (coarse-grained bulk vs.
   fine-grained sparsity-aware) based on data sparsity and network tier.

## Features

- **Tile-based distributed SpMM**: the output is partitioned column-wise into tiles,
  distributed across GPUs, with NVSHMEM symmetric-memory one-sided `get` communication.
- **Sparse-aware communication**: per tile, DistSpMM adaptively chooses between dense
  block communication and sparse communication, transferring only the dense columns
  corresponding to non-zero columns (`comm_type=2`).

## Repository layout

```
DistSPMM-repo/
├── src/                        # Core source code
│   ├── dist_spmm.cu            # Multi-node SpMM (IB + NVLink two-level pipeline)
│   ├── dist_spmm_node.cu       # Single-node multi-GPU SpMM
│   ├── comm_man.h              # Communication manager (block/sparse/adaptive, hierarchical pipeline)
│   ├── h_sdma.h                # HSDMA hierarchical task allocation
│   ├── utils.h                 # Utilities, topology discovery, init
│   ├── sparse_matrix.h         # CSR matrix loading
│   └── comm_test.cu            # Communication microbenchmark
├── scripts/                    # SLURM / mpirun launcher scripts
├── tools/                      # Data & modeling tools (Graph500 / SuiteSparse / fit_comm_model)
├── Makefile                    # Build configuration (platform-aware)
├── dist_env.example.sh         # Environment template (copy to dist_env.sh)
├── run_e2e.sh                  # End-to-end smoke test (gen data -> build -> run)
├── run_all.sh                  # Full experiment driver
├── .gitignore
└── README.md
```

## Dependencies

- NVIDIA GPU (**A800/A100** `sm_80`, or **H100** `sm_90`)
- CUDA Toolkit (>= 12.x)
- cuSPARSE (CUDA Math Library)
- OpenMPI / HPC-X (or any MPI-3 with shared-memory support)
- NVSHMEM (with `libnvshmem` and headers)
- Python (only for data & modeling tools): `numpy`, `scipy`, `networkit`

## Build

```bash
# 1. Prepare the environment (copy the template and fill in your paths)
cp dist_env.example.sh dist_env.sh
vim dist_env.sh    # set CUDA_HOME / CUDA_MATH_HOME / MPI_HOME / NVSHMEM_HOME

# 2. Compile (platform is auto-detected from nvidia-smi)
source dist_env.sh
make

#   Or force a specific platform:
#   make PLATFORM=a800    # A800 / A100 (sm_80)
#   make PLATFORM=h100    # H100 (sm_90)
```

`PLATFORM` selects the CUDA architecture and the matching adaptive-communication
model parameters compiled into `src/comm_man.h` (see *Adaptive communication model*).

Binaries are written to `build/`; each `.cu` source file produces one executable
(e.g. `build/dist_spmm_node`).

## Data preparation

DistSPMM loads matrices in a binary CSR format. For each matrix `<name>`, three
files are required (see `src/sparse_matrix.h::load_from_csr`):

```
data/<name>/<name>_indptr.bin   # int64 row-pointer array
data/<name>/<name>_indices.bin  # int64 column-index array
data/<name>/<name>_values.bin   # float32 non-zero values
```

The experiments in the paper use two families of datasets:

### 1. Generated datasets (Graph500)

Synthetic R-MAT matrices generated with NetworKit. Requires `networkit`:

```bash
python3 tools/gen_graph500_matrix.py                     # Graph500_16, Graph500_32
python3 tools/gen_graph500_matrix.py --base-scale 15     # smaller, quick smoke test
python3 tools/gen_graph500_matrix.py --num-gpu 16 32 64  # custom GPU counts
```

### 2. Real datasets (SuiteSparse)

Downloaded SuiteSparse matrices, converted to DistSPMM's binary format. This is
the pipeline used for the real datasets in the paper:

```bash
python3 tools/download_suitesparse.py                       # all datasets
python3 tools/download_suitesparse.py --dataset twitter7    # a single dataset
```

Each dataset is downloaded from the SuiteSparse Matrix Collection
(`sparse.tamu.edu`), extracted, and written as `data/<name>/<name>_indptr.bin`
plus `_indices.bin` / `_values.bin`. You can also add entries to `DATA_DICT` in
`tools/download_suitesparse.py` to fetch other matrices.

### End-to-end example (download -> run)

Here is a complete, minimal example that downloads a small SuiteSparse graph,
converts it, and runs a single-node multi-GPU SpMM on it. It uses `web-Google`
(SNAP, ~0.9M x 0.9M, ~5 MB), already listed in `tools/download_suitesparse.py`.

```bash
# 1. (optional) build the kernels
source dist_env.sh
make

# 2. download and convert the dataset (writes data/web-Google/*.bin)
python3 tools/download_suitesparse.py --dataset web-Google

# 3. run a single-node, 2-GPU SpMM on it
cd scripts
sbatch test_gpu_a800_2gpu.sh
# or run directly (single node, 2 GPUs):
mpirun -np 2 -npernode 2 ./build/dist_spmm_node \
    data/web-Google/web-Google 128 2 2 4 10 4
```

Expected output (abridged) after the run:

```
task_distribution: 2 and rolls:4
1 1 0 0 0 0 1 1
Running dist_spmm programs on data/web-Google/web-Google with [128] dims
Processing time: (ms)
Re-allocation time(all): 0.225 Hira Time: 0.005
Max memory used: 2.985 GB
Average time used: 2.048 ms
Communication Traffic Intra : 0.349 GB
Communication Traffic Inter : 0.000 GB
Throughput: 638.014 GFLOPS
```

### Fitting the adaptive-communication model

The `comm_type=2` thresholds in `src/comm_man.h` are fitted from measured
communication data.

> The `MEASURED` table inside `tools/fit_comm_model.py` contains the critical
> densities measured **by the authors** on their test platforms. They depend on
> your NVLink / InfiniBand bandwidth, so if you run on a **different platform
> you should re-measure and update the table before fitting** (see the
> instructions at the top of the script).

To fit and print the intra/inter model parameters:

```bash
python3 tools/fit_comm_model.py
```

**To re-measure on your own platform**, run the communication microbenchmark:

```bash
cd scripts
sbatch comm_test_slurm.sh                # measures intra (com_type=0) & inter (com_type=1)
```

Each `comm_test` run prints one `<dim>,<threshold>` line per dimension; collect
these per matrix size `N` and per tier into `MEASURED`, then re-run
`fit_comm_model.py`.

All data tools write into `./data/` inside the repository.

## Running

### Quick end-to-end test

```bash
./run_e2e.sh [ngpus]
```

This generates a small Graph500 matrix (the generated dataset family), builds the
project, and runs a short `dist_spmm_node` job end to end.

### Full experiments

```bash
./run_all.sh [ngpus]
```

Edit `DATASETS` / `DIMS` in `run_all.sh` to match your data and experiment plan.

### SLURM launchers

The scripts under `scripts/` assume data in `<repo>/data` and an environment file
`<repo>/dist_env.sh`; submit them with `sbatch`.

Two ready-to-run examples are provided:

```bash
cd scripts
sbatch test_gpu_a800_2gpu.sh    # single-node example (dist_spmm_node, 2 GPUs)
sbatch test_multi_node_2x2.sh   # multi-node example (dist_spmm, 2 nodes x 2 GPUs)
```

To re-measure the adaptive-communication thresholds on your hardware, also submit
the communication microbenchmark:

```bash
cd scripts
sbatch comm_test_slurm.sh       # communication microbenchmark
```

### Manual invocation

All kernels share the same argument layout:

```
./build/dist_spmm_node <file_name> <dims> <task_dist> <comm_type> <n_rolls> <repeat> [nvlink_num_producers]
```

| Argument | Meaning |
|----------|---------|
| `file_name` | matrix file prefix (e.g. `data/Graph500_16/Graph500_16`) |
| `dims` | number of columns of the dense matrix (embedding dimension) |
| `task_dist` | tile distribution: `0`=slice, `1`=random, `2`=HSDMA (minimize comm.) |
| `comm_type` | communication mode: `0`=block, `1`=sparse, `2`=adaptive |
| `n_rolls` | number of tiles per GPU |
| `repeat` | number of repetitions |

Example:

```bash
source dist_env.sh
mpirun -np 4 ./build/dist_spmm_node data/Graph500_16/Graph500_16 128 2 2 4 10 4
```

## Kernels

| Executable | Scope | Notes |
|------------|-------|-------|
| `dist_spmm` | multi-node | IB (inter-node) + NVLink (intra-node) two-level pipeline. |
| `dist_spmm_node` | single node | NVLink-only pipeline. |
| `comm_test` | microbenchmark | measures PUT/GET bandwidth vs. message size and density. |

## Adaptive communication model

When `comm_type=2`, DistSpMM uses a LogGP-based performance model to decide for each
tile whether coarse-grained bulk communication or fine-grained sparsity-aware
communication is cheaper, per communication tier (intra-node NVLink vs. inter-node
InfiniBand). The decision threshold is controlled by the platform-specific model
parameters below (paper Table 2), selected automatically by the Makefile's
`PLATFORM` variable.

The two sets of parameters (DGX-H100 and SXM-A800) were measured by the authors on
the hardware described in *Evaluated platforms* (paper Table 3). **They are specific
to those machines** — the thresholds depend on your NVLink / InfiniBand bandwidth.
If you run on different hardware, you are strongly advised to **measure and fit your
own parameters** (see *Fitting the adaptive-communication model*).

| Platform | Tier | a | b | c | u |
|----------|------|-----|------|------|------|
| DGX-H100 | Intra-node (NVLink) | 12873.61 | 125.27 | 105.50 | -4.65 |
| DGX-H100 | Inter-node (IB) | 1452.17 | -1407.82 | 47.34 | -1.00 |
| SXM-A800 | Intra-node (NVLink) | 4075.35 | 119.61 | 98.21 | -3.78 |
| SXM-A800 | Inter-node (IB) | 1440.80 | -1394.99 | 43.65 | -1.00 |

## Evaluated platforms

Experiments in the paper were conducted on two multi-GPU platforms:

| Component | DGX-H100 | SXM-A800 |
|-----------|----------|----------|
| GPU | 8× NVIDIA H100 SXM5 (80 GB) | 4× NVIDIA A800-SXM4 (80 GB) |
| NVLink (intra-node) | 450 GB/s/GPU (NV18) | 200 GB/s/GPU (NV8) |
| InfiniBand (inter-node) | 50 GB/s/GPU (8× NDR) | 25 GB/s/GPU (4× HDR) |
| CUDA / cuSPARSE / NVSHMEM | 12.2 / 12.2 / 3.3.9 | 12.9 / 12.5 / 3.4.5 |

## Core design

### Two-level communication (`dist_spmm.cu` + `comm_man.h`)

Within a node, tiles owned by local GPUs are fetched directly over NVLink. For
cross-node tiles, data is first pulled from the remote node over InfiniBand into an
outer buffer (`ib_stream`), then distributed within the node over NVLink via an inner
buffer (`nvlink_stream`). The compute stream (`compute_stream`) is chained to the
communication streams through CUDA events, forming a multi-buffered
producer-consumer pipeline that overlaps computation and communication.

### HSDMA allocation (`h_sdma.h`)

`hierarchical_iterative_refinement_allocation` models task placement as a
communication-demand matrix and refines it hierarchically: node-level swap
refinement first, then GPU-level refinement within each node, minimizing total
communication. Enable it with `task_dist=2`.

## Output

The kernels report (aggregated by rank 0):
- per-stage timings (matrix read, slicing, allocation, communication setup, ...);
- average runtime (ms);
- peak memory usage (GB);
- intra-node / inter-node communication traffic (GB);
- throughput (GFLOPS).

## License

[MIT License](LICENSE)

## Citation

If you use this work in your research, please cite:

```bibtex
@article{gudistspmm,
  title   = {DistSpMM: Accelerating Sparse Matrix Dense Matrix Multiplication on GPUs},
  author  = {Gu, Junyu and Wang, Jue and Xin, Zhikuang and Zhou, Chunbao and
             Liang, Zhiqiang and Pang, Yuchen and Cao, Rongqiang and Wang, Zongguo
             and Liu, Fang and Wang, Jing and Wang, Yangang},
  journal = {ACM Transactions on Architecture and Code Optimization (TACO)},
  year    = {2026}
}
```

(Update the journal/volume/pages once the final metadata is available.)
