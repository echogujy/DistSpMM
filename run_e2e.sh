#!/bin/bash
# End-to-end smoke test for DistSPMM.
#
# Full pipeline:
#   1. generate a small Graph500 matrix (the generated/synthetic dataset family)
#   2. build the project
#   3. run a distributed SpMM with the generated matrix
#
# Run from the repo root (on a GPU compute node):
#   ./run_e2e.sh [ngpus]
#
# `ngpus` defaults to the number of local GPUs (via nvidia-smi), falling back to 1.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- environment -----------------------------------------------------------------
if [ -f "$REPO_DIR/dist_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/dist_env.sh"
else
    echo "!! dist_env.sh not found. Create it from dist_env.example.sh first." >&2
    exit 1
fi

# --- number of GPUs --------------------------------------------------------------
if [ $# -ge 1 ]; then
    NGPUS="$1"
elif [ -n "${SLURM_NTASKS:-}" ]; then
    NGPUS="$SLURM_NTASKS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NGPUS=$(nvidia-smi -L | wc -l)
else
    NGPUS=1
fi

# Graph500 dataset used for the smoke test (base scale 15 -> 2^15 nodes).
E2E_NAME="Graph500_16"

echo "========================================"
echo " DistSPMM end-to-end test"
echo " Repo        : $REPO_DIR"
echo " #GPUs       : $NGPUS"
echo " Dataset     : $E2E_NAME (Graph500, generated)"
echo "========================================"

# --- 1. generate data (Graph500) ---------------------------------------------------
echo ">> [1/4] Generating Graph500 matrix (tools/gen_graph500_matrix.py)"
python3 tools/gen_graph500_matrix.py --base-scale 15 --num-gpu 16

# --- 2. build -----------------------------------------------------------------------
echo ">> [2/4] Building (make)"
make

# --- 3. run -------------------------------------------------------------------------
# dist_spmm_node: single-node multi-GPU kernel.
#   usage: <file> <dims> <task_dist> <comm_type> <n_rolls> <repeat> [nvlink_producers]
echo ">> [3/4] Running distributed SpMM (task_dist=2 HSDMA, comm_type=2 adaptive)"
mpirun -np "$NGPUS" -npernode "$NGPUS" --bind-to none --map-by slot \
    ./build/dist_spmm_node \
    "data/$E2E_NAME/$E2E_NAME" 128 2 2 4 3 4

echo ">> [4/4] Finished successfully"
