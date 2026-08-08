#!/bin/bash
# Full experiment driver for DistSPMM.
#
# Runs the single-node kernel (`dist_spmm_node`) over a list of datasets and
# embedding dimensions, using the H-SDMA task distribution (`task_dist=2`) and
# adaptive communication (`comm_type=2`).
#
# Run from the repo root:
#   ./run_all.sh [ngpus]
#
# `ngpus` defaults to the number of local GPUs (via nvidia-smi), falling back to 1.
# Adjust DATASETS / DIMS below for your experiments.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- environment -----------------------------------------------------------------
# dist_env.sh is machine-specific; create it from dist_env.example.sh if absent.
if [ -f "$REPO_DIR/dist_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/dist_env.sh"
else
    echo "!! dist_env.sh not found. Create it from dist_env.example.sh first." >&2
    exit 1
fi

# --- number of GPUs -------------------------------------------------------------
if [ $# -ge 1 ]; then
    NGPUS="$1"
elif [ -n "${SLURM_NTASKS:-}" ]; then
    NGPUS="$SLURM_NTASKS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NGPUS=$(nvidia-smi -L | wc -l)
else
    NGPUS=1
fi

# --- experiment configuration ---------------------------------------------------
# Datasets must exist under ./data/<name>/ as <name>_indptr.bin etc.
#   - Generated datasets (Graph500):  tools/gen_graph500_matrix.py
#   - Real datasets (SuiteSparse):    tools/download_suitesparse.py
# List the dataset names you prepared in ./data/, e.g.:
DATASETS=(
    "Graph500_16"
    "Graph500_32"
)
DIMS=(
    "128"
)

# Kernel flags: <task_dist> <comm_type> <n_rolls> <repeat> [nvlink_producers]
TASK_DIST=2      # 0: slice, 1: random, 2: H-SDMA (minimize communication)
COMM_TYPE=2      # 0: block, 1: sparse, 2: adaptive
N_ROLLS=4
REPEAT=10
NVLINK_PRODUCERS=4

APP="./build/dist_spmm_node"

echo "========================================"
echo " DistSPMM full experiment"
echo " #GPUs       : $NGPUS"
echo " task_dist   : $TASK_DIST"
echo " comm_type   : $COMM_TYPE"
echo " n_rolls     : $N_ROLLS"
echo " repeat      : $REPEAT"
echo "========================================"

# --- build ----------------------------------------------------------------------
echo ">> Building (make)"
make

# --- run ------------------------------------------------------------------------
for dataset in "${DATASETS[@]}"; do
    for dims in "${DIMS[@]}"; do
        file_path="data/${dataset}/${dataset}"
        if [ ! -f "${file_path}_indptr.bin" ]; then
            echo "!! Skipping $dataset: $file_path missing (data not prepared)."
            continue
        fi
        echo "----------------------------------------"
        echo ">> Running $dataset dims=$dims on $NGPUS GPU(s)"
        echo "   mpirun -np $NGPUS $APP $file_path $dims $TASK_DIST $COMM_TYPE $N_ROLLS $REPEAT $NVLINK_PRODUCERS"
        mpirun -np "$NGPUS" -npernode "$NGPUS" --bind-to none --map-by slot \
            "$APP" "$file_path" "$dims" "$TASK_DIST" "$COMM_TYPE" "$N_ROLLS" "$REPEAT" "$NVLINK_PRODUCERS"
        echo "----------------------------------------"
    done
done

echo "========================================"
echo " Experiment finished."
echo "========================================"
