#!/bin/bash
# Quick multi-node smoke test: 2 nodes x 2 GPUs = 4 GPUs.
# Runs dist_spmm (multi-node kernel) on a single real dataset to verify the
# two-level (IB + NVLink) pipeline works.

#SBATCH -J distspmm_2x2
#SBATCH -p gpu_a800
#SBATCH -N 2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --qos=gpugpu
#SBATCH --time=00:20:00
#SBATCH --output=./logs/test_multi_node_2x2-%j.out

# Create logs directory
mkdir -p logs

# Source environment (users should create ../dist_env.sh from ../dist_env.example.sh)
source ../dist_env.sh

num_gpu=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
num_gpu_per_node=2
dims=128
task_dist=2      # HSDMA task allocation
comm_type=2      # adaptive communication
n_rolls=4
iters=10         # repetitions; must be >= 5 (see dist_spmm.cu run_record slice)

# NVSHMEM settings for multi-node
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_RC_MAP_BY="warp"
export NVSHMEM_IBGDA_NUM_DCI=0
export NVSHMEM_IBGDA_NUM_RC_PER_PE=$num_gpu

# Dataset: web-Google (SNAP, 0.9M rows), downloaded via tools/download_suitesparse.py
dataset="web-Google"
file_path=../data/${dataset}/${dataset}

echo "========================================"
echo "DistSPMM multi-node smoke test (2 nodes x 2 GPUs)"
echo "Start Time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs: $num_gpu"
echo "Dataset: $file_path"
echo "========================================"

mpirun -np ${num_gpu} -npernode ${num_gpu_per_node} \
        --bind-to none --map-by slot \
        -x LD_LIBRARY_PATH \
        -x PATH \
        -x NVSHMEM_IB_ENABLE_IBGDA \
        -x NVSHMEM_IBGDA_RC_MAP_BY \
        -x NVSHMEM_IBGDA_NUM_RC_PER_PE \
        -x NVSHMEM_IBGDA_NUM_DCI \
        ../build/dist_spmm ${file_path} ${dims} ${task_dist} ${comm_type} ${n_rolls} ${iters} 4

echo "========================================"
echo "End Time: $(date)"
echo "Test completed. Check test_multi_node_2x2-${SLURM_JOB_ID}.out for results."
echo "========================================"
