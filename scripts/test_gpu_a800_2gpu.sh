#!/bin/bash
# Quick smoke test on the gpu_a800 partition (2 GPUs, currently available).
# Runs dist_spmm_node on a single real dataset to verify the pipeline works.

#SBATCH -J distspmm_test
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --qos=gpugpu
#SBATCH --output=./logs/test_gpu_a800_2gpu-%j.out

# Create logs directory
mkdir -p logs

# Source environment (users should create ../dist_env.sh from ../dist_env.example.sh)
source ../dist_env.sh

num_gpu=${SLURM_NTASKS}
dims=128
task_dist=2      # HSDMA task allocation
comm_type=2      # adaptive communication
n_rolls=4
iters=10         # repetitions (>= 5 is recommended)

# Dataset: web-Google (SNAP, 0.9M rows), downloaded via tools/download_suitesparse.py
dataset="web-Google"
file_path=../data/${dataset}/${dataset}

echo "========================================"
echo "DistSPMM smoke test on gpu_a800 (2 GPUs)"
echo "Start Time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_NTASKS"
echo "Dataset: $file_path"
echo "========================================"

mpirun -np ${num_gpu} -npernode ${num_gpu} \
        --bind-to none --map-by slot \
        ../build/dist_spmm_node ${file_path} ${dims} ${task_dist} ${comm_type} ${n_rolls} ${iters} 4

echo "========================================"
echo "End Time: $(date)"
echo "Test completed. Check test_gpu_a800_2gpu-${SLURM_JOB_ID}.out for results."
echo "========================================"
