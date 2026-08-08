#!/bin/bash
#SBATCH -J comm_test
#SBATCH -p gpu_a800
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=gpugpu
#SBATCH --time=01:30:00
#SBATCH --output=./logs/slurm-comm_test-%j.out
# Create logs directory
mkdir -p logs

# Source environment (users should create ../dist_env.sh from ../dist_env.example.sh)
source ../dist_env.sh

# Test parameters
gpu_shift=0
bins=100
Ns=(400000 800000 1000000 1200000 1400000 1600000 1800000 2000000)
dim_s=1
dim_e=2048
stride=2
apps=../build/comm_test

# NVSHMEM settings for multi-node
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_RC_MAP_BY="warp"
export NVSHMEM_IBGDA_NUM_RC_PER_PE=4
export NVSHMEM_IBGDA_NUM_DCI=0

echo "========================================"
echo "Running comm_test on SLURM..."
echo "Start Time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "========================================"

# Run tests with parameter 0
echo "========================================"
echo "Running tests with parameter 0 (PUT)"
echo "========================================"
for N in "${Ns[@]}"; do
    echo "Testing with N = $N, comm_test ${N} ${dim_s} ${dim_e} ${stride} ${gpu_shift} ${bins} 0"
    mpirun -np ${SLURM_NTASKS} \
        --bind-to none --map-by slot \
        -x LD_LIBRARY_PATH \
        -x PATH \
        -x NVSHMEM_IB_ENABLE_IBGDA \
        -x NVSHMEM_IBGDA_RC_MAP_BY \
        -x NVSHMEM_IBGDA_NUM_RC_PER_PE \
        -x NVSHMEM_IBGDA_NUM_DCI \
        $apps ${N} ${dim_s} ${dim_e} ${stride} ${gpu_shift} ${bins} 0
    echo "=============================="
done

# Run tests with parameter 1
echo "========================================"
echo "Running tests with parameter 1 (GET)"
echo "========================================"
for N in "${Ns[@]}"; do
    echo "Testing with N = $N, comm_test ${N} ${dim_s} ${dim_e} ${stride} ${gpu_shift} ${bins} 1"
    mpirun -np ${SLURM_NTASKS} \
        --bind-to none --map-by slot \
        -x LD_LIBRARY_PATH \
        -x PATH \
        -x NVSHMEM_IB_ENABLE_IBGDA \
        -x NVSHMEM_IBGDA_RC_MAP_BY \
        -x NVSHMEM_IBGDA_NUM_RC_PER_PE \
        -x NVSHMEM_IBGDA_NUM_DCI \
        $apps ${N} ${dim_s} ${dim_e} ${stride} ${gpu_shift} ${bins} 1
    echo "=============================="
done

echo ""
echo "End Time: $(date)"
echo "Test completed. Check slurm-comm_test-${SLURM_JOB_ID}.out for results."
