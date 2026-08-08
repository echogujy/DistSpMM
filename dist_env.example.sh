#!/bin/bash
# Environment setup for DistSPMM.
#
# Usage:
#   1. Copy this file to `dist_env.sh` in the repo root.
#   2. Adjust the paths below for your own machine (the values shown are the
#      authors' local SXM-A800 example and are used for the end-to-end test).
#   3. Run scripts from the repo root (they `source ./dist_env.sh`).
#
# Required dependencies:
#   - CUDA Toolkit / NVIDIA HPC SDK
#   - cuSPARSE (CUDA Math Library)
#   - OpenMPI / HPC-X (or any MPI-3 with shared-memory support)
#   - NVSHMEM
#
# This file is intentionally NOT committed to the repo (see .gitignore).

# --- Python (only needed for the data & modeling tools in tools/) --------------
# Example (authors' conda env "datapass"):
__conda_setup="$('/data/apps/miniforge3/25.11.0-1/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh" ]; then
        . "/data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh"
    else
        export PATH="/data/apps/miniforge3/25.11.0-1/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate datapass

# --- CUDA Toolkit --------------------------------------------------------------
# Example (SXM-A800, NVIDIA HPC SDK 25.11 / CUDA 12.9):
export CUDA_HOME=/data/apps/nvhpc/25.11/Linux_x86_64/25.11/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_INCLUDE_PATH=$CUDA_HOME/include:$LD_INCLUDE_PATH

# --- CUDA Math Library (cuSPARSE) ----------------------------------------------
export CUDA_MATH_HOME=/data/apps/nvhpc/25.11/Linux_x86_64/25.11/math_libs/12.9
export PATH=$CUDA_MATH_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_MATH_HOME/lib64:$LD_LIBRARY_PATH
export LD_INCLUDE_PATH=$CUDA_MATH_HOME/include:$LD_INCLUDE_PATH

# --- MPI (OpenMPI / HPC-X) ------------------------------------------------------
export MPI_HOME=/data/apps/nvhpc/25.11/Linux_x86_64/25.11/comm_libs/12.9/hpcx/hpcx-2.20/ompi
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export LD_INCLUDE_PATH=$MPI_HOME/include:$LD_INCLUDE_PATH

# --- Communication libraries (NCCL + NVSHMEM) ----------------------------------
export COMM_LIB_HOME=/data/apps/nvhpc/25.11/Linux_x86_64/25.11/comm_libs/12.9
export NCCL_HOME=$COMM_LIB_HOME/nccl
export PATH=$NCCL_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LD_INCLUDE_PATH=$NCCL_HOME/include:$LD_INCLUDE_PATH

export NVSHMEM_HOME=$COMM_LIB_HOME/nvshmem
export PATH=$NVSHMEM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export LD_INCLUDE_PATH=$NVSHMEM_HOME/include:$LD_INCLUDE_PATH
