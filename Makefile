# Makefile for DistSPMM.
#
# Supported platforms (select via PLATFORM):
#   a800   -> NVIDIA A800 / A100, compute_80 (default if auto-detection fails)
#   h100   -> NVIDIA H100,        compute_90
#   auto   -> detect from nvidia-smi (default)
#
# Examples:
#   make                          # auto-detect platform
#   make PLATFORM=a800            # build for A800 (sm_80)
#   make PLATFORM=h100            # build for H100 (sm_90)
#
# The PLATFORM_DEFS macro is passed to nvcc so that src/comm_man.h can select the
# matching adaptive-communication model parameters (see Table 2 of the paper).

NVCC := nvcc
CC := g++

# --- platform detection / selection --------------------------------------------
PLATFORM ?= auto

ifeq ($(PLATFORM),auto)
# Detect the local GPU architecture from nvidia-smi (fall back to sm_80).
CC_MAJOR := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
ifeq ($(CC_MAJOR),9)
PLATFORM := h100
else
PLATFORM := a800
endif
endif

ifeq ($(PLATFORM),h100)
GENCODE_FLAGS := -gencode arch=compute_90,code=sm_90
PLATFORM_DEFS := -DDISTSPMM_PLATFORM_H100
else
GENCODE_FLAGS := -gencode arch=compute_80,code=sm_80
PLATFORM_DEFS := -DDISTSPMM_PLATFORM_A800
endif

# --- toolchain checks -----------------------------------------------------------
ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set. source dist_env.sh first.)
endif

NVCC_FLAGS := -rdc=true -ccbin=$(CC) $(GENCODE_FLAGS) $(PLATFORM_DEFS) -std=c++14 -O2 \
              -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I$(CUDA_MATH_HOME)/include

NVCC_LDFLAGS := -L$(NVSHMEM_HOME)/lib -lnvshmem \
                -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml \
                -L$(CUDA_MATH_HOME)/lib64 -lcusparse \
                -L$(MPI_HOME)/lib -lmpi -lmlx5

# --- build layout ---------------------------------------------------------------
$(shell mkdir -p build)
OBJ_DIR := ./build/

SRC_DIR := ./src
HEADERS_DIR := ./src
HEADERS := $(wildcard $(SRC_DIR)/*.h)

# Auto-detect all .cu files and generate a binary per source file
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
PROGRAMS := $(patsubst $(SRC_DIR)/%.cu,build/%,$(CU_SOURCES))

all: $(PROGRAMS)

# Generic build rule
build/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -I $(HEADERS_DIR) $< -c -o $@

build/%: build/%.o
	$(NVCC) $(GENCODE_FLAGS) $(NVCC_LDFLAGS) $< -o $@

.PHONY: clean all
clean:
	rm -f *.o
	rm -f $(OBJ_DIR)*
