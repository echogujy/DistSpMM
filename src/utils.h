#ifndef UTILS_H
#define UTILS_H
#include <cassert>
#include <unordered_set>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>
#include <vector>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include "h_sdma.h" 
#ifndef warpSize
#define warpSize 32
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess!= result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)
#endif

#ifndef NVSHMEM_CHECK
#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS!= result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)
#endif

#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(stmt) \
    do { \
        cusparseStatus_t status = (stmt); \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "[%s:%d] CUSPARSE failed with error %d: %s\n", \
                    __FILE__, __LINE__, status, cusparseGetErrorString(status)); \
            exit(-1); \
        } \
    } while (0)
#endif

#ifndef MPI_CHECK
#define MPI_CHECK(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS!= mpi_status) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL!= mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            exit( mpi_status );                                                       \
        }                                                                             \
    }
#endif




// Function to check, parse, and set NVSHMEM_SYMMETRIC_SIZE
void check_parse_and_set_nvshmem_symmetric_size(long long unsigned int need_size) {
    char * value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value){
        long long unsigned int units, size;

        assert(value != NULL);

        if (strchr(value, 'G') != NULL) {
            units=1e9;
        } else if (strchr(value, 'M') != NULL) {
            units=1e6;
        } else if (strchr(value, 'K') != NULL) {
            units=1e3;
        } else {
            units=1;
        }

        assert(atof(value) >= 0);
        size = (long long unsigned int) atof(value) * units;
        if (size > need_size)
            return;
        // if (size < need_size ){
        //     printf("NVSHMEM_SYMMETRIC_SIZE is too small, setting to %llu\n", need_size);
        //     size = need_size;
        //     setenv("NVSHMEM_SYMMETRIC_SIZE", std::to_string(size).c_str(), 1);
        // }
    }
    else{
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", need_size);
        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }
}

// Kernel for GPU warmup
__global__ void warmup_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        for (int i = 0; i < 100; ++i)
            data[idx] = data[idx] * idx * 2.0f;  // Simple operation
}

void gpu_warmup(int size=1024 * 32) {
    float *d_data;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemset(d_data, 0, size * sizeof(float));
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    warmup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();
    cudaFree(d_data);
}

// Device function for sleep functionality in kernels
__forceinline__ __device__ void cuda_sleep(unsigned int clocks) {
    auto start_clock = clock64();
    auto current_clock = clock64();
    do {
        current_clock = clock64();
    } while (current_clock - start_clock < clocks);
}

void set_mem_pool_threshold(int device, uint64_t threshold = UINT64_MAX) {
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, device);
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

// Helper to get local GPU count
int get_local_gpu_count() {
    int gpu_count = 0;
    cudaError_t err = cudaGetDeviceCount(&gpu_count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    return gpu_count;
}

/**
 * @brief Initializes the parallel environment (CUDA, NVSHMEM) and discovers the hardware topology.
 * This function integrates the user's original initialization logic with the automatic topology discovery.
 * @param mpi_comm The MPI communicator to use for initialization.
 * @param gpu_topology Output parameter, will be filled with the discovered GPU-to-node mapping.
 * @param num_nodes Output parameter, will be filled with the total number of discovered nodes.
 */
void nvshmem_init_use_mpi(
    MPI_Comm mpi_comm,
    std::vector<GpuInfo>& gpu_topology,
    int& num_nodes, 
    bool set_mem_pool = true) 
{
    // 1. Standard MPI and local rank discovery
    int world_rank, world_size;
    MPI_Comm_rank(mpi_comm, &world_rank);
    MPI_Comm_size(mpi_comm, &world_size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);

    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);

    // 2. Set CUDA device based on local rank (from user's original logic)
    int local_gpu_count = get_local_gpu_count();
    assert(local_rank < local_gpu_count && "More MPI processes on node than available GPUs.");
    cudaSetDevice(local_rank);

    // 3. Initialize NVSHMEM (from user's original logic)
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    // Set memory pool threshold, for async memcpy
    if (set_mem_pool) set_mem_pool_threshold(local_rank);


    // 4. Discover node IDs and total number of nodes (integrated logic)
    MPI_Comm leader_comm;
    int color = (local_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(mpi_comm, color, world_rank, &leader_comm);

    int my_node_id = -1;
    num_nodes = 0;

    if (leader_comm != MPI_COMM_NULL) { // If I am a node leader...
        MPI_Comm_size(leader_comm, &num_nodes);
        MPI_Comm_rank(leader_comm, &my_node_id);
    }

    // Each leader broadcasts its node_id and the total num_nodes to its local peers.
    MPI_Bcast(&my_node_id, 1, MPI_INT, 0, node_comm);
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, node_comm);

    // 5. Gather all node IDs to build the final topology map
    std::vector<int> all_node_ids(world_size);
    MPI_Allgather(&my_node_id, 1, MPI_INT, all_node_ids.data(), 1, MPI_INT, mpi_comm);

    gpu_topology.clear();
    for (int i = 0; i < world_size; ++i) {
        gpu_topology.push_back({i, all_node_ids[i]});
    }

    // 6. Clean up communicators
    MPI_Comm_free(&node_comm);
    if (leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&leader_comm);
    }
}

// int get_gpu_count() {
//     int gpu_count = 0;
//     cudaError_t err = cudaGetDeviceCount(&gpu_count);
//     if (err != cudaSuccess) {
//         std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
//         return -1;
//     }
//     return gpu_count;
// }

// void nvshmem_init_use_mpi(MPI_Comm mpi_comm = MPI_COMM_WORLD, int device_shift = 0, bool set_mem_pool = true) {
//     nvshmemx_init_attr_t attr;
//     // check mpi has initialize
//     int initialized;
//     MPI_Initialized(&initialized);
//     assert(initialized);
//     attr.mpi_comm = &mpi_comm;
//     int world_rank = 0;
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//     // Create a new communicator containing only processes on the same node
//     MPI_Comm node_comm;
//     MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, 
//                         MPI_INFO_NULL, &node_comm);
//     int local_rank;
//     int node_size;
//     // Get the rank of the current process within the local node communicator (0, 1, 2, ...)
//     MPI_Comm_rank(node_comm, &local_rank);
//     // Get the size of the local node communicator (i.e. the total number of processes on the node)
//     MPI_Comm_size(node_comm, &node_size);
//     assert(node_size <= get_gpu_count());
//     CUDA_CHECK(cudaSetDevice(local_rank));
//     NVSHMEM_CHECK(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
//     if (set_mem_pool) set_mem_pool_threshold(local_rank);
// }


void check_pointer_type(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get pointer attributes: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    switch (attributes.type) {
        case cudaMemoryTypeUnregistered:
            std::cout << "Pointer points to unregistered memory" << std::endl;
            break;
        case cudaMemoryTypeHost:
            std::cout << "Pointer points to host (system) memory" << std::endl;
            break;
        case cudaMemoryTypeDevice:
            std::cout << "Pointer points to device (GPU) memory" << std::endl;
            break;
        case cudaMemoryTypeManaged:
            std::cout << "Pointer points to managed (unified) memory" << std::endl;
            break;
        default:
            std::cout << "Unknown memory type" << std::endl;
            break;
    }
}

void global_barrier(MPI_Comm &mpi_comm){
    CUDA_CHECK( cudaDeviceSynchronize());
    nvshmem_barrier_all();
    MPI_Barrier(mpi_comm);
}

void debug_point(const std::string &info, const MPI_Comm &mpi_comm=MPI_COMM_WORLD) {
    MPI_CHECK(MPI_Barrier(mpi_comm));
    int mype = 0;
    MPI_CHECK(MPI_Comm_rank(mpi_comm, &mype));
    if (mype == 0) {
        std::cout << info << std::endl;
    }
}

std::string formatBytes(long bytes) {
    const long KiB = 1024;             // 1 KiB = 1024 bytes
    const long MiB = KiB * 1024;      // 1 MiB = 1024 KiB
    const long GiB = MiB * 1024;      // 1 GiB = 1024 MiB

    if (bytes < KiB) {
        return std::to_string(bytes) + " B";
    } else if (bytes < MiB) {
        double kibibytes = static_cast<double>(bytes) / KiB;
        return std::to_string(kibibytes).substr(0, std::to_string(kibibytes).find('.') + 3) + " KiB"; 
    } else if (bytes < GiB) {
        double mebibytes = static_cast<double>(bytes) / MiB;
        return std::to_string(mebibytes).substr(0, std::to_string(mebibytes).find('.') + 3) + " MiB"; 
    } else {
        double gibibytes = static_cast<double>(bytes) / GiB;
        return std::to_string(gibibytes).substr(0, std::to_string(gibibytes).find('.') + 3) + " GiB"; 
    }
}

float gpu_mem_used() {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    float usedMem = totalMem - freeMem;
    return usedMem / (1024.0 * 1024.0 * 1024.0);
}

/**
 * @brief An MPI wrapper to compute the H-SDMA task allocation in a distributed environment.
 * This version assumes the hardware topology has already been discovered.
 * Each process provides its local vector of communication demands. The root process
 * gathers all demands, computes the global allocation, and broadcasts the result.
 * @param local_comm_demands The vector of communication demands for the tasks, from the perspective of the current MPI process (GPU).
 * @param gpu_topology The pre-computed hardware topology map.
 * @param num_nodes The pre-computed total number of nodes.
 * @param tasks_per_gpu The number of tasks to assign to each GPU.
 * @param remap_result Output vector that will be populated with the final allocation map. Its size must be equal to the total number of tasks.
 * @param mpi_comm The MPI communicator.
 * @param root_rank The rank of the process that will perform the computation.
 * @param verbose_print If true, the root rank will print detailed debug information.
 */
double ComputeAllocation_HSDMA_MPI(
    const std::vector<int>& local_comm_demands,
    const std::vector<GpuInfo>& gpu_topology,
    int num_nodes,
    int tasks_per_gpu,
    std::vector<int>& remap_result,
    MPI_Comm mpi_comm,
    int root_rank = 0,
    bool verbose_print = false)
{
    int my_rank, num_procs;
    MPI_Comm_rank(mpi_comm, &my_rank);
    MPI_Comm_size(mpi_comm, &num_procs);

    int num_local_tasks = local_comm_demands.size();
    if (remap_result.size() != num_local_tasks) {
        throw std::runtime_error("remap_result size must match the total number of tasks.");
    }

    // The root process prepares a flat, contiguous buffer for MPI_Gather.
    std::vector<int> gather_buffer;
    if (my_rank == root_rank) {
        gather_buffer.resize(num_procs * num_local_tasks);
    }

    // Gather all local demand vectors onto the root process's contiguous flat buffer.
    MPI_Gather(local_comm_demands.data(),   // Send buffer
               num_local_tasks,             // Send count
               MPI_INT,                     // Send type
               gather_buffer.data(),        // Recv buffer (contiguous)
               num_local_tasks,             // Recv count (per process)
               MPI_INT,                     // Recv type
               root_rank,                   // Root rank
               mpi_comm);

    double return_time = 0;
    if (my_rank == root_rank) {
        // Reconstruct the 2D matrix from the flat buffer for the H_SDMA function.
        std::vector<std::vector<int>> globalCommDemandMatrix(num_procs, std::vector<int>(num_local_tasks));
        for (int i = 0; i < num_procs; ++i) {
            for (int j = 0; j < num_local_tasks; ++j) {
                globalCommDemandMatrix[i][j] = gather_buffer[i * num_local_tasks + j];
            }
        }
        double t0, t1;
        t0 = MPI_Wtime();
        // printf("H-SDMA");
        // H_SDMA(globalCommDemandMatrix, gpu_topology, num_nodes, tasks_per_gpu, remap_result);
        hierarchical_iterative_refinement_allocation(globalCommDemandMatrix, gpu_topology, tasks_per_gpu, remap_result,5,5,verbose_print);
        t1 = MPI_Wtime();
        // --- Verbose Printing Section ---
        if (verbose_print) {
            std::cout << "\n======================================================" << std::endl;
            std::cout << "      H-SDMA VERBOSE OUTPUT (Rank " << root_rank << ")           " << std::endl;
            std::cout << "======================================================" << std::endl;
            std::cout << "Total number of nodes: " << num_nodes << std::endl;
            std::cout << "Final remapping: " << std::endl;
            for (int gpu_id = 0; gpu_id < gpu_topology.size(); ++gpu_id) {
                std::cout << "GPU " << gpu_id << " (Node " << gpu_topology[gpu_id].nodeId << "): ";
                for (int task_id = 0; task_id < remap_result.size(); ++task_id) {
                    if (remap_result[task_id] == gpu_id) {
                        std::cout << task_id << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "======================================================\n" << std::endl;
            std::cout << "Reallocation used: " << (t1 - t0) * 1000 << " ms" << std::endl << std::flush;
        }
        return_time =  (t1 - t0) * 1000; // Return time in milliseconds
    }

    // Broadcast the final remap result from the root to all other processes.
    MPI_Bcast(remap_result.data(),          // Buffer
              remap_result.size(),          // Count
              MPI_INT,                      // Type
              root_rank,                    // Root rank
              mpi_comm);
    return return_time;
}

void double_level_comm_node_processing(
    MPI_Comm &mpi_comm,
    std::vector<int> &n_col_list,
    std::vector<std::unordered_set<int>> &col_ind_unique,
    bool detail = false)
{
    // 1. Standard MPI and local rank discovery
    int world_rank, world_size;
    MPI_Comm_rank(mpi_comm, &world_rank);
    MPI_Comm_size(mpi_comm, &world_size);

    // 2. Split the communicator based on shared memory access (topology-aware)
    MPI_Comm node_comm;
    // FIX 1: Use the passed communicator, not MPI_COMM_WORLD
    MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // Debug print
    if (detail && node_rank == 0) {
        // Use world_rank to uniquely identify the node leader
        std::cout << "Node leader (World Rank " << world_rank << ") reports " << node_size << " processes on this node." << std::endl;
        // print n_col_list
        std::cout << "Initial n_col_list: { ";
        for (const auto &n_col : n_col_list) {
            std::cout << n_col << " ";
        }
        std::cout << "}" << std::endl << std::flush;
    }

    // 3. Gather the number of columns for each tile from all processes on the node
    int tile_size = n_col_list.size();
    std::vector<int> gather_n_col_list(node_size * tile_size);
    MPI_Allgather(n_col_list.data(), tile_size, MPI_INT, gather_n_col_list.data(), tile_size, MPI_INT, node_comm);

    // 4. Prepare for Allgatherv: calculate counts and displacements
    std::vector<int> col_ind_unique_counts(node_size);
    for (int i = 0; i < node_size; ++i) {
        // Calculate total number of indices for process i on this node
        col_ind_unique_counts[i] = std::accumulate(gather_n_col_list.begin() + i * tile_size,
                                                   gather_n_col_list.begin() + (i + 1) * tile_size, 0);
    }
    
    int sum_all = std::accumulate(col_ind_unique_counts.begin(), col_ind_unique_counts.end(), 0);

    // Flatten local data for sending
    std::vector<int> col_ind_unique_local;
    col_ind_unique_local.reserve(col_ind_unique_counts[node_rank]);
    for (const auto &col_set : col_ind_unique) {
        col_ind_unique_local.insert(col_ind_unique_local.end(), col_set.begin(), col_set.end());
    }

    // Prepare receive buffer and displacements for Allgatherv
    std::vector<int> col_ind_unique_all(sum_all);
    std::vector<int> col_ind_unique_displs(node_size + 1, 0); // Use size+1 for safer access later
    for (int i = 0; i < node_size; ++i) {
        col_ind_unique_displs[i + 1] = col_ind_unique_displs[i] + col_ind_unique_counts[i];
    }

    // 5. Allgatherv the unique column indices across all processes in the node communicator
    MPI_Allgatherv(col_ind_unique_local.data(), col_ind_unique_local.size(), MPI_INT,
                   col_ind_unique_all.data(), col_ind_unique_counts.data(), col_ind_unique_displs.data(),
                   MPI_INT, node_comm);
                   
    // Correct reconstruction logic ---
    col_ind_unique.clear();
    col_ind_unique.resize(tile_size);

    for (int tile_id = 0; tile_id < tile_size; ++tile_id) {
        // For each tile, iterate through all processes on the node and collect their indices for this tile
        for (int rank_in_node = 0; rank_in_node < node_size; ++rank_in_node) {
            // Start of this process's entire data block in the flattened 'col_ind_unique_all' array
            int proc_data_start = col_ind_unique_displs[rank_in_node];
            
            // Calculate the offset to find this specific tile's data within the process's data block
            int tile_offset = 0;
            for (int t = 0; t < tile_id; ++t) {
                tile_offset += gather_n_col_list[rank_in_node * tile_size + t];
            }

            // Determine the start and end indices for the current tile's data from the current process
            int start_idx = proc_data_start + tile_offset;
            int num_indices_for_tile = gather_n_col_list[rank_in_node * tile_size + tile_id];
            
            // Insert the indices into the set for the current tile_id
            if (num_indices_for_tile > 0) {
                col_ind_unique[tile_id].insert(col_ind_unique_all.begin() + start_idx, 
                                               col_ind_unique_all.begin() + start_idx + num_indices_for_tile);
            }
        }
        // Update n_col_list with the size of the merged set of unique columns for this tile
        n_col_list[tile_id] = col_ind_unique[tile_id].size();
    }

    if(detail && node_rank == 0) {
        std::cout << "Node leader (World Rank " << world_rank << ") has processed the unique column indices." << std::endl;
        // print n_col_list
        std::cout << "Processed n_col_list: { ";
        for (const auto &n_col : n_col_list) {
            std::cout << n_col << " ";
        }
        std::cout << "}" << std::endl << std::flush;
    }

    // 6. Clean up communicators
    MPI_Comm_free(&node_comm);
}


#endif // UTILS_H