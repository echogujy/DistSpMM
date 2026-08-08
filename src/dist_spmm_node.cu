
#include <numeric>
#include <algorithm>
#include <cassert>
#include <vector>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <type_traits>
#include <stdexcept>
#include <mpi.h>
#include "sparse_matrix.h" 
#include "utils.h"
#include "comm_man.h"

using Idx_t = int64_t;  
using Val_t = float;  

int main(int argc, char *argv[]) {
    int mype, npes;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mype));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &npes));
    
    if (argc < 7)   {
        std::cerr << "Usage: ./dist_spmm_node <file_name> <dims> <task_dist> <comm_type> <n_rolls> <repeat> " << std::endl;
        return EXIT_FAILURE;
    }
    
    // ===  get parameters ===
    std::string file_name = argv[1];
    int dims = std::stoi(argv[2]);
    int task_dist = std::stoi(argv[3]); // task(tile) distributed:  0: slice , 1: random, 2: trgcr-com-min
    int comm_type = std::stoi(argv[4]); // comm type: 0: block, 1: sparse, 11: no communication (test only)
    int n_rolls = std::stoi(argv[5]);
    int repeat = std::stoi(argv[6]);
    int nvlink_num_producers = 4; 
    int ib_num_producers = 0;

    if (argc == 8) {
        nvlink_num_producers = std::stoi(argv[7]);
    }
    // std::string result_dir_file = std::string(argv[9]) + "_dist_spmm_" + std::to_string(npes) + "_dims_" + std::to_string(dims) 
    //     + "_type_" + std::to_string(task_dist) + std::to_string(tile_order)  + std::to_string(comm_type)
    //     + "_producers_" + std::to_string(num_producer) + "_rolls_" + std::to_string(n_rolls) + ".txt";

    std::vector<GpuInfo> gpu_topology;
    int num_nodes;
    nvshmem_init_use_mpi(mpi_comm,gpu_topology,num_nodes);
    static constexpr cusparseIndexType_t index_type = std::is_same<Idx_t, int>::value? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
    static constexpr cudaDataType_t value_type = std::is_same<Val_t, float>::value? CUDA_R_32F : CUDA_R_64F;

    // std::ofstream output_record(result_dir_file);

    std::vector<double> time_costs;
    MPI_CHECK( MPI_Barrier(mpi_comm));
    double t0, t1;

    // 1. read sparse matrix
    t0 = MPI_Wtime();
    SparseMatrix<Idx_t, Val_t> sp_matrix;
    if (file_name.substr(file_name.length() - 4) == ".mtx") {
        if (!sp_matrix.load_from_mtx(file_name)){
            std::cerr << "Failed to load the matrix from the mtx file." << std::endl;
            return EXIT_FAILURE;
        }
    }
    else {
        if (!sp_matrix.load_from_csr(file_name)) {
            std::cerr << "Failed to load the matrix from (CSR) files." << std::endl;
            return EXIT_FAILURE;
        }
    }
    t1 = MPI_Wtime();
    // read sparse data time: ms
    time_costs.push_back((t1 - t0) * 1000);  // read sparse matrix

    // debug_point("1. Finish loading the Sparse matrix.");

    // 2. sparse matrix partition
    MPI_CHECK( MPI_Barrier(mpi_comm));
    t0 = MPI_Wtime();
    Idx_t M = sp_matrix.num_rows;
    Idx_t N = sp_matrix.num_cols;
    Idx_t m_local, m_lb, m_ub;
    m_local = (M + npes - 1) / npes;
    m_lb = mype * m_local;
    m_ub = std::min(m_lb + m_local, M);
    m_local = m_ub - m_lb;
    sp_matrix.turn_local_l_u_csr(m_lb, m_ub);
    sp_matrix.release_global_data();
    const auto &part_out = sp_matrix.get_csr();
    auto local_indptr  = std::get<0>(part_out);
    auto local_indices = std::get<1>(part_out);
    auto local_values = std::get<2>(part_out);
    t1 = MPI_Wtime();
    time_costs.push_back((t1 - t0) * 1000);  // partition sparse matrix

    // debug_point("2. Finish partitioning the Sparse matrix.");
    // 3. slice sparse matrix to [n_rolls * npes] tiles
    MPI_CHECK( MPI_Barrier(mpi_comm));
    t0 = MPI_Wtime();
    int total_tiles = n_rolls * npes;
    int col_per_tile = (N + total_tiles - 1) / total_tiles;
    std::vector<cusparseSpMatDescr_t> matA_list(total_tiles);
    
    std::vector<std::vector<Idx_t>> row_ptr_parts(total_tiles);
    for (int i = 0; i < total_tiles; ++i) {
        row_ptr_parts[i].resize(m_local + 1, 0);
    }
    std::vector<std::vector<Idx_t>> col_ind_parts(total_tiles);
    std::vector<std::vector<Val_t>> val_parts(total_tiles);
    std::vector<std::unordered_set<int>> col_ind_unique(total_tiles);
    std::vector<int> n_col_list(total_tiles, 0);
    for (Idx_t row = 0; row < m_local; ++row) {
        for (Idx_t j = local_indptr[row]; j < local_indptr[row + 1]; ++j) {
            Idx_t col_local = local_indices[j] % col_per_tile;
            Idx_t part_id = local_indices[j] / col_per_tile;
            row_ptr_parts[part_id][row + 1]++;
            col_ind_parts[part_id].push_back(col_local);
            val_parts[part_id].push_back(local_values[j]);
            col_ind_unique[part_id].insert((int)col_local);
        }
    }
    for (int i = 0; i < total_tiles; ++i) {
        n_col_list[i] = col_ind_unique[i].size();
    }

    t1 = MPI_Wtime();
    time_costs.push_back((t1 - t0) * 1000);
    // debug_point("3. Finish slicing the Sparse matrix portions.");
    // 4. Distribute tiles to distributed PEs
    MPI_CHECK( MPI_Barrier(mpi_comm));
    t0 = MPI_Wtime();
    std::vector<int> remaps(total_tiles);
    std::iota(remaps.begin(), remaps.end(), 0);
    double re_allocation_t= 0;
    if (task_dist == 0) {
        // slice
        for (int i = 0; i < total_tiles; ++i) {
            remaps[i] = i / n_rolls;
        }
    }
    else if (task_dist == 1) {
        // random shuffle
        std::random_shuffle(remaps.begin(), remaps.end());
    }
    else if (task_dist == 2) {
        // trgcr-com-min
        re_allocation_t = ComputeAllocation_HSDMA_MPI(n_col_list, gpu_topology, num_nodes, n_rolls, remaps, mpi_comm, 0, false);
    }
    else {
        // wrong task distribution type
        std::cerr << "Wrong task distribution type." << std::endl;
        return EXIT_FAILURE;
    }
    t1 = MPI_Wtime();
    if (mype == 0) {
        // output task distribution
        std::cout << "task_distribution: " << task_dist << " and rolls:" << n_rolls << std::endl;
        for (int i = 0; i < total_tiles; ++i) {
            std::cout << remaps[i] << " ";
        }
        std::cout << std::endl;
    }
    time_costs.push_back((t1 - t0) * 1000);

    // MPI_CHECK( MPI_Barrier(mpi_comm));
    // return 0;

    // debug_point("4. Finish distributing the Densy matrix basic on the Sparse matrix portions.");
    // 5. Set tile compute(communicate) order
    MPI_CHECK( MPI_Barrier(mpi_comm));
    t0 = MPI_Wtime();
    std::vector<int> remaps_tile_order(total_tiles);
    std::iota(remaps_tile_order.begin(), remaps_tile_order.end(), 0);
    double_level_comm_node_processing(mpi_comm, n_col_list, col_ind_unique,0);
    t1 = MPI_Wtime();
    time_costs.push_back((t1 - t0) * 1000);
    MPI_CHECK( MPI_Barrier(mpi_comm));
    // debug_point("5. Finish setting the Densy matrix compute(communicate) order.");
    // 6. Device memory allocation
    t0 = MPI_Wtime();
    // Dense matrix
    std::vector<Val_t *> sh_ptrs;
    for (int i = 0; i < total_tiles; ++i) {
        if (mype == remaps[i]) {
            Val_t * tmp_ptr = (Val_t *)nvshmem_malloc(col_per_tile  * dims * sizeof(Val_t));
            // initialize dense matrix, now with 1
            std::vector<Val_t> init_data(col_per_tile * dims, 1.0);
            CUDA_CHECK( cudaMemcpy(tmp_ptr, init_data.data(), col_per_tile * dims * sizeof(Val_t), cudaMemcpyHostToDevice));
            sh_ptrs.push_back(tmp_ptr);
        }
    }


    std::vector<Val_t *> d_stores(ib_num_producers + nvlink_num_producers, nullptr);
    for (int i = 0; i < ib_num_producers + nvlink_num_producers; ++i) {
        d_stores[i] = (Val_t *)nvshmem_malloc(col_per_tile  * dims * sizeof(Val_t));
        CUDA_CHECK( cudaMemset(d_stores[i], 0, col_per_tile  * dims * sizeof(Val_t)));
    }
    
    // Sparse matrix
    std::vector<Idx_t *> d_row_ptrs(total_tiles, nullptr);
    std::vector<Idx_t *> d_col_inds(total_tiles, nullptr);
    std::vector<Val_t *> d_vals(total_tiles, nullptr);
    std::vector<Idx_t> nnz_list(total_tiles, 0);
    std::vector<int> n_slice_list(total_tiles, 0);
    // std::vector<Idx_t> n_col_list(total_tiles, 0);
    std::vector<int *> d_col_ids(total_tiles, nullptr);
    for (const auto & i: remaps_tile_order) {
        if (val_parts[i].size() > 0) {

            auto nnz = col_ind_parts[i].size();
            std::vector<Idx_t> row_ids(nnz,0);
            std::partial_sum(row_ptr_parts[i].begin(), row_ptr_parts[i].end(), row_ptr_parts[i].begin());
            for (int j = 0; j < m_local; ++j) {
                std::fill(row_ids.begin() + row_ptr_parts[i][j], row_ids.begin() + row_ptr_parts[i][j + 1], j);
            }
            nnz_list[i] = nnz;
            n_slice_list[i] = std::min((Idx_t)col_per_tile, N - i * col_per_tile);

            CUDA_CHECK( cudaMalloc((void**)&d_row_ptrs[i], nnz * sizeof(Idx_t)) );
            CUDA_CHECK( cudaMalloc((void**)&d_col_inds[i], nnz * sizeof(Idx_t)) );
            CUDA_CHECK( cudaMalloc((void**)&d_vals[i], nnz * sizeof(Val_t)) );

            CUDA_CHECK( cudaMemcpy(d_row_ptrs[i], row_ids.data(), nnz * sizeof(Idx_t), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_col_inds[i], col_ind_parts[i].data(), nnz * sizeof(Idx_t), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_vals[i], val_parts[i].data(), nnz * sizeof(Val_t), cudaMemcpyHostToDevice) );

            CUSPARSE_CHECK( cusparseCreateCoo(&matA_list[i], m_local, n_slice_list[i], nnz, d_row_ptrs[i], d_col_inds[i], d_vals[i],
                index_type, CUSPARSE_INDEX_BASE_ZERO, value_type)); 

            std::vector<int> vec_data(col_ind_unique[i].begin(), col_ind_unique[i].end());
            // n_col_list[i] = vec_data.size();
            CUDA_CHECK( cudaMalloc((void**)&d_col_ids[i], vec_data.size() * sizeof(int)) );
            CUDA_CHECK( cudaMemcpy(d_col_ids[i], vec_data.data(), vec_data.size() * sizeof(int), cudaMemcpyHostToDevice) );
        }
    }

    // Result matrix
    cusparseDnMatDescr_t matC;
    Val_t* d_C;
    CUDA_CHECK( cudaMalloc((void**)&d_C, m_local *   dims * sizeof(Val_t)) );
    CUDA_CHECK( cudaMemset(d_C, 0, m_local *   dims * sizeof(Val_t)) );
    CUSPARSE_CHECK( cusparseCreateDnMat(&matC, m_local, dims, dims, d_C, value_type, CUSPARSE_ORDER_ROW));

    t1 = MPI_Wtime();
    time_costs.push_back((t1 - t0) * 1000);
    // debug_point("6. Finish allocating the Densy matrix and Sparse matrix on the Device.");
    // 7. Set communication manager
    MPI_CHECK( MPI_Barrier(mpi_comm));
    t0 = MPI_Wtime();
    CommMan_hp<Val_t> comm_manager(mype, npes, sh_ptrs, d_stores,  remaps, n_slice_list, n_col_list, d_col_ids, dims, comm_type, num_nodes, ib_num_producers);
    t1 = MPI_Wtime();
    time_costs.push_back((t1 - t0) * 1000);
    int my_node_id = comm_manager.get_node_id();
    int pe_per_node = comm_manager.get_pe_per_node();
    int pe_in_node = comm_manager.get_pe_in_node();
    // debug_point("7. Finish setting the communication manager.");
    // 8. Create Compute Resource
    MPI_CHECK( MPI_Barrier(mpi_comm));
    t0 = MPI_Wtime();
    // cudaStream_t compute_stream, ib_stream, nvlink_stream;
    // CUDA_CHECK( cudaStreamCreate(&compute_stream) );
    // CUDA_CHECK( cudaStreamCreate(&ib_stream) );
    // CUDA_CHECK( cudaStreamCreate(&nvlink_stream) );
    cudaStream_t compute_stream, ib_stream, nvlink_stream;
    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    CUDA_CHECK(cudaStreamCreateWithPriority(&nvlink_stream, cudaStreamNonBlocking, greatestPriority));

    CUDA_CHECK(cudaStreamCreateWithPriority(&ib_stream, cudaStreamNonBlocking, (greatestPriority + leastPriority) / 2));

    CUDA_CHECK(cudaStreamCreateWithPriority(&compute_stream, cudaStreamNonBlocking, leastPriority));


    cudaEvent_t * nvlink_release_events = new cudaEvent_t[nvlink_num_producers];
    cudaEvent_t * nvlink_require_events = new cudaEvent_t[nvlink_num_producers];

    cudaEvent_t * ib_release_events = new cudaEvent_t[ib_num_producers];
    cudaEvent_t * ib_require_events = new cudaEvent_t[ib_num_producers];

    for (int i = 0; i < ib_num_producers; ++i) {
        CUDA_CHECK( cudaEventCreate(&ib_release_events[i]) );
        CUDA_CHECK( cudaEventCreate(&ib_require_events[i]) );
    }
    for (int i = 0; i < nvlink_num_producers; ++i) {
        CUDA_CHECK( cudaEventCreate(&nvlink_release_events[i]) );
        CUDA_CHECK( cudaEventCreate(&nvlink_require_events[i]) );
    }

    cusparseHandle_t handle;
    CUSPARSE_CHECK( cusparseCreate(&handle) );
    CUSPARSE_CHECK( cusparseSetStream(handle, compute_stream) );


    t1 = MPI_Wtime();
    time_costs.push_back((t1 - t0) * 1000);
    // debug_point("8. Finish creating the compute resource.");

    MPI_CHECK( MPI_Barrier(mpi_comm));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: fail before run dist_spmm programs): %s\n", cudaGetErrorString(err));
    }
    else{
        if (mype == 0){
            std::cout << "Running dist_spmm programs on " << file_name << " with [" << dims << "] dims"<< std::endl; 
        }
    }
    gpu_warmup();
    CUDA_CHECK( cudaDeviceSynchronize());
    // debug_point("9. Finish warmup.");
    // 9. Start run
    // int repeat = 20;
    float alpha = 1.0, beta = 1.0;
    std::vector<cusparseDnMatDescr_t> matB_list(total_tiles);
    float max_mem_used = gpu_mem_used();
    void* cusparse_buffers[total_tiles];
    size_t bufferSizes[total_tiles] = {0}; 
    cusparseSpMMAlg_t algorithm = CUSPARSE_SPMM_ALG_DEFAULT; // T


    for (int time_it = 0; time_it < repeat; ++time_it) {
        CUDA_CHECK( cudaMemset(d_C, 0, m_local *   dims * sizeof(Val_t)));
        CUDA_CHECK( cudaDeviceSynchronize());
        MPI_CHECK( MPI_Barrier(mpi_comm));
        t0 = MPI_Wtime();
        // constexpr int buff_num = 2;
        std::queue<int> release_ib_buff;

        int node_id = 0;
        int in_buff_id = 0; 
        for (int in_gpu_id = 0; in_gpu_id < pe_per_node; ++in_gpu_id) {
            for (int local_tile_id = 0; local_tile_id < n_rolls; ++local_tile_id) {
                int gpu_id = (pe_in_node + in_gpu_id) % pe_per_node;

                int global_tile_id = comm_manager.get_gobal_tile_id(node_id, gpu_id, local_tile_id);
                if (nnz_list[global_tile_id] == 0) continue;

                if (in_gpu_id > nvlink_num_producers) {
                    CUDA_CHECK( cudaStreamWaitEvent(nvlink_stream, nvlink_release_events[in_buff_id], 0));
                }
                Val_t * d_buff = comm_manager.comm_on_stream_nvlink(in_buff_id, local_tile_id, gpu_id, node_id ,nvlink_stream); // both local and remote getmem
                
                CUDA_CHECK( cudaEventRecord(nvlink_require_events[in_buff_id], nvlink_stream));
                CUDA_CHECK( cudaStreamWaitEvent(compute_stream, nvlink_require_events[in_buff_id], 0));

                auto & matA = matA_list[global_tile_id];
                auto & matB = matB_list[global_tile_id];
                CUSPARSE_CHECK( cusparseCreateDnMat(&matB, n_slice_list[global_tile_id], dims, dims, d_buff, value_type, CUSPARSE_ORDER_ROW));
                CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                    handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, 
                    matA,
                    matB,
                    &beta,
                    matC,
                    value_type,
                    algorithm,
                    &bufferSizes[global_tile_id]
                ));
                CUDA_CHECK( cudaMallocAsync(&cusparse_buffers[global_tile_id], bufferSizes[global_tile_id], compute_stream));
                CUSPARSE_CHECK(cusparseSpMM(
                    handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    matA,
                    matB,
                    &beta,
                    matC,
                    value_type,
                    algorithm,
                    cusparse_buffers[global_tile_id]
                ));
                CUDA_CHECK( cudaFreeAsync(cusparse_buffers[global_tile_id], compute_stream));
                CUDA_CHECK( cudaEventRecord(nvlink_release_events[in_buff_id], compute_stream));
                in_buff_id = (in_buff_id + 1) % nvlink_num_producers;
            }
        }
        CUDA_CHECK( cudaDeviceSynchronize());
        MPI_CHECK( MPI_Barrier(mpi_comm));
        t1 = MPI_Wtime();
        time_costs.push_back((t1 - t0) * 1000);
        max_mem_used = std::max(gpu_mem_used(), max_mem_used);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: fail in running SPMM %s\n", cudaGetErrorString(err));
    }
    MPI_Barrier(mpi_comm);
    long comm_traffics[3] = {0,0,0};
    comm_traffics[0] = comm_manager.get_comm_traffic_intra();
    comm_traffics[1] = comm_manager.get_comm_traffic_inter();
    comm_traffics[2] = comm_manager.get_reduction_comm_traffic_inter();

    MPI_Allreduce(MPI_IN_PLACE, &comm_traffics, 3, MPI_LONG, MPI_SUM, mpi_comm);
    size_t size_time = time_costs.size();
    MPI_Allreduce(MPI_IN_PLACE, time_costs.data(), size_time, MPI_DOUBLE, MPI_MAX, mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &max_mem_used, 1, MPI_FLOAT, MPI_MAX, mpi_comm);
    // std::vector<double> pre_time   = std::vector<double>(time_costs.begin(),   time_costs.begin() + 8);
    // run_record keeps the per-repetition runtimes (skip 8 setup stages + 5 warmups).
    // Guard the slice: time_costs only holds 8 + repeat samples, so for repeat < 5
    // begin()+13 would exceed end() and throw std::length_error.
    size_t run_start = std::min<size_t>(8 + 5, time_costs.size());
    std::vector<double> run_record(time_costs.begin() + run_start, time_costs.end());
    if (mype == 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Processing time: (ms) " << std::endl;
        std::cout << "Re-allocation time(all): " << time_costs[3] << " Hira Time: " << re_allocation_t << std::endl;
        double avg_time = 0.0;
        for (auto &t: run_record) {
            avg_time += t;
        }
        if (!run_record.empty()) {
            avg_time /= run_record.size();
        }
        
        std::cout << "Max memory used: " << max_mem_used << " GB" << std::endl << std::flush;
        std::cout << "Average time used: " << avg_time << " ms" << std::endl << std::flush;
        std::cout << "Communication Traffic Intra : " << comm_traffics[0] /(1024.0 * 1024.0 * 1024.0) << " GB" << std::endl << std::flush;
        std::cout << "Communication Traffic Inter : " << comm_traffics[1] /(1024.0 * 1024.0 * 1024.0)<< " GB" << std::endl << std::flush;
        // std::cout << "Communication Traffic Reduc : " << comm_traffics[2]/(1024.0 * 1024.0 * 1024.0) << " GB" << std::endl << std::flush;

        // Flops
        double gflops = 2.0 * sp_matrix.nnz * dims / avg_time / 1e6;
        std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    }
    
    nvshmemx_buffer_unregister_all();
    nvshmem_finalize();
    MPI_CHECK( MPI_Barrier(mpi_comm));
    MPI_CHECK( MPI_Finalize());
    return EXIT_SUCCESS;
}