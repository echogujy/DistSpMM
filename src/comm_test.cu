#include <numeric>
#include <algorithm>
#include <cassert>
#include <vector>
#include <random>
#include <fstream>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include "utils.h"
#include "comm_man.h"
using Idx_t = int;  
using Val_t = float;  


int main(int argc, char *argv[]) {
    int mype, npes;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mype));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &npes));
    
    if (argc < 5)   {
        std::cerr << "Usage: ./comm_test <N> <dims_s> <dims_e> <stride> <gpu_shift> <bins> <com_type>" << std::endl;
        return EXIT_FAILURE;
    }

    int N = std::stoi(argv[1]);
    int dims_s = std::stoi(argv[2]);
    int dims_e = std::stoi(argv[3]);
    int stride = std::stoi(argv[4]);

    int gpu_shift = std::stoi(argv[5]);
    int sparse_bins = std::stoi(argv[6]);
    int com_type = std::stoi(argv[7]);

    int64_t nvshmem_mem_size = N  * dims_e * sizeof(Val_t);
    // check_parse_and_set_nvshmem_symmetric_size(nvshmem_mem_size); use VMM
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);

    Val_t * sh_ptr = (Val_t*)nvshmem_malloc(nvshmem_mem_size);

    Val_t * d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, nvshmem_mem_size));
    CUDA_CHECK(cudaMemset(d_ptr, 0,nvshmem_mem_size));
    int reg_id = nvshmemx_buffer_register(d_ptr,nvshmem_mem_size);
    if (mype == 0) {
        std::cout<< "comm buffer size of send and recv : " << formatBytes(nvshmem_mem_size) << " GiB"  << std::endl;
        if (com_type ==0 ) {
            std::cout << "Communication type: intra node communication " << std::endl;
        }
        else {
            std::cout << "Communication type: inter node communication " << std::endl;
        }

    }

    std::vector<int> sequence(N);
    std::iota(sequence.begin(), sequence.end(), 0);

    // shuffle
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sequence.begin(), sequence.end(), g);

    int * d_sequence;
    CUDA_CHECK(cudaMalloc((void**)&d_sequence, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sequence, sequence.data(), N * sizeof(int), cudaMemcpyHostToDevice));


    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    int dims = dims_s;
    int get_idp = com_type == 0 ? mype^1: (mype + npes_node) % npes;
    MPI_CHECK(MPI_Barrier(mpi_comm));
    // return 0;
    while (dims <= dims_e) {
        if (mype == 0) {
            std::cout<< "Dim: " << dims << " Comm-size: " << formatBytes(N * dims * sizeof(Val_t)) << " message size: " << formatBytes(dims * sizeof(Val_t)) << std::endl;
        }
        int rolls = 5;
        // Resize the recorder, dropping the comm_sparse_aware_thread test items
        std::vector<std::vector<float>> time_recorders(rolls);

        float t0, t1;


        for (int r = 0; r < rolls; ++r) {
            MPI_CHECK(MPI_Barrier(mpi_comm));
            
            t0 = MPI_Wtime();
            nvshmemx_getmem_on_stream(d_ptr, sh_ptr, N * dims * sizeof(Val_t), get_idp , stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            t1 = MPI_Wtime();
            MPI_CHECK(MPI_Barrier(mpi_comm));
            time_recorders[r].push_back( (t1 - t0) * 1000.0f );

            for (int i = 1; i < sparse_bins + 1; ++i) {
                float density = static_cast<float>(i) / static_cast<float>(sparse_bins);
                int seq_len = N * density;
                CUDA_CHECK(cudaStreamSynchronize(stream));
                t0 = MPI_Wtime();
                comm_sparse_aware<Val_t>(d_ptr, sh_ptr, seq_len, dims, d_sequence, get_idp, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                t1 = MPI_Wtime();
                time_recorders[r].push_back((t1 - t0) * 1000.0f);
            }
        }
        time_recorders.erase(time_recorders.begin(), time_recorders.begin() + 3);
        int rolls_now = rolls - 3;
        // Resize the average recorder, dropping the comm_sparse_aware_thread slots
        std::vector<float> time_recorders_avg(sparse_bins + 1,0);
        for (int i = 0; i < sparse_bins + 1; ++i) {
            for (int j = 0; j < rolls_now; ++j) {
                time_recorders_avg[i] += time_recorders[j][i];
            }
            time_recorders_avg[i] /= rolls_now;
        }
        MPI_CHECK(MPI_Barrier(mpi_comm));
        // Reduce all the time records
        
        if ( mype == 0) {
            
            MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, time_recorders_avg.data(), time_recorders_avg.size(), MPI_FLOAT, MPI_SUM, 0, mpi_comm))
            // average time
            std::for_each(time_recorders_avg.begin(), time_recorders_avg.end(), [&](float &x) {x /= npes; });
            int i = 1;
            while (time_recorders_avg[0] > time_recorders_avg[i] && i < sparse_bins + 1){
                i++;
            }
            std::cout << "========================== " << std::endl;
            if (mype == 0)  {
                std::cout << dims << "," <<  static_cast<float>(i-1) / static_cast<float>(sparse_bins+1) << std::endl;
            }
            
        }
        else {
            MPI_CHECK(MPI_Reduce(time_recorders_avg.data(), time_recorders_avg.data(), time_recorders_avg.size(), MPI_FLOAT, MPI_SUM, 0, mpi_comm))
        }
        dims *= stride;
    }
    MPI_CHECK(MPI_Barrier(mpi_comm));

    nvshmem_finalize();
    MPI_CHECK(MPI_Finalize());
    return EXIT_SUCCESS;
}