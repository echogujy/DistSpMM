
#ifndef comm_man_H
#define comm_man_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <queue>
int SMs = 0;

int get_sm_num() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

// template<typename T>
// __global__ void comm_sparse_kernel_warp(T *dest, const T * src, size_t nelems_col, int dims, const int * d_col_start_count, int pe)
// {
//     const int laneId = threadIdx.x % warpSize;
//     const int mem_size = dims * sizeof(T);

//     int warpId = threadIdx.x / warpSize;
//     extern __shared__ int mem_ptr[];
//     int * col_start_count = mem_ptr + warpId * warpSize;
//     warpId += blockIdx.x  * (blockDim.x / warpSize) ;
//     for (int i = warpId * warpSize; i < nelems_col; i += gridDim.x * blockDim.x){
//         int work_id = i + laneId;
//         col_start_count[laneId] = (work_id < nelems_col) ? __ldg(d_col_start_count + work_id) : -1;
//         __syncwarp();
//         #pragma unroll 32
//         for (int k = 0; k < warpSize; ++k) {
//             int shift = col_start_count[k] * dims;
//             if (shift >= 0) nvshmemx_getmem_nbi_warp(dest + shift, src + shift, mem_size, pe);
//         }
//     }
// }

// template<typename T>
// void comm_sparse_aware_warp(T *dest, const T *src, size_t nelems_col, int dims, const int * d_col_start_count,int pe, cudaStream_t stream) {
//     int blocks_size = warpSize * 4;
//     if (SMs == 0)   {
//         SMs = get_sm_num();
//     }
//     comm_sparse_kernel_warp<T><<<SMs, blocks_size, blocks_size * sizeof(int), stream>>>(dest, src, nelems_col, dims, d_col_start_count, pe);
// }

// template<typename T>
// __global__ void comm_sparse_kernel_thread(T *dest, const T * src, size_t nelems_col, int dims, const int * d_col_start_count, int pe)
// {
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     const int mem_size = dims * sizeof(T);
//     if (tid < nelems_col) {
//         const int shift = __ldg(d_col_start_count + tid);
//         nvshmem_getmem_nbi(dest + shift, src + shift, mem_size, pe);
//     }
// }

// template<typename T>
// void comm_sparse_aware_thread(T *dest, const T *src, size_t nelems_col, int dims, const int * d_col_start_count,int pe, cudaStream_t stream) {
//     int blocks_size = warpSize * 8;
//     int blocks_num = (nelems_col + blocks_size - 1) / blocks_size;
//     comm_sparse_kernel_thread<<<blocks_num, blocks_size, 0, stream>>>(dest, src, nelems_col, dims, d_col_start_count, pe);
// }

template<typename T>
__global__ void comm_sparse_kernel(T *dest, const T * src, size_t nelems_col, int dims, const int * d_col_start_count, int pe)
{
    const int laneId = threadIdx.x % warpSize;
    const int mem_size = dims * sizeof(T);

    int warpId = threadIdx.x / warpSize;
    extern __shared__ int mem_ptr[];
    int * col_start_count = mem_ptr + warpId * warpSize;
    warpId += blockIdx.x  * (blockDim.x / warpSize) ;
    for (int i = warpId * warpSize; i < nelems_col; i += gridDim.x * blockDim.x){
        int work_id = i + laneId;
        col_start_count[laneId] = (work_id < nelems_col) ? __ldg(d_col_start_count + work_id) : -1;
        __syncwarp();
        #pragma unroll 32
        for (int k = 0; k < warpSize; ++k) {
            int shift = col_start_count[k] * dims;
            if (shift >= 0) nvshmemx_getmem_nbi_warp(dest + shift, src + shift, mem_size, pe);
        }
    }
}

template<typename T>
void comm_sparse_aware(T *dest, const T *src, size_t nelems_col, int dims, const int * d_col_start_count,int pe, cudaStream_t stream) {
    int blocks_size = warpSize * 8;
    if (SMs == 0)   {
        SMs = get_sm_num();
    }
    comm_sparse_kernel<T><<<SMs, blocks_size, blocks_size * sizeof(int), stream>>>(dest, src, nelems_col, dims, d_col_start_count, pe);
}


template<typename Val_t>
class CommMan {
    int mype;
    int npes;
    int dims;
    int tile_size;
    std::vector<Val_t *> shared_buffers;
    std::vector<Val_t *> d_buff;
    std::vector<int> remap;

    // 
    std::vector<int> n_slices;
    std::vector<int> n_nozero_cols;
    std::vector<int *> d_nozero_cols;

    // Sparse communication
    std::vector<bool> is_sparse_comm;

public:
    CommMan(int mype, int npes, const std::vector<Val_t *> & shared_buffers, std::vector<Val_t *> &  d_stores, const std::vector<int> remap, const std::vector<int> &n_slices, 
        const std::vector<int> &n_nozero_cols, const std::vector<int*> &d_nozero_cols, const int dims, const int comm_type=0) 
        : mype(mype), shared_buffers(shared_buffers), remap(remap), n_slices(n_slices), d_buff(d_stores),
        n_nozero_cols(n_nozero_cols), d_nozero_cols(d_nozero_cols), dims(dims)
    {
        assert(shared_buffers.size() == remap.size());
        tile_size = remap.size();
        this->initialize_comm_model(comm_type);
        this->fill_shared_buffers_pointer();
        SMs = get_sm_num();
    }

    void fill_shared_buffers_pointer() {
        std::vector<std::vector<int>> task_assignments(npes);
        for (int i = 0; i < tile_size; i++) {
            task_assignments[remap[i]].push_back(i);
        }

        for (int i = 0; i < npes; i++) {
            for (int j = 0; j < task_assignments[i].size(); j++) {
                shared_buffers[task_assignments[i][j]] = shared_buffers[task_assignments[mype][j]];
            }
        }
    }
    void initialize_comm_model(const int comm_type) {
        if (comm_type == 0)
            for (int i = 0; i < tile_size; i++) {
                is_sparse_comm.push_back(false);
            }
        else if (comm_type == 1) {
            for (int i = 0; i < tile_size; i++) {
                is_sparse_comm.push_back(true);
            }
        }
        else if (comm_type == 2) {
            // float a = 12873.61472, b = 125.27085, c =  105.50392, u = -4.65;  // H100
            float a = 0.07700, b = 64.48303, c = 75.28834, u = 0.31; // A800            
            float k = std::min((float)c, (float)dims);
            float threds_intra = 1 / (a * pow(k, u) + b / k);

            // float inter_a = 1452.17110, inter_b = -1407.82558, inter_c = 47.34021, inter_u = -1.00; // H100
            float inter_a = 1593.97237, inter_b = -1142.47471, inter_c = 262.17718, inter_u = -1.00; // A800
            float inter_k = std::min((float)inter_c, (float)dims);
            float threds_inter = 1 / (inter_a * pow(inter_k, inter_u) + inter_b / inter_k);
            for (int i = 0; i < tile_size; i++) {
                float d_i = (float)n_nozero_cols[i] / (float)(n_slices[i]+1);
                if (remap[i] / 8 == mype / 8 ) {
                    is_sparse_comm.push_back(threds_intra > d_i);
                }
                else {
                    is_sparse_comm.push_back(threds_inter > d_i);
                }
            }
        }
        else {
            assert(false);
        }
    }

    Val_t * comm_on_stream(int producer_id, int tile_id, cudaStream_t stream) {
        Val_t * output_buff = nullptr;
        if (this->remap[tile_id] == mype) {
            output_buff = shared_buffers[tile_id];
        }
        else {
            output_buff = this->d_buff[producer_id];
            if (is_sparse_comm[tile_id]) {
                comm_sparse_aware<Val_t>(output_buff, shared_buffers[tile_id], n_nozero_cols[tile_id], dims, d_nozero_cols[tile_id], remap[tile_id], stream);
            }
            else {
                nvshmemx_getmem_on_stream(output_buff, shared_buffers[tile_id], sizeof(Val_t) * dims * n_slices[tile_id], remap[tile_id], stream);
            }
        }
        return output_buff;
    }   

    long get_comm_traffic() {
        long comm_traffic = 0;
        for (int i = 0; i < tile_size; i++) {
            if (this->remap[i] != mype) {
                if (is_sparse_comm[i]) {
                    comm_traffic += sizeof(Val_t) * dims * n_nozero_cols[i];
                }
                else {
                    comm_traffic += sizeof(Val_t) * dims * n_slices[i];
                }
            }
        }
        return comm_traffic;
    }
};

// template<typename Val_t>
// class CommMan {
//     int mype;
//     int npes;
//     int dims;
//     int tile_size;
    
//     std::vector<Val_t *> shared_buffers;
//     std::vector<Val_t *> d_buff;
//     std::vector<int> remap;

//     // 
//     std::vector<int> n_slices;
//     std::vector<int> n_nozero_cols;
//     std::vector<int *> d_nozero_cols;
//     int comm_type;

// public:
//     CommMan(int mype, int npes, const std::vector<Val_t *> & shared_buffers, std::vector<Val_t *> &  d_stores, const std::vector<int> remap, const std::vector<int> &n_slices, 
//         const std::vector<int> &n_nozero_cols, const std::vector<int*> &d_nozero_cols, const int dims, const int comm_type=0) 
//         : mype(mype), shared_buffers(shared_buffers), remap(remap), n_slices(n_slices), d_buff(d_stores),
//         n_nozero_cols(n_nozero_cols), d_nozero_cols(d_nozero_cols), dims(dims)
//     {
//         assert(shared_buffers.size() == remap.size());
//         tile_size = remap.size();
  
//         this->fill_shared_buffers_pointer();
//         SMs = get_sm_num();
//         this->comm_type = comm_type;
//     }

//     void fill_shared_buffers_pointer() {
//         std::vector<std::vector<int>> task_assignments(npes);
//         for (int i = 0; i < tile_size; i++) {
//             task_assignments[remap[i]].push_back(i);
//         }

//         for (int i = 0; i < npes; i++) {
//             for (int j = 0; j < task_assignments[i].size(); j++) {
//                 shared_buffers[task_assignments[i][j]] = shared_buffers[task_assignments[mype][j]];
//             }
//         }
//     }

//     Val_t * comm_on_stream(int producer_id, int tile_id, cudaStream_t stream) {
//         Val_t * output_buff = nullptr;
//         if (this->remap[tile_id] == mype) {
//             output_buff = shared_buffers[tile_id];
//         }
//         else {
//             output_buff = this->d_buff[producer_id];
//             if (comm_type == 0) { // dense comm
//                 nvshmemx_getmem_on_stream(output_buff, shared_buffers[tile_id], sizeof(Val_t) * dims * n_slices[tile_id], remap[tile_id], stream);
//             }
//             else if (comm_type == 1) { // sparse comm
//                 comm_sparse_aware<Val_t>(output_buff, shared_buffers[tile_id], n_nozero_cols[tile_id], dims, d_nozero_cols[tile_id], remap[tile_id], stream);
//             }
//             else if (comm_type == 11) {
//                 // W/o communication operations for get the time of communication and computation
//             }
//             else {
//                 assert(false);
//             }
//         }
//         return output_buff;
//     }   

//     long get_comm_traffic() {
//         long comm_traffic = 0;
//         for (int i = 0; i < tile_size; i++) {
//             if (this->remap[i] != mype) {
//                 if (comm_type == 0) {
//                     comm_traffic += sizeof(Val_t) * dims * n_slices[i];
//                 }
//                 else if (comm_type == 1) {
//                     comm_traffic += sizeof(Val_t) * dims * n_nozero_cols[i];
//                 }
//                 else if (comm_type == 11) {
//                     comm_traffic += 0;
//                 }
//                 else {
//                     assert(false);
//                 }
//             }
//         }
//         return comm_traffic;
//     }

// };



template<typename Val_t>
class CommMan_hp {
    int mype;
    int npes;
    int dims;
    int tile_size;
    int local_tile_num;
    std::vector<Val_t *> shared_buffers;
    std::vector<Val_t *> outer_buff;
    std::vector<Val_t *> inner_buff;
    // Val_t * outer_buff[2] = {nullptr, nullptr};
    // Val_t * inner_buff[2] = {nullptr, nullptr};

    std::vector<int> remap;

    std::vector<int> n_slices;
    std::vector<int> n_nozero_cols;
    std::vector<int *> d_nozero_cols;
    std::vector<std::vector<int>> task_assignments; 
    int comm_type;

    int node_nums;
    int mype_node;
    int npes_node;
    int node_id;

    int set_holding_outer_buff_id;

    std::queue<int> outer_buff_queue;

    std::vector<bool> is_sparse_comm_intra;
    std::vector<bool> is_sparse_comm_inter;

public:
    CommMan_hp(int mype, int npes, const std::vector<Val_t *> & shared_buffers, std::vector<Val_t *> &  d_stores, 
        const std::vector<int> remap, const std::vector<int> &n_slices, 
        const std::vector<int> &n_nozero_cols, const std::vector<int*> &d_nozero_cols, 
        const int dims, const int comm_type=0, const int node_nums=2,
        const int ib_buff_num=2) 
        : mype(mype), shared_buffers(shared_buffers), remap(remap), n_slices(n_slices), 
        n_nozero_cols(n_nozero_cols), d_nozero_cols(d_nozero_cols), dims(dims),  comm_type(comm_type), node_nums(node_nums)
    {
        local_tile_num = shared_buffers.size();
        tile_size = remap.size();
  
        this->fill_task_assignments();

        this->mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        this->npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);


        for (int i = 0; i < ib_buff_num; i++) {
            outer_buff.push_back(d_stores[i]);
        }
        for (int i = ib_buff_num; i < d_stores.size(); i++) {
            inner_buff.push_back(d_stores[i]);
        }

        this->node_id = mype / npes_node;

        this->initialize_comm_model(comm_type);
    }
    void initialize_comm_model(const int comm_type) {
        if (comm_type == 0)
            for (int i = 0; i < tile_size; i++) {
                is_sparse_comm_intra.push_back(false);
                is_sparse_comm_inter.push_back(false);
            }
        else if (comm_type == 1) {
            for (int i = 0; i < tile_size; i++) {
                is_sparse_comm_intra.push_back(true);
                is_sparse_comm_inter.push_back(true);
            }
        }
        else if (comm_type == 2) {
            // Adaptive-communication model parameters (paper Table 2), measured by
            // the authors on the hardware in the paper (paper Table 3). Select the
            // platform with the Makefile's PLATFORM variable (-> DISTSPMM_PLATFORM_*).
            // These values are machine-specific: on different hardware, re-measure
            // them with scripts/comm_test_slurm.sh and fit via tools/fit_comm_model.py.
#ifdef DISTSPMM_PLATFORM_A800
            // SXM-A800 (NVLink 200 GB/s/GPU, HDR IB 25 GB/s/GPU)
            float a = 4075.35, b = 119.61, c = 98.21, u = -3.78;
            float inter_a = 1440.80, inter_b = -1394.99, inter_c = 43.65, inter_u = -1.00;
#else
            // DGX-H100 (NVLink 450 GB/s/GPU, NDR IB 50 GB/s/GPU) [default]
            float a = 12873.61472, b = 125.27085, c =  105.50392, u = -4.65;
            float inter_a = 1452.17110, inter_b = -1407.82558, inter_c = 47.34021, inter_u = -1.00;
#endif
            float k = std::min((float)c, (float)dims);
            float threds_intra = 1 / (a * pow(k, u) + b / k);

            float inter_k = std::min((float)inter_c, (float)dims);
            float threds_inter = 1 / (inter_a * pow(inter_k, inter_u) + inter_b / inter_k);
            for (int i = 0; i < tile_size; i++) {
                float d_i = (float)n_nozero_cols[i] / (float)(n_slices[i]+1);
                is_sparse_comm_intra.push_back(threds_intra > d_i);
                is_sparse_comm_inter.push_back(threds_inter > d_i);
            }
        }
        else {
            assert(false);
        }
    }

    int get_node_id() {
        return node_id;
    }
    int get_pe_per_node() {
        return npes_node;
    }
    int get_pe_in_node() {
        return mype_node;
    }

    int get_gobal_tile_id(int node_id_r, int gpu_id, int tile_id) {
        int pe_id = node_id_r * npes_node + gpu_id;
        return task_assignments[pe_id][tile_id];
    }
    void fill_task_assignments() {
        task_assignments.resize(npes);
        for (int i = 0; i < tile_size; i++) {
            task_assignments[remap[i]].push_back(i);
        }
    }
    int outer_buff_size() {
        return outer_buff_queue.size();
    }
    int outer_buff_front() {
        if (outer_buff_queue.empty()) {
            return -1;
        }
        return outer_buff_queue.front();
    }
    int outer_buff_back() {
        if (outer_buff_queue.empty()) {
            return -1;
        }
        return outer_buff_queue.back();
    }

    void comm_on_stream_ib(int outer_buff_id, int tile_id, int node_id_remote, cudaStream_t stream) {
        if (node_id_remote == node_id)  return;

        int remote_pe = node_id_remote * npes_node + mype_node;
        int original_tile_id = task_assignments[remote_pe][tile_id];

            if (is_sparse_comm_inter[original_tile_id]) {
                comm_sparse_aware<Val_t>(outer_buff[outer_buff_id], shared_buffers[tile_id], n_nozero_cols[original_tile_id], dims, d_nozero_cols[original_tile_id], remote_pe, stream);
            }
            else {
                nvshmemx_getmem_on_stream(outer_buff[outer_buff_id], shared_buffers[tile_id], sizeof(Val_t) * dims * n_slices[original_tile_id], remote_pe, stream);
            }

        outer_buff_queue.push(outer_buff_id);
    }  

    void set_outer_buff_id() {
        if (!outer_buff_queue.empty()) {
            set_holding_outer_buff_id = outer_buff_queue.front();
            outer_buff_queue.pop();
        }
        else {
            std::cerr << "outer_buff_queue is empty[1], please check the code logic." << std::endl;
        }
    }

    Val_t * comm_on_stream_nvlink(int inner_id, int tile_id, int gpu_id, int node_id_remote,cudaStream_t stream) {
        Val_t * output_buff = nullptr;
        Val_t * shared_r_buff = nullptr;
        if (node_id_remote == node_id) {
            if (gpu_id == mype_node) {
                output_buff = shared_buffers[tile_id];
                return output_buff;
            }
            shared_r_buff = shared_buffers[tile_id];
        }else {
            shared_r_buff = outer_buff[set_holding_outer_buff_id];
        }
        output_buff = inner_buff[inner_id];
        int remote_pe = node_id * npes_node + gpu_id;
        int original_tile_id = task_assignments[remote_pe][tile_id];

        // if (comm_type == 0) { // dense comm
        //     nvshmemx_getmem_on_stream(output_buff, shared_r_buff, sizeof(Val_t) * dims * n_slices[original_tile_id], remote_pe, stream);
        // }
        // else if (comm_type == 1) { // sparse comm
        //     comm_sparse_aware<Val_t>(output_buff, shared_r_buff, n_nozero_cols[original_tile_id], dims, d_nozero_cols[original_tile_id], remote_pe, stream);
        // }
        // else if (comm_type == 11) {
        //     // W/o communication operations for get the time of communication and computation
        // }
        if (is_sparse_comm_intra[original_tile_id]) {
            comm_sparse_aware<Val_t>(output_buff, shared_r_buff, n_nozero_cols[original_tile_id], dims, d_nozero_cols[original_tile_id], remote_pe, stream);
        }
        else {
            nvshmemx_getmem_on_stream(output_buff, shared_r_buff, sizeof(Val_t) * dims * n_slices[original_tile_id], remote_pe, stream);
        }
        return output_buff;
    }  

    long get_comm_traffic_inter() {
        long comm_traffic = 0;
        if (node_nums < 2) {
            return 0;
        }
        for (int node_id_remote = 0; node_id_remote < node_nums; node_id_remote++) {
            if (node_id_remote == node_id)  continue;
            int remote_pe = node_id_remote * npes_node + mype_node;
            for (int tile_id = 0; tile_id < task_assignments[remote_pe].size(); tile_id++) {
                int original_tile_id = task_assignments[remote_pe][tile_id];
                if (is_sparse_comm_inter[original_tile_id]) {
                    comm_traffic += sizeof(Val_t) * dims * n_nozero_cols[original_tile_id];
                }
                else {
                    comm_traffic += sizeof(Val_t) * dims * n_slices[original_tile_id];
                }
            }
        }
        return comm_traffic;
    }

    long get_comm_traffic_intra() { 
        long comm_traffic = 0;
        for (int i = 0; i < tile_size; i++) {
            if (this->remap[i] != mype) {
                if (is_sparse_comm_intra[i]) {
                    comm_traffic += sizeof(Val_t) * dims * n_nozero_cols[i];
                }
                else {
                    comm_traffic += sizeof(Val_t) * dims * n_slices[i];
                }
            }
        }
        return comm_traffic;
    }

    long get_reduction_comm_traffic_inter() {
        long comm_traffic = 0;
        if (node_nums < 2) {
            return 0;
        }
        for (int node_id_remote = 0; node_id_remote < node_nums; node_id_remote++) {
            if (node_id_remote == node_id)  continue;
            for (int gpu_id_p = 1; gpu_id_p < npes_node; gpu_id_p++) {
                int remote_pe = node_id_remote * npes_node + (mype_node+gpu_id_p) % npes_node;
                for (int tile_id = 0; tile_id < task_assignments[remote_pe].size(); tile_id++) {
                    int original_tile_id = task_assignments[remote_pe][tile_id];
                    if (is_sparse_comm_inter[original_tile_id]) {
                        comm_traffic += sizeof(Val_t) * dims * n_nozero_cols[original_tile_id];
                    }
                    else {
                        comm_traffic += sizeof(Val_t) * dims * n_slices[original_tile_id];
                    }
                }
            }
        }
        return comm_traffic;
    }

    long get_comm_traffic() {
        return get_comm_traffic_intra() + get_comm_traffic_inter();
    }

};

#endif // comm_man_H