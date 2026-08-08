#ifndef H_SDMA_H
#define H_SDMA_H

#include <vector>
#include <map>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <algorithm> // For std::sort

// Represents the hardware topology for a single GPU
struct GpuInfo {
    int id;       // Global GPU ID (e.g., 0 to P-1)
    int nodeId;   // ID of the node it belongs to (e.g., 0 to x-1)
};

// --- Helper Functions based on your Python implementation ---

/**
 * @brief Calculates the reduction in communication cost for a potential swap.
 * @param comm_demand_matrix The communication demand matrix for the current context.
 * @param task_a Index of the first task to swap.
 * @param task_b Index of the second task to swap.
 * @param task_to_map Current mapping of tasks to processing units (nodes or GPUs).
 * @return The calculated reduction in cost (a positive value means improvement).
 */
double swap_refinement(
    const std::vector<std::vector<double>>& comm_demand_matrix,
    int task_a,
    int task_b,
    const std::vector<int>& task_to_map)
{
    int map_a = task_to_map[task_a];
    int map_b = task_to_map[task_b];
    double delta_reduce_cost = (comm_demand_matrix[map_a][task_b] + comm_demand_matrix[map_b][task_a]) -
                               (comm_demand_matrix[map_a][task_a] + comm_demand_matrix[map_b][task_b]);
    return delta_reduce_cost;
}

/**
 * @brief Calculates the total communication traffic for the current task mapping.
 * @param comm_demand_matrix The communication demand matrix for the current context.
 * @param task_to_map Current mapping of tasks to processing units.
 * @return The total communication traffic.
 */
double get_total_cost(
    const std::vector<std::vector<double>>& comm_demand_matrix,
    const std::vector<int>& task_to_map)
{
    double comm_traffic = 0;
    int num_units = comm_demand_matrix.size();
    int num_tasks = task_to_map.size();

    for (int i = 0; i < num_units; ++i) {
        double unit_traffic = 0;
        for (int j = 0; j < num_tasks; ++j) {
            if (task_to_map[j] != i) {
                unit_traffic += comm_demand_matrix[i][j];
            }
        }
        comm_traffic += unit_traffic;
    }
    return comm_traffic;
}


/**
 * @brief Finds a high-quality assignment using a hierarchical, two-stage iterative refinement heuristic.
 * C++ implementation of your Python logic.
 * @param commDemandMatrix The global communication demand matrix.
 * @param gpu_topology The hardware topology map.
 * @param tasks_per_gpu The number of tasks each GPU must have.
 * @param max_inter_node_iterations The number of refinement passes for node-level assignment.
 * @param max_intra_node_iterations The number of refinement passes for GPU-level assignment.
 * @param detail If true, prints detailed progress information.
 * @param remap Output vector to be filled with the assignment.
 */
void hierarchical_iterative_refinement_allocation(
    const std::vector<std::vector<int>>& commDemandMatrix,
    const std::vector<GpuInfo>& gpu_topology,
    int tasks_per_gpu,
    std::vector<int>& remap,
    int max_inter_node_iterations = 5,
    int max_intra_node_iterations = 5,
    bool detail = false)
{
    int num_gpus = commDemandMatrix.size();
    if (num_gpus == 0) return;
    int num_tasks = commDemandMatrix[0].size();
    if (num_tasks == 0) return;

    int num_nodes = 0;
    for(const auto& gpu : gpu_topology) {
        if (gpu.nodeId + 1 > num_nodes) num_nodes = gpu.nodeId + 1;
    }

    int tasks_per_node = num_tasks / num_nodes;
    int gpu_per_node = num_gpus / num_nodes;
    
    std::vector<int> gpu_to_node(num_gpus);
    for (const auto& gpu : gpu_topology) {
        gpu_to_node[gpu.id] = gpu.nodeId;
    }

    std::vector<std::vector<int>> tasks_for_node(num_nodes);

    if (num_nodes <= 1) {
        if (detail) std::cout << "\n--- [Heuristic] Single-Node Case Detected: Skipping Phase 1 ---" << std::endl;
        tasks_for_node[0].resize(num_tasks);
        std::iota(tasks_for_node[0].begin(), tasks_for_node[0].end(), 0);
    } else {
        // Phase 1: Node-level refinement
        std::vector<std::vector<double>> node_comm_matrix(num_nodes, std::vector<double>(num_tasks, 0.0));
        for (int i = 0; i < num_gpus; ++i) {
            int node_id = gpu_to_node[i];
            for (int j = 0; j < num_tasks; ++j) {
                node_comm_matrix[node_id][j] += commDemandMatrix[i][j];
            }
        }

        std::vector<int> node_remap(num_tasks);
        for(int i=0; i<num_tasks; ++i) node_remap[i] = i / tasks_per_node;

        if (detail) {
            std::cout << "\n--- [Heuristic] Phase 1: Refining Node-level assignment... ---" << std::endl;
            std::cout << "Initial Inter-Node Cost: " << get_total_cost(node_comm_matrix, node_remap) << std::endl;
        }

        for (int inter_it = 0; inter_it < max_inter_node_iterations; ++inter_it) {
            double best_reduce_cost = 0;
            int best_task_a = -1, best_task_b = -1;

            for (int task_a = 0; task_a < num_tasks; ++task_a) {
                for (int task_b = task_a + 1; task_b < num_tasks; ++task_b) {
                    if (node_remap[task_a] == node_remap[task_b]) continue;
                    
                    double delta_reduce_cost = swap_refinement(node_comm_matrix, task_a, task_b, node_remap);
                    if (delta_reduce_cost > best_reduce_cost) {
                        best_reduce_cost = delta_reduce_cost;
                        best_task_a = task_a;
                        best_task_b = task_b;
                    }
                }
            }

            if (best_reduce_cost > 0) {
                std::swap(node_remap[best_task_a], node_remap[best_task_b]);
                if (detail) {
                    std::cout << "Phase 1, Iteration " << inter_it + 1 << ": Swapped tasks " << best_task_a 
                              << " and " << best_task_b << ", cost reduction: " << best_reduce_cost << std::endl;
                    std::cout << "New cost: " << get_total_cost(node_comm_matrix, node_remap) << std::endl;
                }
            } else {
                if (detail) std::cout << "No further inter-node improvements found. Stopping early." << std::endl;
                break;
            }
        }

        for (int task_id = 0; task_id < num_tasks; ++task_id) {
            tasks_for_node[node_remap[task_id]].push_back(task_id);
        }
    }

    // Phase 2: GPU-level refinement
    if (detail) std::cout << "\n--- [Heuristic] Phase 2: Refining GPU-level assignment... ---" << std::endl;
    remap.assign(num_tasks, -1);

    for (int node_id = 0; node_id < num_nodes; ++node_id) {
        if (tasks_for_node[node_id].empty()) continue;

        int num_tasks_on_node = tasks_for_node[node_id].size();
        std::vector<int> sorted_task_list = tasks_for_node[node_id];
        std::sort(sorted_task_list.begin(), sorted_task_list.end());

        std::vector<std::vector<double>> gpu_comm_matrix(gpu_per_node, std::vector<double>(num_tasks_on_node, 0.0));
        
        for (int i = 0; i < gpu_per_node; ++i) {
            int gpu_id_global = i + node_id * gpu_per_node;
            for (int j = 0; j < num_tasks_on_node; ++j) {
                gpu_comm_matrix[i][j] = commDemandMatrix[gpu_id_global][sorted_task_list[j]];
            }
        }

        std::vector<int> gpu_remap(num_tasks_on_node);
        for(int i=0; i<num_tasks_on_node; ++i) gpu_remap[i] = i / tasks_per_gpu;

        if (detail) {
            std::cout << "Node " << node_id << ", Initial Intra-Node Cost: " << get_total_cost(gpu_comm_matrix, gpu_remap) << std::endl;
        }

        for (int intra_it = 0; intra_it < max_intra_node_iterations; ++intra_it) {
            double best_reduce_cost = 0;
            int best_idx_a = -1, best_idx_b = -1;

            for (int idx_a = 0; idx_a < num_tasks_on_node; ++idx_a) {
                for (int idx_b = idx_a + 1; idx_b < num_tasks_on_node; ++idx_b) {
                    if (gpu_remap[idx_a] == gpu_remap[idx_b]) continue;

                    double delta_reduce_cost = swap_refinement(gpu_comm_matrix, idx_a, idx_b, gpu_remap);
                    if (delta_reduce_cost > best_reduce_cost) {
                        best_reduce_cost = delta_reduce_cost;
                        best_idx_a = idx_a;
                        best_idx_b = idx_b;
                    }
                }
            }

            if (best_reduce_cost > 0) {
                std::swap(gpu_remap[best_idx_a], gpu_remap[best_idx_b]);
                if (detail) {
                     std::cout << "Node " << node_id << ", Iteration " << intra_it + 1 << ": Swapped tasks " 
                               << sorted_task_list[best_idx_a] << " and " << sorted_task_list[best_idx_b] 
                               << ", cost reduction: " << best_reduce_cost << std::endl;
                     std::cout << "New cost: " << get_total_cost(gpu_comm_matrix, gpu_remap) << std::endl;
                }
            } else {
                if (detail) std::cout << "Node " << node_id << ": No further intra-node improvements found. Stopping early." << std::endl;
                break;
            }
        }

        for (int i = 0; i < num_tasks_on_node; ++i) {
            int task_id = sorted_task_list[i];
            int local_gpu_idx = gpu_remap[i];
            remap[task_id] = local_gpu_idx + node_id * gpu_per_node;
        }
    }
}

#endif // H_SDMA_H
