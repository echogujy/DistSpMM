#ifndef __SPARSE_MATRIX_H__
#define __SPARSE_MATRIX_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <utility>
// #include <ranges>
#include <algorithm>
#include <tuple>
#include <numeric>
#include <random>

// CSR (Compressed Sparse Row) format for sparse matrix
template<typename Idx_t, typename Val_t>
class SparseMatrix {
public:

    std::vector<Idx_t> g_row_ptr; // Points to the start of each row in col_idx and values
    std::vector<Idx_t> g_col_idx; // Column indices for each non-zero element
    std::vector<Val_t> g_values;  // Non-zero values

    std::vector<Idx_t> row_ptr; // Points to the start of each row in col_idx and values
    std::vector<Idx_t> col_idx; // Column indices for each non-zero element
    std::vector<Val_t> values;  // Non-zero values

    Idx_t num_rows; // Number of rows in the matrix
    Idx_t num_cols; // Number of columns in the matrix (optional, depending on use case)
    size_t nnz;     // Number of non-zero elements

    Idx_t local_m, local_n;
    size_t local_nnz;


public:
    SparseMatrix() : num_rows(0), num_cols(0), nnz(0) {}

    ~SparseMatrix() = default;

    // Load a sparse matrix from MGG files.
    bool load_from_csr(const std::string& file_path);
    bool load_from_mtx(const std::string& mtx_file);
    void print_info(size_t print_limit = 5) const;

    void turn_local_l_u_csr(Idx_t m_lb, Idx_t m_ub);
    void turn_local_range_constrained_binary_search(Idx_t &m_lb, Idx_t &m_ub, int mype, int npes);

    void release_global_data(){
        // use swap to really clear the memory
        std::vector<Idx_t>().swap(g_row_ptr);
        std::vector<Idx_t>().swap(g_col_idx);
        std::vector<Val_t>().swap(g_values);
        // g_row_ptr.clear();
        // g_col_idx.clear();
        // g_values.clear();
    };
    std::tuple<std::vector<Idx_t>, std::vector<Idx_t>,std::vector<Val_t>>get_csr() {
        return {row_ptr, col_idx, values};
    }
    
};

// Load a sparse matrix from a .mtx file
template<typename Idx_t, typename Val_t>
bool SparseMatrix<Idx_t, Val_t>::load_from_mtx(const std::string& mtx_file) {
    // TODO: Implement this function
    // raise a eror when file not found
    std::cout << "SparseMatrix::load_from_mtx not implemented" << std::endl;
    return false;
    // std::ifstream file(mtx_file);
    // if (!file.is_open()) {
    //     std::cerr << "Failed to open .mtx file: " << mtx_file << std::endl;
    //     return false;
    // }

    // return 1;
}

template<typename Idx_t, typename Val_t>
bool SparseMatrix<Idx_t, Val_t>::load_from_csr(const std::string& file_path) {
    using FileIdx_t = int64_t;  
    using FileVal_t = float;   
    std::ifstream file;

    std::string indptr_files = file_path + "_indptr.bin";
    // Read the row pointer file
    file.open(indptr_files, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open row pointer file: " << indptr_files << std::endl;
        return false;
    }

    // Calculate number of rows based on file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    num_rows = file_size / sizeof(FileIdx_t) - 1;
    std::vector<FileIdx_t> temp_row_ptr(num_rows + 1);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(temp_row_ptr.data()), file_size);
    file.close();

    // Convert and copy row pointers
    g_row_ptr.resize(num_rows + 1);

    for (size_t i = 0; i <= num_rows; ++i) {
        g_row_ptr[i] = static_cast<Idx_t>(temp_row_ptr[i]);
    }

    // Calculate the number of non-zeros from the last element of row_ptr
    nnz = static_cast<size_t>(g_row_ptr[num_rows]);

    // Read the column index file
    std::string indices_file = file_path + "_indices.bin";
    file.open(indices_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open column index file: " << indices_file << std::endl;
        return false;
    }
    std::vector<FileIdx_t> temp_col_idx(nnz);
    file.read(reinterpret_cast<char*>(temp_col_idx.data()), nnz * sizeof(FileIdx_t));
    file.close();

    // Convert and copy column indices
    g_col_idx.resize(nnz);


    for (size_t i = 0; i < nnz; ++i) {
        g_col_idx[i] = static_cast<Idx_t>(temp_col_idx[i]);
    }


    // Read the value file
    std::string values_file = file_path + "_values.bin";
    file.open(values_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open value file: " << values_file << std::endl;
        return false;
    }
    std::vector<FileVal_t> temp_values(nnz);
    file.read(reinterpret_cast<char*>(temp_values.data()), nnz * sizeof(FileVal_t));
    file.close();

    // Convert and copy values
    g_values.resize(nnz);

    for (size_t i = 0; i < nnz; ++i) {
        g_values[i] = static_cast<Val_t>(temp_values[i]);
    }

    // get the number of columns
    // auto iter_max =std::max_element(col_idx.begin(), col_idx.end());
    // num_cols = *iter_max + 1;
    num_cols = num_rows;
    return true;
}

template<typename Idx_t, typename Val_t>
void SparseMatrix<Idx_t, Val_t>::print_info(size_t print_limit) const {
    std::cout << "Matrix loaded successfully:" << std::endl;
    std::cout << "Number of rows: " << num_rows << std::endl;
    std::cout << "Number of columns: " << num_cols << std::endl;
    std::cout << "Number of non-zero elements: " << nnz << std::endl;

    // Print first few rows for verification
    for (size_t i = 0; i < num_rows && i < print_limit; ++i) {
        std::cout << "Row " << i << ": ";
        for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            std::cout << "(" << col_idx[j] << ", " << values[j] << ") ";
        }
        std::cout << std::endl;
    }
}

template<typename Idx_t, typename Val_t>
void SparseMatrix<Idx_t, Val_t>::turn_local_l_u_csr(Idx_t m_lb, Idx_t m_ub){
    if (m_lb < 0 || m_ub > num_rows)
        throw std::out_of_range("Invalid partition range.");

    // std::cout << "matrix info: " << num_rows << " " << num_cols << " " << nnz << std::endl;
    auto start = g_row_ptr[m_lb];
    auto end = g_row_ptr[m_ub];
    // std::cout << " stat (m_lb) " << m_lb << " " << m_ub << " " << start << " " << end << std::endl;
    row_ptr.resize(m_ub - m_lb + 1);
    col_idx.resize(end - start);
    values.resize(end - start);
    std::move(g_row_ptr.begin() + m_lb, g_row_ptr.begin() + m_ub + 1, row_ptr.begin());
    std::move(g_col_idx.begin() + start, g_col_idx.begin() + end, col_idx.begin());
    std::move(g_values.begin() + start, g_values.begin() + end, values.begin());

    for (auto& x : row_ptr) x -= start;

    local_m = m_ub - m_lb;
    local_n = num_cols;
    local_nnz = end - start;
}

#endif // __SPARSE_MATRIX_H__