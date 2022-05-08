#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "schur_opt_gpu.h"

using namespace std;

void SchurOpt::from_g2o(/* parameters */) {

}

void SchurOpt::to_g2o(/* parameters */) {

}

SchurOpt::~SchurOpt() {
    if(A_gpu) {
        cudaFree(A_gpu);
        A_gpu = nullptr;
    }
    if(Ainv_gpu) {
        cudaFree(Ainv_gpu);
        Ainv_gpu = nullptr;
    }
    if(A_gpu_batch) {
        cudaFree(A_gpu_batch);
        A_gpu_batch = nullptr;
    }
    if(Ainv_gpu_batch) {
        cudaFree(Ainv_gpu_batch);
        Ainv_gpu_batch = nullptr;
    }
    if(B_gpu) {
        cudaFree(B_gpu);
        B_gpu = nullptr;
    }
    if(B_gpu_batch) {
        cudaFree(B_gpu_batch);
        B_gpu_batch = nullptr;
    }
    if(C_gpu) {
        cudaFree(C_gpu);
        C_gpu = nullptr;
    }
    if(C_gpu_batch) {
        cudaFree(C_gpu_batch);
        C_gpu_batch = nullptr;
    }
    if(CAinv_gpu) {
        cudaFree(CAinv_gpu);
        CAinv_gpu = nullptr;
    }
    if(CAinv_gpu_batch) {
        cudaFree(CAinv_gpu_batch);
        CAinv_gpu_batch = nullptr;
    }
}

// For CUBLAS, this needs to be column major
inline int pair_to_idx_col(const int r, const int c, const int max_r, const int max_c) {
    return r + c * max_r;
}

inline int pair_to_idx_row(const int r, const int c, const int max_r, const int max_c) {
    return c + r * max_c;
}

void print_error_and_exit(const string& msg) {
    cout << msg << endl;
    exit(1);
}


/*void SchurOpt::read_sparse(const string& fname, WhichBlock which_block) {
    ifstream fin(fname);

    if(!fin.is_open()) {
        cerr << "Error opening: " << fname << endl;
        exit(1);
    }

    string line;
    getline(fin, line); // first 3 are unimportant
    getline(fin, line);
    getline(fin, line);

    int num_rows, num_cols;
    string garbage;

    fin >> garbage >> garbage >> num_rows >> garbage >> garbage >> num_cols;

    // cout << "num_rows = " << num_rows << " num_cols = " << num_cols << endl;

    assert(num_rows % 3 == 0);
    assert(num_cols % 3 == 0);

    int row, col;
    double val;

    if(which_block == WhichBlock::isA) {
        int num_blocks = num_rows / block_size;

        A_sparse = vector<double>(num_blocks * block_squared, 0);
        L = num_blocks * block_size;

        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            assert(abs(row - col) < 3);

            int block_id = row / 3;
            assert(block_id < num_blocks);

            int i_offset = row % 3, j_offset = col % 3;
            A_sparse[block_id * block_squared + pair_to_idx(i_offset, j_offset, block_size, block_size)] = val;
        }
    } else if (which_block == WhichBlock::isB || which_block == WhichBlock::isC){
        if(which_block == WhichBlock::isC) {
            // swap max rows and cols because we're reading the transpose
            int temp = num_rows;
            num_rows = num_cols;
            num_cols = temp;
        }
        P = num_cols;
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;

        B = vector<double>(num_row_blocks * num_col_blocks * block_squared, 0);
        B_used = vector<bool>(num_row_blocks * num_col_blocks, false);
        
        while(fin >> row >> col >> val) {
            if(which_block == WhichBlock::isC) {
                // Swap col and row because we're reading the transpose
                int temp = row;
                row = col;
                col = temp;
            }

            row--;  // index by 1
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int row_block = row / block_size;
            int col_block = col / block_size;

            int block_idx = pair_to_idx(row_block, col_block, num_row_blocks, num_col_blocks);
            int i_offset = row % block_size;
            int j_offset = col % block_size;
            int idx = pair_to_idx(i_offset, j_offset, block_size, block_size);
            B[block_idx * block_squared + idx] = val;
            B_used[block_idx] = true;

            // cout << row_block << " " << col_block << " " << block_idx << " " << i_offset << " " << j_offset << " " << val << endl;
        }
    } else if (which_block == WhichBlock::isD){
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;

        cout << "read D" << num_row_blocks << " " << num_col_blocks << endl;

        D = vector<vector<double>>(num_row_blocks * num_col_blocks, vector<double>(block_squared, 0));
        D_used = vector<bool>(num_row_blocks * num_col_blocks, false);
        
        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int row_block = row / block_size;
            int col_block = col / block_size;

            int block_idx = pair_to_idx(row_block, col_block, num_row_blocks, num_col_blocks);
            int i_offset = row % block_size;
            int j_offset = col % block_size;
            int idx = pair_to_idx(i_offset, j_offset, block_size, block_size);

            D[block_idx][idx] = val;
            D_used[block_idx] = true;
        }
    }
}*/

void SchurOpt::read_sparse(const string& fname, WhichBlock which_block) {
    ifstream fin(fname);

    if(!fin.is_open()) {
        cerr << "Error opening: " << fname << endl;
        exit(1);
    }

    string line;
    getline(fin, line); // first 3 are unimportant
    getline(fin, line);
    getline(fin, line);

    int num_rows, num_cols;
    string garbage;

    fin >> garbage >> garbage >> num_rows >> garbage >> garbage >> num_cols;

    // cout << "num_rows = " << num_rows << " num_cols = " << num_cols << endl;

    assert(num_rows % 3 == 0);
    assert(num_cols % 3 == 0);

    int row, col;
    double val;

    if(which_block == WhichBlock::isA) {
        int num_blocks = num_rows / block_size;

        A_sparse = vector<double>(num_blocks * block_squared, 0);
        L = num_blocks * block_size;

        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            assert(abs(row - col) < 3);

            int block_id = row / 3;
            assert(block_id < num_blocks);

            int i_offset = row % 3, j_offset = col % 3;
            A_sparse[block_id * block_squared + pair_to_idx_col(i_offset, j_offset, block_size, block_size)] = val;
        }
    } else if (which_block == WhichBlock::isB || which_block == WhichBlock::isC){
        if(which_block == WhichBlock::isB) {
            // swap max rows and cols because we're reading the transpose
            int temp = num_rows;
            num_rows = num_cols;
            num_cols = temp;
        }
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;

        C = vector<double>(num_row_blocks * num_col_blocks * block_squared, 0);
        
        while(fin >> row >> col >> val) {
            if(which_block == WhichBlock::isB) {
                // Swap col and row because we're reading the transpose
                int temp = row;
                row = col;
                col = temp;
            }

            row--;  // index by 1
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int idx = pair_to_idx_col(row, col, num_rows, num_cols);
            C[idx] = val;

            // cout << num_rows << " " << num_cols << " " << row << " " << col << " " << idx << " " << val << endl;
        }
    } else if (which_block == WhichBlock::isD){
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;

        P = num_rows;
        // cout << "read D" << num_row_blocks << " " << num_col_blocks << endl;

        D = vector<vector<double>>(num_row_blocks * num_col_blocks, vector<double>(block_squared, 0));
        D_used = vector<bool>(num_row_blocks * num_col_blocks, false);
        
        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int row_block = row / block_size;
            int col_block = col / block_size;

            int block_idx = pair_to_idx_col(row_block, col_block, num_row_blocks, num_col_blocks);
            int i_offset = row % block_size;
            int j_offset = col % block_size;
            int idx = pair_to_idx_col(i_offset, j_offset, block_size, block_size);

            D[block_idx][idx] = val;
            D_used[block_idx] = true;
        }
    }
}

void SchurOpt::compute_Ainv() {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS initialization failed");
    }

    cudaStat = cudaMalloc((void**) &A_gpu, A_sparse.size() * sizeof(double));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure");
    }
    stat = cublasSetVector(A_sparse.size(), sizeof(double), A_sparse.data(), 1, A_gpu, 1);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS vector copy failed");
    }

    cudaStat = cudaMalloc((void**) &Ainv_gpu, A_sparse.size() * sizeof(double));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure: Aing_gpu");
    }

    assert(A_sparse.size() % block_squared == 0);
    batch_size = A_sparse.size() / block_squared;

    cudaStat = cudaMalloc((void**) &A_gpu_batch, batch_size * sizeof(double*));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure for A_gpu_batch");
    }

    cudaStat = cudaMalloc((void**) &Ainv_gpu_batch, batch_size * sizeof(double*));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure for Ainv_gpu_batch");
    }
    
    vector<double*> A_batch(batch_size, nullptr);
    vector<double*> Ainv_batch(batch_size, nullptr);
    for(int i = 0; i < batch_size; i++) {
        A_batch[i] = A_gpu + i * block_squared;
        Ainv_batch[i] = Ainv_gpu + i * block_squared;
    }

    stat = cublasSetVector(batch_size, sizeof(double*), A_batch.data(), 1, A_gpu_batch, 1);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS vector copy failed: A_gpu_batch");
    }

    stat = cublasSetVector(batch_size, sizeof(double*), Ainv_batch.data(), 1, Ainv_gpu_batch, 1);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS vector copy failed: Ainv_gpu_batch");
    }

    int* info_gpu;
    cudaStat = cudaMalloc((void**) &info_gpu, batch_size * sizeof(int));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure: info_gpu");
    }


    stat = cublasDmatinvBatched(handle, 3, A_gpu_batch, 3, Ainv_gpu_batch, 3, info_gpu, batch_size);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("Error: Inverting A");
    }

    Ainv = vector<double>(A_sparse.size(), 0);
    stat = cublasGetVector(A_sparse.size(), sizeof(double), Ainv_gpu, 1, Ainv.data(), 1);   
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("Error: Getting vector");
    }

    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 3; k++) {
                cout << A_sparse[i * 9 + j * 3 + k] << " "; 
            }
            cout << endl;
        }
        cout << endl;
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 3; k++) {
                cout << Ainv[i * 9 + j * 3 + k] << " "; 
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    cudaFree(info_gpu);
    info_gpu = nullptr;
    cublasDestroy(handle);
}

void SchurOpt::compute_schur(/* parameters */) {
    compute_Ainv();

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS initialization failed");
    }

    cudaStat = cudaMalloc((void**) &C_gpu, C.size() * sizeof(double));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure: C_gpu");
    }
    stat = cublasSetVector(C.size(), sizeof(double), C.data(), 1, C_gpu, 1);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS vector copy failed: C_gpu");
    }

    cudaStat = cudaMalloc((void**) &CAinv_gpu, C.size() * sizeof(double));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure: CAinv_gpu");
    }

    cudaStat = cudaMalloc((void**) &C_gpu_batch, batch_size * sizeof(double*));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure for C_gpu_batch");
    }

    cudaStat = cudaMalloc((void**) &CAinv_gpu_batch, batch_size * sizeof(double*));
    if(cudaStat != cudaSuccess) {
        print_error_and_exit("Error: Device memory allocation failure for CAinv_gpu_batch");
    }

    int panel_size = block_size * P;

    vector<double*> C_batch(batch_size, nullptr);
    vector<double*> CAinv_batch(batch_size, nullptr);
    for(int i = 0; i < batch_size; i++) {
        C_batch[i] = C_gpu + i * panel_size;
        CAinv_batch[i] = CAinv_gpu + i * panel_size;
    }

    stat = cublasSetVector(batch_size, sizeof(double*), C_batch.data(), 1, C_gpu_batch, 1);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS vector copy failed: C_gpu_batch");
    }

    stat = cublasSetVector(batch_size, sizeof(double*), CAinv_batch.data(), 1, CAinv_gpu_batch, 1);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS vector copy failed: Ainv_gpu_batch");
    }

    double alpha = 1, beta = 0;

    stat = cublasDgemmBatched(handle, 
                              transa, 
                              transb, 
                              P, block_size, block_size, 
                              &alpha, 
                              C_gpu_batch, P, 
                              Ainv_gpu_batch, block_size, 
                              &beta, 
                              CAinv_gpu_batch, P, 
                              batch_size);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("CUBLAS batched dgemm failed");
    }

    vector<double> CAinv(C.size(), 0);
    stat = cublasGetVector(C.size(), sizeof(double), CAinv_gpu, 1, CAinv.data(), 1);   
    if(stat != CUBLAS_STATUS_SUCCESS) {
        print_error_and_exit("Error getting data: CAinv");
    }

    cout << "CAinv: " << endl;
    for(int i = 0; i < 100; i++) {
        cout << CAinv[i] << endl;
    }

    cublasDestroy(handle);
}