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

inline int pair_to_idx(const int r, const int c, const int max_r, const int max_c) {
    return r * max_c + c;
}


void SchurOpt::read_sparse(const string& fname, SchurOpt& schur_opt, WhichBlock which_block) {
    ifstream fin(fname);

    if(!fin.is_open()) {
        cerr << "Error opening: " << fname << endl;
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

        A_sparse = vector<vector<double>>(num_blocks, vector<double>(block_squared, 0));
        L = num_blocks * block_size;

        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            assert(abs(row - col) < 3);

            int block_id = row / 3;
            assert(block_id < num_blocks);

            int i_offset = row % 3, j_offset = col % 3;
            A_sparse[block_id][pair_to_idx(i_offset, j_offset, block_size, block_size)] = val;
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

        B = vector<vector<double>>(num_row_blocks * num_col_blocks, vector<double>(block_squared, 0));
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
            B[block_idx][idx] = val;
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
}

void SchurOpt::compute_schur(/* parameters */) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    double* A_gpu;
    cudaStat = cudaMalloc((void**) &A_gpu, block_squared * A_sparse.size());
    cout << cudaStat << endl;
}
