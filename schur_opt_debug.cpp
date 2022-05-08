#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <eigen3/Eigen/Eigen>

#include "schur_opt_gpu.h"

using namespace std;

void SchurOpt::from_g2o(/* parameters */) {

}

void SchurOpt::to_g2o(/* parameters */) {

}

inline int pair_to_idx_row(const int r, const int c, const int max_r, const int max_c) {
    return c + r * max_c;
}

// For CUBLAS, this needs to be column major
inline int pair_to_idx_col(const int r, const int c, const int max_r, const int max_c) {
    return r + c * max_r;
}

void print_error_and_exit(const string& msg) {
    cout << msg << endl;
    exit(1);
}

SchurOpt::~SchurOpt() {}


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
    assert(A_sparse.size() % block_squared == 0);
    batch_size = A_sparse.size() / block_squared;
    Ainv = A_sparse;

    // cout << batch_size << endl;
    // cout << "Ainv: " << endl; 
    for(int i = 0; i < batch_size; i++) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> Aii_m(&A_sparse[i * block_squared]);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> Ainv_ii_m(&Ainv[i * block_squared]);
        Ainv_ii_m = Aii_m.inverse();

        // cout << Ainv_ii_m << endl << endl;
    }
}

void SchurOpt::compute_schur(/* parameters */) {
    compute_Ainv();
    vector<double> CAinv(C.size(), 0);
    cout << "CAinv: " << endl;
    for(int i = 0; i < batch_size; i++) {
        Eigen::Map<Eigen::MatrixXd> Ci_m(&C[i * block_size * P], P, block_size);
        Eigen::Map<Eigen::MatrixXd> Ainvi_m(&Ainv[i * block_squared], block_size, block_size);
        Eigen::Map<Eigen::MatrixXd> CAinvi_m(&CAinv[i * block_size * P], P, block_size);
        CAinvi_m = Ci_m * Ainvi_m;

        cout << CAinvi_m << endl;
    }
}
