#include "schur_opt.h"
#include <omp.h>

#include <cmath>
#include <string>
#include <cassert>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <utility>
#include <chrono>

using namespace std;
using std::tie;
using Eigen::Matrix;
using Eigen::RowMajor;

void SchurOpt::from_g2o(/* parameters */) {
    // TO DO
}

void SchurOpt::to_g2o(/* parameters */) {
    // TO DO
}

// (row, col) -> index in row-major storage
inline int pair_to_idx(const int r, const int c, const int max_r, const int max_c) {
    return r * max_c + c; // row major access
}

/**
 * Read Sparse Matrixes
 * Only handle oct formats
 * 
 */
void SchurOpt::read_sparse(const string& fname, SchurOpt& schur_opt, WhichBlock which_block) {
    ifstream fin(fname);

    if(!fin.is_open()) {
        cerr << "Error opening: " << fname << endl;
    }

    string line;
    // first 3 are unimportant, as it contain name, type, nnz
    getline(fin, line); 
    getline(fin, line);
    getline(fin, line);

    int num_rows, num_cols;
    string garbage;

    fin >> garbage >> garbage >> num_rows >> garbage >> garbage >> num_cols;

    // cout << "num_rows = " << num_rows << " num_cols = " << num_cols << endl;

    assert(num_rows % block_size == 0);
    assert(num_cols % block_size == 0);

    int row, col;
    double val;

    if(which_block == WhichBlock::isA) {
        int num_blocks = num_rows / block_size; // number of sparse block on the diagonal

        A_sparse = vector<vector<double>>(num_blocks, vector<double>(block_squared, 0));
        L = num_blocks * block_size;

        // oct file data in row, col, val format
        while(fin >> row >> col >> val) {
            row--;  // decrease index by 1 as we are zero indexed
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
        // process everything in terms of B
        P = num_cols;
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;
        // cout << num_row_blocks << " " << num_col_blocks << endl;

        // Create B in blocked storage format
        B = vector<vector<double>>(num_row_blocks * num_col_blocks, vector<double>(block_squared, 0));
        B_used = vector<bool>(num_row_blocks * num_col_blocks, false);
        
        while(fin >> row >> col >> val) {
            if(which_block == WhichBlock::isC) {
                // Swap col and row because we're reading the transpose
                int temp = row;
                row = col;
                col = temp;
            }

            row--;  // decrease index by 1 as we are zero indexed
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int row_block = row / block_size;
            int col_block = col / block_size;

            // row major storage
            int block_idx = pair_to_idx(row_block, col_block, num_row_blocks, num_col_blocks);
            int i_offset = row % block_size;
            int j_offset = col % block_size;
            int idx = pair_to_idx(i_offset, j_offset, block_size, block_size);
            B[block_idx][idx] = val;
            B_used[block_idx] = true; // this bool is true as long as some entry in the block is filled

            // cout << row_block << " " << col_block << " " << block_idx << " " << i_offset << " " << j_offset << " " << val << endl;
        }
    } else if (which_block == WhichBlock::isD){
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;

        // cout << "read D" << num_row_blocks << " " << num_col_blocks << endl;

        D = vector<vector<double>>(num_row_blocks * num_col_blocks, vector<double>(block_squared, 0));
        D_used = vector<bool>(num_row_blocks * num_col_blocks, false);
        
        while(fin >> row >> col >> val) {
            row--; // decrease index by 1 as we are zero indexed
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int row_block = row / block_size;
            int col_block = col / block_size;

            // row major storage
            int block_idx = pair_to_idx(row_block, col_block, num_row_blocks, num_col_blocks);
            int i_offset = row % block_size;
            int j_offset = col % block_size;
            int idx = pair_to_idx(i_offset, j_offset, block_size, block_size);

            D[block_idx][idx] = val;
            D_used[block_idx] = true; // this bool is true as long as some entry in the block is filled
        }
    } else if (which_block == WhichBlock::isDschur_ref){
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;

        Dschur_ref = vector<vector<double>>(num_row_blocks * num_col_blocks, vector<double>(block_squared, 0));
        Dschur_ref_used = vector<bool>(num_row_blocks * num_col_blocks, false);
        
        while(fin >> row >> col >> val) {
            row--; // decrease index by 1 as we are zero indexed
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int row_block = row / block_size;
            int col_block = col / block_size;

            // row major storage
            int block_idx = pair_to_idx(row_block, col_block, num_row_blocks, num_col_blocks);
            int i_offset = row % block_size;
            int j_offset = col % block_size;
            int idx = pair_to_idx(i_offset, j_offset, block_size, block_size);

            Dschur_ref[block_idx][idx] = val;
            Dschur_ref_used[block_idx] = true; // this bool is true as long as some entry in the block is filled
        }
    }
}

/**
 * Compute the Schur component S = D-Cinv(A)B
 * First compute inv(A) B
 */
void SchurOpt::compute_schur(double* runtime) {

    // cout << "Using Simon's version " << endl;
    omp_set_num_threads(omp_num_threads);
    int num_eigen_threads = Eigen::nbThreads();

    // cout << "Number of Eigen Threads " << num_eigen_threads << endl;

    // We are going to assume all the matrices are set in a nice way
    std::chrono::steady_clock::time_point t_schur_start = std::chrono::steady_clock::now();

    assert(P % block_size == 0);
    assert(L % block_size == 0);

    int num_P_blocks = P / block_size;
    int num_L_blocks = L / block_size;
    // note this is shared across threads but there won't be datarace.
    // Create A in blocked storage format
    A_inv_B = vector<vector<double>>(num_P_blocks * num_L_blocks, vector<double>(block_squared, 0));
    A_inv_B_used = vector<bool>(num_P_blocks * num_L_blocks, false);


    // 1. Calculate inv(A)B
    #pragma omp parallel for
    for(int i = 0; i < A_sparse.size(); i++) {
        // get inv(Ak)
        vector<double>& Aii = A_sparse[i];
        Eigen::Map<Matrix<double, 3, 3, RowMajor>> Aii_m(Aii.data());
        Matrix<double, 3, 3, RowMajor> Ainv_m = Aii_m.inverse(); 

        for (int j = 0; j < num_P_blocks; j++) {
            // Get Block B_ij
            int Bij_idx = pair_to_idx(i, j, num_L_blocks, num_P_blocks);
            if(!B_used[Bij_idx]) { // skip sparse entry
                continue;
            }
            vector<double>& Bij = B[Bij_idx];
            Eigen::Map<Matrix<double, 3, 3, RowMajor>> Bij_m(Bij.data());

            int A_inv_B_ij_idx = Bij_idx;
            A_inv_B_used[A_inv_B_ij_idx] = true; 

            vector<double>& A_inv_B_ij = A_inv_B[A_inv_B_ij_idx];
            Eigen::Map<Matrix<double, 3, 3, RowMajor>> A_inv_B_ij_m(A_inv_B_ij.data());

            // Matrix<double, 3, 3, RowMajor> A_inv_B_m = Ainv_m * Bij_m;
            A_inv_B_ij_m = Ainv_m * Bij_m;
            
            // for (int i = 0; i < 9; i++) {
            //     cout << A_inv_B_ij[i] << " ";
            // }
            // cout << endl;
            
        }

    }

    // // examine structure of A_inv_B
    // int count = 0;
    // for (int i = 0; i < A_inv_B_used.size(); i++) {
    //     if (!A_inv_B_used[i]) count++;
    // }
    // cout << count << " out of " << A_inv_B_used.size() << " blocks of A_inv_B is sparse" << endl;

    // 2. Calculate C (PxL) * A_inv_B (LxP) -> C A_inv_B (PxP)
    // TO DO
    // Approach 1: convert C and A_inv_B into an Eigen matrix

    Dschur = vector<vector<double>>(num_P_blocks * num_P_blocks, vector<double>(block_squared, 0)); 
    Dschur_used = vector<bool>(num_P_blocks * num_P_blocks, false);

    // Apporach 2: do operation directly on existing data structure
    // this is naive but i just want to get something working, we can optimize laterss
    #pragma omp parallel for
    for (int i = 0; i < num_P_blocks; i++) {
        for (int j = 0; j < num_P_blocks; j++) {
            // initialize Dschur entry as corresponding data in D
            int ij_idx = pair_to_idx(i, j, num_P_blocks, num_P_blocks);
            Dschur_used[ij_idx] = D_used[ij_idx];
            if (D_used[ij_idx]) {
                Dschur[ij_idx] = D[ij_idx]; // Dschur = D - CA-1B
                // memcpy(&Dschur[ij_idx], &Dschur[ij_idx], block_squared*sizeof(double));
            }

            vector<double>& Dschur_ij = Dschur[ij_idx];
            Eigen::Map<Matrix<double, 3, 3, RowMajor>> Dschur_ij_m(Dschur_ij.data());

            // add inner product result, iterate over k, accumulate C_ik * A_inv_B_kj
            for (int k = 0; k < num_L_blocks; k++) {

                // C_ik -> B_ki as C and B are tranpose
                int ik_idx = pair_to_idx(k, i, num_L_blocks, num_P_blocks);
                int kj_idx = pair_to_idx(k, j, num_L_blocks, num_P_blocks);

                // skip if either entry is sparse 
                if (!B_used[ik_idx] || !A_inv_B_used[kj_idx]) {
                    continue;
                }
                Dschur_used[ij_idx] = true;

                // Get the C_ik block
                vector<double>& C_ik = B[ik_idx];
                Eigen::Map<Matrix<double, 3, 3, RowMajor>> C_ik_m(C_ik.data());

                // Get th A_inv_B_kj block
                vector<double>& A_inv_B_kj = A_inv_B[kj_idx];
                Eigen::Map<Matrix<double, 3, 3, RowMajor>> A_inv_B_kj_m(A_inv_B_kj.data());

                // need to transpose C_ik_m as data is stored in B originally
                Dschur_ij_m = Dschur_ij_m - C_ik_m.transpose() * A_inv_B_kj_m;
            }
        }
    }


    // 3. Calculate Dschur = D - C * A_inv_B
    // TO DO

    chrono::steady_clock::time_point t_schur_end = chrono::steady_clock::now();
    double t_schur = chrono::duration_cast<chrono::duration<double, milli>>(t_schur_end - t_schur_start).count();
    // cout << "num_threads= " << omp_num_threads << " t_schur= " << t_schur << endl;
    *runtime=t_schur;
}



/**
 * Dschur - Schur matrix the solver calculated
 * Dschur_ref - Schur matrix the G2O block solver outputted
 * Compare dimension, sparisty, and calculate the MSE. 
 */
void SchurOpt::verify_correctness(/* parameters */) {

    // If you set this, MSE will be 0
    // Dschur_ref = Dschur;
    // Dschur_ref_used = Dschur_used;

    // verify Dschur has the right size
    assert(Dschur.size() * block_squared == P * P);
    
    // actual code
    assert(Dschur.size() == Dschur_ref.size()); // comparing number of 3x3 blocks
    assert(Dschur_used.size() == Dschur_ref_used.size());

    // cout << "Dschur size=" << Dschur.size() << endl;
    double se = 0.0;  // squared error
    double diff;
    for (int block_idx = 0; block_idx < Dschur.size(); block_idx++) {
        // assert(Dschur_used[block_idx] == Dschur_ref_used[block_idx]); // agree on block sparsity
        // if (!Dschur_used[block_idx]) {
        //     continue; // skip as it is sparse
        // }
        // cout << "Block block_idx = " << block_idx << endl;
        for (int val_idx = 0; val_idx < block_squared; val_idx++) {
            // cout << Dschur[block_idx][val_idx] << " " << Dschur_ref[block_idx][val_idx] << endl;
            diff = Dschur_ref[block_idx][val_idx] - Dschur[block_idx][val_idx];
            se += diff*diff;
        }
    }
    double mse = se / (double) (P*P);
    if (mse < 1) {
        cout << "[Correctness Passed] " << "MSE: " << mse << endl;
    } else {
        cout << "[Correctness FAILED] " << "MSE: " << mse << endl;
    }

}
