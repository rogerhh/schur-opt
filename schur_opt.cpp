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
        exit(1);
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
    } else if (which_block == WhichBlock::isD) {
        int num_row_blocks = num_rows / block_size;
        int num_col_blocks = num_cols / block_size;

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
    }  else if (which_block == WhichBlock::isDschur_ref){
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
 * Wrapper data structure for reducing Dschur matrix
 * contains pointer to stored data as well as bool flag
 */
struct DPair {
    vector<vector<double>> Dschur_ptr;
    vector<bool> Dschur_used_ptr;

    DPair(int num_blocks, int block_squared) {
        Dschur_ptr = vector<vector<double>>(num_blocks, vector<double>(block_squared, 0));
        Dschur_used_ptr = vector<bool>(num_blocks, false);

    }

    DPair(vector<vector<double>>& Dschur_in, vector<bool>& Dschur_used_in) {
        Dschur_ptr = Dschur_in;
        Dschur_used_ptr = Dschur_used_in;
    }
};

/**
 * Reduction function on DSchur (PxP) matrixes
 */
void matrix_add(DPair& omp_out, 
                DPair& omp_in) {
    vector<vector<double>>& Dschur = (omp_out.Dschur_ptr);
    vector<bool>& Dschur_used = (omp_out.Dschur_used_ptr);
    vector<vector<double>>& Dschur_local = (omp_in.Dschur_ptr);
    vector<bool>& Dschur_used_local = (omp_in.Dschur_used_ptr);

    // cout << "size = " << Dschur.size() << " " << Dschur_local.size() << endl;
    assert(Dschur.size() == Dschur_local.size());
    assert(Dschur_used.size() == Dschur_used_local.size());
    assert(Dschur.size() == Dschur_used.size());

    // reduction on self
    if(omp_in.Dschur_ptr == omp_out.Dschur_ptr) {
        return;
    }

    // cout << "thread id = " << omp_get_thread_num() << " before loop " << Dschur[0][0] << " " << Dschur_local[0][0] << " " << endl;

    for(int i = 0; i < Dschur_local.size(); i++) {
        if(!Dschur_used_local[i]) {
            continue;
        }

        Dschur_used[i] = true;
        assert(Dschur[i].size() == Dschur_local[i].size());
        for(int j = 0; j < Dschur_local[i].size(); j++) {
            Dschur[i][j] += Dschur_local[i][j];
        }
    }
    // cout << "thread id = " << omp_get_thread_num() << " before loop " << omp_out.Dschur_ptr[0][0] << " " << Dschur_local[0][0] << " " << endl;
    return;
}

DPair dpair_init() {
    cout << "init" << endl;
    return DPair(4, 9);
}

// Declaration of OpenMP reduction on DSchur
#pragma omp declare reduction(matrix_add : DPair : matrix_add(omp_out, omp_in)) initializer(omp_priv=omp_orig)

/**
 * Compute the Schur component S = D-Cinv(A)B
 * dpair contains the final resultant Dschur
 * By exploiting the block sparisty pattern of A, we can calculate Ck*inv(Ak)*Bk in parallel
 * each of Ck*inv(Ak)*Bk results in PxP summand which we will reduce in the end
 */
void SchurOpt::compute_schur(double* runtime) {
    assert(P % block_size == 0);
    assert(L % block_size == 0);
    int num_P_blocks = P / 3;
    int num_L_blocks = L / 3;

    // We are going to assume all the matrices are set in a nice way
    std::chrono::steady_clock::time_point t_schur_start = std::chrono::steady_clock::now();
    
    omp_set_num_threads(omp_num_threads);

    Dschur = D;
    Dschur_used = D_used;
    DPair dpair(Dschur, Dschur_used);
    DPair dpair_init(DPair(num_P_blocks * num_P_blocks, block_squared));
    vector<DPair> dpair_arr;
    for(int i = 0; i < omp_num_threads; i++) {
        // dpair_arr.push_back(DPair(Dschur_local_arr[i], Dschur_used_local_arr[i]));
        dpair_arr.push_back(DPair(num_P_blocks * num_P_blocks, block_squared));
    }

    #pragma omp parallel default(shared) 
    {
        int omp_id = omp_get_thread_num();
        
        // local summand of Ck*inv(Ak)*Bk wich is PxP, stored in blocks (bxb)
        vector<vector<double>>& Dschur_local = (dpair_arr[omp_id].Dschur_ptr);
        assert(Dschur_local.size() == D.size());
        vector<bool>& Dschur_used_local = (dpair_arr[omp_id].Dschur_used_ptr);
        vector<double> bschur_local;

        // parallelize across number of block diagonal matrix in A
        // calculate Ck*inv(Ak)*Bk in parallel
        #pragma omp for
        for(int i = 0; i < A_sparse.size(); i++) {
            
            // Ck*inv(Ak)*Bk can be calculated as an outer product between
            // Ck (Pxb) * inv(Ak) (bxb) * Bk (bxP)

            vector<double>& Aii = A_sparse[i];
            Eigen::Map<Matrix<double, 3, 3, RowMajor>> Aii_m(Aii.data());
            Matrix<double, 3, 3, RowMajor> Ainv_m = Aii_m.inverse(); // get inv(Ak)

            // Iterate over blocks of C panel
            int num_P_blocks = P / block_size;
            int num_L_blocks = L / block_size;

            // calculate outer product
            for(int j = 0; j < num_P_blocks; j++) {
                // We want block Cji, which is block Bij
                int Bij_idx = pair_to_idx(i, j, num_L_blocks, num_P_blocks);

                if(!B_used[Bij_idx]) {
                    continue;
                }

                vector<double>& Bij = B[Bij_idx];

                Eigen::Map<Matrix<double, 3, 3, RowMajor>> Bij_m(Bij.data());

                Matrix<double, 3, 3, RowMajor> CAinv_m = Bij_m.transpose() * Ainv_m;

                for(int k = 0; k < num_P_blocks; k++) {
                    // We want block Bik
                    int Bik_idx = pair_to_idx(i, k, num_L_blocks, num_P_blocks);
                    if(!B_used[Bik_idx]) {
                        continue;
                    }

                    vector<double>& Bik = B[Bik_idx];
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>> Bik_m(Bik.data());

                    int Djk_idx = pair_to_idx(j, k, num_P_blocks, num_P_blocks);
                    vector<double>& Djk = Dschur_local[Djk_idx];
                    Dschur_used_local[Djk_idx] = true;
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>> Djk_m(Djk.data());

                    // multiply by -1 here, later we just need to do sum reduction
                    Djk_m -= CAinv_m * Bik_m;
                }
            }
        }
    #pragma omp barrier
    }

    // cout << omp_num_threads << endl;

    #pragma omp parallel for reduction(matrix_add : dpair_init)
    for(int i = 0; i < omp_num_threads; i++) {
        matrix_add(dpair_init, dpair_arr[i]);
    }
    matrix_add(dpair, dpair_init);
    Dschur = dpair.Dschur_ptr;
    // cout << "end " << Dschur[0][0] << endl;

    chrono::steady_clock::time_point t_schur_end = chrono::steady_clock::now();
    double t_schur = chrono::duration_cast<chrono::duration<double, milli>>(t_schur_end - t_schur_start).count();
    // cout << "num_cores= " << omp_num_threads << " t_schur = " << t_schur << endl;
    // cout << "Dschur[0, 0] = " << Dschur[0][0] << endl;
    // cout << "dschur addr at end = " << &Dschur << endl;
    // cout << "num_threads= " << omp_num_threads << " t_schur= " << t_schur << " P= " << P << " L= " << L << endl;
    *runtime = t_schur;
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
