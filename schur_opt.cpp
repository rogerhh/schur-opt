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

struct DPair {
    vector<vector<double>>* Dschur_ptr;
    vector<bool>* Dschur_used_ptr;

    DPair(vector<vector<double>>& Dschur_in, vector<bool>& Dschur_used_in) {
        Dschur_ptr = &Dschur_in;
        Dschur_used_ptr = &Dschur_used_in;
    }
};

void matrix_sub(DPair omp_out, 
                const DPair omp_in) {
    vector<vector<double>>& Dschur = *(omp_out.Dschur_ptr);
    vector<bool>& Dschur_used = *(omp_out.Dschur_used_ptr);
    vector<vector<double>>& Dschur_local = *(omp_in.Dschur_ptr);
    vector<bool>& Dschur_used_local = *(omp_in.Dschur_used_ptr);

    assert(Dschur.size() == Dschur_local.size());
    assert(Dschur_used.size() == Dschur_used_local.size());
    assert(Dschur.size() == Dschur_used.size());

    for(int i = 0; i < Dschur_local.size(); i++) {
        if(!Dschur_used_local[i]) {
            continue;
        }

        Dschur_used[i] = true;
        assert(Dschur[i].size() == Dschur_local[i].size());
        for(int j = 0; j < Dschur_local[i].size(); j++) {
            Dschur[i][j] -= Dschur_local[i][j];
        }
    }
    return;
}


#pragma omp declare reduction(matrix_sub : DPair : matrix_sub(omp_out, omp_in)) initializer(omp_priv=omp_orig)

void SchurOpt::compute_schur(/* parameters */) {
    // We are going to assume all the matrices are set in a nice way
    std::chrono::steady_clock::time_point t_schur_start = std::chrono::steady_clock::now();
    
    omp_set_num_threads(omp_num_threads);

    DPair dpair(Dschur, Dschur_used);

    #pragma omp parallel default(shared) 
    {
        assert(P % block_size == 0);
        assert(L % block_size == 0);
        int num_P_blocks = P / 3;
        int num_L_blocks = L / 3;

        vector<vector<double>> Dschur_local(num_P_blocks * num_P_blocks, vector<double>(block_squared, 0));
        
        assert(Dschur_local.size() == D.size());
        vector<bool> Dschur_used_local(num_P_blocks * num_P_blocks, false);
        vector<double> bschur_local;

        #pragma omp for
        for(int i = 0; i < A_sparse.size(); i++) {
            vector<double>& Aii = A_sparse[i];
            Eigen::Map<Matrix<double, 3, 3, RowMajor>> Aii_m(Aii.data());
            Matrix<double, 3, 3, RowMajor> Ainv_m = Aii_m.inverse();

            // Iterate over blocks of C panel
            int num_P_blocks = P / block_size;
            int num_L_blocks = L / block_size;
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

                    vector<double>& Bik = B[Bij_idx];
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>> Bik_m(Bik.data());

                    int Djk_idx = pair_to_idx(j, k, num_P_blocks, num_P_blocks);
                    vector<double>& Djk = Dschur_local[Djk_idx];
                    Dschur_used_local[Djk_idx] = true;
                    Eigen::Map<Matrix<double, 3, 3, RowMajor>> Djk_m(Djk.data());

                    Djk_m += CAinv_m * Bik_m;
                }
            }
        }


        #pragma omp for private(Dschur_local, Dschur_used_local) reduction(matrix_sub : dpair)
        for(int i = 0; i < omp_num_threads; i++) {
            matrix_sub(dpair, DPair(Dschur_local, Dschur_used_local));
        }

    }

    chrono::steady_clock::time_point t_schur_end = chrono::steady_clock::now();
    double t_schur = chrono::duration_cast<chrono::duration<double, milli>>(t_schur_end - t_schur_start).count();
    cout << "num_cores= " << omp_num_threads << " t_schur = " << t_schur << endl;

}
