#include "schur_opt.h"

#include <cmath>
#include <string>
#include <cassert>
#include <fstream>
#include <iostream>

using namespace std;

void SchurOpt::from_g2o(/* parameters */) {

}

void SchurOpt::to_g2o(/* parameters */) {

}

#pragma omp declare reduction(matrix_add : vector<double> : matrix_add(omp_out, omp_in)) initializer(omp_priv=omp_orig)

void SchurOpt::compute_schur(/* parameters */) {
    // // We are going to assume all the matrices are set in a nice way
    // 
    // omp_thread
    // #pragma omp parallel default(shared) schedule(static)
    // {

    //     vector<double> Dschur_local;
    //     vector<double> bschur_local;
    //     vector<double> scratchpad(block_size * block_size, 0);

    //     #pragma omp for
    //     for(int i = 0; i < A_sparse.size(); i++) {
    //         const vector<double>& A_block = A_sparse[i];
    //         auto Ainv = compute_inverse(A_block);

    //         for(int j = 0; j < P/b; j++) {

    //             CAinv = C[j] * Ainv;

    //             bschur_local = CAinv * b[j*block_size:(j+1)*block_size];

    //             for(int k = 0; k < P/b; k++) {
    //                 Dschur_local[j, k] += CAinv * C_transpose;
    //             }
    //         }
    //     }

    //     #pragma omp for private(Dschur_local, bschur_local) reduction(matrix_add : sum_matrix)
    //     for(int i = 0; i < omp_num_threads; i++) {
    //         Dschur -= Dschur_local;
    //         bschur -= bschur_local;
    //     }

    // }

}

void SchurOpt::read_sparse(const string& fname, SchurOpt& schur_opt, WhichBlock which_block) {
    ifstream fin(fname);

    if(!fin.is_open()) {
        cerr << "Error opening: " << fname << endl;
    }

    string line;
    getline(fin, line); // first 3 are unimportant
    cout << line << endl;
    getline(fin, line);
    cout << line << endl;
    getline(fin, line);
    cout << line << endl;

    int num_rows, num_cols;
    string garbage;

    fin >> garbage >> garbage >> num_rows >> garbage >> garbage >> num_cols;

    cout << "num_rows = " << num_rows << " num_cols = " << num_cols << endl;

    int num_blocks = num_rows / block_size + 1;
    int block_squared = block_size * block_size;
    assert(num_rows % block_size == 0 && "Number of rows must be divisible by block_size");

    int row, col;
    double val;

    if(which_block == WhichBlock::isA) {
        A_sparse = vector<vector<double>>(num_blocks, vector<double>(block_squared, 0));

        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            assert(abs(row - col) < 3);

            int block_id = row / 3;
            assert(block_id < num_blocks);

            int x_offset = row % 3, y_offset = col % 3;
            A_sparse[block_id][y_offset * block_size + x_offset] = val;
        }
    } else if (which_block == WhichBlock::isB){
        B = vector<double>(num_rows * num_cols, 0);
        
        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            assert(row < num_rows);
            assert(col < num_cols);

            int idx = row * num_cols + col;
            B[idx] = val;

            cout << row << " " << col << " " << val << endl;
        }
    } else if (which_block == WhichBlock::isC){
        // swap max rows and cols because we're reading the transpose
        int temp = num_rows;
        num_rows = num_cols;
        num_cols = temp;

        B = vector<double>(num_rows * num_cols, 0);
        
        while(fin >> row >> col >> val) {
            row--;  // index by 1
            col--;

            // Swap col and row because we're reading the transpose
            int temp = row;
            row = col;
            col = temp;

            assert(row < num_rows);
            assert(col < num_cols);

            int idx = row * num_cols + col;
            B[idx] = val;

            cout << row << " " << col << " " << val << endl;
        }
    }


}
