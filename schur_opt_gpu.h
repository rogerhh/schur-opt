#ifndef SCHUR_OPT_GPU_H
#define SCHUR_OPT_GPU_H

#include <vector>
#include <string>

class SchurOpt {

public:

    // SchurOpt() {}

    enum WhichBlock { isA, isB, isC, isD };

    void from_g2o(/* parameters */);
    void to_g2o(/* parameters */);

    void compute_schur(/* parameters */);
    void read_sparse(const std::string& fname, SchurOpt& schur_opt, WhichBlock which_block);

private:

    const int block_size = 3;
    const int block_squared = block_size * block_size;
    int omp_num_threads = 0;
    int L = -1, P = -1;

    // 5-4-2022: Decided on using row-major matrices
    std::vector<std::vector<double>> A_sparse;
    std::vector<std::vector<double>> B;  // C is B^T
    std::vector<bool> B_used;  // C is B^T
    std::vector<std::vector<double>> D;
    std::vector<bool> D_used;
    std::vector<double> b1;
    std::vector<double> b2;

    std::vector<std::vector<double>> Dschur;
    std::vector<bool> Dschur_used;
    std::vector<double> bschur;
};

#endif
