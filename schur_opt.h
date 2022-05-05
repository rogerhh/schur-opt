#ifndef SCHUR_OPT_H
#define SCHUR_OPT_H

#include <vector>
#include <string>

class SchurOpt {

public:

    SchurOpt(int omp_num_threads_in)
    : omp_num_threads(omp_num_threads_in)
    {}

    enum WhichBlock { isA, isB, isC, isD };

    void from_g2o(/* parameters */);
    void to_g2o(/* parameters */);

    void compute_schur(/* parameters */);
    void read_sparse(const std::string& fname, SchurOpt& schur_opt, WhichBlock which_block);

private:

    int block_size = 3;
    int omp_num_threads = 0;

    // 5-4-2022: Decided on using row-major matrices
    std::vector<std::vector<double>> A_sparse;
    std::vector<double> B;  // C is B^T
    std::vector<double> D;
    std::vector<double> b1;
    std::vector<double> b2;

    std::vector<double> Dschur;
    std::vector<double> bschur;
};

#endif
