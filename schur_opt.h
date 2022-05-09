#ifndef SCHUR_OPT_H
#define SCHUR_OPT_H

#include <vector>
#include <string>

class SchurOpt {

public:

    SchurOpt(int omp_num_threads_in)
    : omp_num_threads(omp_num_threads_in)
    {}

    enum WhichBlock { isA, isB, isC, isD, isDschur_ref };

    void from_g2o(/* parameters */);
    void to_g2o(/* parameters */);

    void compute_schur(double* runtime);
    void read_sparse(const std::string& fname, SchurOpt& schur_opt, WhichBlock which_block);

    void verify_correctness(/* parameters */);
    
private:

    const int block_size = 3;
    const int block_squared = block_size * block_size;
    int omp_num_threads = 0;
    int L = -1, P = -1;

    // 5-4-2022: Decided on using row-major matrices
    // A_sparse is block diagonal
    // Each A_sparse entry is the block matrix in row-major order
    std::vector<std::vector<double>> A_sparse;
    
    // B stored in 2D row major, C is B^T
    std::vector<std::vector<double>> B;  
    std::vector<bool> B_used;  

    // D stored in 2D row major
    std::vector<std::vector<double>> D;
    std::vector<bool> D_used;

    // b are vectors
    std::vector<double> b1;
    std::vector<double> b2;

    std::vector<std::vector<double>> Dschur;
    std::vector<bool> Dschur_used;
    std::vector<double> bschur;

    // for verifying result
    std::vector<std::vector<double>> Dschur_ref;
    std::vector<bool> Dschur_ref_used;
    std::vector<double> bschur_ref;
};

#endif
