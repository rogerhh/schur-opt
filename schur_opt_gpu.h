#ifndef SCHUR_OPT_GPU_H
#define SCHUR_OPT_GPU_H

#include <vector>
#include <string>

class SchurOpt {

public:

    // SchurOpt() {}

    enum WhichBlock { isA, isB, isC, isD, isDschur_ref };

    void from_g2o(/* parameters */);
    void to_g2o(/* parameters */);

    void compute_schur(/* parameters */);
    void read_sparse(const std::string& fname, WhichBlock which_block);

    void verify_correctness(/* parameters */);

    ~SchurOpt();

private:

    void compute_Ainv();

    const int block_size = 3;
    const int block_squared = block_size * block_size;
    int omp_num_threads = 0;
    int L = -1, P = -1;

    // 5-4-2022: Decided on using row-major matrices
    std::vector<double> A_sparse;
    std::vector<double> Ainv;
    std::vector<double> B;     // C is B^T
    std::vector<bool> B_used;  // C is B^T
    std::vector<double> C;     // C is B^T
    std::vector<bool> C_used;  // C is B^T
    std::vector<double> D;
    std::vector<bool> D_used;
    std::vector<double> b1;
    std::vector<double> b2;

    std::vector<double> Dschur;
    std::vector<bool> Dschur_used;
    std::vector<double> bschur;

    std::vector<double> Dschur_ref;

    double* A_gpu = nullptr;
    double** A_gpu_batch = nullptr;
    double* Ainv_gpu = nullptr;
    double** Ainv_gpu_batch = nullptr;
    double* B_gpu = nullptr;
    double** B_gpu_batch = nullptr;
    double* C_gpu = nullptr;
    double** C_gpu_batch = nullptr;
    double* CAinv_gpu = nullptr;
    double** CAinv_gpu_batch = nullptr;
    double* D_gpu = nullptr;

    int batch_size = 0;
};

#endif
