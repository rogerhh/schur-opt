#include <fstream>
#include <iostream>
#include <getopt.h>
#include <string>
#include <regex>
#include <cassert>
#include <cmath>
#include "schur_opt.h"

using namespace std;


int main(int argc, char** argv) {

    // OpenMP scaling experiment 
    // for(int num_cores = 0; num_cores < 32; num_cores += 2) {
    int opt = -1;
    int omp_num_threads = 1; // default single threaded
    
    while((opt = getopt(argc, argv, "n:")) != -1) {
        switch(opt) {
            case 'n':
                omp_num_threads = atoi(optarg);
                break;
            default:
                cerr << "Unrecognized flag" << endl;
                exit(1);
        }
    }

    // initialize solver based on num_threads set
    SchurOpt schur_opt(omp_num_threads);

    string list_fname = "filelist_sample.csv";
    ifstream list_fin(list_fname);
    if(!list_fin.is_open()) {
        cerr << "Error opening file: " << list_fname << endl;
        exit(1);
    }

    string A_fname, C_fname, D_fname, b_fname, Dschur_fname, bschur_fname;

    while(list_fin >> A_fname >> C_fname >> D_fname >> b_fname >> Dschur_fname >> bschur_fname) {

        regex regexp("[0-9]+_[0-9]+_[0-9]+"); 
        smatch m; 
        regex_search(A_fname, m, regexp); 

        for (auto x : m) 
            cout << "Loaded dataset " << x << endl; 

        // cout << A_fname << ", " << C_fname << ", " << D_fname << ", " << b_fname << ", " 
        //         << Dschur_fname << ", " << bschur_fname << endl;

        schur_opt.read_sparse(A_fname, schur_opt, SchurOpt::WhichBlock::isA); // Hll
        schur_opt.read_sparse(C_fname, schur_opt, SchurOpt::WhichBlock::isC); // Hpl
        schur_opt.read_sparse(D_fname, schur_opt, SchurOpt::WhichBlock::isD); // Hpp
        // reference
        schur_opt.read_sparse(Dschur_fname, schur_opt, SchurOpt::WhichBlock::isDschur_ref); // Hschur_ref
    }

    schur_opt.compute_schur();
    schur_opt.verify_correctness();
    // }
    return 0;
}
