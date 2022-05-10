#include <fstream>
#include <iostream>
#include <getopt.h>
#include <string>
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
        // string list_fname = "../data/filelist.csv";
        ifstream list_fin(list_fname);
        if(!list_fin.is_open()) {
            cerr << "Error opening file: " << list_fname << endl;
        }

    string A_fname, C_fname, D_fname, b_fname, Dschur_fname, bschur_fname;

    while(list_fin >> A_fname >> C_fname >> D_fname >> b_fname >> Dschur_fname >> bschur_fname) {
        cout << A_fname << ", " << C_fname << ", " << D_fname << ", " << b_fname << ", " 
                << Dschur_fname << ", " << bschur_fname << endl;

        schur_opt.read_sparse(A_fname, SchurOpt::WhichBlock::isA);
        schur_opt.read_sparse(C_fname, SchurOpt::WhichBlock::isC);
        schur_opt.read_sparse(D_fname, SchurOpt::WhichBlock::isD);

        schur_opt.read_sparse(Dschur_fname, SchurOpt::WhichBlock::isDschur_ref); // Hschur_ref
    }

    schur_opt.compute_schur();
    schur_opt.verify_correctness();

        // TO-DO: add correctness check with regard to the reference Dschur
    // }
    return 0;
}
