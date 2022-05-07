#include <fstream>
#include <iostream>
#include <getopt.h>
#include <string>
#include <cassert>
#include <cmath>
#include "schur_opt.h"

using namespace std;


int main(int argc, char** argv) {

    for(int num_cores = 0; num_cores < 24; num_cores += 2) {
        int opt = -1;
        int omp_num_threads = num_cores;

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

        SchurOpt schur_opt(omp_num_threads);

        string list_fname = "/app/schur_opt/filelist.csv";
        ifstream list_fin(list_fname);
        if(!list_fin.is_open()) {
            cerr << "Error opening file: " << list_fname << endl;
        }

        string A_fname, B_fname, D_fname, b_fname, Dschur_fname, bschur_fname;

        while(list_fin >> A_fname >> B_fname >> D_fname >> b_fname >> Dschur_fname >> bschur_fname) {
            cout << A_fname << ", " << B_fname << ", " << D_fname << ", " << b_fname << ", " 
                 << Dschur_fname << ", " << bschur_fname << endl;

            schur_opt.read_sparse(A_fname, SchurOpt::WhichBlock::isA);
            schur_opt.read_sparse(B_fname, SchurOpt::WhichBlock::isC);
            schur_opt.read_sparse(D_fname, SchurOpt::WhichBlock::isD);

        }

        schur_opt.compute_schur();

    }
    return 0;
}
