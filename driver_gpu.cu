#include <fstream>
#include <iostream>
#include <getopt.h>
#include <string>
#include <cassert>
#include <cmath>
#include "schur_opt_gpu.h"

using namespace std;

int main(int argc, char** argv) {
    int opt = -1;

    while((opt = getopt(argc, argv, "n:")) != -1) {
        switch(opt) {
            default:
                cerr << "Unrecognized flag" << endl;
                exit(1);
        }
    }

    SchurOpt schur_opt();

    string list_fname = "/app/schur_opt/filelist.csv";
    ifstream list_fin(list_fname);
    if(!list_fin.is_open()) {
        cerr << "Error opening file: " << list_fname << endl;
    }

    string A_fname, B_fname, D_fname, b_fname, Dschur_fname, bschur_fname;

    while(list_fin >> A_fname >> B_fname >> D_fname >> b_fname >> Dschur_fname >> bschur_fname) {
        cout << A_fname << ", " << B_fname << ", " << D_fname << ", " << b_fname << ", " 
            << Dschur_fname << ", " << bschur_fname << endl;

        // schur_opt.read_sparse(A_fname, SchurOpt::WhichBlock::isA);
        // schur_opt.read_sparse(B_fname, SchurOpt::WhichBlock::isC);
        // schur_opt.read_sparse(D_fname, SchurOpt::WhichBlock::isD);

    }

    // schur_opt.compute_schur();

    return 0;
}
