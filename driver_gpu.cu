#include <fstream>
#include <iostream>
#include <getopt.h>
#include <string>
#include <regex>
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

    SchurOpt schur_opt;

    string list_fname = "filelist_sample.csv";
    ifstream list_fin(list_fname);
    if(!list_fin.is_open()) {
        cerr << "Error opening file: " << list_fname << endl;
        exit(1);
    }

    string A_fname, C_fname, D_fname, b_fname, Dschur_fname, bschur_fname;

    while(list_fin >> A_fname >> C_fname >> D_fname >> b_fname >> Dschur_fname >> bschur_fname) {
        // cout << A_fname << ", " << C_fname << ", " << D_fname << ", " << b_fname << ", " 
        //     << Dschur_fname << ", " << bschur_fname << endl;

        
        regex regexp("[0-9]+_[0-9]+_[0-9]+"); 
        smatch m; 
        regex_search(A_fname, m, regexp); 

        for (auto x : m) 
            cout << "[DATASET] Loaded dataset " << x << endl; 


        schur_opt.read_sparse(A_fname, SchurOpt::WhichBlock::isA);
        schur_opt.read_sparse(C_fname, SchurOpt::WhichBlock::isC);
        schur_opt.read_sparse(D_fname, SchurOpt::WhichBlock::isD);

        schur_opt.read_sparse(Dschur_fname, SchurOpt::WhichBlock::isDschur_ref); // Hschur_ref

        for (int i = 0;  i < 5; i++) {
            schur_opt.compute_schur();
        }
        // schur_opt.verify_correctness();
    }

    return 0;
}
