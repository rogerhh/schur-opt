#include <fstream>
#include <iostream>
#include <getopt.h>
#include <string>
#include <cassert>
#include <cmath>
#include "schur_opt.h"

using namespace std;


int main(int argc, char** argv) {

    int opt = -1;
    int omp_num_threads = 1;

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

    ifstream list_fin("filelist.csv");

    string A_fname, B_fname, D_fname, b_fname, Dschur_fname, bschur_fname;

    while(list_fin >> A_fname >> B_fname >> D_fname >> b_fname >> Dschur_fname >> bschur_fname) {
        cout << A_fname << ", " << B_fname << ", " << D_fname << ", " << b_fname << ", " 
             << Dschur_fname << ", " << bschur_fname << endl;

        schur_opt.read_sparse(A_fname, schur_opt, SchurOpt::WhichBlock::isA);
        schur_opt.read_sparse(B_fname, schur_opt, SchurOpt::WhichBlock::isC);

    }


    return 0;
}
