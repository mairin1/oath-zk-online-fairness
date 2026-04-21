#ifndef _ZKPOF_FAIRNESS_METRICS
#define _ZKPOF_FAIRNESS_METRICS

#include "emp-zk/emp-zk.h"
#include <iostream>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "constant.cpp"
#include "utils.cpp"

using namespace emp;
using namespace std;

void certify_postproc_CF(
    vector<Bit> & predicted_outcomes_original,
    vector<Bit> & predicted_outcomes_flipped,
    const int NUM_POINTS,
    bool verbose = false)
{
    Bit TRU = Bit(1, PUBLIC);
    Bit fair_check = Bit(1, PUBLIC);  // starts true; set to false on first violation

    for (int i = 0; i < NUM_POINTS; i++) {
        Bit differs = predicted_outcomes_original[i] ^ predicted_outcomes_flipped[i];
        fair_check = fair_check & (differs ^ TRU);  // fair_check &= !differs
    }

    if (verbose) {
        cout << "Counterfactually fair? " << fair_check.reveal() << endl;
    }
}

#endif