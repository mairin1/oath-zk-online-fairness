#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "zk-pof/utils.cpp"
#include "zk-pof/constant.cpp"
#include "zk-pof/lr_zk.cpp"
#include "zk-pof/fairness_zk.cpp"

using namespace emp;
using namespace std;

int port, party;
const int threads = 12;


// D is the number of queries in the dataset
// two_nu is the number of queries in the consistency check sample
// R is the number of values in the input
void bench_correctness_lr(int two_nu, int R, BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    // initialization of test
    vector<Float> input;
    vector<Float> weights;
    Float threshold_a(0.5, ALICE);
    Float threshold_b(0.6, ALICE);
    Bit input_sa(1, ALICE);
    Bit output(1, ALICE);
    

    
    for (int i=0; i<R; ++i) {
        // dummy values, since runtime is independent of value
        input.push_back(Float(3.5, ALICE)); 
        weights.push_back(Float(2.5, ALICE));
    }

    vector<Bit> predicted_outcomes_sample;
    vector<Bit> sensitive_attributes_sample;
    example_bit_vectors_DP(predicted_outcomes_sample, sensitive_attributes_sample, two_nu, 0.8, 0.78, 0.5);

    auto start = clock_start();
    //uint64_t comm1_individual_inf;
    //uint64_t comm2_individual_inf;
    for (int i=0; i<two_nu; ++i) {
        if ( (party==BOB) && ((i % 1000) == 0)) {
            cout << i << "/" << two_nu << "\n";
            
        }
        fair_binary_LR(input_sa, input, weights, threshold_a, threshold_b, output);
    }
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");    
    double runtime = emp::time_from(start);


    if (party==ALICE) {
        
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << two_nu << "\t" << R << "\t" << runtime << "\t" << "LR" << "\t" << "3 correctness" << "\n";
            fclose(fp);
        }
        
        
    } else {
        cout << "bench_phase3b_lr runtime: " << runtime << "\t2nu: " << two_nu << "\tR: " << R << endl;
        //cout << "bob communication: " << c_indiv_inf << " (individual inference)\t" << c_all_inf << " (total inf)\t" << c_faircheck << " (faircheck)\n";
        //cout << "bob total comm: " << c_all_inf + c_faircheck << "\n";
    }
}


int main(int argc, char **argv) {
    // initialize logfile
    using namespace std::chrono;
	time_t rawtime;
    tm* timeinfo;
    char buffer [80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);
    puts(buffer);
	string logfile_str(buffer);

	logfile_str = "experiments/phase3corrcheck_benchmark" + logfile_str + ".txt";


    parse_party_and_port(argv, &party, &port);

    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    //NetIO *randomness_io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + 1000);
    //test_circuit_zk(ios, party);
    //setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);

    int two_nus[4] = {1000, 2000, 7600, 10000};
    //int two_nus[3] = { 1000, 2000, 7600}; // see trials.py for corresponding analysis
    //int Rs[5] = {8, 22, 23, 14, 61}; // more than double the number of parameters in any of the fairness datasets

    
    if (party==ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "w", stdout);
        cout << "2nu\tR\truntime (microsec)\tmodel\tphase\n";    
        fclose(fp);
    }
    

    for (int i=0; i<4; ++i) {
        int two_nu = two_nus[i];
        for (int j=0; j<1; ++j) {
            //int R = Rs[j];
            int R = 50;
            for (int k=0; k<5; ++k) {
                bench_correctness_lr(two_nu, R, ios, party, logfile_str);
            }

        }
    }

    //bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    //if (cheat)error("cheat!\n");

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }

    return 0;
}