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


uint64_t comm(BoolIO<NetIO> *ios[threads]) {
    uint64_t c = 0;
    for (int i = 0; i < threads; ++i)
      c += ios[i]->counter;
    return c;
}

uint64_t comm2(BoolIO<NetIO> *ios[threads]) {
    uint64_t c = 0;
    for (int i = 0; i < threads; ++i)
      c += ios[i]->io->counter;
    return c;
}

void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer a(32, 3, ALICE);
    Integer b(32, 2, ALICE);
    cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void nn_unit(size_t in_sz, size_t hl1_sz, size_t hl2_sz, BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    vector<Float> input;
    vector<Float> hl1_weights;
    vector<Float> hl2_weights;
    vector<Float> hl3_weights;
    // initialize weights
    for (int i=0; i<in_sz; ++i) {
        // value doesn't matter since runtime invariant to value
        hl1_weights.push_back(Float(1.5, ALICE));
    }
    for (int i=0; i<hl1_sz; ++i) {
        hl2_weights.push_back(Float(1.5, ALICE));
    }
    for (int i=0; i<in_sz; ++i) {
        input.push_back(Float(11.1, ALICE));
    }

    Bit input_sa(1, ALICE);
    Float t_a(0.5, ALICE);
    Float t_b(0.5, ALICE);
    Bit output(0, ALICE);

    auto start = clock_start();
    fair2layer_NN(14, 8, 2, input_sa, input, hl1_weights, hl2_weights, t_a, t_b, output);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");    
    double runtime = emp::time_from(start);
    cout << "party: " << party << "\t runtime: " << runtime << endl;
}

void nn3l_unit(size_t in_sz, size_t hl1_sz, size_t hl2_sz, size_t hl3_sz, BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    vector<Float> input;
    vector<Float> hl1_weights;
    vector<Float> hl2_weights;
    vector<Float> hl3_weights;
    // initialize weights
    for (int i=0; i<in_sz; ++i) {
        // value doesn't matter since runtime invariant to value
        hl1_weights.push_back(Float(1.5, ALICE));
    }
    for (int i=0; i<hl1_sz; ++i) {
        hl2_weights.push_back(Float(1.5, ALICE));
    }
    for (int i=0; i<hl2_sz; ++i) {
        hl3_weights.push_back(Float(1.5, ALICE));
    }
    for (int i=0; i<in_sz; ++i) {
        input.push_back(Float(11.1, ALICE));
    }

    Bit input_sa(1, ALICE);
    Float t_a(0.5, ALICE);
    Float t_b(0.5, ALICE);
    Bit output(0, ALICE);

    auto start = clock_start();
    //fair2layer_NN(14, 8, 2, input_sa, input, hl1_weights, hl2_weights, t_a, t_b, output);
    fair3layer_NN(in_sz, hl1_sz, hl2_sz, hl3_sz, input_sa, input, hl1_weights, hl2_weights, hl3_weights, t_a, t_b, output);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");    
    double runtime = emp::time_from(start);
    cout << "party: " << party << "\t runtime: " << runtime << endl;
}

// T is the number of rows in training dataset
// R is the number of Float features per row 
void bench_phase1_lr(int T, int R, BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    // initialization of test
    vector<Float> input;
    vector<Float> weights;
    Float threshold_a(0.5, ALICE);
    Float threshold_b(0.6, ALICE);
    Bit input_sa(1, ALICE);
    Bit output(1, ALICE);

    double dp_thresh_clear = 0.25;
    int dp_thresh_int = dp_thresh_clear * 100000;
    Integer DP_thresh = Integer(32, dp_thresh_int, PUBLIC); // 0.15 acceptable DP gap
    

    
    for (int i=0; i<R; ++i) {
        // dummy values, since runtime is independent of value
        input.push_back(Float(3.5, ALICE)); 
        weights.push_back(Float(2.5, ALICE));
    }

    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, T, 0.8, 0.78, 0.5);
    
    auto start = clock_start();
    for (int i=0; i<T; ++i) {
        if ( (party==BOB) && ((i % 1000) == 0)) {
            cout << i << "/" << T << "\n";
        }
        fair_binary_LR(input_sa, input, weights, threshold_a, threshold_b, output);
    }
    certify_postproc_DP(predicted_outcomes, sensitive_attributes, DP_thresh, T);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");    
    double runtime = emp::time_from(start);


    if (party==ALICE) {
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << T << "\t" << R << "\t" << runtime << "\t" << "LR" << "\t" << "1\n";
            fclose(fp);
        }
        
    } else {
        cout << "bench_phase1_lr runtime: " << runtime << "\tT: " << T << "\tR: " << R << endl;
    }
}


// T is the number of rows in training dataset
// R is the number of Float features per row 
void bench_baseline_3lnn(int n_queries, int R, size_t hl1_sz, size_t hl2_sz, size_t hl3_sz, BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    // initialization of test
    vector<Float> input;
    vector<Float> weights;
    Float threshold_a(0.5, ALICE);
    Float threshold_b(0.6, ALICE);
    Bit input_sa(1, ALICE);
    Bit output(1, ALICE);

    double dp_thresh_clear = 0.25;
    int dp_thresh_int = dp_thresh_clear * 100000;
    Integer DP_thresh = Integer(32, dp_thresh_int, PUBLIC); // 0.15 acceptable DP gap
    

    
    for (int i=0; i<R; ++i) {
        // dummy values, since runtime is independent of value
        input.push_back(Float(3.5, ALICE)); 
        //weights.push_back(Float(2.5, ALICE));
    }
    vector<Float> hl1_weights;
    vector<Float> hl2_weights;
    vector<Float> hl3_weights;
    // initialize weights and thresholds
    // values don't matter since runtime invariant to value
    for (int i=0; i<R; ++i) {
        // value doesn't matter since runtime invariant to value
        hl1_weights.push_back(Float(1.5, ALICE));
    }
    for (int i=0; i<hl1_sz; ++i) {
        hl2_weights.push_back(Float(1.5, ALICE));
    }
    for (int i=0; i<hl2_sz; ++i) {
        hl3_weights.push_back(Float(1.5, ALICE));
    }
    Float t_a(0.5, ALICE); // threshold for group a
    Float t_b(0.5, ALICE); // threshold for group b

    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, n_queries, 0.8, 0.78, 0.5);
    
    auto start = clock_start();
    for (int i=0; i<n_queries; ++i) {
        if ( (party==BOB) && (((i % 1000) == 0) || i==1 || i==2 || i==3 )) {
            cout << i << "/" << n_queries << "\n";
        }
        fair3layer_NN(R, hl1_sz, hl2_sz, hl3_sz, input_sa, input, hl1_weights, hl2_weights, hl3_weights, t_a, t_b, output);
        //fair_binary_LR(input_sa, input, weights, threshold_a, threshold_b, output);
    }
    certify_postproc_DP(predicted_outcomes, sensitive_attributes, DP_thresh, n_queries);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");    
    double runtime = emp::time_from(start);


    if (party==ALICE) {
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << n_queries << "\t" << R << "\t" << runtime << "\t" << "3LNN (" << hl1_sz << ", " << hl2_sz << ", " << hl3_sz << ")\t" << "3b\n";
            fclose(fp);
        }
        
    } else {
        cout << "bench_baseline_3lnn runtime: " << runtime << "\t2nu: " << n_queries << "\tR: " << R << "\tParam: (" << hl1_sz << ", " << hl2_sz << ", " << hl3_sz << ")\n";
    }
}


// benchmark for Adult dataset
// N is number of data points in validation set
// B is the number of bins in the calibration plot
void bench_baseline_tabular_adult(BoolIO<NetIO> *ios[threads], int party, int n_queries, int fp_threshold) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    uint64_t com1 = comm(ios);
    uint64_t com11 = comm2(ios);
    int fractional_bits = 5;  

    // set up dummy values for tabular model
    size_t in_sz = 57;
    size_t hr1_sz = 64;
    size_t hr2_sz = 32;
    size_t out_sz = 2;
    vector< vector<Float> > W1 = gen_dummy_weights(in_sz, hr1_sz, 0.01);
    vector< vector<Float> > W2 = gen_dummy_weights(hr1_sz, hr2_sz, 0.01);
    vector< vector<Float> > W3 = gen_dummy_weights(hr2_sz, out_sz, 0.01);
    vector<Float> bnd1 = gen_dummy_vec(hr1_sz, 1.1);
    vector<Float> bns1 = gen_dummy_vec(hr1_sz, 0.001);
    vector<Float> bnd2 = gen_dummy_vec(hr2_sz, 1.1);
    vector<Float> bns2 = gen_dummy_vec(hr2_sz, 0.001);

    double dp_thresh_clear = 0.25;
    int dp_thresh_int = dp_thresh_clear * 100000;
    Integer DP_thresh = Integer(32, dp_thresh_int, PUBLIC); // 0.15 acceptable DP gap


    vector<Float> x;
    for (int i=0; i<in_sz; ++i) {
        x.push_back(Float(1.0, ALICE));
    }

    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, n_queries, 0.8, 0.78, 0.5);
    auto start = clock_start();
    for (int i=0; i<n_queries; ++i) {
        if ( (party==BOB) && (((i % 1000) == 0) || i==1 || i==2 || i==3 )) {
            cout << i << "/" << n_queries << "\n";
        }

        Integer y(32, 2, ALICE); // dummy true label
        vector<Float> res = tabular_model(x, hr1_sz, W1, bnd1, bns1, hr2_sz, W2, bnd2, bns2, out_sz, W3);

    }
    certify_postproc_DP(predicted_outcomes, sensitive_attributes, DP_thresh, n_queries);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");    
    double runtime = emp::time_from(start);
    cout << "bench_baseline_tabular_adult -- n_queries: " << n_queries << "  runtime (sec): " << runtime / 1000000 << endl;

    #ifdef DEBUG
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << yhats[i].reveal<int>() << " ";
    }
    cout << ")\n";
    cout << "(";
    for (int i=0; i<N; ++i) {
        cout << " " << phats[i].reveal<double>() << " ";
    }
    cout << ")\n";
    #endif


    

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

	logfile_str = "experiments/phase1_benchmark" + logfile_str + ".txt";


    parse_party_and_port(argv, &party, &port);

    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    //NetIO *randomness_io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + 1000);
    //test_circuit_zk(ios, party);
    //setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);

    //nn_unit(14, 8, 2, ios, party);
    //nn3l_unit(14, 25, 25, 2, ios, party);
    size_t n_queries = 7600;
    bench_baseline_tabular_adult(ios, party, n_queries, 15);


    //int Ts[4] = {6151, 1993, 30000, 45222};
    //int Rs[4] = {8, 22, 23, 14};
    /*
    if (party==ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "w", stdout);
        cout << "T\tR\truntime (microsec)\tmodel\tphase\n";    
        fclose(fp);
    }
    */

    /*
    for (int i=0; i<1; ++i) {
        for (int j=0; j<5; ++j) {
            int T = Ts[i];
            int R = Rs[i];
            bench_phase1_lr(T, R, ios, party, logfile_str);
        }
    }
    */

    //bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    //if (cheat)error("cheat!\n");

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }

    return 0;
}

