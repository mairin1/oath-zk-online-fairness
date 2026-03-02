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
const int threads = 1;
// benchmarks for the class-balanced random sample part of phase 3
// uses ZKRORAM so can only use single thread


void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer a(32, 3, ALICE);
    Integer b(32, 2, ALICE);
    cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}


// NOTE: only accurately records runtime if not verbose
string bench_phase3a_lr(int D, int nu, BoolIO<NetIO> * ios[threads], int party, bool verbose) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    // initialization of test

    Bit TRU(1, PUBLIC);
    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    double proportion_a0 = 0.25;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, D, 0.8, 0.35, proportion_a0);
    std::default_random_engine rng(ZKPOF_SEED);
    std::shuffle(begin(sensitive_attributes), end(sensitive_attributes), rng);

    auto start = clock_start();
    int ind_sz = ceil(log2(D));
    // init P0
    ROZKRAM<BoolIO<NetIO>> * pi_in_a0 = init_pi_in(party, D * proportion_a0, ind_sz);
    vector<Integer> pi_out_a0 = vector<Integer>(D);
    Bit temp_sa(0, ALICE);
    class_specific_rank_permutation(D, temp_sa, pi_in_a0, pi_out_a0, sensitive_attributes);
    // init P1
    ROZKRAM<BoolIO<NetIO>> * pi_in_a1 = init_pi_in(party, D * (1-proportion_a0), ind_sz);
    vector<Integer> pi_out_a1 = vector<Integer>(D);
    class_specific_rank_permutation(D, temp_sa^TRU, pi_in_a1, pi_out_a1, sensitive_attributes);

    // initialize S with all 0s
    vector<Bit> sample_vec;
    for (int i=0; i<D; ++i) {
        sample_vec.push_back(Bit(0, PUBLIC));
    }
    class_balanced_sample(D, nu, pi_out_a0, pi_out_a1, sample_vec);
    pi_in_a0->check();
    pi_in_a1->check();

    // only accurately records runtime if not verbose
    if (verbose) {
        cout << "sensitive_attributes: [ ";
        for (int i=0; i<32; ++i) {
            cout << sensitive_attributes[i].reveal<bool>() << " ";
        }
        cout << "]\n";
        //cout << "csrp_v1 pi_in:  [ ";
        cout << "csrp p0: [ ";
        for (int i=0; i<32; ++i) {
            cout << pi_out_a0[i].reveal<int>() << " ";
        }
        cout << "]\n";
        cout << "csrp p1: [ ";
        for (int i=0; i<32; ++i) {
            cout << pi_out_a1[i].reveal<int>() << " ";
        }
        cout << "]\n";
        cout << "sample_vec : [ ";
        for (int i=0; i<32; ++i) {
            cout << sample_vec[i].reveal() << " ";
        }
    }

    delete pi_in_a0;
    delete pi_in_a1;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    double runtime = emp::time_from(start);
    string outstr = "";        
    outstr = outstr + to_string(D) + "\t" + to_string(nu) + "\t" + to_string(runtime) + "\t" + "doesntmatter" + "\t" + "3a\n";  
    if (party==BOB) {
        cout << "phase3a (model agnostic) -- D: " << D << "\t nu: " << nu << "\t runtime: " << runtime << "\n";
    }
    return outstr;
}


// something about how im opening the logfile is very haunted.... decided to try a different approach
// NOTE: only accurately records runtime if not verbose
void bench_phase3a_lr_v1(int D, int nu, BoolIO<NetIO> * ios[threads], int party, int test_ind, bool verbose, string logfile_str="") {
    cout << "??\n";
    cout << "logfile_str: " << logfile_str << "\n";
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    cout << "1\n";
    // initialization of test

    Bit TRU(1, PUBLIC);
    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    double proportion_a0 = 0.25;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, D, 0.8, 0.35, proportion_a0);
    std::default_random_engine rng(ZKPOF_SEED);
    std::shuffle(begin(sensitive_attributes), end(sensitive_attributes), rng);

    cout << "2\n";
    auto start = clock_start();
    int ind_sz = ceil(log2(D));
    // init P0
    ROZKRAM<BoolIO<NetIO>> * pi_in_a0 = init_pi_in(party, D * proportion_a0, ind_sz);
    vector<Integer> pi_out_a0 = vector<Integer>(D);
    Bit temp_sa(0, ALICE);
    class_specific_rank_permutation(D, temp_sa, pi_in_a0, pi_out_a0, sensitive_attributes);
    cout << "3\n";
    // init P1
    ROZKRAM<BoolIO<NetIO>> * pi_in_a1 = init_pi_in(party, D * (1-proportion_a0), ind_sz);
    vector<Integer> pi_out_a1 = vector<Integer>(D);
    class_specific_rank_permutation(D, temp_sa^TRU, pi_in_a1, pi_out_a1, sensitive_attributes);

    // initialize S with all 0s
    vector<Bit> sample_vec;
    for (int i=0; i<D; ++i) {
        sample_vec.push_back(Bit(0, PUBLIC));
    }
    cout << "4\n";
    class_balanced_sample(D, nu, pi_out_a0, pi_out_a1, sample_vec);
    cout << "5\n";
    pi_in_a0->check();
    pi_in_a1->check();

    // only accurately records runtime if not verbose
    if (verbose) {
        cout << "sensitive_attributes: [ ";
        for (int i=0; i<32; ++i) {
            cout << sensitive_attributes[i].reveal<bool>() << " ";
        }
        cout << "]\n";
        //cout << "csrp_v1 pi_in:  [ ";
        cout << "csrp p0: [ ";
        for (int i=0; i<32; ++i) {
            cout << pi_out_a0[i].reveal<int>() << " ";
        }
        cout << "]\n";
        cout << "csrp p1: [ ";
        for (int i=0; i<32; ++i) {
            cout << pi_out_a1[i].reveal<int>() << " ";
        }
        cout << "]\n";
        cout << "sample_vec : [ ";
        for (int i=0; i<32; ++i) {
            cout << sample_vec[i].reveal() << " ";
        }
        cout << "]\n";
    }
    cout << "6\n";

    delete pi_in_a0;
    delete pi_in_a1;
    cout << "7\n";

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    double runtime = emp::time_from(start);
    cout << "there should be 2 of this message\n";
    if (party==ALICE) {
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << D << "\t" << nu << "\t" << runtime << "\t" << "doesntmatter" << "\t" << "3a\n";
            if (test_ind==0) {cout << "first time\n";}
            else {cout << "second time\n";}
            fclose(fp);
        }
    } else {
        cout << "phase3a (model agnostic) -- D: " << D << "\t nu: " << nu << "\t runtime: " << runtime << "\n";
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

	logfile_str = "experiments/phase3a_benchmark_" + logfile_str + ".txt";

    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    
    /*
    // ?
    if (party==ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
        cout << "D\tnu\truntime (microsec)\tmodel\tphase\n";    
        fclose(fp);
    }
    */
    string l = "";
    int Ds[4] = {50000, 100000, 500000, 1000000};
    //int nus[4] = {50, 500, 1000, 3800};
    for (int i=0; i<4; ++i) {
        for (int j=0; j<5; ++j) {
            l += bench_phase3a_lr(Ds[i], 1000, ios, party, false);
        }
    }


    std::cout << l << "\n";

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    if (party==ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
        cout << "D\tnu\truntime (microsec)\tmodel\tphase\n";
        cout << l;  
        fclose(fp);
    }
    return 0;
}