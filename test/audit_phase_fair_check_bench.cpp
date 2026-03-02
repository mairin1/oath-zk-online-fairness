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

void bench_faircheck_DP(int D, BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    double dp_thresh_clear = 0.25;
    int dp_thresh_int = dp_thresh_clear * 100000;
    Integer DP_thresh = Integer(32, dp_thresh_int, PUBLIC); // 0.15 acceptable DP gap
    vector<Bit> predicted_outcomes_pool;
    vector<Bit> sensitive_attributes_pool;
    example_bit_vectors_DP(predicted_outcomes_pool, sensitive_attributes_pool, D, 0.8, 0.78, 0.5);

    auto start = clock_start();
    certify_postproc_DP(predicted_outcomes_pool, sensitive_attributes_pool, DP_thresh, D);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");    
    double runtime = emp::time_from(start);
    if (party==ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
        cout << D << "\t" << runtime << "\t" << "doesntmatter\t" << "3 faircheck\n";  
        fclose(fp);
    } else {
        cout << "certify dem parity D: " << D << "\t runtime: " << runtime << "\n";
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

	logfile_str = "experiments/audit_phase_faircheck_benchmark" + logfile_str + ".txt";


    parse_party_and_port(argv, &party, &port);

    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    //NetIO *randomness_io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + 1000);
    //test_circuit_zk(ios, party);
    //setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);

    int Ds[3] = {100000, 500000, 1000000};
    
    if (party==ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "w", stdout);
        cout << "D\truntime (microsec)\tmodel\tphase\n";    
        fclose(fp);
    }
    

    for (int i=0; i<3; ++i) {
        int D = Ds[i];
        for (int k=0; k<5; ++k) {
            bench_faircheck_DP(D, ios, party, logfile_str);
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