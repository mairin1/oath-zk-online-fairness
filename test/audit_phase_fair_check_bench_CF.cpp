// counterfactual fairness check benchmark

#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "zk-pof/utils.cpp"
#include "zk-pof/constant.cpp"
#include "zk-pof/lr_zk.cpp"
#include "zk-pof/fairness_zk.cpp"
#include "zk-pof/fairness_metrics.cpp"

using namespace emp;
using namespace std;

int port, party;
const int threads = 12;

void example_bit_vectors_CF(
    vector<Bit> & predicted_outcomes_original,
    vector<Bit> & predicted_outcomes_flipped,
    vector<Bit> & sensitive_attributes,
    int num_points,
    double a0_pos,
    double a1_pos,
    double sa_split
) {
    example_bit_vectors_DP(predicted_outcomes_original, sensitive_attributes, num_points, a0_pos, a1_pos, sa_split);
    predicted_outcomes_flipped = predicted_outcomes_original;
}

void bench_faircheck_CF(int D, BoolIO<NetIO> *ios[threads], int party, string logfile_str = "") {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);

    vector<Bit> predicted_outcomes_original;
    vector<Bit> predicted_outcomes_flipped;
    vector<Bit> sensitive_attributes;

    example_bit_vectors_CF(
        predicted_outcomes_original, predicted_outcomes_flipped,
        sensitive_attributes, D, 0.8, 0.78, 0.5);

    auto start = emp::clock_start();
    certify_postproc_CF(predicted_outcomes_original, predicted_outcomes_flipped, D, true);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat) error("cheat!\n");
    double runtime = emp::time_from(start);

    if (party == ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
        cout << D << "\t" << runtime << "\t" << "doesntmatter\t" << "CF faircheck\n";
        fclose(fp);
    } else {
        cout << "certify counterfactual fairness D: " << D << "\t runtime: " << runtime << "\n";
    }
}

int main(int argc, char **argv) {
    using namespace std::chrono;
    time_t rawtime;
    tm* timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo);
    puts(buffer);
    string logfile_str(buffer);
    logfile_str = "experiments/audit_phase_faircheck_benchmark_CF" + logfile_str + ".txt";

    parse_party_and_port(argv, &party, &port);

    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(
            new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i),
            party == ALICE);

    int Ds[3] = {100000, 500000, 1000000};

    if (party == ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "w", stdout);
        cout << "D\truntime (microsec)\tmodel\tphase\n";
        fclose(fp);
    }

    for (int i = 0; i < 3; ++i) {
        int D = Ds[i];
        for (int k = 0; k < 5; ++k) {
            bench_faircheck_CF(D, ios, party, logfile_str);
        }
    }

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
