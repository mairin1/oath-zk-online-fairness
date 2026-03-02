#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "zk-pof/fairness_zk.cpp"
#include "zk-pof/constant.cpp"

using namespace emp;
using namespace std;

int port, party;
const int threads = 12;


void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer a(32, 3, ALICE);
    Integer b(32, 2, ALICE);
    cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}


void sensitive_attr_bench(BoolIO<NetIO> *ios[threads], int party, size_t n_queries) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);

    auto start = clock_start();
    // initialization
    vector<Integer> alpha_xs_dummy = Int_zeros(n_queries);
    vector<Integer> alpha_zeros_dummy = Int_zeros(n_queries);
    vector<Integer> alpha_ones_dummy = Int_ones(n_queries);
    vector<Bit> xs_dummy = Bit_vec(n_queries, 0);

    sensitive_attr_check(alpha_xs_dummy, alpha_zeros_dummy, alpha_ones_dummy, xs_dummy, n_queries);

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    cout << "sensitive attr check -- n_queries: " << n_queries << "\t time (sec): " << time_from(start) / 1000000 << " " << party << endl;
}




int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    test_circuit_zk(ios, party);

    sensitive_attr_bench(ios, party, 50000);


    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}