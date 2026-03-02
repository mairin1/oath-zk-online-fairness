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

// benchmarks for the final part of phase 3, IT-MAC computation to verify consistency


void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer a(32, 3, ALICE);
    Integer b(32, 2, ALICE);
    cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void zk_plain_verify_mac(BoolIO<NetIO> *ios[threads], int party, size_t num_authentications) {
    auto start = clock_start();
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer p(32, 11, PUBLIC); // prime modulus
    
    // authenticate IT-MAC tags held by P
    vector<Integer> tags;
    vector<Integer> vals;
    for (int i=0; i < num_authentications; ++i) {
        tags.push_back(Integer(32, 2, ALICE));
        vals.push_back(Integer(32, 1, ALICE));
    }

    // authenticate IT-MAC key values revealed by V
    vector<Integer> A_keys;
    vector<Integer> B_keys;
    for (int i=0; i < num_authentications; ++i) {
        A_keys.push_back(Integer(32, 1, PUBLIC));
        B_keys.push_back(Integer(32, 1, PUBLIC));
    }

    Bit consistency_check(1, PUBLIC);
    for (int i=0; i < num_authentications; ++i) {
        Bit temp = (tags[i] == ((vals[i] * A_keys[i] + B_keys[i])) % p );
        consistency_check = consistency_check & temp;
    }
    cout << "consistency check: " << consistency_check.reveal() << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    double tt = time_from(start);
    std::cout << "time: " << tt << std::endl;
}



int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    
    int Q = 7300;
    int R = 63;

    zk_plain_verify_mac(ios, party, Q * R);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}