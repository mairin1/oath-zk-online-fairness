#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "zk-pof/utils.cpp"
#include "zk-pof/fairness_zk.cpp"


using namespace emp;
using namespace std;

int port, party;
const int threads = 1;

int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    //NetIO *randomness_io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + 1000);
    //test_circuit_zk(ios, party);


    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    const int TEST_IND_SZ = 5;
   
    ROZKRAM<BoolIO<NetIO>> * pi = init_pi_in(party, 14, TEST_IND_SZ);
    std::cout << "init_pi_in: [ ";
    int test_n = 1 << TEST_IND_SZ;
  
    for (int i=0; i < test_n; ++i) {
        Integer IND(TEST_IND_SZ, i, PUBLIC);
        Integer o = pi->read(IND);
        std::cout << o.reveal<int>() << " ";
    }
    std::cout << "]\n";


    pi->check();
    delete pi;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    


    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}