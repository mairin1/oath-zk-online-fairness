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
//const int threads = 1;

/*
void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer a(32, 3, ALICE);
    Integer b(32, 2, ALICE);
    cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}
*/



int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    //const size_t input_sz = 150592; // 224 x 224 x 3 image in bytes
    const size_t input_sz = 61*32; // size of input in bytes
    const size_t random_sz = 8; // size of 64-bit random seed in bytes
    const size_t output_sz = 1; // single byte of classification output
    const size_t witness_sz = input_sz + random_sz + output_sz;
    const size_t signature_sz = 350; // size of base64 encoded signature in bytes
    const size_t hash_sz = 256 / 8; // size of hash in bytes 

    auto start = emp::clock_start();

    BoolIO<NetIO> *alice_send_ios[1];
    BoolIO<NetIO> *bob_send_ios[1];
    for (int i = 0; i < 1; ++i) {
        alice_send_ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
        bob_send_ios[i] = new BoolIO<NetIO>(new NetIO(party==BOB ? nullptr : "127.0.0.1", port + 100 + i), party==BOB);
    }
    // fair coin flipping
    // each party hashes a random string to commit to it, sends it to the other party
    system("openssl dgst -sha256 data/dummy_file.txt"); 
    // initialize buffers
    uint8_t hash_placeholder[hash_sz + random_sz];
    memset(hash_placeholder, 0, hash_sz + random_sz);
    uint8_t a_recv_h[hash_sz + random_sz];
    memset(a_recv_h, 0, hash_sz + random_sz); 
    uint8_t b_recv_h[hash_sz + random_sz];
    memset(b_recv_h, 0, hash_sz + random_sz);
    // client sends hash
    if (party==ALICE) {
        alice_send_ios[0]->io->send_data(&hash_placeholder, hash_sz+random_sz);
    } else {
        alice_send_ios[0]->io->recv_data(&b_recv_h, hash_sz+random_sz);
    }
    alice_send_ios[0]->io->flush();

    // P sends hash + committed randomness
    if (party==BOB) {
        bob_send_ios[0]->io->send_data(&hash_placeholder, hash_sz+random_sz);
    } else {
        bob_send_ios[0]->io->recv_data(&a_recv_h, hash_sz+random_sz);
    }
    bob_send_ios[0]->io->flush();

    // then client sends witness incl committed randomness
    // initialize buffers
    uint8_t x[witness_sz + signature_sz];
    memset(x, 1, witness_sz + signature_sz);
    uint8_t a_recv[witness_sz + signature_sz];
    memset(a_recv, 0, witness_sz + signature_sz); 
    uint8_t b_recv[witness_sz + signature_sz];
    memset(b_recv, 0, witness_sz + signature_sz);

    if (party==ALICE) {
        system("openssl dgst -sha256 -sign data/testprivkey.pem -out data/temp-sign.sha256 data/dummy_file.txt");
        system("openssl base64 -in data/temp-sign.sha256 -out data/test_sig");
    }

    if (party==ALICE) {
        alice_send_ios[0]->io->send_data(&x, witness_sz + signature_sz);
    } else {
        alice_send_ios[0]->io->recv_data(&b_recv, witness_sz + signature_sz);
    }
    alice_send_ios[0]->io->flush();

    // P validates signature and hash of randomness, xors with his randomness to get seed
    if (party==BOB) {
        system("openssl dgst -sha256 data/dummy_file.txt");
        system("openssl base64 -d -in data/test_sig -out data/temp-sign.sha256 && openssl dgst -sha256 -verify data/testpubkey.pem -signature data/temp-sign.sha256 data/dummy_file.txt");
    }


    // then BOB locally computes model output (skipped for simplicity)

    if (party==BOB) {
        system("openssl dgst -sha256 -sign data/testprivkey.pem -out data/temp-sign.sha256 data/dummy_file.txt");
        system("openssl base64 -in data/temp-sign.sha256 -out data/test_sig");
    }

    // bob sends back witness w/ seed
    if (party==BOB) {
        cout << "party: " << party << "\tsending.\n";
        bob_send_ios[0]->io->send_data(&x, witness_sz + signature_sz);
    } else {
        cout << "party: " << party << "\treceiving.\n";
        bob_send_ios[0]->io->recv_data(&a_recv, witness_sz + signature_sz);
    }
    bob_send_ios[0]->io->flush();

    // alice validates signature, verifies hash of the seed
    if (party==ALICE) {
        system("openssl dgst -sha256 data/dummy_file.txt");
        system("openssl base64 -d -in data/test_sig -out data/temp-sign.sha256 && openssl dgst -sha256 -verify data/testpubkey.pem -signature data/temp-sign.sha256 data/dummy_file.txt");
        system("openssl dgst -sha256 data/dummy_file.txt");
    }
        if (party==ALICE) {
        alice_send_ios[0]->io->send_data(&x, witness_sz + signature_sz);
    } else {
        alice_send_ios[0]->io->recv_data(&b_recv, witness_sz + signature_sz);
    }
    alice_send_ios[0]->io->flush();
    uint64_t total_comm = 0;
    total_comm += alice_send_ios[0]->io->counter;
    total_comm += bob_send_ios[0]->io->counter;

    int t = emp::time_from(start);
    cout << "phase 2 -- party: " << party << "\t runtime: " << t << "\n";
    cout << "party: " << party << "\tcomm: " << total_comm << "\n";
    for (int i = 0; i < 1; ++i) {
        delete alice_send_ios[i]->io;
        delete bob_send_ios[i]->io;
    }
    return 0;
}