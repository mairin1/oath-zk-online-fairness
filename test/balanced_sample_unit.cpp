#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "zk-pof/utils.cpp"
#include "zk-pof/constant.cpp"
#include "zk-pof/fairness_zk.cpp"

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;


void unit_test_01_class_specific_rank_permutation(int party) {
    const int NUM_POINTS = 31;
    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    double proportion_a0 = 0.25;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, NUM_POINTS, 0.8, 0.35, proportion_a0);
    sensitive_attributes.push_back((Bit(0, ALICE)));

    int ind_sz = 5;
    ROZKRAM<BoolIO<NetIO>> * pi_in = init_pi_in(party, 32 * proportion_a0, ind_sz);
    vector<Integer> pi_out = vector<Integer>(32);
    
    class_specific_rank_permutation(32, Bit(0, ALICE), pi_in, pi_out, sensitive_attributes);
    cout << "class_specific_rank_permutation_v1: [ ";
    for (int i=0; i<32; ++i) {
        cout << pi_out[i].reveal<int>() << " ";
    }
    cout << "]\n";
    pi_in->check();
    delete pi_in;
    
}

void unit_test_02_class_specific_rank_permutation(int party) {
    const int NUM_POINTS = 32;
    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    double proportion_a0 = 0.25;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, NUM_POINTS, 0.8, 0.35, proportion_a0);
    std::default_random_engine rng(ZKPOF_SEED);
    std::shuffle(begin(sensitive_attributes), end(sensitive_attributes), rng);

    int ind_sz = 5;
    ROZKRAM<BoolIO<NetIO>> * pi_in = init_pi_in(party, 32 * proportion_a0, ind_sz);
    vector<Integer> pi_out = vector<Integer>(32);
    class_specific_rank_permutation(32, Bit(0, ALICE), pi_in, pi_out, sensitive_attributes);
    cout << "sensitive_attributes: [ ";
    for (int i=0; i<32; ++i) {
        cout << sensitive_attributes[i].reveal<bool>() << " ";
    }
    cout << "]\n";
    //cout << "csrp_v1 pi_in:  [ ";
    cout << "csrp_v1 pi_out: [ ";
    for (int i=0; i<32; ++i) {
        cout << pi_out[i].reveal<int>() << " ";
    }
    cout << "]\n";
    pi_in->check();
    delete pi_in;
}


void unit_test_01_class_balanced_sample(int party) {
    const int NUM_POINTS = 32;
    Bit TRU(1, PUBLIC);
    vector<Bit> predicted_outcomes;
    vector<Bit> sensitive_attributes;
    double proportion_a0 = 0.25;
    example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, NUM_POINTS, 0.8, 0.35, proportion_a0);
    std::default_random_engine rng(ZKPOF_SEED);
    std::shuffle(begin(sensitive_attributes), end(sensitive_attributes), rng);

    int ind_sz = 5;
    // init P0
    ROZKRAM<BoolIO<NetIO>> * pi_in_a0 = init_pi_in(party, 32 * proportion_a0, ind_sz);
    vector<Integer> pi_out_a0 = vector<Integer>(32);
    Bit temp_sa(0, ALICE);
    class_specific_rank_permutation(32, temp_sa, pi_in_a0, pi_out_a0, sensitive_attributes);

    // init P1
    ROZKRAM<BoolIO<NetIO>> * pi_in_a1 = init_pi_in(party, 32 * (1-proportion_a0), ind_sz);
    vector<Integer> pi_out_a1 = vector<Integer>(32);
    class_specific_rank_permutation(32, temp_sa^TRU, pi_in_a1, pi_out_a1, sensitive_attributes);

    // initialize S with all 0s
    vector<Bit> sample_vec;
    for (int i=0; i<NUM_POINTS; ++i) {
        sample_vec.push_back(Bit(0, PUBLIC));
    }
    class_balanced_sample(NUM_POINTS, 5, pi_out_a0, pi_out_a1, sample_vec);
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
    pi_in_a0->check();
    pi_in_a1->check();
    delete pi_in_a0;
    delete pi_in_a1;
}

/*
# for imbalanced classes, we can instead take a random sample from each of the two groups
# this can be done easily if we leak the size of each group membership
# 
# this can be done by seeding a PRG with fair coins.
# in detail: 
# commit to indices i \in I_0 such that:
# if d[i].a == 1, i = -1
# o/w d[i] \in [0, N_0], with d[i] != d[i']
# where N_0 is the number of points in a0
# and similarly define indices I_1
# and it must be the case that N_0 + N_1 = N, the size of the dataset.
# need to formalize that a bit better but I think the idea comes across
# 
# then, V can provide two random permutations, oh but I guess they need to know N_0 and N_1 to do so hmmmmmm
# yeah if they know N_0 and N_1 it's easy, just provide a random permutation on those 
# prove that there exists some sequence of original indices such that their I_0 indices are all ascending by 1
# maybe we can rebrand this as a summary statistics phase ?
# if *absolutely* necessary, we could use a polynomial equivalence check like in ZKRAM, but
# that seems a bit too high power for this use case.

# define oblivious protocol for a class-balanced sample:
# from full dataset D, randomly sample an initial set of size nu, call it S
# let b = <num a0 in sample> - <num a1 in sample>
# go to index i of S
# randomly (without replacement) pick a point D[j]
# if S[i] \in a0 and b > 0: 
#                if D[j] \in a1, replace S[i] with D[j]
#                   o/w do nothing
# else if S[i] \in a1 and b < 0:
#                if D[j] \in a0, replace S[i] with D[j]
#                   o/w do nothing
# analyze how many steps this requires to make a class-balanced sample

# better one, requires only linear pass through D:
# assign D indices I_0 and I_1 as before, prove their correctness
# obtain class-specific random permutations Pi_0 and Pi_1 on I_0 and I_1 using the following process:
# Pi_0(i) <- b_a0 * T[counter] // where T is a random permutation of [1, N_0]
# counter += b_a0
# so Pi_0(i) maps all entries in a0 to a unique entry in [1, N_0], and all entries in a1 to 0.
# linear pass through D -- for i in range(len(D)):
# let j = b_a0 * Pi_0(i) + b_a1 * Pi_1(i)  \\ map through Pi_0 if D[i] \in a0, map through Pi_1 o/w
# R[j] <- D[i] \\ update ZKRAM (D[i] need not be a ZKRAM, can just be an array)

# note: a random permutation on [1, N] with entries (N_0, N] ablated, is a random permutation on [1, N_0]
# just need to show that they have same rank
*/


int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);

    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    //ZKRAM<BoolIO<NetIO>> * sample_ram = new ZKRAM<BoolIO<NetIO>>(party, CONST_SAMPLE_INDEX_SZ, CONST_STEP_SZ, CONST_VAL_SZ);
    //unit_test_02_class_specific_rank_permutation(party);
    //Integer x = Integer(32, 3, PUBLIC);
    unit_test_01_class_balanced_sample(party);

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
