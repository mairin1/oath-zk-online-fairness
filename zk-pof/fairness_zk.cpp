#ifndef _ZKPOF_FAIRNESS
#define _ZKPOF_FAIRNESS

#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "constant.cpp"
#include "utils.cpp"

using namespace emp;
using namespace std;

void check() {
    cout << "check\n";
}

double compute_bit_l2_norm(vector<Bit> & a, vector<Bit> & b, const int NUM_POINTS){
    // can xor the queries (equivalent to subtracting)
    // no need to square b/c we are only dealing with bits
    double ans = 0;
    for (int i=0; i<a.size(); i++){ //  for (int i=0; i<NUM_POINTS; i++){
        ans += (a[i] ^ b[i]).reveal();
    }
    return sqrt(ans);
}

// given a set of predicted outcomes and sensitive attributes, prove that the model is eps-individually fair an eps_thresh
// predicted outcomes and sensitive attributes should be 1 or 0 valued -- this can be proven at low cost,
// but for this subroutine we take it as given (as it can be ensured by proofs upstream of this function)
void certify_postproc_IF(vector<vector<Bit>> & queries, vector<Bit> & predicted_outcomes, vector<Bit> & sensitive_attributes, Integer eps_thresh, const int NUM_POINTS, bool verbose=false) {
    // initialize constant values and counters
    Integer ZERO = Integer(32, 0, PUBLIC);
    // Bit TRU = Bit(1, PUBLIC);
    Bit FAL = Bit(0, PUBLIC);
    vector<Bit> pairPasses;
    Integer HUNDRED_THOUSAND = Integer(32, 100000, PUBLIC); // for normalizing

    for (int i=0; i<NUM_POINTS; ++i) {
        for (int j=i+1; j<NUM_POINTS; j++) {
            Bit pass = 1;
            vector<Bit> queryI = queries[i];
            vector<Bit> queryJ = queries[j];
            // zero out sensitive values so they don't affect the norm
            // TODO: assuming sensitive values are in the first position of the query as in the paper
            queryI[0] = FAL;
            queryJ[0] = FAL;

            double l2_norm = compute_bit_l2_norm(queryI, queryJ, NUM_POINTS);
            Integer l2_norm_int = Integer(32, l2_norm * 100000, PUBLIC);
            if (eps_thresh.geq(l2_norm_int).reveal() == 1){
                pass = Bit(predicted_outcomes[i] == predicted_outcomes[j]);
            }
            // set values back
            queryI[0] = sensitive_attributes[i];
            queryJ[0] = sensitive_attributes[j];
            pairPasses.push_back(pass);
        }
    }
    // prove that for all pairs of inputs with l2 norm < eps, the model produces the same output
    Bit fair_check = 1;
    for (int i=0; i<pairPasses.size(); i++) {
        fair_check = fair_check & pairPasses[i];
    }
    if (verbose) {
        cout << "Fair? " << fair_check.reveal() << endl;
    }
}

// given a set of predicted outcomes and sensitive attributes, prove that the demographic parity gap is beneath a threshold.
// predicted outcomes and sensitive attributes should be 1 or 0 valued -- this can be proven at low cost,
// but for this subroutine we take it as given (as it can be ensured by proofs upstream of this function)
void certify_postproc_DP(vector<Bit> & predicted_outcomes, vector<Bit> & sensitive_attributes, Integer dp_gap_thresh, const int NUM_POINTS, bool verbose=false) {
    // initialize constant values and counters
    Integer ZERO = Integer(32, 0, PUBLIC);
    Bit TRU = Bit(1, PUBLIC);
    Integer count_sa_zero = Integer(32, 0, PUBLIC);
    Integer count_sa_one = Integer(32, 0, PUBLIC);
    Integer count_zero_pos = Integer(32, 0, PUBLIC);
    Integer count_one_pos = Integer(32, 0, PUBLIC);

    // count positive outcomes and class sizes in zero knowledge
    for (int i=0; i<NUM_POINTS; ++i) {
        // define indicator bits for sensitive attribute; sensitive attribute and positive outcome
        Bit sa_zero_indicator = sensitive_attributes[i] ^ TRU; // negate
        Bit sa_one_indicator = sensitive_attributes[i]; // not technically necessary to alias this, just showing my work
        Bit zero_pos_indicator = sa_zero_indicator & predicted_outcomes[i];
        Bit one_pos_indicator = sa_one_indicator & predicted_outcomes[i];

        // append indicator bits to LSB of integers for addition
        Integer int_sa_zero_indicator = Integer(32, 0, PUBLIC);
        int_sa_zero_indicator[0] = sa_zero_indicator;
        Integer int_sa_one_indicator = Integer(32, 0, PUBLIC);
        int_sa_one_indicator[0] = sa_one_indicator;
        Integer int_zero_pos_indicator = Integer(32, 0, PUBLIC);
        int_zero_pos_indicator[0] = zero_pos_indicator;
        Integer int_one_pos_indicator = Integer(32, 0, PUBLIC);
        int_one_pos_indicator[0] = one_pos_indicator;

        // add to counters
        count_sa_zero = count_sa_zero + int_sa_zero_indicator;
        count_sa_one = count_sa_one + int_sa_one_indicator;
        count_zero_pos = count_zero_pos + int_zero_pos_indicator;
        count_one_pos = count_one_pos + int_one_pos_indicator;
    }

    // evaluate fairness
    // prove that DP gap is underneath threshold via:
    // prove equation \theta >= | Pos_a0 / N_a0   -   Pos_a1 / N_a1 |
    // which can be rewritten \theta * N_a0 * N_a1 >= | Pos_a0 * N_a1 - Pos_a1 * N_a0 | (to save the divisions)
    Integer HUNDRED_THOUSAND = Integer(32, 100000, PUBLIC); // normalize DP thresh
    Integer temp = (count_zero_pos * count_sa_one - count_one_pos * count_sa_zero).abs() * HUNDRED_THOUSAND;
    Bit fair_check = (dp_gap_thresh * count_sa_zero * count_sa_one).geq(temp);
    if (verbose) {
        cout << "Fair? " << fair_check.reveal() << endl;
    }
}

// given a set of predicted classes and sensitive attributes, prove that the model satisfies demographic parity for each class
// per-class fairness: for each class k, verify that |P(Ŷ=k|A=0) - P(Ŷ=k|A=1)| <= threshold
// predicted classes should be in range [0, num_classes-1]
void certify_postproc_multiclass_DP(vector<Integer> & predicted_classes, vector<Bit> & sensitive_attributes, Integer dp_gap_thresh, const int NUM_POINTS, const int num_classes, bool verbose=false) {
    // initialize constant values
    Integer ZERO = Integer(32, 0, PUBLIC);
    Bit TRU = Bit(1, PUBLIC);
    Integer HUNDRED_THOUSAND = Integer(32, 100000, PUBLIC); // normalize DP thresh

    Bit all_classes_fair = Bit(1, PUBLIC); // all classes must pass fairness check

    // for each class k, check demographic parity independently
    for (int k=0; k<num_classes; ++k) {
        Integer count_sa_zero = Integer(32, 0, PUBLIC);
        Integer count_sa_one = Integer(32, 0, PUBLIC);
        Integer count_zero_class_k = Integer(32, 0, PUBLIC);
        Integer count_one_class_k = Integer(32, 0, PUBLIC);
        Integer class_k_int = Integer(32, k, PUBLIC);

        // count instances where class==k and class sizes in zero knowledge
        for (int i=0; i<NUM_POINTS; ++i) {
            // define indicator bits for sensitive attribute; sensitive attribute and prediction==k
            Bit sa_zero_indicator = sensitive_attributes[i] ^ TRU; // negate
            Bit sa_one_indicator = sensitive_attributes[i];
            Bit is_class_k = (predicted_classes[i] == class_k_int);
            Bit zero_class_k_indicator = sa_zero_indicator & is_class_k;
            Bit one_class_k_indicator = sa_one_indicator & is_class_k;

            // convert bits to integers for addition
            Integer int_sa_zero_indicator = Integer(32, 0, PUBLIC);
            int_sa_zero_indicator[0] = sa_zero_indicator;
            Integer int_sa_one_indicator = Integer(32, 0, PUBLIC);
            int_sa_one_indicator[0] = sa_one_indicator;
            Integer int_zero_class_k_indicator = Integer(32, 0, PUBLIC);
            int_zero_class_k_indicator[0] = zero_class_k_indicator;
            Integer int_one_class_k_indicator = Integer(32, 0, PUBLIC);
            int_one_class_k_indicator[0] = one_class_k_indicator;

            // add to counters
            count_sa_zero = count_sa_zero + int_sa_zero_indicator;
            count_sa_one = count_sa_one + int_sa_one_indicator;
            count_zero_class_k = count_zero_class_k + int_zero_class_k_indicator;
            count_one_class_k = count_one_class_k + int_one_class_k_indicator;
        }

        // evaluate fairness for class k
        // prove that DP gap is underneath threshold via:
        // θ * N_a0 * N_a1 >= | Pos_a0_k * N_a1 - Pos_a1_k * N_a0 |
        Integer temp = (count_zero_class_k * count_sa_one - count_one_class_k * count_sa_zero).abs() * HUNDRED_THOUSAND;
        Bit class_k_fair = (dp_gap_thresh * count_sa_zero * count_sa_one).geq(temp);

        if (verbose) {
            cout << "Class " << k << " fair? " << class_k_fair.reveal() << endl;
        }

        all_classes_fair = all_classes_fair & class_k_fair;
    }

    if (verbose) {
        cout << "All classes fair? " << all_classes_fair.reveal() << endl;
    }
}

// given a set of predicted classes and sensitive attributes, prove that the model is eps-individually fair
// for multiclass: similar inputs must receive the same predicted class
void certify_postproc_multiclass_IF(vector<vector<Bit>> & queries, vector<Integer> & predicted_classes, vector<Bit> & sensitive_attributes, Integer eps_thresh, const int NUM_POINTS, bool verbose=false) {
    // initialize constant values and counters
    Integer ZERO = Integer(32, 0, PUBLIC);
    Bit FAL = Bit(0, PUBLIC);
    vector<Bit> pairPasses;
    Integer HUNDRED_THOUSAND = Integer(32, 100000, PUBLIC); // for normalizing
    int query_len = queries[0].size(); // get the query vector size

    for (int i=0; i<NUM_POINTS; ++i) {
        for (int j=i+1; j<NUM_POINTS; j++) {
            Bit pass = 1;
            vector<Bit> queryI = queries[i];
            vector<Bit> queryJ = queries[j];
            // zero out sensitive values so they don't affect the norm
            // TODO: assuming sensitive values are in the first position of the query as in the paper
            queryI[0] = FAL;
            queryJ[0] = FAL;

            double l2_norm = compute_bit_l2_norm(queryI, queryJ, query_len);
            Integer l2_norm_int = Integer(32, l2_norm * 100000);
            if (eps_thresh.geq(l2_norm_int).reveal() == 1){
                pass = Bit(predicted_classes[i] == predicted_classes[j]);
            }

            // set values back
            queryI[0] = sensitive_attributes[i];
            queryJ[0] = sensitive_attributes[j];
            pairPasses.push_back(pass);
        }
    }
    // prove that for all pairs of inputs with l2 norm < eps, the model produces the same class
    Bit fair_check = 1;
    for (size_t i=0; i<pairPasses.size(); i++) {
        fair_check = fair_check & pairPasses[i];
    }
    if (verbose) {
        cout << "Multiclass IF Fair? " << fair_check.reveal() << endl;
    }
}

void sensitive_attr_check(vector<Integer> alpha_xs, vector<Integer> alpha_zeros, vector<Integer> alpha_ones, vector<Bit> xs, size_t n_queries) {
    Bit b_pass(1, PUBLIC);
    for (int i=0; i<n_queries; ++i) {
        Bit x_prime = (alpha_ones[i] == alpha_xs[i]);
        Bit b = (alpha_xs[i] == alpha_zeros[i]) ^ (x_prime);
        b_pass = b_pass & b & (x_prime == xs[i]);
    }
    cout << "sensitive attr check: " << b_pass.reveal() << endl;
}


// COMPONENTS OF CLASS-BALANCED RANDOM SAMPLE:

// ind_sz should be ceil( log_2(N0) )
ROZKRAM<BoolIO<NetIO>>* init_pi_in(int party, int N0, int ind_sz) {
    if (N0+1 >= (1<<ind_sz)) {
        cout << "ERROR: init_pi_in needs bigger ind_sz\n";
        //exit;
    }

    vector<int> xs;

    for(int i=0; i<N0; ++i) {
        xs.push_back(i);
    }
    std::default_random_engine rng(ZKPOF_SEED);
    std::shuffle(begin(xs), end(xs), rng);

    xs.push_back(-1);
    xs[N0] = xs[0];
    xs[0] = -1;

    for (; xs.size() < (1<<ind_sz);) {
        xs.push_back(-1);
    }
    //cout << "size: " << xs.size() << endl;
    vector<Integer> data;
    for (int i=0; i<xs.size(); ++i) {
        Integer temp = Integer(CONST_VAL_SZ, xs[i], PUBLIC);
        data.push_back(temp);
    }

    ROZKRAM<BoolIO<NetIO>> * out = new ROZKRAM<BoolIO<NetIO>>(party, ind_sz, CONST_VAL_SZ);
    out->init(data);
    return out;
}


// given random permutation:
// set all outputs that are above N0 to -1
/*
# obtain class-specific random permutations Pi_0 and Pi_1 on I_0 and I_1 using the following process:
# Pi_0(i) <- b_a0 * T[counter] // where T is a random permutation of [1, N_0]
# counter += b_a0
# so Pi_0(i) maps all entries in a0 to a unique entry in [1, N_0], and all entries in a1 to 0.
*/
// component of class balanced random sample
void class_specific_rank_permutation(int N, Bit permutation_sensitive_attr, ROZKRAM<BoolIO<NetIO>> * Pi_in, vector<Integer> & Pi_out, vector<Bit> & sensitive_attributes) {
    int ind_sz = ceil(log2(N));
    Integer rank_counter = Integer(ind_sz, 1, PUBLIC);
    for (int i=0; i<N; ++i) {
        Bit indicator = (permutation_sensitive_attr == sensitive_attributes[i]);
        Integer int_indicator = bit_to_int(indicator, ind_sz);
        Integer mask_indicator = bit_to_mask(indicator, ind_sz);
        Integer ind = mask_indicator & rank_counter; // 0 if in group, rank_counter o/w
        Pi_out[i] = Pi_in->read(ind); // if ind == 0, maps to -1, o/w maps to Pi_in[ class rank ]
        rank_counter = rank_counter + int_indicator; // ++ if in group, no-op o/w
    }
}

// makes sample_vec a vector indicating whether a row is included in the random sample
void class_specific_sample(int N, int nu, vector<Integer> & Pi_class_rank, vector<Bit> & sample_vec) {
    Integer NU = Integer(32, nu, PUBLIC);
    for (int i=0; i<N; ++i) {
        Bit temp = (Pi_class_rank[i] < NU) & (Pi_class_rank[i] > Integer(32, -1, PUBLIC));
        sample_vec[i] = temp;
    }
}


// creates a Bit vector S encoding a class-balanced sample of size 2 \nu
// S is guaranteed to include \nu entries from one class, and \nu entries from the other, selected uniform randomly from within the classes
// (an entry of the query database D[i] is in the sample iff S[i] == 1)
// works by filtering over two class-specific random permutations
void class_balanced_sample(int N, int nu, vector<Integer> & P0, vector<Integer> & P1, vector<Bit> & sample_vec) {
    Integer NU = Integer(32, nu, PUBLIC);
    Integer NEG_ONE = Integer(32, -1, PUBLIC);
    for (int i=0; i<N; ++i) {
        Bit temp = ((P0[i] < NU) & (P0[i] > NEG_ONE)) | ((P1[i] < NU) & (P1[i] > NEG_ONE));
        sample_vec[i] = temp;
    }
}

#endif
