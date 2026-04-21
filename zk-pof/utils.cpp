#ifndef _ZKPOF_UTILS
#define _ZKPOF_UTILS

#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "constant.cpp"
#include <algorithm>
#include <random>

using namespace emp;
using namespace std;




vector<Float> gen_dummy_vec(size_t sz, float v) {
    vector<Float> ret;
    for (int i=0; i<sz; ++i) {
        ret.push_back(Float(v, ALICE));
    }
    return ret;
}


Float bit_to_float(Bit input) {
    Float ret(0.0, PUBLIC);
    Float ONE(1.0, PUBLIC);
    for (int i=0; i<32; ++i) {
        ret[i] = ONE[i] & input;
        //cout << i << ": " << ONE[i].reveal() << "  " << input.reveal() << endl;
    }
    return ret;
}

Integer float_argmax(vector<Float> & xs, Integer & argmax_output, Float & max_output) {
    Integer ret(32, 0, ALICE);
    size_t n = xs.size();
    Float curr_max = xs[0];
    for (int i=1; i<n; ++i) {
        Integer curr_ind(32, i, PUBLIC);
        Bit flag = (curr_max.less_than(xs[i]));
        ret = ret.select(flag, curr_ind);
        //#ifdef DEBUG
        //cout << "iteration " << i << ":" << endl;
        //cout << "curr_max: " << curr_max.reveal<double>() << endl;
        //cout << "xs[i]: " << xs[i].reveal<double>() << endl;
        //cout << "flag: " << flag.reveal() << endl;
        //cout << "ret (post select): " << ret.reveal<int>() << endl;
        //#endif
        Float t = bit_to_float(flag);
        Float not_t = bit_to_float(flag ^ Bit(1,PUBLIC));
        curr_max = curr_max * not_t + xs[i] * t;
    }
    argmax_output = ret;
    max_output = curr_max;
    return ret;
}

vector< vector<Float> > gen_dummy_weights(size_t in_sz, size_t out_sz, float v) {
    vector< vector<Float> > ret;
    for (int i=0; i<out_sz; ++i) {
        vector<Float> t;
        for (int j=0; j<in_sz; ++j) {
            t.push_back(Float(v, ALICE));
        }
        ret.push_back(t);
    }
    return ret;
}


// appends a private Bit to an Integer for verified arithmetic operations
// 32 bits by default
Integer bit_to_int(Bit input, int int_sz=32) {
    Integer x = Integer(int_sz, 0, PUBLIC);
    x[0] = input;
    return x;
}

// if given a True Bit, returns an Int with binary rep 111...111
// if given a False Bit, returns an Int with binary rep 000...000
Integer bit_to_mask(Bit input, int int_sz=32) {
    Integer x = Integer(int_sz, 0, PUBLIC);
    for (int i=0; i<int_sz; ++i) {
        x[i] = input;
    }
    return x;
}

// helper function for unit testing, generates bit vectors that simulate class and success probability distributions
// predicted_outcomes and sensitive_attributes should be vector<Bit>s that have just been initialized
// a0_pos is the proportion of points with class a0 that will have predicted outcome 1
// a1_pos is same for points with class a1
// sa_split gives the proportion of points with class a0
// example usage:
/*    
vector<Bit> predicted_outcomes;
vector<Bit> sensitive_attributes;
const int NUM_POINTS = 20;
example_bit_vectors_DP(predicted_outcomes, sensitive_attributes, NUM_POINTS, 0.8, 0.35, 0.5);
*/

vector<Integer> Int_zeros(size_t n, size_t int_sz=32) {
    vector<Integer> out;
    for (int i=0; i<n; ++i) {
        out.push_back(Integer(int_sz, 0, ALICE));
    }
    return out;
}

vector<Integer> Int_ones(size_t n, size_t int_sz=32) {
    vector<Integer> out;
    for (int i=0; i<n; ++i) {
        out.push_back(Integer(int_sz, 1, ALICE));
    }
    return out;
}

vector<Bit> Bit_vec(size_t n, bool val) {
    vector<Bit> out;
    for (int i=0; i<n; ++i) {
        out.push_back(Bit(val, ALICE));
    }
    return out;
}

void example_bit_vectors_DP(vector<Bit> & predicted_outcomes, vector<Bit> & sensitive_attributes, int num_points, double a0_pos, double a1_pos, double sa_split, bool verbose=false) {
    int num_s0 = num_points * sa_split;
    int num_s1 = num_points - num_s0;
    int num_s0_pos = num_s0 * a0_pos;
    //int num_s0_neg = num_s0 - num_s0_pos;
    int num_s1_pos = num_s1 * a1_pos;
    //int num_s1_neg = num_s1 - num_s1_pos;

    int s0_count = 0;
    int s0_pos_count = 0;
    int s1_pos_count = 0;
    for (int i=0; i<num_points; i++) {
        if (s0_count < num_s0) {
            sensitive_attributes.push_back(Bit(0, ALICE));
            s0_count++;
            if (s0_pos_count < num_s0_pos) {
                predicted_outcomes.push_back(Bit(1, ALICE));
                s0_pos_count++;
            } else {
                predicted_outcomes.push_back(Bit(0, ALICE));
            } 
        } else {
            sensitive_attributes.push_back(Bit(1, ALICE));
            if (s1_pos_count < num_s1_pos) {
                predicted_outcomes.push_back(Bit(1, ALICE));
                s1_pos_count++;
            } else {
                predicted_outcomes.push_back(Bit(0, ALICE));
            }
        }
    }
    if (verbose) {
        cout << "predicted outcomes:   [";
        for (int i=0; i<num_points; ++i) {
            cout << " " << predicted_outcomes[i].reveal() << " ";
        }
        cout << "]\n";
        cout << "sensitive attributes: [";
        for (int i=0; i<num_points; ++i) {
            cout << " " << sensitive_attributes[i].reveal() << " ";
        }
        cout << "]\n";
    }
}

void example_queries_IF(vector<vector<Bit>> & queries, vector<Bit> & sensitive_attributes, vector<Bit> & predicted_outcomes, int num_points, int query_len){
    // should be fair for certain eps values
    for (int i=0; i<num_points; i++){
        vector<Bit> query;
        query.push_back(sensitive_attributes[i]);
        // only one position + sensitive attribute should be different for queries with the same outcome
        bool pred_outcome = (predicted_outcomes[i] == Bit(0)).reveal();
        for (int j=1; j<query_len; j++){
            bool bit_val;
            if (j==0){
                bit_val = (i ^ j) % 2;
            } else if (pred_outcome){
                bit_val = 0;
            } else {
                bit_val = 1;
            }
            query.push_back(Bit(bit_val));
        }
        queries.push_back(query);
    }
}

#endif