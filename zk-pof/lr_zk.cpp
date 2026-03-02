#ifndef _ZKPOF_LR
#define _ZKPOF_LR

#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"

using namespace emp;
using namespace std;

Float _sigmoid(Float z) {
    Float ONE = Float(1, PUBLIC);
    Float ret = ONE / (ONE + z.exp());
    return ret;
}

Float _linear(vector<Float> & input, vector<Float> & weights) {
    Float ZERO = Float(0, PUBLIC);
    Float ret = inner_product(begin(weights), end(weights), begin(input), ZERO);
    return ret;
}

Float binary_LR(vector<Float> & input, vector<Float> & weights, Float & ood_threshold) {
    Float logit = _linear(input, weights);
    Bit is_above_ood_thresh = ood_threshold.less_equal(logit);
    cout << "Above Threshold? " << is_above_ood_thresh.reveal() << endl;
    Float ret = _sigmoid(logit);
    return ret;
}

void fair_binary_LR(Bit & input_sa, vector<Float> & input_features, vector<Float> & weights, Float & t_a, Float & t_b, Bit & output) {
    Bit TRU(1, PUBLIC);
    Float logit = _linear(input_features, weights);
    Float score = _sigmoid(logit);
    Bit is_above_ta = score.less_equal(t_a);
    Bit is_above_tb = score.less_equal(t_b);
    Bit a_indicator = input_sa;
    Bit b_indicator = input_sa ^ TRU;
    output = (is_above_ta & a_indicator) | (is_above_tb & b_indicator);
}

vector<Float> _softmax(vector<Float> & logits, const int num_classes) {
    vector<Float> exp_logits;
    vector<Float> ret;
    Float sum = Float(0, PUBLIC);
    for (int c=0; c<num_classes; c++) {
        Float temp = logits[c].exp();
        sum = sum + temp;
        exp_logits.push_back(temp);
    }

    for (int c=0; c<num_classes; c++) {
        ret.push_back(exp_logits[c] / sum);
    }
    return ret;
}

vector<Float> softmax_LR(vector<Float> & input, vector<vector<Float>> & weights, Float & ood_threshold, const int num_classes) {
    Bit is_above_ood_thresh = Bit(0, PUBLIC);
    vector<Float> logits;
    for (int c=0; c<num_classes; c++) {
        cout << c << "\n";
        Float logit = _linear(input, weights[c]);
        logits.push_back(logit);
        is_above_ood_thresh = (ood_threshold.less_equal(logit)) | is_above_ood_thresh; // if any logits are above threshold, not ood
    }
    cout << "Above Threshold? " << is_above_ood_thresh.reveal() << endl;
    vector<Float> ret = _softmax(logits, num_classes);
    return ret;
}

// sz should be the size of the input vector
void _ReLU(vector<Float> & input, size_t sz) {
    Float ZERO = Float(0, PUBLIC);
    Bit TRU(true, PUBLIC);
    for (int i=0; i<sz; ++i){
        input[i] = input[i].If(input[i].less_than(ZERO), ZERO);
    }
}

void fair2layer_NN(size_t input_sz, size_t hr1_sz, size_t hr2_sz, Bit & input_sa, vector<Float> & input, vector<Float> & weights1, vector<Float> & weights2, Float & t_a, Float & t_b, Bit & output) {
    Float ZERO(0, PUBLIC);
    vector<Float> hr1; // first hidden representation
    for (int i=0; i<hr1_sz; ++i) {
        hr1.push_back(inner_product(begin(weights1), end(weights1), begin(input), ZERO));
    }
    _ReLU(hr1, hr1_sz);
    vector<Float> hr2; // second hidden representation
    for (int i=0; i<hr2_sz; ++i) {
        hr2.push_back(inner_product(begin(weights2), end(weights2), begin(hr1), ZERO));
    }
    _ReLU(hr2, hr2_sz);

    // output layer 
    Float logit = _linear(hr2, weights2); 
    //Float score = _sigmoid(logit);
    Float score = logit; // FairProof doesn't seem to use sigmoid so this is a more direct comparison

    Bit TRU(1, PUBLIC);
    Bit is_above_ta = score.less_equal(t_a);
    Bit is_above_tb = score.less_equal(t_b);
    Bit a_indicator = input_sa;
    Bit b_indicator = input_sa ^ TRU;
    output = (is_above_ta & a_indicator) | (is_above_tb & b_indicator);
}


void fair3layer_NN(size_t input_sz, size_t hr1_sz, size_t hr2_sz, size_t hr3_sz, Bit & input_sa, vector<Float> & input, vector<Float> & weights1, vector<Float> & weights2, vector<Float> & weights3, Float & t_a, Float & t_b, Bit & output) {
    Float ZERO(0, PUBLIC);
    vector<Float> hr1; // first hidden representation
    for (int i=0; i<hr1_sz; ++i) {
        hr1.push_back(inner_product(begin(weights1), end(weights1), begin(input), ZERO));
    }
    _ReLU(hr1, hr1_sz);
    vector<Float> hr2; // second hidden representation
    for (int i=0; i<hr2_sz; ++i) {
        hr2.push_back(inner_product(begin(weights2), end(weights2), begin(hr1), ZERO));
    }
    _ReLU(hr2, hr2_sz);
    vector<Float> hr3; // third hidden representation
    for (int i=0; i<hr3_sz; ++i) {
        hr3.push_back(inner_product(begin(weights3), end(weights3), begin(hr2), ZERO));
    }
    _ReLU(hr3, hr3_sz);


    // output layer 
    vector<Float> ol;
    ol.push_back(Float(1.5, ALICE));
    ol.push_back(Float(1.5, ALICE));
    //Float logit = _linear(hr3, ol); 
    //Float score = _sigmoid(logit);
    Float score = hr2[0]; // FairProof doesn't seem to use sigmoid so this is a more direct comparison

    Bit TRU(1, PUBLIC);
    Bit is_above_ta = score.less_equal(t_a);
    Bit is_above_tb = score.less_equal(t_b);
    Bit a_indicator = input_sa;
    Bit b_indicator = input_sa ^ TRU;
    output = (is_above_ta & a_indicator) | (is_above_tb & b_indicator);
}


void _batch_norm(vector<Float> & input, vector<Float> & bn_recip, vector<Float> & bn_subtracts, size_t sz){
    for (int i=0; i<sz; ++i) {
        input[i] = input[i] * bn_recip[i] - bn_subtracts[i];
    }
}

// batch norm divisors should contain *reciprocals* of divisor terms
vector<Float> tabular_model(vector<Float> & input, size_t hr1_sz, vector< vector<Float> > & hweights1, vector<Float> & batch_norm_divisors1, vector<Float> & batch_norm_subtracts1, size_t hr2_sz, vector< vector<Float>> & hweights2, vector<Float> & batch_norm_divisors2, vector<Float> & batch_norm_subtracts2, size_t hr3_sz, vector< vector<Float>> hweights3) {
    Float ZERO(0, PUBLIC);
    vector<Float> hr1; // hidden representation
    for (int i=0; i<hr1_sz; ++i) {
        hr1.push_back(inner_product(begin(hweights1[i]), end(hweights1[i]), begin(input), ZERO));
    }

    _ReLU(hr1, hr1_sz);
    _batch_norm(hr1, batch_norm_divisors1, batch_norm_subtracts1, hr1_sz);

    vector<Float> hr2;
    for (int i=0; i<hr2_sz; ++i) {
        hr2.push_back(inner_product(begin(hweights2[i]), end(hweights2[i]), begin(hr1), ZERO));
    }

    _ReLU(hr2, hr2_sz);
    _batch_norm(hr2, batch_norm_divisors2, batch_norm_subtracts2, hr2_sz);

    vector<Float> hr3;
    for (int i=0; i<hr3_sz; ++i) {
        hr3.push_back(inner_product(begin(hweights3[i]), end(hweights3[i]), begin(hr2), ZERO));
    }
    _ReLU(hr3, hr3_sz);
    return hr3;
}


#endif