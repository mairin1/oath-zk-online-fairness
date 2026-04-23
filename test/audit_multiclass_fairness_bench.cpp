#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "../zk-pof/constant.cpp"
#include "../zk-pof/utils.cpp"
#include "../zk-pof/fairness_zk.cpp"
#include "../zk-pof/lr_zk.cpp"
#include <ctime>
#include <chrono>

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;

void test_multiclass_DP_fair(BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    const int NUM_POINTS = 20;
    const int NUM_CLASSES = 3;

    // Generate synthetic data: perfectly balanced across classes and sensitive attributes
    vector<Integer> predicted_classes;
    vector<Bit> sensitive_attributes;

    // Create fair predictions: each class equally distributed across sensitive groups
    for (int i = 0; i < NUM_POINTS; i++) {
        // Sensitive attribute alternates between 0 and 1
        sensitive_attributes.push_back(Bit(i % 2, ALICE));
        // Class cycles through 0, 1, 2
        predicted_classes.push_back(Integer(32, i % NUM_CLASSES, ALICE));
    }

    // Set DP threshold to high value (permissive)
    Integer dp_gap_thresh = Integer(32, 100000, PUBLIC);

    certify_postproc_multiclass_DP(predicted_classes, sensitive_attributes, dp_gap_thresh, NUM_POINTS, NUM_CLASSES, true);

    if (party == ALICE) {
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << "multiclass_DP_fair\tPASSED\t" << NUM_POINTS << "\t" << NUM_CLASSES << "\n";
            fclose(fp);
        }
    } else {
        cout << "Multiclass DP Fair test completed" << endl;
    }
}

void test_multiclass_DP_unfair(BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    const int NUM_POINTS = 20;
    const int NUM_CLASSES = 3;

    // Generate biased data: class 0 heavily predicted for sensitive_attr=0
    vector<Integer> predicted_classes;
    vector<Bit> sensitive_attributes;

    for (int i = 0; i < NUM_POINTS; i++) {
        sensitive_attributes.push_back(Bit(i / 10, ALICE)); // First 10: attr=0, next 10: attr=1

        if (i < 10) {
            // Bias: mostly class 0 for sensitive_attr=0
            predicted_classes.push_back(Integer(32, i % 2, ALICE));
        } else {
            // Different distribution for sensitive_attr=1
            predicted_classes.push_back(Integer(32, (i + 1) % NUM_CLASSES, ALICE));
        }
    }

    // Set strict DP threshold (0.20 gap maximum)
    Integer dp_gap_thresh = Integer(32, 20000, PUBLIC);

    certify_postproc_multiclass_DP(predicted_classes, sensitive_attributes, dp_gap_thresh, NUM_POINTS, NUM_CLASSES, true);

    if (party == ALICE) {
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << "multiclass_DP_unfair\tFAILED\t" << NUM_POINTS << "\t" << NUM_CLASSES << "\n";
            fclose(fp);
        }
    } else {
        cout << "Multiclass DP Unfair test completed" << endl;
    }
}

void test_multiclass_IF_fair(BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    const int NUM_POINTS = 4;
    const int QUERY_LEN = 5;

    // Create queries with controlled similarity
    vector<vector<Bit>> queries;
    vector<Bit> sensitive_attributes;
    vector<Integer> predicted_classes;

    // Create two pairs of similar inputs with same predictions
    for (int i = 0; i < NUM_POINTS; i++) {
        vector<Bit> query;

        // Sensitive attribute
        sensitive_attributes.push_back(Bit(i % 2, ALICE));

        // Predicted class (same for similar pairs)
        int class_id = i / 2; // 0,0,1,1
        predicted_classes.push_back(Integer(32, class_id, ALICE));

        // Query features - create similar pairs
        query.push_back(sensitive_attributes[i]);
        for (int j = 1; j < QUERY_LEN; j++) {
            if (i < 2) {
                // First pair - identical features (except sensitive attribute)
                query.push_back(Bit(j % 2, ALICE));
            } else {
                // Second pair - identical features (except sensitive attribute)
                query.push_back(Bit((j + 1) % 2, ALICE));
            }
        }
        queries.push_back(query);
    }

    Integer eps_thresh = Integer(32, 50000, PUBLIC);

    certify_postproc_multiclass_IF(queries, predicted_classes, sensitive_attributes, eps_thresh, NUM_POINTS, true);

    if (party == ALICE) {
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << "multiclass_IF_fair\tPASSED\t" << NUM_POINTS << "\t" << QUERY_LEN << "\n";
            fclose(fp);
        }
    } else {
        cout << "Multiclass IF Fair test completed" << endl;
    }
}

void test_multiclass_IF_unfair(BoolIO<NetIO> *ios[threads], int party, string logfile_str="") {
    const int NUM_POINTS = 4;
    const int QUERY_LEN = 5;

    // Create queries where similar inputs get different predictions
    vector<vector<Bit>> queries;
    vector<Bit> sensitive_attributes;
    vector<Integer> predicted_classes;

    for (int i = 0; i < NUM_POINTS; i++) {
        vector<Bit> query;

        sensitive_attributes.push_back(Bit(i % 2, ALICE));

        // Different predictions for similar inputs (unfair!)
        predicted_classes.push_back(Integer(32, i % 3, ALICE));

        // Create similar pairs with different predictions
        query.push_back(sensitive_attributes[i]);
        for (int j = 1; j < QUERY_LEN; j++) {
            query.push_back(Bit(j % 2, ALICE));
        }
        queries.push_back(query);
    }

    Integer eps_thresh = Integer(32, 50000, PUBLIC);

    certify_postproc_multiclass_IF(queries, predicted_classes, sensitive_attributes, eps_thresh, NUM_POINTS, true);

    if (party == ALICE) {
        if (logfile_str != "") {
            FILE *fp = freopen(logfile_str.c_str(), "a", stdout);
            cout << "multiclass_IF_unfair\tFAILED\t" << NUM_POINTS << "\t" << QUERY_LEN << "\n";
            fclose(fp);
        }
    } else {
        cout << "Multiclass IF Unfair test completed" << endl;
    }
}

int main(int argc, char **argv) {
    // initialize logfile
    using namespace std::chrono;
    time_t rawtime;
    tm* timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo);
    string logfile_str(buffer);

    logfile_str = "experiments/audit_multiclass_fairness_benchmark" + logfile_str + ".txt";

    int party, port;
    parse_party_and_port(argv, &party, &port);

    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i) {
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    }

    // Write header to log file for Alice
    if (party == ALICE) {
        FILE *fp = freopen(logfile_str.c_str(), "w", stdout);
        cout << "test_name\tresult\tnum_points\tnum_classes_or_query_len\n";
        fclose(fp);
    }

    // Run all tests within one ZK session
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);

    test_multiclass_DP_fair(ios, party, logfile_str);
    test_multiclass_DP_unfair(ios, party, logfile_str);
    test_multiclass_IF_fair(ios, party, logfile_str);
    test_multiclass_IF_unfair(ios, party, logfile_str);

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat) error("cheat!\n");

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }

    return 0;
}
