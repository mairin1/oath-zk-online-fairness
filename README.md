This repository details our proof of concept implementation of OATH, from the NeurIPS 2025 paper: Secure and Confidential Certificates of Online Fairness.

## Installation
=====
1. `wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py`
2. `python[3] install.py --deps --tool --ot --zk`
3. `gh repo clone cleverhans-lab/oath-zk-online-fairness`
4. `mkdir experiments`
5. `cmake . && make`

Test installation correctness using `./run bin/test_basic_zk` to execute a basic 'hello world'-like program in emp-zk.


## Benchmarks
=====

The service phase of OATH can be benchmarked using the code in `service_phase_bench.cpp` e.g. `./run bin/test_service_phase_bench`

The audit phase of OATH consists of a few parts, which we implement individually to produce an estimate of end-to-end runtime. 

1. The verified Measurement of Group Fairness step is benchmarked using `./run bin/test_audit_phase_fair_check_bench`
2. The verified Group Balanced Sampling is benchmarked using `./run bin/test_audit_phase_balanced_sample_bench`
3. The Sensitive Attribute Check is benchmarked using `./run bin/test_audit_phase_sensitive_attr_bench`
4. The Inference Correctness Check is benchmarked using `./run bin/audit_correctness_check_bench` (note that for larger models e.g. ResNet101, we estimate runtimes by using ZKP inference times reported in [Mystique](https://eprint.iacr.org/2021/730.pdf) as a drop in replacement for ZKP inference on the smaller models in our implementation).
5. The Consistency Correctness Check runtimes are computed using the method from [Mystique](https://eprint.iacr.org/2021/730.pdf) -- verification of n 64-bit blocks of input are given by one ZKP verification of SHA256 plus 2n-1 invocations of [LowMC](https://eprint.iacr.org/2016/687). These can be respectively benchmarked using `./run bin/test_sha256` and `./run bin/test_lowmc` .