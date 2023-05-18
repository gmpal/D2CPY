#!/usr/bin/env python
# coding: utf-8

from d2c import SimulatedDAGs, D2C

N_JOBS = 1
ndag = 1
n = 10
no_nodes = 10
function_types = ["linear", "quadratic", "sigmoid"]
quantize = True
seed = 1
sdn = 0.1
verbose = True
if __name__ == "__main__":

    simulated_dags = SimulatedDAGs(ndag, n, no_nodes, function_types, quantize, seed, sdn, verbose, n_jobs=N_JOBS)
    simulated_dags.generate_dags()
    simulated_dags.simulate_observations()
    d2c = D2C(simulated_dags, n_jobs=N_JOBS)
    d2c.initialize()
    dataframe = d2c.get_df()
    score = d2c.get_score(metric="f1")

