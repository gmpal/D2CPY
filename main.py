#!/usr/bin/env python
# coding: utf-8

from d2c import SimulatedDAGs, D2C

N_JOBS = 1

n_dags = 1
n_observations = 10
n_nodes = 10

if __name__ == "__main__":

    simulated_dags = SimulatedDAGs(n_dags, n_observations, n_nodes, n_jobs=N_JOBS)
    simulated_dags.generate_dags()
    simulated_dags.simulate_observations()
    d2c = D2C(simulated_dags, n_jobs=N_JOBS)
    d2c.initialize()
    dataframe = d2c.get_df()
    score = d2c.get_score(metric="f1")

