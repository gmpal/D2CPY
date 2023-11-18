import argparse
import sys
sys.path.append("..")
sys.path.append("../d2c/")
import numpy as np
from d2c.simulated import Simulated
from d2c.simulatedTimeSeries import SimulatedTimeSeries
import pickle

def generate_time_series(n_series, n_observations, n_variables, maxlags, not_acyclic, n_jobs, name, random_state):
    generator = SimulatedTimeSeries(n_series, n_observations, n_variables, not_acyclic=not_acyclic, maxlags=maxlags, n_jobs=n_jobs, random_state=random_state)
    generator.generate()
    observations = generator.get_observations()
    dags = generator.get_dags()
    updated_dags = generator.get_updated_dags()
    #pickle everything
    with open(f'../data/{name}.pkl', 'wb') as f:
        pickle.dump((observations, dags, updated_dags), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Simulated Time Series')
    parser.add_argument('--n_series', type=int, default=100, help='Number of series')
    parser.add_argument('--n_observations', type=int, default=150, help='Number of observations per series')
    parser.add_argument('--n_variables', type=int, default=3, help='Number of variables per observation')
    parser.add_argument('--maxlags', type=int, default=3, help='Maximum lags for the time series')
    parser.add_argument('--not_acyclic', type=bool, default=True, help='Whether the DAGs should allow cyclic cause-effect pairs when looking at the past')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for parallel processing')
    parser.add_argument('--name', type=str, default='ts3', help='Name of the file to save the data')
    parser.add_argument('--random_state', type=int, default=0, help='Random state for reproducibility')


    args = parser.parse_args()

    generate_time_series(args.n_series, args.n_observations, args.n_variables, args.maxlags, args.not_acyclic, args.n_jobs, args.name, args.random_state)
