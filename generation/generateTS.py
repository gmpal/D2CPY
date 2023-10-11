import sys
sys.path.append("..")
sys.path.append("../d2c/")
import numpy as np
from d2c.simulated import Simulated
from d2c.simulatedTimeSeries import SimulatedTimeSeries
import pickle

if __name__ == "__main__":
    n_series = 50 # You can change this as needed
    n_observations = 150
    n_variables = 3
    maxlags = 3
    generator = SimulatedTimeSeries(n_series,  n_observations, n_variables, not_acyclic=True, maxlags = maxlags, n_jobs=10)
    generator.generate()

    observations = generator.get_observations()
    dags = generator.get_dags()
    updated_dags = generator.get_updated_dags()

    #pickle everything
    with open('../data/ts.pkl', 'wb') as f:
        pickle.dump((observations,dags,updated_dags), f)

