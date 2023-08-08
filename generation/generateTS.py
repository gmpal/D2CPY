import sys
sys.path.append("..")
sys.path.append("../d2c/")
import numpy as np
from d2c.simulated import Simulated
from d2c.simulatedTimeSeries import SimulatedTimeSeries
import pickle

if __name__ == "__main__":
    n_series = 100  # You can change this as needed
    n_observations = 300
    n_variables = 3
    maxlags = 5
    generator = SimulatedTimeSeries(n_series,  n_observations, n_variables, maxlags, n_jobs=35)
    generator.generate()

    observations = generator.get_observations()
    dags = generator.get_dags()
    updated_dags = generator.get_updated_dags()

    #pickle everything
    with open('observations.pkl', 'wb') as f:
        pickle.dump(observations, f)

    with open('dags.pkl', 'wb') as f:
        pickle.dump(dags, f)

    with open('updated_dags.pkl', 'wb') as f:
        pickle.dump(updated_dags, f)



