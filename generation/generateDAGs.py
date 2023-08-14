import sys
sys.path.append("..")
sys.path.append("../d2c/")
import numpy as np
from d2c.simulated import Simulated
from d2c.simulatedDAGs import SimulatedDAGs
import pickle
import multiprocessing as mp

if __name__ == "__main__":
    n_dags = 5
    # Other parameters that might be needed for your class, adjust as needed
    n_observations = 150
    n_variables = 10
    generator = SimulatedDAGs(n_dags,  n_observations, n_variables, n_jobs=10)
    generator.generate()

    observations = generator.get_observations()
    dags = generator.get_dags()

    #pickle everything
    with open('../data/dag.pkl', 'wb') as f:
        pickle.dump((observations,dags), f)




