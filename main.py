from d2c.simulatedDAGs import SimulatedDAGs
from d2c.D2C import D2C
import pandas as pd
import time

N_JOBS = 1

# Usage example
ndag = 10
n = 10
no_nodes = 10
function_types = ["linear", "quadratic", "sigmoid"]
quantize = True
seed = 1
sdn = 0.1
verbose = True

if __name__ == "__main__":
    start = time.time()
    simulated_dags = SimulatedDAGs(ndag, n, no_nodes, function_types, quantize, seed, sdn, verbose, n_jobs=N_JOBS)
    simulated_dags.generate_dags()
    simulated_dags.simulate_observations()

    # Create an instance of the D2C class
    d2c = D2C(simulated_dags, n_jobs=N_JOBS)

    # Initialize the D2C instance
    d2c.initialize()

    # Access the generated X and Y values
    data = pd.concat([d2c.X,d2c.Y], axis=1)
    
    # Print the time taken rounded to 2 decimal places
    end = time.time()
    print("Time taken: " + str(round(end - start, 2)) + " seconds")