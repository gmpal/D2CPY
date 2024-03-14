import random
import pandas as pd
import numpy as np
from src.data_generation.models import model_registry

class TSBuilder():
    def __init__(self, observations_per_time_series=200, maxlags=3, n_variables=5, time_series_per_process=10, processes_to_use=list(range(1, 21)), noise_std=0.1, max_neighborhood_size = 6, seed=42, verbose=True):
        self.observations_per_time_series = observations_per_time_series
        self.maxlags = maxlags
        self.n_variables = n_variables
        self.ts_per_process = time_series_per_process
        self.noise_std = noise_std
        self.seed = seed
        self.verbose = verbose
        self.processes_to_use = processes_to_use
        self.max_neighborhood_size = min(max_neighborhood_size, n_variables)


        self.generated_observations = {}
        self.generated_dags = {}     

        random.seed(seed)

    def build(self):
        for model_id in self.processes_to_use:
            self.generated_observations[model_id] = {}
            self.generated_dags[model_id] = {}
            model_instance = model_registry.get_model(model_id)()
            max_time_lag = model_instance.get_maximum_time_lag() #each model has a different time lag

            for ts_index in range(self.ts_per_process):
                valid_observations = False
                while not valid_observations:

                    W = np.random.normal(0, self.noise_std, (self.observations_per_time_series + max_time_lag, self.n_variables))
                    size_N_j = np.random.randint(1, self.max_neighborhood_size + 1, self.n_variables)
                    N_j = [np.random.choice(range(self.n_variables), size, replace=False) for size in size_N_j]
                    Y_n = np.empty(shape=(self.observations_per_time_series + max_time_lag, self.n_variables))
                    
                    # Initialize the first `max_time_lag` rows with starting values if needed
                    Y_n[:max_time_lag] = np.random.uniform(-1, 1, (max_time_lag, self.n_variables))
                    
                    for t in range(max_time_lag, self.observations_per_time_series + max_time_lag - 1):  # Adjusted to -1 to avoid index error
                        for j in range(self.n_variables):
                            Y_n[t+1, j] = model_instance.update(Y_n, t, j, N_j[j], W)
                    
                    # Check if the generated observations are valid (no inf no nans)
                    valid_observations = not np.any(np.isnan(Y_n)) and not np.any(np.isinf(Y_n))

                self.generated_dags[model_id][ts_index] = model_instance.build_dag(T=self.maxlags,N_j=N_j, N=self.n_variables)
                self.generated_observations[model_id][ts_index] = Y_n[max_time_lag:]

        
    def get_generated_observations(self):
        return self.generated_observations
    
    def get_generated_dags(self):
        return self.generated_dags
