"""
This module is responsible for generating time series data based on the specified parameters.
"""

import numpy as np
import warnings
import pickle
import networkx as nx
from d2c.data_generation.models import model_registry

class TSBuilder():
    def __init__(self, observations_per_time_series=200, maxlags=3, n_variables=5, time_series_per_process=10, processes_to_use=list(range(1, 21)), noise_std=0.1, max_neighborhood_size = 6, seed=42, verbose=True):
        """
        Initializes the TSBuilder object with the specified parameters.

        Args:
            observations_per_time_series (int): Number of observations per time series.
            maxlags (int): Maximum number of time lags.
            n_variables (int): Number of variables in the time series.
            time_series_per_process (int): Number of time series per process.
            processes_to_use (list): List of process IDs to use.
            noise_std (float): Standard deviation of the noise.
            max_neighborhood_size (int): Maximum size of the neighborhood.
            seed (int): Seed for random number generation.
            verbose (bool): Whether to print verbose output.
        """
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

        np.random.seed(seed)

    def build(self, max_attempts = 10):
        """
        Builds the time series data.

        Args:
            max_attempts (int): Maximum number of attempts to generate valid time series.

        Raises:
            ValueError: If a non-finite value is detected in the generated time series.

        """
        warnings.filterwarnings('error', category=RuntimeWarning)

        for process_id in self.processes_to_use:
            self.generated_observations[process_id] = {}
            self.generated_dags[process_id] = {}
            model_instance = model_registry.get_model(process_id)()
            max_time_lag = model_instance.get_maximum_time_lag() #each model has a different time lag

            for ts_index in range(self.ts_per_process):
                attempts = 0
                while attempts < max_attempts:
                    try:
                        W = np.random.normal(0, self.noise_std, (self.observations_per_time_series + max_time_lag, self.n_variables))
                        size_N_j = np.random.randint(1, self.max_neighborhood_size + 1, self.n_variables)
                        N_j = [np.random.choice(range(self.n_variables), size, replace=False) for size in size_N_j]
                        Y_n = np.empty(shape=(self.observations_per_time_series + max_time_lag, self.n_variables))
                        
                        # Initialize the first `max_time_lag` rows with starting values if needed
                        Y_n[:max_time_lag] = np.random.uniform(-1, 1, (max_time_lag, self.n_variables))
                        
                        for t in range(max_time_lag, self.observations_per_time_series + max_time_lag - 1):  # Adjusted to -1 to avoid index error
                            for j in range(self.n_variables):
                                Y_n[t+1, j] = model_instance.update(Y_n, t, j, N_j[j], W)
                                if not np.isfinite(Y_n[t+1, j]):
                                    raise ValueError("Non-finite value detected. Trying again.")
                        
                        if not np.any(np.isnan(Y_n)) and not np.any(np.isinf(Y_n)):
                            # Check if the generated observations are valid (no inf no nans)
                            self.generated_dags[process_id][ts_index] = model_instance.build_dag(T=self.maxlags, N_j=N_j, N=self.n_variables)
                            self.generated_observations[process_id][ts_index] = Y_n[max_time_lag:]
                            break # Valid observations generated, exit while loop
                    except (ValueError, OverflowError, RuntimeWarning) as e:
                        print("Non-finite value detected in the current TS. Generating a new TS...")
                        attempts += 1
                        if attempts == max_attempts:
                            print(e,f"Failed to generate valid TS for model {process_id}, TS index {ts_index} after {max_attempts} attempts. Try again with a different seed.")

                self.generated_dags[process_id][ts_index] = model_instance.build_dag(T=self.maxlags,N_j=N_j, N=self.n_variables)
                self.generated_observations[process_id][ts_index] = Y_n[max_time_lag:]

        # After the critical section, reset warnings to default behavior
        warnings.filterwarnings('default', category=RuntimeWarning)

    def _prepare_dags(self):
        """
        This method renames the nodes in the generated DAGs based on a different naming convention.
        Specifically, the nodes are renamed from "Y[t][j]" to "{j}_t-{maxlags-t}".
        It also assigns an index to each DAG based on the process ID and time series index.
        TODO: handle the case when not all processes are considered! 
        Returns:
            None
        """
        rename_dict = {f"Y[{t}][{j}]": f"{j}_t-{self.maxlags-t}" for t in range(self.maxlags+1) for j in range(self.n_variables)}
        for process_id in self.generated_dags:
            for ts_index in self.generated_dags[process_id]:
                self.generated_dags[process_id][ts_index] = nx.relabel_nodes(self.generated_dags[process_id][ts_index], rename_dict)
                self.generated_dags[process_id][ts_index].graph['index'] = (process_id - 1) * self.ts_per_process + ts_index

    def get_generated_observations(self):
        """
        Returns the generated observations.

        Returns:
            dict: A dictionary containing the generated observations.
        """
        return self.generated_observations
    
    def get_generated_dags(self):
        """
        Returns the generated DAGs.

        Returns:
            dict: A dictionary containing the generated DAGs.
        """
        self._prepare_dags()
        return self.generated_dags

    def to_pickle(self, path):
        """
        Saves the generated data to a pickle file.

        Args:
            path (str): Path to save the data.
        """
        self._prepare_dags()
        with open(path, 'wb') as f:
            pickle.dump((self.generated_observations, self.generated_dags), f)