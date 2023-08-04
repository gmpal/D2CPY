import numpy as np
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
from multiprocessing import Pool
import pandas as pd
import random

from typing import List

from simulated import Simulated

#TODO: the two classes share a lot, consider merging them into one class and inherit from it.

class SimulatedTimeSeries(Simulated):
    def __init__(self, n_series: int, n_observations: int, n_variables: int, n_jobs: int = 1, random_state: int = 42):
        """
        SimulatedTimeSeries is a class to generate time series data based on a Vector Autoregressive (VAR) model.
        
        Args:
            n_series (int): Number of time series to generate.
            n_observations (int): Number of observations per series.
            n_variables (int): Number of variables in each series.
            random_state (int): Seed for the random number generator. Defaults to 42.
        """
        self.n_series = n_series
        self.n_observations = n_observations
        self.n_variables = n_variables
        self.random_state = random_state
        self.list_time_series = []
        self.list_initial_dags = []
        self.list_updated_dags = []
        self.sdn = 0.1

        self.n_jobs = n_jobs

        np.random.seed(self.random_state)

    def generate(self):
        """
        Generates n_series number of time series.
        """
        if self.n_jobs == 1: 
            for _ in range(self.n_series):
                data, initial_DAG, updated_DAG = self._generate_single_time_series(_)
                self.list_time_series.append(data)
                self.list_initial_dags.append(initial_DAG)
                self.list_updated_dags.append(updated_DAG)
        else:
            with Pool(self.n_jobs) as pool:
                results = pool.map(self._generate_single_time_series, range(self.n_series))

            for data, initial_DAG, updated_DAG in results:
                self.list_time_series.append(data)
                self.list_initial_dags.append(initial_DAG)
                self.list_updated_dags.append(updated_DAG)

    def _generate_single_time_series(self, index: int = 0):
        """
        Generates a single time series.
        """
        # Initialize a DataFrame to hold the time series data
        data = pd.DataFrame(index=range(self.n_observations), columns=[f'Var{i}' for i in range(self.n_variables)])

        initial_DAG = self._generate_single_dag(0)
        # self.print_DAG(initial_DAG)
        data = self._generate_initial_observations(initial_DAG)
        updated_DAG = self._update_dag_for_timestep(initial_DAG, data, 1)
        # self.print_DAG(updated_DAG)

        for t in range(1, self.n_observations):
            data = self._generate_timestep_observations(updated_DAG, data, t)
            # data = pd.concat([data, timestep_data])
        return data, initial_DAG, updated_DAG



    def _generate_initial_observations(self, dag: nx.DiGraph) -> pd.DataFrame:
        """
        Generates the initial observations for a single DAG.
        
        Args:
            dag (nx.DiGraph): The DAG for which to simulate initial observations.

        Returns:
            pd.DataFrame: A DataFrame containing the initial observations.
        """
        data = pd.DataFrame(index=[0], columns=dag.nodes)

        for node in nx.topological_sort(dag):
            parents = list(dag.predecessors(node))
            node_data = dag.nodes[node]
            bias = node_data['bias']
            sigma = node_data['sigma']

            if not parents:
                data[node] = np.random.normal(loc=bias, scale=sigma, size = 1)
            else:
                data[node] = bias
                for parent in parents:
                    edge_data = dag.edges[parent, node]
                    # weight = edge_data['weight'] #TODO: check weight implementation
                    H = edge_data['H']
                    if H == "linear":
                            a = np.random.uniform(-1, 1, 2).reshape(2, 1)
                            X = np.array([data[parent] ** i for i in range(2)])    # data[node] += a0 * 1 + a1 * data[parent] 
                            data[node] += np.sum(X * a, axis=0)
                    elif H == "quadratic":
                        a = np.random.uniform(-1, 1, 3).reshape(3, 1)
                        X = np.array([data[parent] ** i for i in range(3)])
                        data[node] += np.sum(X * a, axis=0)
                    elif H == "exponential":
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(0, 1)
                        data[node] += a * np.exp(b * data[parent])
                    elif H == "logarithmic": #could capture a slowing or saturating effect.
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(1, 2)
                        data[node] += a * np.log(b + data[parent])
                    elif H == "sigmoid": #could model a system that has a thresholding or saturating effect.
                        a = np.random.uniform(-5, 5)
                        b = np.random.uniform(0, 1)
                        data[node] += 1 / (1 + np.exp(-a * (data[parent] - b)))
                    elif H == "sinusoidal": #can model periodic effects
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(0, 2 * np.pi)
                        data[node] += a * np.sin(b + data[parent])
                data[node] += np.random.normal(scale=sigma)

        return data


    def _update_dag_for_timestep(self, dag: nx.DiGraph, data: pd.DataFrame, timestep: int) -> nx.DiGraph:
        """
        Updates the given DAG for a new timestep by adding past values as new nodes.
        
        Args:
            dag (nx.DiGraph): The original DAG.
            data (pd.DataFrame): The DataFrame containing past observations.
            timestep (int): The current timestep.

        Returns:
            nx.DiGraph: The updated DAG.
        """
        past_dag = dag.copy(as_view=False)
        # print(data)
        # Add past nodes and edges to the DAG
        for node in dag.nodes:
            past_node = f"{node}_t-{timestep}"
            past_dag.add_node(past_node, **dag.nodes[node])  # Copy attributes from the original node)
            weight = np.round(np.random.uniform(low=0, high=1),2)
            past_dag.add_edge(past_node, node, weight=weight, H='linear')

            # Add edges from past nodes to current nodes that the original node had edges to
            for successor in dag.successors(node):
                weight = np.round(np.random.uniform(low=0, high=1),2)
                past_dag.add_edge(past_node, successor, **dag.edges[node, successor])  # Copy attributes from the original edge

        return past_dag

    def _generate_timestep_observations(self, dag: nx.DiGraph, data: pd.DataFrame, timestep: int) -> pd.DataFrame:
        """
        Generates observations for a single timestep.
        
        Args:
            dag (nx.DiGraph): The DAG for which to simulate observations.
            data (pd.DataFrame): The DataFrame containing past observations.
            timestep (int): The current timestep.

        Returns:
            pd.DataFrame: A DataFrame containing the observations.
        """
        data.loc[len(data)] = np.nan
        # print(data)
        for node in nx.topological_sort(dag):
            #if t not in node
            if f"_t-" not in str(node):
                parents = list(dag.predecessors(node))
                node_data = dag.nodes[node]
                bias = node_data['bias']
                sigma = node_data['sigma']

                # if not parents: #impossible
                #     obs.loc[timestep, node] = np.random.normal(loc=bias, scale=sigma)
                # else:
                data.loc[timestep, node] = bias
                for parent in parents:
                    edge_data = dag.edges[parent, node]
                    #weight = edge_data['weight'] #TODO: check weight implementation
                    H = edge_data['H']
                    parent_idx = int(str(parent)[0])
                    if H == "linear":
                            a = np.random.uniform(-1, 1, 2).reshape(2, 1)
                            X = np.array([data.loc[timestep-1, parent_idx] ** i for i in range(2)])    # data[node] += a0 * 1 + a1 * data[parent] 
                            data.loc[timestep, node] += np.sum(X * a, axis=0)
                    elif H == "quadratic":
                        a = np.random.uniform(-1, 1, 3).reshape(3, 1)
                        X = np.array([data.loc[timestep-1, parent_idx] ** i for i in range(3)])
                        data.loc[timestep, node] += np.sum(X * a, axis=0)
                    elif H == "exponential":
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(0, 1)
                        data.loc[timestep, node] += a * np.exp(b * data.loc[timestep-1, parent_idx])
                    elif H == "logarithmic": #could capture a slowing or saturating effect.
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(1, 2)
                        data.loc[timestep, node] += a * np.log(b + data.loc[timestep-1, parent_idx])
                    elif H == "sigmoid": #could model a system that has a thresholding or saturating effect.
                        a = np.random.uniform(-5, 5)
                        b = np.random.uniform(0, 1)
                        data.loc[timestep, node] += 1 / (1 + np.exp(-a * (data.loc[timestep-1, parent_idx] - b)))
                    elif H == "sinusoidal": #can model periodic effects
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(0, 2 * np.pi)
                        data.loc[timestep, node] += a * np.sin(b + data.loc[timestep-1, parent_idx])
                data.loc[timestep, node] += np.random.normal(scale=sigma)

        return data




    def get_observations(self) -> List[pd.DataFrame]:
        """
        Returns the generated time series.
        """
        return self.list_time_series

    def get_dags(self) -> List[nx.DiGraph]:
        """
        Returns the generated DAGs.
        #TODO: the main problem here is the following 
        Should the DAGs contain the past nodes or not?
        How does this impact D2C?
        The assumption now is that the DAGs do not contain the past nodes.
        """
        return self.list_initial_dags

    def get_updated_dags(self) -> List[nx.DiGraph]:
        """
        Returns the generated DAGs including the past nodes.
        """
        return self.list_updated_dags


if __name__ == "__main__":
    import time
    n_series = 1000  # You can change this as needed
    n_observations = 10
    n_variables = 5
    # Testing with a single process
    start_time = time.time()
    generator = SimulatedTimeSeries(n_series, n_observations, n_variables,  n_jobs=1)
    generator.generate()
    end_time = time.time()
    print(f"Time taken with a single process: {end_time - start_time:.2f} seconds")

    # Testing with multiprocessing, up to 6 cores
    for cores in range(2, 7):
        start_time = time.time()
        generator = SimulatedTimeSeries(n_series,  n_observations, n_variables, n_jobs=cores)
        generator.generate()
        end_time = time.time()
        print(f"Time taken with {cores} cores: {end_time - start_time:.2f} seconds")


