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
    def __init__(self, n_series: int, n_observations: int, n_variables: int, maxlags=1, n_jobs: int = 1, random_state: int = 42):
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
        self.maxlags = maxlags
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
        initial_DAG = self._generate_single_dag()
        updated_DAG = self._update_dag_for_timestep(initial_DAG)
        data = pd.DataFrame(np.random.rand(self.maxlags, self.n_variables))
        for _ in range(1, self.n_observations):
            self._generate_timestep_observation(updated_DAG, data)
        return data, initial_DAG, updated_DAG


    def _update_dag_for_timestep(self, dag: nx.DiGraph) -> nx.DiGraph:
        """
        Updates the given DAG for a new timestep by adding past values as new nodes.
        
        Args:
            dag (nx.DiGraph): The original DAG.
            data (pd.DataFrame): The DataFrame containing past observations.
            timestep (int): The current timestep.

        Returns:
            nx.DiGraph: The updated DAG.
        """
        # Add past nodes and edges to the DAG
        past_dag = dag.copy(as_view=False)

        # Add past nodes and edges to the DAG
        for node in dag.nodes:
            for lag in range(1, self.maxlags + 1):
                past_node = f"{node}_t-{lag}"
                past_dag.add_node(past_node, **dag.nodes[node])  # Copy attributes from the original node
                if lag > 1: 
                    past_dag.add_edge(past_node, f"{node}_t-{lag-1}", weight=1, H='linear')
                weight = np.round(np.random.uniform(low=0, high=1), 2)
                past_dag.add_edge(past_node, node, weight=weight, H='linear')

                # Add edges from past nodes to current nodes that the original node had edges to
                for successor in dag.successors(node):
                    weight = np.round(np.random.uniform(low=0, high=1), 2)
                    past_dag.add_edge(past_node, successor, **dag.edges[node, successor])  # Copy attributes from the original edge

        return past_dag


    
    def _generate_timestep_observation(self, dag: nx.DiGraph, data: pd.DataFrame) -> pd.DataFrame:
        
        #inizialize the first row of the dataframe
        current_len = len(data)
        for node in nx.topological_sort(dag):
            if f"_t-" in str(node):
                variable = int(str(node)[0])
                timestamp = int(str(node)[-1])
                dag.nodes[node]['value'] = data.loc[len(data) - timestamp, variable] 
            else:
                parents = list(dag.predecessors(node))
                bias = dag.nodes[node]['bias']
                sigma = dag.nodes[node]['sigma']
                
                data.loc[current_len, node] = 0
                for parent in parents:
                    edge_data = dag.edges[parent, node]
                    data.loc[current_len, node] += self.compute_value(bias, edge_data, dag.nodes[parent]['value'])
                data.loc[current_len, node] += np.random.normal(scale=sigma)
                dag.nodes[node]['value'] = data.loc[current_len, node]
        

        

    def compute_value(self, value, edge_data, parent_value):
        weight = edge_data['weight'] 
        H = edge_data['H']
        if H == "linear":
            a = np.random.uniform(-1, 1, 2).reshape(2, 1)
            X = np.array([parent_value ** i for i in range(2)])    # data[node] += a0 * 1 + a1 * data[parent] 
            value += np.sum(X * a)
        elif H == "quadratic":
            a = np.random.uniform(-1, 1, 3).reshape(3, 1)
            X = np.array([parent_value ** i for i in range(3)])
            value += np.sum(X * a)
        elif H == "exponential":
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(0, 1)
            value += a * np.exp(b * parent_value)
        elif H == "logarithmic": #could capture a slowing or saturating effect.
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(1, 2)
            value += a * np.log(b + parent_value)
        elif H == "sigmoid": #could model a system that has a thresholding or saturating effect.
            a = np.random.uniform(-5, 5)
            b = np.random.uniform(0, 1)
            value += 1 / (1 + np.exp(-a * (parent_value - b)))
        elif H == "sinusoidal": #can model periodic effects
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(0, 2 * np.pi)
            value += a * np.sin(b + parent_value)

        return value 



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
    n_series = 3  # You can change this as needed
    n_observations = 10
    n_variables = 5
    # Testing with a single process
    generator = SimulatedTimeSeries(n_series, n_observations, n_variables, maxlags=3)
    generator.generate()
    dags = generator.get_dags()
    DAG = generator.get_updated_dags()[0]
    

