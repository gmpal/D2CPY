from abc import ABC, abstractmethod
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
import random
import numpy as np
import pandas as pd
from typing import List
from utils import print_DAG

# Base class
class Simulated(ABC):

    #TODO: implement verbosity, quantize
    def __init__(self, n_dags: int, n_observations: int, n_variables: int, function_types: List[str] = ["linear", "quadratic", "sigmoid"], sdn: int = 0.2,
                 verbose: bool = True, random_state: int = 42, n_jobs: int = 1):
        """
        SimulatedDAGs is a class to generate directed acyclic graphs (DAGs) and simulate observations on them.

        Args:
            n_dags (int): Number of DAGs to generate.
            n (int): Number of observations to generate per DAG.
            no_nodes (int): Number of nodes in each DAG.
            function_types (List[str]): List of function types for DAGs.
            quantize (bool): Whether to quantize the observations. Defaults to False.
            additive (bool): if TRUE the output is the sum of the H transformation of the inputs, othervise it is the H transformation of the sum of the inputs.
            sdn (int): Standard deviation of noise in observations. Defaults to None.
            verbose (bool): Whether to print verbose output. Defaults to False.
            random_state (int): Seed for the random number generator. Defaults to None.
            n_jobs (int): Number of jobs to run in parallel. Defaults to 1.
        """
        self.n_dags = n_dags
        self.n_observations = n_observations
        self.n_variables = n_variables
        
        self.sdn = sdn
        self.function_types = function_types

        self.list_initial_dags = []
        self.list_observations = []

        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def get_observations(self) -> List[pd.DataFrame]:
        pass

    def _generate_single_dag(self) -> nx.DiGraph:
        """
        Generates a single directed acyclic graph (DAG).

        Args:
            index (int): The index number for the DAG.

        Returns:
            nx.DiGraph: Generated DAG.
        """

        # randomly at 50/50
        G = nx.DiGraph()
        for i in range(self.n_variables):
            G.add_node(i)

        edges = [(i, j) for i in range(self.n_variables) for j in range(self.n_variables) if i != j]
            
        G.add_edges_from(edges)

        while not is_directed_acyclic_graph(G):
            # If it's not a DAG, remove a random edge
            edge_to_remove = random.choice(list(G.edges()))
            G.remove_edge(*edge_to_remove)


        for node in G.nodes:
            # G.nodes[node]['bias'] = np.round(np.random.uniform(low=-0.1, high=0.1),5)
            G.nodes[node]['bias'] = 0
            G.nodes[node]['sigma'] = self.sdn
            G.nodes[node]['seed'] = self.random_state

        for edge in G.edges:
            G.edges[edge]['weight'] = np.round(np.random.uniform(low=-0.5, high=0.5),5)
            G.edges[edge]['H'] = random.choice(self.function_types)

        return G

