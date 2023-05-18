import numpy as np
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
from multiprocessing import Pool
import pandas as pd
import random

from typing import List, Union, Optional


class SimulatedDAGs:
    def __init__(self, ndag: int, n: int, no_nodes: int, function_types: List[str], quantize: bool = False, 
                 seed: Optional[int] = None, sdn: Optional[int] = None, verbose: bool = False, n_jobs: int = 1):
        """
        SimulatedDAGs is a class to generate directed acyclic graphs (DAGs) and simulate observations on them.

        Args:
            ndag (int): Number of DAGs to generate.
            n (int): Number of observations to generate per DAG.
            no_nodes (int): Number of nodes in each DAG.
            function_types (List[str]): List of function types for DAGs.
            quantize (bool, optional): Whether to quantize the observations. Defaults to False.
            seed (Optional[int], optional): Seed for the random number generator. Defaults to None.
            sdn (Optional[int], optional): Standard deviation of noise in observations. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
        """
        self.ndag = ndag
        self.n = n
        self.no_nodes = no_nodes
        self.function_types = function_types



        self.ndag = ndag
        self.n = n
        self.no_nodes = no_nodes
        self.function_types = function_types
        self.quantize = quantize
        self.seed = seed
        self.sdn = sdn
        self.verbose = verbose
        self.list_DAGs = []
        self.list_observations = []
        self.n_jobs = n_jobs

    def generate_dags(self) -> None:
        """Generates ndag number of DAGs."""
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._generate_single_dag, range(self.ndag))
        self.list_DAGs = results

    def _generate_single_dag(self, index: int) -> nx.DiGraph:
        """
        Generates a single directed acyclic graph (DAG).
        
        Args:
            index (int): The index number for the DAG.

        Returns:
            nx.DiGraph: Generated DAG.
        """
        G = nx.DiGraph()
        edges = [(i, j) for i in range(self.no_nodes) for j in range(i)]
        G.add_edges_from(edges)
        
        while not is_directed_acyclic_graph(G):
            # If it's not a DAG, remove a random edge
            edge_to_remove = random.choice(list(G.edges()))
            G.remove_edge(*edge_to_remove)

        return G

    def simulate_observations(self) -> None:
        """Simulates observations for all the DAGs."""
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._simulate_single_dag_observations, self.list_DAGs)
        self.list_observations = results

    def _simulate_single_dag_observations(self, dag: nx.DiGraph) -> pd.DataFrame:
        """
        Simulates observations for a single DAG.
        
        Args:
            dag (nx.DiGraph): The DAG for which to simulate observations.

        Returns:
            pd.DataFrame: A DataFrame containing the simulated observations.
        """
        N = self.n  # number of observations to generate
        data = pd.DataFrame(index=range(N), columns=dag.nodes)

        for node in nx.topological_sort(dag):
            parents = list(dag.predecessors(node))
            if parents:
                # For simplicity, we'll use edge weights as coefficients in a linear model.
                # coefficients = np.array([dag.edges[parent, node]['weight'] for parent in parents])

                # For simplicity, we'll use random coefficients in a linear model.
                coefficients = np.random.normal(size=len(parents))

                parent_values = data[parents]
                # Simulate data for this node based on its parents.
                data[node] = parent_values.dot(coefficients) + np.random.normal(size=N)
            else:
                # If the node has no parents, simulate data from a standard normal distribution.
                data[node] = np.random.normal(size=N)

        return data
    
