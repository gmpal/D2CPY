import numpy as np
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
# from multiprocessing import Pool
import pandas as pd
import random

from scipy.special import expit


from typing import List



class SimulatedDAGs:
    #TODO: implement verbosity, quantize
    def __init__(self, n_dags: int, n_observations: int, n_nodes: int, function_types: List[str] = ["linear", "quadratic", "sigmoid"], sdn: int = 0.2,
                 quantize: bool = False, additive: bool = True, verbose: bool = True, random_state: int = 42, n_jobs: int = 1):
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
        self.n_nodes = n_nodes
        self.function_types = function_types
        
        self.quantize = quantize
        self.additive = additive
        
        self.sdn = sdn
        self.verbose = verbose
        self.list_DAGs = []
        self.list_observations = []

        self.random_state = random_state
        self.n_jobs = n_jobs
        
        np.random.seed(self.random_state) #TODO: reposition this


    def generate_dags(self) -> None:
        """Generates ndag number of DAGs."""
        results = [self._generate_single_dag(i) for i in range(self.n_dags)]
        # if self.n_jobs == 1:
        #     results = [self._generate_single_dag(i) for i in range(self.n_dags)]
        # else:
        #     with Pool(processes=self.n_jobs) as pool:
        #         results = pool.map(self._generate_single_dag, range(self.n_dags))
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
        edges = [(i, j) for i in range(self.n_nodes) for j in range(i)]
        G.add_edges_from(edges)
        
        while not is_directed_acyclic_graph(G):
            # If it's not a DAG, remove a random edge
            edge_to_remove = random.choice(list(G.edges()))
            G.remove_edge(*edge_to_remove)

        for node in G.nodes:
            G.nodes[node]['bias'] = np.random.normal(loc=0, scale=1)
            G.nodes[node]['sigma'] = self.sdn
            G.nodes[node]['seed'] = self.random_state + index

        for edge in G.edges:
            G.edges[edge]['weight'] = np.random.uniform(low=0, high=1) #TODO: check weight implementation
            # G.edges[edge]['H'] = random.choice(self.function_types)
            G.edges[edge]['H'] = 'linear' #TODO: implement other functions

        return G

    def simulate_observations(self) -> None:
        """Simulates observations for all the DAGs."""
        results = [self._simulate_single_dag_observations(dag) for dag in self.list_DAGs]
        # if self.n_jobs == 1:
        #     results = [self._simulate_single_dag_observations(dag) for dag in self.list_DAGs]
        # else:
        #     with Pool(processes=self.n_jobs) as pool:
        #         results = pool.map(self._simulate_single_dag_observations, self.list_DAGs)
        self.list_observations = results

    def _simulate_single_dag_observations(self, dag: nx.DiGraph) -> pd.DataFrame:
        """
        Simulates observations for a single DAG.
        
        Args:
            dag (nx.DiGraph): The DAG for which to simulate observations.

        Returns:
            pd.DataFrame: A DataFrame containing the simulated observations.
        """
        data = pd.DataFrame(index=range(self.n_observations), columns=dag.nodes)

        for node in nx.topological_sort(dag):
            parents = list(dag.predecessors(node))
            node_data = dag.nodes[node]
            bias = node_data['bias']
            sigma = node_data['sigma']

            if not parents:
                data[node] = np.random.normal(loc=bias, scale=sigma, size=self.n_observations)
            else:
                data[node] = bias
                if self.additive: #TODO: implement nonadditive cases
                    for parent in parents:
                        edge_data = dag.edges[parent, node]
                        weight = edge_data['weight']
                        H = edge_data['H']
                        if H == "linear":
                            a = np.random.uniform(-1, 1, 2).reshape(2, 1)
                            X = np.array([data[parent] ** i for i in range(2)])    
                            data[node] += np.sum(X * a, axis=0)

                data[node] += np.random.normal(scale=sigma, size=self.n_observations)

        return data
    

   