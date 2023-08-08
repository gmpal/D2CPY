from abc import ABC, abstractmethod
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
import random
import numpy as np
import pandas as pd
from typing import List

# Base class
class Simulated(ABC):

    FUNCTION_TYPES = ["linear", "quadratic"]

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
        edges = [(i, j) for i in range(self.n_variables) for j in range(i)]

        complex = random.random() < 0.5
        if complex: 
            
            G.add_edges_from(edges)
            
            while not is_directed_acyclic_graph(G):
                # If it's not a DAG, remove a random edge
                edge_to_remove = random.choice(list(G.edges()))
                G.remove_edge(*edge_to_remove)
        else:  # simple
            random.shuffle(edges)

            for edge in edges:
                G.add_edge(*edge)
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(*edge)

        for node in G.nodes:
            G.nodes[node]['bias'] = np.random.normal(loc=0, scale=1)
            G.nodes[node]['sigma'] = self.sdn
            G.nodes[node]['seed'] = self.random_state

        for edge in G.edges:
            G.edges[edge]['weight'] = np.random.uniform(low=0, high=1) #TODO: check weight implementation
            G.edges[edge]['H'] = random.choice(self.FUNCTION_TYPES) 

        return G

