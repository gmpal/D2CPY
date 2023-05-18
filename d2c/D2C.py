import numpy as np 
import networkx as nx
from multiprocessing import Pool
from scipy.stats import skew
import pandas as pd

class D2C:
    def __init__(self, simulatedDAGs, rev=True, n_jobs=1):
        self.DAGs = simulatedDAGs.list_DAGs
        self.observations = simulatedDAGs.list_observations
        self.rev = rev
        self.X = None
        self.Y = None
        self.n_jobs = n_jobs

    def initialize(self):
        with Pool(processes=self.n_jobs) as pool:
            results = pool.starmap(
                self._compute_descriptor_parallel,
                zip(self.observations),
            )

        X_list, Y_list = zip(*results)
        self.X = pd.concat([pd.DataFrame(X) for X in X_list], axis=0)
        self.Y = pd.concat([pd.DataFrame(Y) for Y in Y_list], axis=0)

    def _compute_descriptor_parallel(self, observation):
        X = []
        Y = []

        edge_pairs = self.generate_edge_pairs("is.child")

        for edge_pair in edge_pairs:
            parent, child = edge_pair[0], edge_pair[1]
            descriptor = compute_descriptor(observation, parent, child)
            X.append(descriptor)
            Y.append(1)  # Label edge as "is.child"

            if self.rev:
                # Reverse edge direction
                descriptor_reverse = compute_descriptor(observation, child, parent)
                X.append(descriptor_reverse)
                Y.append(0)  # Label reverse edge as NOT "is.child"

        return X, Y
    
    def generate_edge_pairs(self, dependency_type):
        edge_pairs = []
        for DAG in self.DAGs:
            if dependency_type == "is.child":
                for parent_node, child_node in DAG.edges:
                    edge_pairs.append((parent_node, child_node))
        return edge_pairs
        


def compute_descriptor(D, ca, ef):
    De = {}

    # Normalize the data matrix
    D = (D - np.mean(D, axis=0)) / np.std(D, axis=0)

    
    # Compute descriptors
    De['correlation'] = np.corrcoef(D.iloc[:, ca], D.iloc[:, ef])[0, 1]
    De['skewness_ca'] = skew(D.iloc[:, ca])
    De['skewness_ef'] = skew(D.iloc[:, ef])
    De['mean_diff'] = np.mean(D.iloc[:, ca]) - np.mean(D.iloc[:, ef])

    return De
