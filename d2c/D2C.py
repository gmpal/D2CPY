import numpy as np 
import networkx as nx
from multiprocessing import Pool
from scipy.stats import skew
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from d2c.simulatedDAGs import SimulatedDAGs

from typing import Union, Tuple, Any


class D2C:
    def __init__(self, simulatedDAGs: SimulatedDAGs, rev: bool = True, n_jobs: int = 1, random_state: int = 42) -> None:
        """
        Class for D2C analysis.

        D2C (Dependency to Causalilty) analysis is a method for inferring causal relationships
        from observational data using simulated directed acyclic graphs (DAGs) and computing 
        asymmetric descriptors from the observations associated with each DAG.

        Args:
            simulatedDAGs (SimulatedDAGs): An instance of the SimulatedDAGs class.
            rev (bool, optional): Whether to consider reverse edges. Defaults to True.
            n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.DAGs = simulatedDAGs.list_DAGs
        self.observations = simulatedDAGs.list_observations
        self.rev = rev
        self.X = None
        self.Y = None
        self.n_jobs = n_jobs
        self.random_state = random_state

    def initialize(self) -> None:
        """
        Initialize the D2C object by computing descriptors in parallel for all observations.

        """
        with Pool(processes=self.n_jobs) as pool:
            results = pool.starmap(
                self._compute_descriptor_parallel,
                zip(self.observations)
            )

        X_list, Y_list = zip(*results)
        self.X = pd.concat([pd.DataFrame(X) for X in X_list], axis=0)
        self.Y = pd.concat([pd.DataFrame(Y) for Y in Y_list], axis=0)

    def _compute_descriptor_parallel(self, observation: Any) -> Tuple[list, list]:
        """
        Compute descriptors in parallel for a given observation.

        Args:
            observation (Any): The observation data.

        Returns:
            Tuple[list, list]: The computed descriptors and corresponding labels.

        """
        X = []
        Y = []

        edge_pairs = self._generate_edge_pairs("is.child")

        for edge_pair in edge_pairs:
            parent, child = edge_pair[0], edge_pair[1]
            descriptor = self._compute_descriptor(observation, parent, child)
            X.append(descriptor)
            Y.append(1)  # Label edge as "is.child"

            if self.rev:
                # Reverse edge direction
                descriptor_reverse = self._compute_descriptor(observation, child, parent)
                X.append(descriptor_reverse)
                Y.append(0)  # Label reverse edge as NOT "is.child"

        return X, Y
    
    def _generate_edge_pairs(self, dependency_type: str) -> list:
        """
        Generate pairs of edges based on the dependency type.

        Args:
            dependency_type (str): The type of dependency.

        Returns:
            list: List of edge pairs.

        """
        edge_pairs = []
        for DAG in self.DAGs:
            if dependency_type == "is.child":
                for parent_node, child_node in DAG.edges:
                    edge_pairs.append((parent_node, child_node))
        return edge_pairs
        


    def _compute_descriptor(self, D: pd.DataFrame, ca: int, ef: int) -> dict:
        """
        Compute the descriptors based on the given data and column indices.

        Args:
            D (pd.DataFrame): The data matrix.
            ca (int): Column index of the cause variable.
            ef (int): Column index of the effect variable.

        Returns:
            dict: Dictionary containing the computed descriptors.

        """
        De = {}

        # Normalize the data matrix
        D = (D - np.mean(D, axis=0)) / np.std(D, axis=0)

        
        # Compute descriptors
        De['correlation'] = np.corrcoef(D.iloc[:, ca], D.iloc[:, ef])[0, 1]
        De['skewness_ca'] = skew(D.iloc[:, ca])
        De['skewness_ef'] = skew(D.iloc[:, ef])
        De['mean_diff'] = np.mean(D.iloc[:, ca]) - np.mean(D.iloc[:, ef])

        return De


    def get_df(self) -> pd.DataFrame:
        """
        Get the concatenated DataFrame of X and Y.

        Returns:
            pd.DataFrame: The concatenated DataFrame of X and Y.

        """
        return pd.concat([self.X,self.Y], axis=1)
    

    def get_score(self, model: RandomForestClassifier = RandomForestClassifier(), test_size: float = 0.2, metric: str = "accuracy") -> Union[float, None]:
        """
        Get the score of a machine learning model using the specified metric.

        Parameters:
            model (RandomForestClassifier): The machine learning model to evaluate.
            test_size (float): The proportion of the data to use for testing.
            metric (str): The evaluation metric to use (default is "accuracy"). Valid metrics are: 'accuracy', 'f1', 'precision', 'recall', 'auc'.

        Returns:
            float: The score of the model using the specified metric.
        
        Raises:
            ValueError: If an invalid metric is provided.

        """
        data = self.X
        labels = self.Y

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=1-test_size, test_size=test_size, random_state=self.random_state)

        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # Create an instance of the Random Forest classifier
        model = RandomForestClassifier(n_jobs=self.n_jobs, random_state=self.random_state)

        # Train the model
        model.fit(X_train, y_train)

        # Get the accuracy of the model
        if metric == "accuracy":
            return model.score(X_test, y_test)
        elif metric == "f1":
            return f1_score(y_test, model.predict(X_test))
        elif metric == "precision":
            return precision_score(y_test, model.predict(X_test))
        elif metric == "recall":
            return recall_score(y_test, model.predict(X_test))
        elif metric == "auc":
            return roc_auc_score(y_test, model.predict(X_test))
        else:
            raise ValueError("Invalid metric. Valid metrics are: 'accuracy', 'f1', 'precision', 'recall'")
