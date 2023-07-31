import numpy as np
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
# from multiprocessing import Pool
import pandas as pd
import random



from typing import List


import numpy as np
import pandas as pd
import networkx as nx
from typing import List

class SimulatedTimeSeries:
    def __init__(self, n_series: int, n_observations: int, n_variables: int, random_state: int = 42):
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

        #TODO: implement multiprocessing

        np.random.seed(self.random_state)

    def generate_time_series(self):
        """
        Generates n_series number of time series.
        """
        for _ in range(self.n_series):
            data, initial_DAG, updated_DAG = self._generate_single_time_series()
            self.list_time_series.append(data)
            self.list_initial_dags.append(initial_DAG)
            self.list_updated_dags.append(updated_DAG)

    def _generate_single_time_series(self):
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


    def _generate_single_dag(self, index: int) -> nx.DiGraph:
        """
        Generates a single directed acyclic graph (DAG).
        
        Args:
            index (int): The index number for the DAG.

        Returns:
            nx.DiGraph: Generated DAG.
        """
        G = nx.DiGraph()
        edges = [(i, j) for i in range(self.n_variables) for j in range(i)]
        G.add_edges_from(edges)
        
        while not is_directed_acyclic_graph(G):
            # If it's not a DAG, remove a random edge
            edge_to_remove = random.choice(list(G.edges()))
            G.remove_edge(*edge_to_remove)

        for node in G.nodes:
            G.nodes[node]['bias'] = np.round(np.random.normal(loc=0, scale=1),2)
            G.nodes[node]['sigma'] = 0.1
            #G.nodes[node]['sigma'] = self.sdn TODO: fix
            G.nodes[node]['seed'] = self.random_state + index

        for edge in G.edges:
            G.edges[edge]['weight'] = np.round(np.random.uniform(low=0, high=1),2) #TODO: check weight implementation
            # G.edges[edge]['H'] = random.choice(self.function_types)
            G.edges[edge]['H'] = 'linear' #TODO: implement other functions

        return G

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
                data[node] = np.random.normal(loc=bias, scale=sigma)
            else:
                data[node] = bias
                for parent in parents:
                    edge_data = dag.edges[parent, node]
                    weight = edge_data['weight']
                    H = edge_data['H']
                    if H == "linear":
                        data[node] += weight * data[parent]

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
                    weight = edge_data['weight']
                    H = edge_data['H']
                    if H == "linear":
                        data.loc[timestep, node] += weight * data.loc[timestep-1, int(str(parent)[0])]  # Update here

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


def main():
    # Instantiate SimulatedTimeSeries object
    sim_ts = SimulatedTimeSeries(n_series = 1 , n_observations=100, n_variables=4, random_state=42)

    # Generate timeseries
    sim_ts.generate_time_series()

    # Retrieve timeseries
    timeseries = sim_ts.get_time_series()

    # Print the timeseries
    print(timeseries)

if __name__ == "__main__":
    main()

