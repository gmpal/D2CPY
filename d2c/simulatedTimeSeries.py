import numpy as np
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
from multiprocessing import Pool
import pandas as pd


from typing import List

from simulated import Simulated

#TODO: the two classes share a lot, consider merging them into one class and inherit from it.

class SimulatedTimeSeries(Simulated):
    
    def __init__(self, n_dags: int, n_observations: int, n_variables: int, not_acyclic: bool = False, maxlags: int = 1,  n_jobs: int = 1, random_state: int = 42, function_types: list = ["linear"], sdn: int = 0.001):
        super().__init__(n_dags, n_observations, n_variables, not_acyclic=not_acyclic, n_jobs=n_jobs, random_state=random_state, function_types=function_types, sdn=sdn)
        self.list_updated_dags = []
        self.maxlags = maxlags


    def generate(self):
        """
        Generates n_dags number of time series.
        """
        if self.n_jobs == 1: 
            for _ in range(self.n_dags):
                data, initial_DAG, updated_DAG = self._generate_single_time_series(_)
                self.list_observations.append(data)
                self.list_initial_dags.append(initial_DAG)
                self.list_updated_dags.append(updated_DAG)
        else:
            with Pool(self.n_jobs) as pool:
                results = pool.map(self._generate_single_time_series, range(self.n_dags))
            
            for data, initial_DAG, updated_DAG in results:
                self.list_observations.append(data)
                self.list_initial_dags.append(initial_DAG)
                self.list_updated_dags.append(updated_DAG)

    def _generate_single_time_series(self, index: int = 0):
        """
        Generates a single time series.
        """
        # print(index)
        # Initialize a DataFrame to hold the time series data
        # pick a random lag between 1 and maxlags
        # np.random.seed(self.random_state + index)
        current_lag = np.random.randint(1, self.maxlags + 1)
        # print(f"current lag: {current_lag}")
        initial_DAG = self._generate_single_dag()
        updated_DAG = self._update_dag_for_timestep(initial_DAG, current_lag, index)
        data = pd.DataFrame(2*np.random.rand(current_lag, self.n_variables)-1).round(5)
        # print(data)
        for _ in range(1, self.n_observations):
            self._generate_timestep_observation(updated_DAG, data)
        # print(index, "done")
        return data, initial_DAG, updated_DAG

    def _update_dag_for_timestep(self, dag: nx.DiGraph, current_lag: int, index: int) -> nx.DiGraph:
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
        for node in nx.topological_sort(dag):
            for lag in range(1, current_lag + 1):
                past_node = f"{node}_t-{lag}"
                past_dag.add_node(past_node, **dag.nodes[node])  # Copy attributes from the original node
                weight = np.round(np.random.uniform(low=-0.5, high=0.5),5)
                h = self.function_types[np.random.randint(0, len(self.function_types))]
                past_dag.add_edge(past_node, node, weight=weight, H=h)
                if lag > 1: 
                    past_dag.add_edge(past_node, f"{node}_t-{lag-1}", weight=weight, H=h)

                # Add edges from past nodes to current nodes that the original node had edges to
                for successor in dag.successors(node):
                    past_dag.add_edge(past_node, successor, **dag.edges[node, successor])  # Copy attributes from the original edge

        #TESTING ONLY THE PAST
        for node in nx.topological_sort(dag):
            if f"_t-" not in str(node):
                #remove all successors 
                successors = list(past_dag.successors(node))
                if len(successors) > 0:
                    for successor in successors:
                        past_dag.remove_edge(node, successor)

        if self.not_acyclic: #if you want to add relationships between past nodes that would be acyclic when added to the current dag
                #select a couple of nodes in dag 
                nodes = list(dag.nodes)
                
                already_selected_couples = []
                for _ in range(len(nodes)): #TODO: evaluate number of iterations, are len(nodes) iterations enough?
                    node1 = nodes[np.random.randint(0, len(nodes))]
                    node2 = nodes[np.random.randint(0, len(nodes))]
                    counter = 0
                    while (node1 == node2 or (node1,node2) in already_selected_couples or nx.has_path(past_dag, node1, node2)) and counter < 10 : #avoid self loops
                        node2 = nodes[np.random.randint(0, len(nodes))]
                        counter +=1
                    
                    if counter == 10: 
                        continue #if we did not find a suitable couple we skip this iteration
                    else:
                        already_selected_couples.append((node1, node2))
            
                        # print(f"Adding edge between {node1} and {node2}")
                        for lag in range(1, current_lag + 1):
                            past_node_1 = f"{node1}_t-{lag}"
                            if lag > 1:
                                past_node_2_lag = f"{node2}_t-{lag-1}"
                            else:
                                past_node_2_lag = f"{node2}"
                            #check if the nodes are already in the dag
                            if past_node_1 not in past_dag.nodes:
                                print(f"{past_node_1} not in past_dag")
                                past_dag.add_node(past_node_1, **dag.nodes[node1])
                            if past_node_2_lag not in past_dag.nodes:
                                print(f"{past_node_2_lag} not in past_dag")
                                past_dag.add_node(past_node_2_lag, **dag.nodes[node2])
                            # print(f"Adding edge between {past_node_1} and {past_node_2_lag}")

                            past_dag.add_edge(past_node_1, past_node_2_lag, weight=0, H="linear")

        #we avoid edge cases for the moment
        # #number of edges 
        # n_edges = len(past_dag.edges)
        # #random number between 0 and n_edges
        # n_edges_to_remove = int(np.round(np.random.uniform(low=0, high=n_edges)))
        # #remove n_edges_to_remove edges
        # for _ in range(n_edges_to_remove):
        #     edge_to_remove = random.choice(list(past_dag.edges))
        #     past_dag.remove_edge(edge_to_remove[0], edge_to_remove[1])
        from graphviz import Digraph

        G_dot = Digraph(engine="dot",format='png')

        for node in past_dag.nodes():
            G_dot.node(str(node))
        for edge in past_dag.edges():
            G_dot.edge(str(edge[0]), str(edge[1]))

        # Render the graph in a hierarchical layout
        G_dot.render(f"pics/{index}", view=False, cleanup=True)
        return past_dag

    
    def _generate_timestep_observation(self, dag: nx.DiGraph, data: pd.DataFrame) -> pd.DataFrame:
        
        #inizialize the first row of the dataframe

        current_len = len(data)
        for node in nx.topological_sort(dag):
            # print("Node", node)
            # print("with bias", dag.nodes[node]['bias'])
            if f"_t-" in str(node):
                variable = int(str(node)[0])
                timestamp = int(str(node)[-1]) #TODO: check if this is correct, if more than 9 timestamps it will not work
                dag.nodes[node]['value'] = data.loc[len(data) - timestamp, variable] 
                # print("Value",dag.nodes[node]['value'])
            else:
                parents = list(dag.predecessors(node))
                column = int(node)
                data.loc[current_len, column] = 0
                # print("Data now is ", data)
                # print("parents", parents, "with values", [dag.nodes[parent]['value'] for parent in parents])
                # print("with edges weight", [dag.edges[parent, node]['weight'] for parent in parents])
                for parent in parents:
                    
                    data.loc[current_len, column] += self.compute_value(dag.nodes[node], dag.edges[parent, node], dag.nodes[parent]['value'])
                data.loc[current_len, column] += dag.nodes[node]['bias']
                dag.nodes[node]['value'] = data.loc[current_len, column]
                # print("Value",dag.nodes[node]['value'])

        

    def compute_value(self, node_data, edge_data, parent_value):
        sigma = node_data['sigma']
        weight = edge_data['weight'] 
        H = edge_data['H']
        value = 0
        if H == "linear":
            value += parent_value * weight
        value += np.random.normal(scale=sigma)
        return np.round(value,5)



    def get_observations(self) -> List[pd.DataFrame]:
        """
        Returns the generated time series.
        """
        return self.list_observations

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
    from graphviz import Digraph
    from utils import print_DAG

    n_dags = 5  # You can change this as needed
    n_observations = 5
    n_variables = 5
    maxlags = 4
    # Testing with a single process
    generator = SimulatedTimeSeries(n_dags, n_observations, n_variables, maxlags)
    generator.generate()
    dags = generator.get_dags()
    DAGs = generator.get_updated_dags()
    data = generator.get_observations()[0]
    

    for idx, DAG in enumerate(dags):
        G_dot = Digraph(engine="dot",format='png')

        for node in DAG.nodes():
            G_dot.node(str(node))
        for edge in DAG.edges():
            G_dot.edge(str(edge[0]), str(edge[1]))

        # Render the graph in a hierarchical layout
        #save the graph
        G_dot.render(f"graph_{idx}", view=True)

