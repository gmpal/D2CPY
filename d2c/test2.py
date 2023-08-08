import networkx as nx
import pandas as pd
import numpy as np

def generate_timestep_observation(dag: nx.DiGraph, data: pd.DataFrame) -> pd.DataFrame:
    
    #inizialize the first row of the dataframe
    current_len = len(data)
    for node in nx.topological_sort(dag):
        if f"_t-" in str(node):
            variable = int(str(node)[0])
            timestamp = int(str(node)[-1])
            dag.nodes[node]['value'] = data.loc[len(data) - timestamp, variable] 
        else:
            parents = list(dag.predecessors(node))
            data.loc[current_len, node] = 0
            for parent in parents:
                data.loc[current_len, node] += compute_value(dag.nodes[node], dag.edges[parent, node], dag.nodes[parent]['value'])
            dag.nodes[node]['value'] = data.loc[current_len, node] + dag.nodes[node]['bias']
    


def compute_value(node_data, edge_data, parent_value):
    sigma = node_data['sigma']
    weight = edge_data['weight'] 
    H = edge_data['H']
    value = 0
    if H == "linear":
        value += parent_value * weight
    # value += np.random.normal(scale=sigma)
    return np.round(value,2)


import pickle

dag = pickle.load(open("DAG1.pkl", "rb"))
first_lines = pickle.load(open("first_lines.pickle", "rb"))


def testfunction(dag, first_lines):
    
    dag_dict = {}
    for node in nx.topological_sort(dag):
        if f"_t-" in str(node):
            idx = int(str(node)[0])
            position = int(str(node)[-1])
            dag_dict[node] = first_lines[idx][position - 1]
        else:
            dag_dict[node] = 0
            parents = list(dag.predecessors(node))
            for parent in parents:
                dag_dict[node] += dag_dict[parent] * dag.edges[parent, node]['weight'] + dag.nodes[node]['bias']
    return dag_dict[0], dag_dict[1], dag_dict[2] 




#expectation 
print(testfunction(dag, first_lines))


#reality
generate_timestep_observation(dag, first_lines)
print(first_lines.iloc[-1,:])