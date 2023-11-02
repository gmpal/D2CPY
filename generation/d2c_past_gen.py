
import pickle
import sys 
import pandas as pd
import networkx as nx
sys.path.append("..")
sys.path.append("../d2c/")

def create_lagged(observations, lag):
    #create lagged observations
    lagged = observations.copy()
    names = observations.columns
    for i in range(1,lag+1):
        lagged = pd.concat([lagged, observations.shift(i)], axis=1)
    lagged.columns = [i for i in range(len(lagged.columns))]
    lagged_column_names = [str(name) + '_lag' + str(i) for i in range(lag+1) for name in names]
    return lagged, lagged_column_names


if __name__ == '__main__': 


    with open('../data/fixed_lags.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)
    
    lagged_observations, lagged_column_names, updated_dags_renamed = [], [], []
     
    for obs in observations:
        lagged_obs, lagged_names = create_lagged(obs, 3)
        lagged_observations.append(lagged_obs.dropna())
        lagged_column_names.append(lagged_names)

    for updated_dag in updated_dags:
        #rename dag nodes
        mapping = {i: index for index,i in enumerate(updated_dag.nodes)}
        updated_dag = nx.relabel_nodes(updated_dag, mapping)
        updated_dags_renamed.append(updated_dag)


    from d2c.D2C import D2C
    d2c = D2C(updated_dags_renamed, lagged_observations,n_jobs=10)
    d2c.initialize()
    d2c.save_descriptors_df('../data/fixed_lags_descriptors.csv')
