
import pickle
import sys 
import pandas as pd
import networkx as nx
import argparse

sys.path.append("..")
sys.path.append("../d2c/")
from d2c.d2c import D2C

def create_lagged(observations, lag):
    #create lagged observations
    lagged = observations.copy()
    names = observations.columns
    for i in range(1,lag+1):
        lagged = pd.concat([lagged, observations.shift(i)], axis=1)
    lagged.columns = [i for i in range(len(lagged.columns))]
    lagged_column_names = [str(name) + '_lag' + str(i) for i in range(lag+1) for name in names]
    return lagged, lagged_column_names

def generate_descriptors(name:str = 'data', maxlags:int = 3, n_jobs:int=1):

    with open('../data/'+name+'.pkl', 'rb') as f:
        observations, _, updated_dags, _ = pickle.load(f)
    
    lagged_observations, updated_dags_renamed = [], []
     
    for obs in observations:
        lagged_obs, _ = create_lagged(obs, maxlags)
        lagged_observations.append(lagged_obs.dropna())

    for updated_dag in updated_dags:
        mapping = {i: index for index,i in enumerate(updated_dag.nodes)}
        updated_dag = nx.relabel_nodes(updated_dag, mapping)
        updated_dags_renamed.append(updated_dag)

    d2c = D2C(updated_dags_renamed, lagged_observations, n_jobs=n_jobs)
    d2c.initialize()
    d2c.save_descriptors_df('../data/'+name+'_descriptors.csv')


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Generate D2C Descriptors')
    parser.add_argument('--name', type=str, default='data', help='Name of the file to load and save the data')
    parser.add_argument('--maxlags', type=int, default=3, help='Maximum lags for the time series')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for parallel processing')

    args = parser.parse_args()

    generate_descriptors(args.name, args.maxlags, args.n_jobs)