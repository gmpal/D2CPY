
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

def generate_descriptors(name:str = 'data', maxlags:int = 3, n_jobs:int=1, for_test:bool = False):

    with open('../data/'+name+'.pkl', 'rb') as f:
        observations, _, updated_dags, _ = pickle.load(f)

    lagged_observations, updated_dags_renamed = [], []
    
    nodes_hardcoded = ['0_t-0', 
    '1_t-0', 
    '2_t-0', 
    '3_t-0', 
    '4_t-0',
    '5_t-0',
    '6_t-0',
    '7_t-0',
    '8_t-0',
    '9_t-0',
    '10_t-0',
    '11_t-0',
    '12_t-0',
    '13_t-0',
    '14_t-0',
    '15_t-0',
    '16_t-0',
    '17_t-0',
    '18_t-0',
    '19_t-0',
    '0_t-1', 
    '1_t-1', 
    '2_t-1', 
    '3_t-1', 
    '4_t-1',
    '5_t-1',
    '6_t-1',
    '7_t-1',
    '8_t-1',
    '9_t-1',
    '10_t-1',
    '11_t-1',
    '12_t-1',
    '13_t-1',
    '14_t-1',
    '15_t-1',
    '16_t-1',
    '17_t-1',
    '18_t-1',
    '19_t-1',
    '0_t-2', 
    '1_t-2', 
    '2_t-2', 
    '3_t-2', 
    '4_t-2',
    '5_t-2',
    '6_t-2',
    '7_t-2',
    '8_t-2',
    '9_t-2',
    '10_t-2',
    '11_t-2',
    '12_t-2',
    '13_t-2',
    '14_t-2',
    '15_t-2',
    '16_t-2',
    '17_t-2',
    '18_t-2',
    '19_t-2',
    '0_t-3', 
    '1_t-3', 
    '2_t-3', 
    '3_t-3', 
    '4_t-3',
    '5_t-3',
    '6_t-3',
    '7_t-3',
    '8_t-3',
    '9_t-3',
    '10_t-3',
    '11_t-3',
    '12_t-3',
    '13_t-3',
    '14_t-3',
    '15_t-3',
    '16_t-3',
    '17_t-3',
    '18_t-3',
    '19_t-3'
    ] #TODO: fix

    for obs in observations:
        lagged_obs, _ = create_lagged(obs, maxlags)
        lagged_observations.append(lagged_obs.dropna())

    for updated_dag in updated_dags:
        mapping = {i: index for index,i in enumerate(nodes_hardcoded)}
        updated_dag = nx.relabel_nodes(updated_dag, mapping)
        updated_dags_renamed.append(updated_dag)

    d2c = D2C(updated_dags_renamed, lagged_observations, n_jobs=n_jobs, n_variables=20, maxlags=maxlags)
    if for_test:
        causal_dataframe = d2c.compute_descriptors_no_dags()
        causal_dataframe.to_csv('../data/test_'+name+'_descriptors.csv')
    else:
    
        d2c.initialize()
        d2c.save_descriptors_df('../data/'+name+'_descriptors.csv')


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Generate D2C Descriptors')
    parser.add_argument('--name', type=str, default='data', help='Name of the file to load and save the data')
    parser.add_argument('--maxlags', type=int, default=3, help='Maximum lags for the time series')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for parallel processing')
    parser.add_argument('--for_test', type=bool, default=False, help='Set to True if you want to generate descriptors for test data')

    args = parser.parse_args()

    generate_descriptors(args.name, args.maxlags, args.n_jobs, args.for_test)