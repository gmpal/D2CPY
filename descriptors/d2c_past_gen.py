
import pickle
import sys 
import pandas as pd
import networkx as nx
import argparse

sys.path.append("..")
sys.path.append("../d2c/")
from d2c.d2c import D2C

class DescriptorsGenerator():
    
    def __init__(self, ts_builder = None, data_path = None, maxlags = 3, n_jobs = 1):
        if ts_builder is not None: 
            self.observations = self.from_dict_of_lists_to_list(ts_builder.get_observations())
            self.dags = self.from_dict_of_lists_to_list(ts_builder.get_dags())
            self.causal_dfs = self.from_dict_of_lists_to_list(ts_builder.get_causal_dfs())
            self.maxlags = ts_builder.get_maxlags()
            self.n_variables = ts_builder.get_n_variables()    

        elif data_path is not None:
            with open(data_path, 'rb') as f:
                self.observations, self.dags, self.causal_dfs = pickle.load(f)
            self.maxlags = maxlags        
            self.n_variables = self.observations[0].shape[1] #TODO: assumption of constant number of variables
        else:
            raise ValueError('Either ts_builder or data_path must be provided')

        self.n_jobs = n_jobs
        self.lagged_observations = []
        self.updated_dags = []
        self.are_testing_descriptors_unseen = False #TODO: when is this actually useful? 
        self.d2c = None

    def from_dict_of_lists_to_list(self, dict_of_lists):
        list_of_lists = []
        for key in dict_of_lists.keys():
            list_of_lists.extend(dict_of_lists[key])
        return list_of_lists

    def generate(self):
        if len(self.updated_dags) == 0 or len(self.lagged_observations) == 0:
            self.create_lagged_multiple_ts()
            self.rename_dags()
        
        self.d2c = D2C(self.updated_dags, self.lagged_observations, n_jobs=self.n_jobs, n_variables=self.n_variables, maxlags=self.maxlags)
        self.d2c.initialize()
        # if self.are_testing_descriptors_unseen: #TODO: when is this actually useful? 
        #     causal_dataframe = d2c.compute_descriptors_no_dags()
        #     causal_dataframe.to_csv('../data/test_'+name+'_descriptors.csv')
        # else:
    
    # def save_data(self):
    #     with open('../data/descriptors.pkl', 'wb') as f:
    #         pickle.dump((self.ts_list, self.dags, self.causal_dfs), f)

    def save(self, output_folder):
        descriptors_df = self.d2c.get_descriptors_df()
        with open(output_folder+'descriptors.pkl', 'wb') as f:
            pickle.dump(descriptors_df, f)


    def create_lagged(self, observations, lag):
        #create lagged observations
        lagged = observations.copy()
        names = observations.columns
        for i in range(1,lag+1):
            lagged = pd.concat([lagged, observations.shift(i)], axis=1)
        lagged.columns = [i for i in range(len(lagged.columns))]
        lagged_column_names = [str(name) + '_lag' + str(i) for i in range(lag+1) for name in names]
        return lagged, lagged_column_names

    def create_lagged_multiple_ts(self):
        #create lagged observations for all the available time series
        for obs in self.observations:
            lagged_obs, _ = self.create_lagged(obs, self.maxlags)
            self.lagged_observations.append(lagged_obs.dropna())

    def rename_dags(self):
        #rename the nodes of the dags to use the same convention as the descriptors
        for dag in self.dags:
            mapping = {node: int(node.split('_')[0]) + int(node.split('-')[1]) * self.n_variables for node in dag.nodes()} #from x_(t-y) to x + y*n_variables
            dag = nx.relabel_nodes(dag, mapping)
            self.updated_dags.append(dag)

    def get_descriptors_df(self):
        return self.d2c.get_descriptors_df()

    def get_causal_dfs(self):
        #TODO: adjust causal_dfs_dict to avoid this
        causal_dfs_dict = {}
        for i in range(len(self.causal_dfs)):
            causal_dfs_dict[i] = self.causal_dfs[i]
        return causal_dfs_dict
    
    def get_dags(self):
        return self.dags

    def get_observations(self):
        return self.observations

    def get_test_couples(self):
        return self.d2c.get_test_couples()

def generate_descriptors(name:str = 'data', maxlags:int = 3, n_jobs:int=1, for_test:bool = False):

    with open('../data/'+name+'.pkl', 'rb') as f:
        observations, updated_dags, _ = pickle.load(f)

    n_variables = observations[0].shape[1]

    lagged_observations, updated_dags_renamed = [], []

    for obs in observations:
        lagged_obs, _ = create_lagged(obs, maxlags)
        lagged_observations.append(lagged_obs.dropna())

    for updated_dag in updated_dags:
        mapping = {node: int(node.split('_')[0]) + int(node.split('-')[1]) * n_variables for node in updated_dag.nodes()}
        updated_dag = nx.relabel_nodes(updated_dag, mapping)
        updated_dags_renamed.append(updated_dag)

    d2c = D2C(updated_dags_renamed, lagged_observations, n_jobs=n_jobs, n_variables=5, maxlags=maxlags)

    if for_test:
        causal_dataframe = d2c.compute_descriptors_no_dags()
        causal_dataframe.to_csv('../data/test_'+name+'_descriptors.csv')
    else:
    
        d2c.initialize()
        d2c.save_descriptors_df('../data/'+name+'_descriptors.csv')


if __name__ == '__main__': 

    # parser = argparse.ArgumentParser(description='Generate D2C Descriptors')
    # parser.add_argument('--name', type=str, default='data', help='Name of the file to load and save the data')
    # parser.add_argument('--maxlags', type=int, default=3, help='Maximum lags for the time series')
    # parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for parallel processing')
    # parser.add_argument('--for_test', type=bool, default=False, help='Set to True if you want to generate descriptors for test data')

    # args = parser.parse_args()

    # generate_descriptors(args.name, args.maxlags, args.n_jobs, args.for_test)

    dg = DescriptorsGenerator(maxlags=3, n_jobs=10, data_path='../new_data/data_1.pkl')    
    dg.generate()
