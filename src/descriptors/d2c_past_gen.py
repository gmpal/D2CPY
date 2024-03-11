
import pickle
import pandas as pd
import networkx as nx
import os 

from src.d2c.d2c import D2C
from src.d2c.utils import from_dict_of_lists_to_list, rename_dags, create_lagged_multiple_ts

class DescriptorsGenerator():
    
    def __init__(self, ts_builder = None, data_path = None, maxlags = 3, n_jobs = 1, mutual_information_proxy = 'linear', proxy_params = None, family={'basic': True ,'var_var': True, 'var_mb':True, 'var_mb_given_var':True,'mb_mb_given_var':True,'structural':True}, couples_to_consider_per_dag = 20, MB_size =5 ):
        if ts_builder is not None: 
            self.observations = from_dict_of_lists_to_list(ts_builder.get_observations())
            self.dags = from_dict_of_lists_to_list(ts_builder.get_dags())
            self.causal_dfs = from_dict_of_lists_to_list(ts_builder.get_causal_dfs())
            self.maxlags = ts_builder.get_maxlags()
            self.n_variables = ts_builder.get_n_variables()    
    

        elif data_path is not None:
            # for each file starting with 'data' in the folder data_path
            # load the data and append it to the list of observations
            
            loaded_observations = {}
            loaded_dags = {}
            loaded_causal_dfs = {}
            for file in os.listdir(data_path):
                if file.startswith('data'):
                    index = file.split('_')[1].split('.')[0]
                    with open(data_path+file, 'rb') as f:
                        loaded_observations[index], loaded_dags[index], loaded_causal_dfs[index], _ = pickle.load(f)
            
            #TODO: add the case of single data file, consider moving always to single data file

            self.observations = from_dict_of_lists_to_list(loaded_observations)
            self.dags = from_dict_of_lists_to_list(loaded_dags)
            self.causal_dfs = from_dict_of_lists_to_list(loaded_causal_dfs)
            self.maxlags = maxlags        
            self.n_variables = self.observations[0].shape[1] #TODO: assumption of constant number of variables
        else:
            raise ValueError('Either ts_builder or data_path must be provided')

        self.n_jobs = n_jobs
        self.lagged_observations = []
        self.updated_dags = []
        self.are_testing_descriptors_unseen = False #TODO: when is this actually useful? 
        self.mutual_information_proxy = mutual_information_proxy
        self.proxy_params = proxy_params
        self.d2c = None
        self.family = family
        self.couples_to_consider_per_dag = couples_to_consider_per_dag
        self.MB_size = MB_size


    def generate(self):
        if len(self.updated_dags) == 0 or len(self.lagged_observations) == 0:
            self.lagged_observations = create_lagged_multiple_ts(self.observations, self.maxlags)
            self.updated_dags = rename_dags(self.dags, self.n_variables)
        
        self.d2c = D2C(self.updated_dags, self.lagged_observations, self.couples_to_consider_per_dag, self.MB_size, n_jobs=self.n_jobs, n_variables=self.n_variables, maxlags=self.maxlags, mutual_information_proxy=self.mutual_information_proxy, proxy_params=self.proxy_params, family=self.family)
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
        if self.proxy_params is None:
            filename = output_folder+'descriptors_'+self.mutual_information_proxy+'.pkl'
        else:
            filename = output_folder+'descriptors_'+self.mutual_information_proxy+'_tau'+ str(self.proxy_params['tau'])+'.pkl'
        if self.family is not None:
            if 'tau' not in filename: #If we are using tau, we don't need to add the family, we use all the descriptors
                filename = filename[:-4]+'family'+filename[-4:]
                for family_index, key in enumerate(self.family):
                    if self.family[key]:
                        filename = filename[:-4]+'-'+str(family_index)+filename[-4:]
        with open(filename, 'wb') as f:
            pickle.dump(descriptors_df, f)


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

    d2c = D2C(updated_dags_renamed, lagged_observations, n_jobs=n_jobs, n_variables=5, maxlags=maxlags, family=family)

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
