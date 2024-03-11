
import pickle
import pandas as pd
import networkx as nx
import os 


class DataLoader():
    """
    This class is in between the data generation process and the descriptors computation. 
    It loads the data and transform it appropriately.
    Transformations are: 
    1. from multiple files to a single list
    2. compute the lagged observations
    3. rename the dags with increasing integers as nodes
    TODO: add the case of single data file, consider moving always to single data file

    """
    
    def __init__(self, maxlags = 3, n_variables=3):
        self.observations = None
        self.dags = None
        self.n_variables = n_variables
        self.maxlags = maxlags


    def _from_dict_of_lists_to_list(self, dict_of_lists):
        """
        Convert a dictionary of lists to a single list.
        This is useful because data is stored in different files, according to the generative process. 
        When we load data, we keep track of the generative process as index of the dictionary.
        """
        list_of_lists = []
        sorted_keys = sorted(dict_of_lists.keys(), key=lambda x: int(x))
        for key in sorted_keys:
            list_of_lists.extend(dict_of_lists[key])
        return list_of_lists

    def _rename_dags(self, dags, n_variables):
        """
        Rename the nodes of the dags to use the same convention as the descriptors.
        Specifically, we rename the nodes from x_(t-y) to x + y*n_variables.
        We move from string to integer and we consider the variables from the past as different.

        Example:
        if n = 3
        - 3_t-0 -> 3
        - 1_t-1 -> 4
        - 3_t-1 -> 6
        """
        #rename the nodes of the dags to use the same convention as the descriptors
        updated_dags = []
        for dag in dags:
            mapping = {node: int(node.split('_')[0]) + int(node.split('-')[1]) * n_variables for node in dag.nodes()} #from x_(t-y) to x + y*n_variables
            dag = nx.relabel_nodes(dag, mapping)
            updated_dags.append(dag)
        return updated_dags

    def _create_lagged_multiple_ts(self, observations, maxlags):
        """
        Create lagged multiple time series from the given observations.

        Parameters:
        observations (list): A list of pandas DataFrames representing the time series observations.
        maxlags (int): The maximum number of lags to create.

        Returns:
        lagged_observations (list): A list of pandas DataFrames representing the lagged time series observations.
        """
        lagged_observations = []
        for obs in observations:
            lagged = obs.copy()
            for i in range(1,maxlags+1):
                lagged = pd.concat([lagged, obs.shift(i)], axis=1)
            lagged.columns = [i for i in range(len(lagged.columns))]
            lagged_observations.append(lagged.dropna())
        return lagged_observations


    def load_data_from_file(self, data_path):
        """
        Data loader from a folder of data files.
        This is the standard way to load data, when we have multiple files for different generative processes.
        """
        # for each file starting with 'data' in the folder data_path
        # load the data and append it to the list of observations
        loaded_observations = {}
        loaded_dags = {}
        for file in os.listdir(data_path):
            if file.startswith('data'):
                index = file.split('_')[1].split('.')[0]
                with open(data_path+file, 'rb') as f:
                    loaded_observations[index], loaded_dags[index], _ , _ = pickle.load(f)
        

        self.observations = self._from_dict_of_lists_to_list(loaded_observations)
        self.dags = self._from_dict_of_lists_to_list(loaded_dags)
        

    def load_data_from_tsbuilder(self, ts_builder):
        """
        Data loader from a TimeSeriesBuilder object. 
        This prevents storing data on disk and allows a unique flow from data generation to descriptors computation to benchmark.
        """
        self.observations = self._from_dict_of_lists_to_list(ts_builder.get_observations())
        self.dags = self._from_dict_of_lists_to_list(ts_builder.get_dags())

    def get_observations(self):
        """
        Get the observations after having created the lagged time series.
        """
        return self._create_lagged_multiple_ts(self.observations, self.maxlags)
    
    def get_dags(self):
        """
        Get the dags after having renamed the nodes.
        """
        return self._rename_dags(self.dags, self.n_variables)