from benchmark.base import BaseCausalInference
import pandas as pd
import pickle

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('../d2c')
from d2c import D2C as D2C_


class D2C(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        self.use_real_MB = kwargs.pop('use_real_MB', False)
        self.train_ratio = kwargs.pop('train_ratio', 1)
        self.flattening = kwargs.pop('flattening', False)
        self.n_variables = kwargs.pop('n_variables', 6)
        self.n_gen_proc = kwargs.pop('n_gen_proc', 17)
        self.recompute_descriptors = kwargs.pop('recompute_descriptors', False)
            
        descriptors_path = kwargs.pop('descriptors_path', None)
        with open(descriptors_path, 'rb') as f:
            self.descriptors = pickle.load(f)
        super().__init__(*args, **kwargs)
        self.returns_proba = True

        n_graphs = len(self.descriptors.graph_id.unique())
        n_graphs_per_process = n_graphs // self.n_gen_proc
        self.descriptors['process_id'] = self.descriptors['graph_id'] // n_graphs_per_process + 1

    def create_lagged(self, observations, lag):
        lagged = observations.copy()
        for i in range(1,lag+1):
            lagged = pd.concat([lagged, observations.shift(i)], axis=1)
        lagged.columns = [i for i in range(len(lagged.columns))]
        return lagged.dropna()

    def infer(self, single_ts, **kwargs):
        ts_index = kwargs.get('ts_index', None)
        if ts_index is None:
            raise ValueError('ts_index is required for D2C inference')
        
        data = self.descriptors

        testing_data = data.loc[data['graph_id'] == ts_index] #TODO: this would not work if self.recompute_descriptors!
        
        generative_process = testing_data.iloc[0]['process_id']
        training_data = data.loc[data['process_id'] != generative_process] 

        ############################
        ########TRAINING############
        ############################
       
        X_train = training_data.drop(['process_id','graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
        y_train = training_data['is_causal']
       
        clf = BalancedRandomForestClassifier(n_estimators=20, n_jobs=1, sampling_strategy='all',replacement=True)
        clf.fit(X_train, y_train)

        ###########################
        ########TESTING############
        ###########################

        if self.recompute_descriptors:
            pass #TODO: when testing on data for which descriptors are not available
            # Recompute descriptors for the test data
            # lagged = self.create_lagged(single_ts, self.maxlags)
            # d2c_test = D2C_(None,[lagged], maxlags=self.maxlags, use_real_MB=self.use_real_MB,n_variables=self.n_variables,dynamic=True)
            # X_test = d2c_test.compute_descriptors_no_dags()
            # X_test = X_test.drop(['graph_id', 'edge_source', 'edge_dest'], axis=1)
            # y_pred = clf.predict_proba(X_test)[:,1]
            # returned = pd.DataFrame(y_pred,index=X_test.index, columns=['is_causal'])
        else:
            # for fairness in the comparison, we test only on pairs that all methods have seen
            testing_data = testing_data[(testing_data['edge_dest'] < self.n_variables) & (testing_data['edge_source'] >= self.n_variables)].sort_values(by=['graph_id','edge_source', 'edge_dest']).reset_index(drop=True)

            X_test = testing_data.drop(['process_id','graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
            y_pred = clf.predict_proba(X_test)[:,1]

            testing_data['truth'] = testing_data['is_causal']
            testing_data['is_causal'] = y_pred
            # returned = pd.concat([pd.DataFrame(testing_data)[['edge_source','edge_dest']], pd.DataFrame(y_pred, columns=['is_causal'])], axis=1, sort=True)
            
        return testing_data[['edge_source','edge_dest','truth','is_causal']]
    
    def build_causal_df(self, results, n_variables):
        
        results.set_index(['edge_source','edge_dest'], inplace=True) #Already set
        results['value'] = results['is_causal']
        results['is_causal'] = results['is_causal'].apply(lambda x: 1 if x > 0.5 else 0)
        results['pvalue'] = 0
        return results


if __name__ == "__main__":
    # Usage
    with open('../data/100_known_ts_all_20_variables.pkl', 'rb') as f:
        observations, dags, updated_dags, causal_dfs = pickle.load(f)

    causal_method = D2C(observations[-4:], maxlags=3,descriptors_path='../data/100_known_ts_all_descriptors_20_variables.pkl', n_variables=20)
    causal_method.run()
    results = causal_method.get_causal_dfs()