from base import BaseCausalInference
import pandas as pd
import pickle

from imblearn.ensemble import BalancedRandomForestClassifier

import sys
sys.path.append('../d2c')
from d2c import D2C as D2C_


class D2C(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        self.use_real_MB = kwargs.pop('use_real_MB', False)
        super().__init__(*args, **kwargs)
        self.returns_proba = True


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
        lagged_obs = self.create_lagged(single_ts, self.maxlags)
        d2c_test = D2C_(None, [lagged_obs], n_jobs=1, dynamic=True, n_variables = single_ts.shape[1], maxlags=self.maxlags) #TODO: parallelism is handled at another spot
        X_test = d2c_test.compute_descriptors_no_dags()
        test_df = pd.DataFrame(X_test).drop(['graph_id','edge_source','edge_dest'], axis=1)
        
        training_data = pd.read_csv('../data/fixed_lags_descriptors_MB_estimated.csv')
        # training_data = pd.read_csv('../data/fixed_lags_descriptors.csv')
        training_data = training_data.loc[training_data['graph_id'] != ts_index] 

        X_train = training_data.drop(['graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
        y_train = training_data['is_causal']

        clf = BalancedRandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=1, sampling_strategy='all',replacement=True)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(test_df)[:,1]
        returned = pd.concat([pd.DataFrame(X_test)[['edge_source','edge_dest']], pd.DataFrame(y_pred, columns=['is_causal'])], axis=1)
      
        return returned
    
    def build_causal_df(self, results, n_variables):
        results.set_index(['edge_source','edge_dest'], inplace=True)
        results['value'] = results['is_causal']
        results['is_causal'] = results['is_causal'].apply(lambda x: 1 if x > 0.5 else 0)
        results['pvalue'] = 0
        return results


if __name__ == "__main__":
    # Usage
    with open('../data/fixed_lags.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    causal_method = D2C(observations[-5:], maxlags=3)
    causal_method.run()
    results = causal_method.get_causal_dfs()