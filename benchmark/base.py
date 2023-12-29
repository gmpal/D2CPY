from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from multiprocessing import Pool
import numpy as np
import pickle

class BaseCausalInference:
    def __init__(self, ts_list, maxlags=3, ground_truth=None, n_jobs=1, suffix=''):
        self.ts_list = ts_list
        self.maxlags = maxlags
        self.causal_dfs = {}
        self.ground_truth = ground_truth
        self.returns_proba = False #TODO: in subclasses
        self.n_jobs = n_jobs
        self.suffix = suffix
        

    def standardize(self, single_ts):
        # Standardize data
        single_ts -= single_ts.mean(axis=0)
        single_ts /= single_ts.std(axis=0)
        return single_ts

    def infer(self, single_ts,**kwargs):
        pass
    
    def build_causal_df(self, results, n_variables):
        pass

    def process_ts(self, ts_tuple):
        ts_index, ts = ts_tuple
        print('\rProcessing', ts_index, 'of', len(self.ts_list), end='', flush=True)
        results = self.infer(self.standardize(ts), ts_index=ts_index)
        causal_df = self.build_causal_df(results, len(ts.columns))
        print('\rProcessed', ts_index, 'of', len(self.ts_list), end='', flush=True)
        return ts_index, causal_df

    def run(self):
        if self.n_jobs == 1:
            for ts_index, ts in enumerate(self.ts_list):
                _ , causal_df = self.process_ts((ts_index, ts))
                self.causal_dfs[ts_index] = causal_df
        else:
            # Create a list of tuples for mapping
            ts_tuples = list(enumerate(self.ts_list))

            # Create a pool of workers
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(self.process_ts, ts_tuples)

            # Store results in self.causal_dfs
            for ts_index, causal_df in results:
                self.causal_dfs[ts_index] = causal_df

        return self
    
    def get_causal_dfs(self):
        return self.causal_dfs

    def set_causal_dfs(self, causal_dfs):
        self.causal_dfs = causal_dfs
    
    def set_ground_truth(self, causal_dfs):
        self.ground_truth = causal_dfs
    
    def filter_causal_dfs(self, d2c_causal_dfs):
        indexes = {}
        for i, df in d2c_causal_dfs.items():
            indexes[i] = list(df.index)

        updated_causal_dfs = {}
        for i, idx_i in indexes.items():
            updated_causal_dfs[i] = self.causal_dfs[i].loc[idx_i]

        updated_ground_truths = {}
        for i, idx_i in indexes.items():
            updated_ground_truths[i] = self.ground_truth[i].loc[idx_i]

        self.set_causal_dfs(updated_causal_dfs)
        self.set_ground_truth(updated_ground_truths)



    def evaluate(self):
        data = []
        for ts_idx in range(len(self.ts_list)):
            print('\revaluating', ts_idx, 'of', len(self.ts_list), end='', flush=True)
            method_name = self.__class__.__name__ + self.suffix
            y_test = self.ground_truth[ts_idx]['is_causal'].astype(int)
            
            # y_test = self.causal_dfs[ts_idx]['truth'].astype(int)
            y_hat = self.causal_dfs[ts_idx]['is_causal'].astype(int)
            
            y_test = y_test.loc[y_hat.index]

            data.append([method_name, 'accuracy', accuracy_score(y_test, y_hat)])
            data.append([method_name, 'precision', precision_score(y_test, y_hat, zero_division=np.nan)])
            data.append([method_name, 'recall', recall_score(y_test, y_hat, zero_division=np.nan)])
            data.append([method_name, 'f1', f1_score(y_test, y_hat, zero_division=np.nan)])
            data.append([method_name, 'balanced_error', 1 - balanced_accuracy_score(y_test, y_hat)])   
            if self.returns_proba: 
                y_prob = self.causal_dfs[ts_idx]['value'].values.astype(float)
                if len(np.unique(y_test)) > 1:
                    auc_test = roc_auc_score(y_test, y_prob)
                    data.append([method_name, 'auc', auc_test])

        return data

    def save_causal_dfs(self, name):
        method_name = self.__class__.__name__ + self.suffix
        with open('../data/'+name+'_'+method_name+'_causal_dfs.pkl', 'wb') as f:
            pickle.dump(self.causal_dfs, f)