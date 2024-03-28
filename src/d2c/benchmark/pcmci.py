from src.benchmark.base import BaseCausalInference

from tigramite.pcmci import PCMCI as PCMCI_
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.data_processing as pp

import pandas as pd
import pickle

class PCMCI(BaseCausalInference):
    #TODO: parameters of PCMCI
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_proba = True

    def infer(self, single_ts,**kwargs):
        dataframe = pp.DataFrame(single_ts.values)
        cond_ind_test = ParCorr()
        pcmci = PCMCI_(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_max=self.maxlags - 1, pc_alpha=None)
        return results
    
    def build_causal_df(self, results, n_variables):
        pvalues = results['p_matrix']
        values = results['val_matrix']

        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['source', 'target'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['is_causal', 'value', 'pvalue'])

        for lag in range(self.maxlags):
            for source in range(n_variables):
                for effect in range(n_variables):
                    current_pvalue = pvalues[source][effect][lag]
                    current_value = values[source][effect][lag]
                    is_causal = 0 if current_pvalue > 0.05 else 0 if abs(current_value) < 0.1 else 1
                    causal_dataframe.loc[(n_variables + source+lag*n_variables, effect)] = is_causal, abs(current_value), current_pvalue

        return causal_dataframe


if __name__ == "__main__":
    # Usage
    with open('../data/fixed_lags.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    causal_method = PCMCI(observations[:5], maxlags=3)
    causal_method.run()
    results = causal_method.get_causal_dfs()
    print(results)