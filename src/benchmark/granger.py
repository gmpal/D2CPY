from src.benchmark.base import BaseCausalInference

from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from statsmodels.tools.sm_exceptions import InfeasibleTestError

import pandas as pd
import pickle

class Granger(BaseCausalInference):
    #TODO: parameters of PCMCI
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, single_ts,**kwargs):
        results = {}
        for x1 in single_ts.columns:
            gc2 = {}
            for x2 in single_ts.columns:
                try:
                    gc_res = grangercausalitytests(single_ts[[x1,x2]], self.maxlags)
                    gc_res_lags = {}
                    for lag in range(1,self.maxlags+1):
                        gc_res_lags[lag] = gc_res[lag][0]['ssr_ftest'][1]
                except InfeasibleTestError:
                    gc_res_lags = {lag:np.nan for lag in range(1,self.maxlags+1)}
                gc2[int(x2)] = gc_res_lags 
            results[int(x1)] = gc2 

        return results
    
    def build_causal_df(self, results, n_variables):
        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['source', 'target'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['is_causal', 'value', 'pvalue'])

        for lag in range(self.maxlags):
            for source in range(n_variables):
                for effect in range(n_variables):
                    current_pvalue = results[source][effect][lag+1]
                    
                    is_causal = 0 if current_pvalue > 0.05 else 1
                    causal_dataframe.loc[(n_variables + source+lag*n_variables, effect)] = is_causal, is_causal, current_pvalue #no 'strenght' of causal for granger

        return causal_dataframe


if __name__ == "__main__":
    # Usage
    with open('../data/fixed_lags.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    causal_method = Granger(observations[:5], maxlags=3)
    causal_method.run()
    results = causal_method.get_causal_dfs()
    print(results)