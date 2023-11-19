from base import BaseCausalInference

from causalnex.structure.dynotears import from_pandas_dynamic

import pandas as pd
import pickle

class DYNOTEARS(BaseCausalInference):
    #TODO: parameters of PCMCI
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_proba = True

    def infer(self, single_ts,**kwargs):
        results = from_pandas_dynamic(single_ts, p=self.maxlags)
        return results
    
    def build_causal_df(self, results, n_variables):
        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['source', 'target'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['is_causal', 'value', 'pvalue'])

        causal_dataframe['is_causal'] = 0
        causal_dataframe['pvalue'] = 0
        causal_dataframe['value'] = 0

        for edge in results.edges:
            source = int(edge[0][0])
            effect = int(edge[1][0])
            lag = int(edge[0][-1]) - 1
            if lag > 0: #we ignore contemporaneous relations for the moment
                value = results.get_edge_data(*edge)['weight']
                causal_dataframe.loc[(n_variables+source+lag*n_variables, effect), 'is_causal'] = 1
                causal_dataframe.loc[(n_variables+source+lag*n_variables, effect), 'value'] = abs(value)
                causal_dataframe.loc[(n_variables+source+lag*n_variables, effect), 'pvalue'] = 0


        return causal_dataframe


if __name__ == "__main__":
    # Usage
    with open('../data/fixed_lags.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    causal_method = DYNOTEARS(observations[:5], maxlags=3)
    causal_method.run()
    results = causal_method.get_causal_dfs()
    print(results)
    