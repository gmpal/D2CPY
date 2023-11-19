
from base import BaseCausalInference
import pandas as pd
import pickle
import statsmodels.tsa.api as tsa

class VAR(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, single_ts,**kwargs):
        model = tsa.var.var_model.VAR(single_ts.values)
        results = model.fit(maxlags=self.maxlags)
        return results
    
    def build_causal_df(self, results, n_variables):
        pvalues = results.pvalues
        values = results.coefs

        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['source', 'target'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['is_causal', 'value', 'pvalue'])

        for lag in range(self.maxlags):
            for source in range(n_variables):
                for effect in range(n_variables):
                    current_pvalue = pvalues[source+lag*n_variables, effect]
                    current_value = values[lag][effect][source]

                    is_causal = 0 if current_pvalue > 0.05 else 0 if abs(current_value) < 0.1 else 1
                    causal_dataframe.loc[(n_variables + source+lag*n_variables, effect)] = is_causal, current_value, current_pvalue

        return causal_dataframe


if __name__ == "__main__":
    # Usage
    with open('data/fixed_lags.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    causal_method = VAR(observations[:5], maxlags=3)
    causal_method.run()
    results = causal_method.get_causal_dfs()
    