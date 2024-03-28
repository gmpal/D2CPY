
import pandas as pd
import pickle
import statsmodels.tsa.api as tsa

from src.benchmark.base import BaseCausalInference

class VAR(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_proba = True

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
                    causal_dataframe.loc[(n_variables + source+lag*n_variables, effect)] = is_causal, abs(current_value), current_pvalue

        return causal_dataframe


if __name__ == "__main__":

    # Usage
    with open('../data/100_known_ts_all_20_variables.pkl', 'rb') as f:
        observations, dags, updated_dags, causal_dfs = pickle.load(f)

    causal_method = VAR(observations[:5], maxlags=3)
    causal_method.run()
    results = causal_method.get_causal_dfs()
    # print(results[2].loc[results[2].index.isin([(20,  0),
    #         (20,  1),
    #         (20,  2),
    #         (20,  3),
    #         (20,  4),
    #         (20,  5),
    #         (20,  6),
    #         (20,  7),
    #         (20,  8),
    #         (20,  9),
    #         (20, 10),
    #         (20, 11),
    #         (20, 12),
    #         (20, 13),
    #         (20, 14),
    #         (20, 15),
    #         (20, 16),
    #         (20, 17),
    #         (20, 18),
    #         (20, 19),
    #         (22,  0),
    #         (29,  4),
    #         (42,  6),
    #         (42, 12),
    #         (42, 14),
    #         (42, 15),
    #         (42, 19),
    #         (49,  6)])]
    #         )
    
    tuples = [
        (20, 0), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9),
        (20, 10), (20, 11), (20, 12), (20, 13), (20, 14), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19),
        (22, 0), (29, 4), (42, 6), (42, 12), (42, 14), (42, 15), (42, 19), (49, 6)
    ]

    print(results[2].loc[tuples].reset_index())