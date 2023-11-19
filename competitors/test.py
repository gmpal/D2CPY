import pandas as pd
import pickle

import sys
sys.path.append('../d2c')

from d2c_wrapper import D2C
from dynotears import DYNOTEARS
from granger import Granger
from pcmci import PCMCI
from var import VAR
from varlingam import VARLiNGAM

import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == "__main__":

    with open('../data/fixed_lags.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    df = pd.read_csv('../data/fixed_lags_descriptors.csv')
    df = pd.read_csv('../data/fixed_lags_descriptors_MB_estimated.csv')

    #flattening
    ground_truth_df = df[(df['edge_dest'] < 3) & (df['edge_source'] > 2)].sort_values(by=['graph_id','edge_source', 'edge_dest']).reset_index()[['graph_id','is_causal']]

    #select last 5 graphs
    ground_truth_df = ground_truth_df.loc[ground_truth_df['graph_id'] > len(ground_truth_df['graph_id'].unique()) - 11]

    ground_truth = []
    for value in ground_truth_df['graph_id'].unique():
        ground_truth.append(ground_truth_df.loc[ground_truth_df['graph_id'] == value]['is_causal'].values)  

    data = observations[-10:]

    d2c_eval = D2C(data, maxlags=3, n_jobs=10, ground_truth=ground_truth).run().evaluate()
    dyno_eval = DYNOTEARS(data, maxlags=3, ground_truth=ground_truth).run().evaluate()
    granger_eval = Granger(data, maxlags=3, ground_truth=ground_truth).run().evaluate()
    pcmci_eval = PCMCI(data, maxlags=3, ground_truth=ground_truth).run().evaluate()
    var_eval = VAR(data, maxlags=3, ground_truth=ground_truth).run().evaluate()
    varlingam_eval = VARLiNGAM(data, maxlags=3, ground_truth=ground_truth).run().evaluate()


    all_eval = [d2c_eval, dyno_eval, granger_eval, pcmci_eval, var_eval, varlingam_eval]
    df_all_eval = pd.DataFrame(columns=['Model', 'Metric', 'Score'])
    for eval in all_eval:
        df_all_eval = pd.concat([df_all_eval,pd.DataFrame(eval,columns=['Model', 'Metric', 'Score'])])


    df_scores = pd.DataFrame(df_all_eval, columns=['Model', 'Metric', 'Score'])

    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Metric', y='Score', hue='Model', data=df_scores)
    plt.title("Comparison of methods Across Different Metrics")
    plt.show()

