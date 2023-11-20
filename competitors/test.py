import argparse
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

#suppress FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def test(name:str = 'data', maxlags:int = 3, n_jobs:int=1):
    
    descriptors_path = '../data/'+name+'_descriptors.csv'
    data_path = '../data/'+name+'.pkl'

    with open(data_path, 'rb') as f:
        data, dags, updated_dags, causal_dfs = pickle.load(f)

    df = pd.read_csv(descriptors_path)

    ground_truth = [df['is_causal'].values for df in causal_dfs]

    for train_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        d2c = D2C(data, maxlags=maxlags, n_jobs=n_jobs, ground_truth=ground_truth,descriptors_path=descriptors_path, train_ratio=train_ratio, suffix=train_ratio).run()
        d2c_causal_dfs = d2c.get_causal_dfs() 
        with open('../data/'+name+'_d2c_causal_dfs_'+str(train_ratio)+'.pkl', 'wb') as f:
            pickle.dump(d2c_causal_dfs, f)
        d2c_eval = d2c.evaluate()

    dyno = DYNOTEARS(data, maxlags=maxlags, ground_truth=ground_truth).run()
    dyno_causal_dfs = dyno.get_causal_dfs()
    with open('../data/'+name+'_dyno_causal_dfs.pkl', 'wb') as f:
        pickle.dump(dyno_causal_dfs, f)
    dyno_eval = dyno.evaluate()

    granger = Granger(data, maxlags=maxlags, ground_truth=ground_truth).run()
    granger_causal_dfs = granger.get_causal_dfs()
    with open('../data/'+name+'_granger_causal_dfs.pkl', 'wb') as f:
        pickle.dump(granger_causal_dfs, f)
    granger_eval = granger.evaluate()

    pcmci = PCMCI(data, maxlags=maxlags, ground_truth=ground_truth).run()
    pcmci_causal_dfs = pcmci.get_causal_dfs()
    with open('../data/'+name+'_pcmci_causal_dfs.pkl', 'wb') as f:
        pickle.dump(pcmci_causal_dfs, f)
    pcmci_eval = pcmci.evaluate()
    
    var = VAR(data, maxlags=maxlags, ground_truth=ground_truth).run()
    var_causal_dfs = var.get_causal_dfs()
    with open('../data/'+name+'_var_causal_dfs.pkl', 'wb') as f:
        pickle.dump(var_causal_dfs, f)
    var_eval = var.evaluate()

    varlingam = VARLiNGAM(data, maxlags=maxlags, ground_truth=ground_truth).run()
    varlingam_causal_dfs = varlingam.get_causal_dfs()
    with open('../data/'+name+'_varlingam_causal_dfs.pkl', 'wb') as f:
        pickle.dump(varlingam_causal_dfs, f)
    varlingam_eval = varlingam.evaluate()


    all_eval = [d2c_eval, dyno_eval, granger_eval, pcmci_eval, var_eval, varlingam_eval]
    df_all_eval = pd.DataFrame(columns=['Model', 'Metric', 'Score'])
    for eval in all_eval:
        df_all_eval = pd.concat([df_all_eval,pd.DataFrame(eval,columns=['Model', 'Metric', 'Score'])])


    df_scores = pd.DataFrame(df_all_eval, columns=['Model', 'Metric', 'Score'])
    #save
    df_scores.to_csv('../data/'+name+'_scores.csv', index=False)
    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Metric', y='Score', hue='Model', data=df_scores)
    plt.title("Comparison of methods Across Different Metrics")
    plt.legend(loc='upper right')
    #save
    plt.savefig('../data/'+name+'_scores.png')
    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test Causal Inference Methods')
    parser.add_argument('--name', type=str, default='data', help='Name of the file to load and save the data')
    parser.add_argument('--maxlags', type=int, default=3, help='Maximum lags for the time series')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for parallel processing')

    args = parser.parse_args()

    test(args.name, args.maxlags, args.n_jobs)    

