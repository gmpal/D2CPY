import argparse
import pandas as pd
import pickle
import time
import numpy as np

import sys
sys.path.append('../d2c')

from benchmark.d2c_wrapper import D2C
from benchmark.dynotears import DYNOTEARS
from benchmark.granger import Granger
from benchmark.pcmci import PCMCI
from benchmark.var import VAR
from benchmark.varlingam import VARLiNGAM

import seaborn as sns
import matplotlib.pyplot as plt

#suppress FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) #TODO:  y_pred contains classes not in y_true

class BenchmarkRunner:
    def __init__(self, data, ground_truth, benchmarks, name, maxlags=3, n_jobs=1):
        self.data = data
        self.ground_truth = ground_truth
        self.benchmarks = benchmarks  # List of benchmark objects
        self.name = name
        self.maxlags = maxlags
        self.n_jobs = n_jobs
        self.results = []

        self.test_couples = None #used to store the subset of pairs of variables to test ! 

    def run_all(self):
        for benchmark in self.benchmarks:
            print(f"\nRunning {benchmark.__class__.__name__}")
            start_time = time.time()
            benchmark.run()
            if benchmark.__class__.__name__ == 'D2C':
                self.test_couples = benchmark.get_causal_dfs()
            else:
                benchmark.filter_causal_dfs(self.test_couples)
            elapsed_time = time.time() - start_time
            print(f"\n{benchmark.__class__.__name__} time: {round(elapsed_time, 2)} seconds")
            self.results.append(benchmark.evaluate())

    def save_results(self, path='../data/3vars/'):
        # Combine and save all results
        df_all_eval = pd.DataFrame(columns=['Model', 'Metric', 'Score'])
        for result in self.results:
            df_all_eval = pd.concat([df_all_eval, pd.DataFrame(result, columns=['Model', 'Metric', 'Score'])])

        df_all_eval.to_csv(path+f'{self.name}_scores.csv', index=False)

    def plot_results(self, path='../data/3vars/'):
        # Load results if not already loaded
        df_scores = pd.read_csv(path+f'{self.name}_scores.csv')

        sns.set_style("whitegrid")
        sns.set_palette("muted")  # Set the Seaborn
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Metric', y='Score', hue='Model', data=df_scores)
        plt.title("Comparison of Methods Across Different Metrics")
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        #bbox
        plt.tight_layout()

        plt.savefig(path+f'{self.name}_scores.png')
        # plt.show()


# def test(name:str = 'data', maxlags:int = 3, n_jobs:int=1, n_variables:int=6, recompute_descriptors:bool=False):
    
#     # descriptors_path = '../data/'+name+'_descriptors.csv'
#     descriptors_linear = '../data/3vars/'+name+'_descriptors_linear.pkl'
#     descriptors_nonlinear = '../data/3vars/'+name+'_descriptors_nonlinear.pkl'
#     data_path = '../data/3vars/'+name+'.pkl'

#     with open(data_path, 'rb') as f:
#         data, _, updated_dags, causal_dfs = pickle.load(f)

#     # df = pd.read_csv(descriptors_path)
#     #list to dict with key = ts_index
#     causal_dfs_dict = {}
#     for i in range(len(causal_dfs)):
#         causal_dfs_dict[i] = causal_dfs[i]

#     ground_truth = causal_dfs_dict


#     print('D2C1')
#     start = time.time()
#     d2c = D2C(data, maxlags=maxlags, n_jobs=n_jobs, ground_truth=ground_truth, descriptors_path=descriptors_linear, n_variables=n_variables, suffix='linear').run()
#     d2c_causal_dfs = d2c.get_causal_dfs() 
#     d2c.filter_causal_dfs(d2c_causal_dfs)
#     d2c.save_causal_dfs(name)
#     d2c_eval = d2c.evaluate()
#     print('D2C1 time:', np.round(time.time()-start,2), 'seconds')
    
#     print('D2C2')
#     start = time.time()
#     d2c2 = D2C(data, maxlags=maxlags, n_jobs=n_jobs, ground_truth=ground_truth, descriptors_path=descriptors_nonlinear, n_variables=n_variables, suffix='nonlinear').run()
#     # d2c2_causal_dfs = d2c2.get_causal_dfs() 
#     d2c2.filter_causal_dfs(d2c_causal_dfs)
#     d2c2.save_causal_dfs(name)
#     d2c2_eval = d2c.evaluate()
#     print('D2C2 time:', np.round(time.time()-start,2), 'seconds')

#     # print('DYNOTEARS')
#     # start = time.time()
#     # dyno = DYNOTEARS(data, maxlags=maxlags, n_jobs=5, ground_truth=ground_truth).run()
#     # dyno.filter_causal_dfs(d2c_causal_dfs)
#     # dyno.save_causal_dfs(name)
#     # dyno_eval = dyno.evaluate()
#     # print('DYNOTEARS time:', np.round(time.time()-start,2), 'seconds')

#     print('Granger')
#     start = time.time()
#     granger = Granger(data, maxlags=maxlags,  n_jobs=n_jobs, ground_truth=ground_truth).run()
#     granger.filter_causal_dfs(d2c_causal_dfs)
#     granger.save_causal_dfs(name)
#     granger_eval = granger.evaluate()
#     print('Granger time:', np.round(time.time()-start,2), 'seconds')
    
#     print('PCMCI')
#     start = time.time()
#     pcmci = PCMCI(data, maxlags=maxlags,  n_jobs=n_jobs, ground_truth=ground_truth).run()
#     pcmci.filter_causal_dfs(d2c_causal_dfs)
#     pcmci.save_causal_dfs(name)
#     pcmci_eval = pcmci.evaluate()
#     print('PCMCI time:', np.round(time.time()-start,2), 'seconds')
    
#     print('VAR')
#     start = time.time()
#     var = VAR(data, maxlags=maxlags,  n_jobs=n_jobs, ground_truth=ground_truth).run()
#     var.filter_causal_dfs(d2c_causal_dfs)
#     var.save_causal_dfs(name)
#     var_eval = var.evaluate()
#     print('VAR time:', np.round(time.time()-start,2), 'seconds')

#     print('VARLiNGAM')
#     start = time.time()
#     varlingam = VARLiNGAM(data, maxlags=maxlags,  n_jobs=n_jobs, ground_truth=ground_truth).run()
#     varlingam.filter_causal_dfs(d2c_causal_dfs)
#     varlingam.save_causal_dfs(name)    
#     varlingam_eval = varlingam.evaluate()
#     print('VARLiNGAM time:', np.round(time.time()-start,2), 'seconds')


#     # all_eval = [dyno_eval, granger_eval, pcmci_eval, var_eval, varlingam_eval]
#     all_eval = [d2c_eval,d2c2_eval, granger_eval, pcmci_eval, var_eval, varlingam_eval]
#     # all_eval = [dyno_eval, granger_eval, pcmci_eval, var_eval]
#     df_all_eval = pd.DataFrame(columns=['Model', 'Metric', 'Score'])
#     for eval_ in all_eval:
#         df_all_eval = pd.concat([df_all_eval,pd.DataFrame(eval_,columns=['Model', 'Metric', 'Score'])])


#     df_scores = pd.DataFrame(df_all_eval, columns=['Model', 'Metric', 'Score'])
#     #save
#     df_scores.to_csv('../data/3vars/'+name+'_scores.csv', index=False)
#     # Plotting
#     sns.set_style("whitegrid")
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(x='Metric', y='Score', hue='Model', data=df_scores.reset_index(drop=True))
#     plt.title("Comparison of methods Across Different Metrics")
#     plt.legend(loc='upper right')
#     #save
#     plt.savefig('../data/3vars/'+name+'_scores.png')
#     # plt.show()


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description='Test Causal Inference Methods')
#     parser.add_argument('--name', type=str, default='data', help='Name of the file to load and save the data')
#     parser.add_argument('--maxlags', type=int, default=3, help='Maximum lags for the time series')
#     parser.add_argument('--n_jobs', type=int, default=35, help='Number of jobs for parallel processing')
#     parser.add_argument('--n_variables', type=int, default=5, help='Number of variables in the time series')
#     parser.add_argument('--recompute_descriptors', type=bool, default=False, help='Whether to recompute descriptors or not')

#     args = parser.parse_args()

#     test(args.name, args.maxlags, args.n_jobs, args.n_variables, args.recompute_descriptors)    

