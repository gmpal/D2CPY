import pandas as pd 
import os 
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import statsmodels.tsa.api as tsa


# Main
if __name__ == "__main__":
    descriptors = {}
    for n_variables in [50]:
            descriptors[n_variables] = {}
            for noise_std in [0.3]:
                maxlags = 3
                output_folder = f'data_N{n_variables}_std{noise_std}/'
                file_name = 'descriptors_Ridgefamily-0-1-2-3-4-5.pkl'

                descriptors = pd.read_pickle('../'+output_folder + file_name)


                data_path = output_folder
                loaded_observations = {}
                loaded_dags = {}
                loaded_causal_dfs = {}
                for file in os.listdir('../'+data_path):
                    if file.startswith('data'):
                        index = file.split('_')[1].split('.')[0]
                        with open('../'+data_path+file, 'rb') as f:
                            loaded_observations[int(index)], loaded_dags[int(index)], loaded_causal_dfs[int(index)], _ = pickle.load(f)

                list_results = []
                for generative_process_idx in tqdm(range(1, 21)):
                    lagged_time_series = create_lagged_multiple_ts(loaded_observations[generative_process_idx], maxlags)
                    n_jobs = 30
                    if n_jobs == 1:
                        results = []
                        for internal_idx, ts in enumerate(lagged_time_series):
                            results.append(process_time_series((generative_process_idx, internal_idx, ts, n_variables, maxlags, loaded_observations, descriptors)))
                    else:
                        with Pool(n_jobs) as pool:
                            args = [(generative_process_idx, internal_idx, ts, n_variables, maxlags, loaded_observations, descriptors) for internal_idx, ts in enumerate(lagged_time_series)]
                            results = pool.map(process_time_series, args)
                    list_results.append(results)

                list_flat = [item for sublist in list_results for item in sublist]
                descriptors_var = pd.concat(list_flat, axis=0)
                descriptors_var = descriptors_var[[c for c in descriptors_var if c not in ['is_causal']] + ['is_causal']]
                descriptors_var.to_pickle('../'+data_path+'descriptors_var.pkl')