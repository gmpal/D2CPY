import random
import math
import networkx as nx
import os
from data_generation.generate_known_TS import TSBuilder
from descriptors.d2c_past_gen import DescriptorsGenerator
import os
from benchmark import BenchmarkRunner, D2C, VAR, VARLiNGAM, PCMCI, Granger, DYNOTEARS


def generate_time_series(output_folder, series_time_lag, n_variables, timesteps_per_series, n_series_per_generative_process, noise_std, seed):
    """
    Generates time series data.

    Args:
        output_folder (str): The folder where the generated data will be saved.
        series_time_lag (int): The time lag for the series.
        n_variables (int): The number of variables in the time series.
        timesteps_per_series (int): The number of timesteps per series.
        n_series_per_generative_process (int): The number of series per generative process.

    Returns:
        TSBuilder: An instance of TSBuilder with the generated time series.
    """
    ts_builder = TSBuilder(timesteps=timesteps_per_series, maxlags=series_time_lag, 
                           n_variables=n_variables, n_iterations=n_series_per_generative_process, noise_std=noise_std, seed=seed)
    ts_builder.build()
    ts_builder.save(output_folder=output_folder,single_file=True)
    return ts_builder


def compute_descriptors(ts_builder = None, data_path = None, n_jobs=1, output_folder= 'new_data/', mutual_information_proxy='Ridge', proxy_params=None, family={'basic': True ,'var_var': True, 'var_mb':True, 'var_mb_given_var':True,'mb_mb_given_var':True,'structural':True}):
    """
    Computes the descriptors for the time series data.

    Args:
        ts_builder (TSBuilder): The TSBuilder instance containing the time series data.
        n_jobs (int): The number of jobs for parallel processing.
        output_folder (str): The folder where the descriptors will be saved.

    Returns:
        DescriptorsGenerator: An instance of DescriptorsGenerator with computed descriptors.
    """
    descr_gen = DescriptorsGenerator(ts_builder=ts_builder, data_path = data_path, n_jobs=n_jobs, mutual_information_proxy=mutual_information_proxy, proxy_params=proxy_params, family=family)
    descr_gen.generate()
    descr_gen.save(output_folder=output_folder)
    data = descr_gen.get_observations()
    ground_truth = descr_gen.get_causal_dfs()
    return descr_gen, data, ground_truth


def run_benchmarks(data, ground_truth, series_time_lag, n_jobs, output_folder, descriptors_folder, n_variables, suffix, n_gen_proc):
    """
    Runs various benchmark algorithms on the time series data.

    Args:
        data: The time series data.
        ground_truth: The ground truth causal dataframes.
        series_time_lag (int): The time lag for the series.
        n_jobs (int): The number of jobs for parallel processing.
        output_folder (str): The folder where the benchmark results will be saved.
        descriptors_folder (str): The path to the saved descriptors file.
        n_variables (int): The number of variables in the time series.
        suffix (str): A suffix to be used in naming the output files.
    """
    # benchmarks = []
    # for descriptors_file in os.listdir(descriptors_folder):
    #     descriptors_path = descriptors_folder + descriptors_file
    #     method = descriptors_path.split('/')[-1].split('_')[1].split('.')[0]
    #     params = ''
    #     if method == 'LOWESS':
    #         params = descriptors_path.split('/')[-1].split('_')[2].split('.pkl')[0]
    #     if method == 'Ridge':
    #         params = descriptors_path.split('/')[-1].split('_')[2].split('.pkl')[0]
    #     d2c = D2C(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth, 
    #         descriptors_path=descriptors_path, n_variables=n_variables, suffix=method + params, n_gen_proc=n_gen_proc)
    #     benchmarks.append(d2c)

    benchmarks = []
    for file in os.listdir(descriptors_folder):
        if file.startswith('descriptors_var'):
            descriptors_path = descriptors_folder + file
            d2c = D2C(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth, 
                descriptors_path=descriptors_path, n_variables=n_variables, suffix='', n_gen_proc=n_gen_proc)
            benchmarks.append(d2c)


    benchmarks.extend([
        # Granger(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth),
        # PCMCI(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth),
        VAR(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth),
        # VARLiNGAM(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth),
        # DYNOTEARS(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth)

    ])

    runner = BenchmarkRunner(data=data, ground_truth=ground_truth, benchmarks=benchmarks,
                             name='test_run', maxlags=series_time_lag, n_jobs=n_jobs)
    runner.run_all()
    runner.save_results(path=output_folder)
    runner.plot_results(path=output_folder)


if __name__ == "__main__":

    # Configuration
    series_time_lag = 3
    timesteps_per_series = 250
    n_gen_proc = 20
    n_series_per_generative_process = 50
    n_jobs = 50

    for n_variables in [3]:
        for noise_std in [0.3]:
            
            seed = 42 
            
            output_folder = f'data_N{n_variables}_std{noise_std}/'
            suffix = f'N{n_variables}_std{noise_std}'

            # Additional parameters for D2C
            descriptors_folder = output_folder
            
            data_path = output_folder

            descr_gen = DescriptorsGenerator(data_path = data_path, n_jobs=n_jobs, mutual_information_proxy='Ridge')    
            data = descr_gen.get_observations()
            ground_truth = descr_gen.get_causal_dfs()
            run_benchmarks(data, ground_truth, series_time_lag, n_jobs, output_folder, descriptors_folder, n_variables, suffix, n_gen_proc)


