import random
import math
import networkx as nx
from data_generation.generate_known_TS import TSBuilder
from descriptors.d2c_past_gen import DescriptorsGenerator
from benchmark import BenchmarkRunner, D2C, VAR, VARLiNGAM, PCMCI, Granger


def generate_time_series(output_folder, series_time_lag, n_variables, timesteps_per_series, n_series_per_generative_process):
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
                           n_variables=n_variables, n_iterations=n_series_per_generative_process)
    ts_builder.build()
    ts_builder.save(output_folder=output_folder)
    return ts_builder


def compute_descriptors(ts_builder, n_jobs, output_folder):
    """
    Computes the descriptors for the time series data.

    Args:
        ts_builder (TSBuilder): The TSBuilder instance containing the time series data.
        n_jobs (int): The number of jobs for parallel processing.
        output_folder (str): The folder where the descriptors will be saved.

    Returns:
        DescriptorsGenerator: An instance of DescriptorsGenerator with computed descriptors.
    """
    descr_gen = DescriptorsGenerator(ts_builder=ts_builder, n_jobs=n_jobs)
    data = descr_gen.get_observations()
    ground_truth = descr_gen.get_causal_dfs()
    # Optionally save the descriptors
    # descr_gen.save(output_folder=output_folder)
    return descr_gen, data, ground_truth


def run_benchmarks(data, ground_truth, series_time_lag, n_jobs, output_folder, descriptors_path, n_variables, suffix):
    """
    Runs various benchmark algorithms on the time series data.

    Args:
        data: The time series data.
        ground_truth: The ground truth causal dataframes.
        series_time_lag (int): The time lag for the series.
        n_jobs (int): The number of jobs for parallel processing.
        output_folder (str): The folder where the benchmark results will be saved.
        descriptors_path (str): The path to the saved descriptors file.
        n_variables (int): The number of variables in the time series.
        suffix (str): A suffix to be used in naming the output files.
    """
    benchmarks = [
        D2C(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth, 
            descriptors_path=descriptors_path, n_variables=n_variables, suffix=suffix),
        Granger(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth),
        PCMCI(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth),
        VAR(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth),
        VARLiNGAM(data, maxlags=series_time_lag, n_jobs=n_jobs, ground_truth=ground_truth)
    ]

    runner = BenchmarkRunner(data=data, ground_truth=ground_truth, benchmarks=benchmarks,
                             name='test_run', maxlags=series_time_lag, n_jobs=n_jobs)
    runner.run_all()
    runner.save_results(path=output_folder)
    runner.plot_results(path=output_folder)


if __name__ == "__main__":
    # Configuration
    series_time_lag = 3
    n_variables = 3 
    timesteps_per_series = 200
    n_series_per_generative_process = 2
    n_jobs = 15
    output_folder = 'new_data/'

     # Additional parameters for D2C
    descriptors_path = output_folder + 'descriptors.pkl'
    suffix = 'linear'


    # PART 1: Generation of observations
    ts_builder = generate_time_series(output_folder, series_time_lag, n_variables, 
                                      timesteps_per_series, n_series_per_generative_process)

    # PART 2: Computation of descriptors
    descr_gen, data, ground_truth = compute_descriptors(ts_builder, n_jobs, output_folder)

    # PART 3: Computation of causal graphs
    run_benchmarks(data, ground_truth, series_time_lag, n_jobs, output_folder, descriptors_path, n_variables, suffix)

