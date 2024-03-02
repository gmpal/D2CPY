"""This module generates time series data and saves it to files.

It defines a function `generate_time_series` to generate the data based on specified parameters,
and executes a series of generation tasks if run as the main program.
"""

import os
from src.data_generation.generate_known_TS import TSBuilder

def generate_time_series(
    output_folder: str,
    series_time_lag: int,
    n_variables: int,
    timesteps_per_series: int,
    n_series_per_generative_process: int,
    noise_std: float,
    seed: int,
) -> TSBuilder:
    """
    Generates and saves a collection of time series data.

    Args:
        output_folder: The directory path to save the generated time series data.
        series_time_lag: The maximum lag in the autoregressive model of the time series.
        n_variables: The number of variables in each time series.
        timesteps_per_series: The number of timesteps in each time series.
        n_series_per_generative_process: The number of series to generate per generative process.
        noise_std: The standard deviation of the noise in the time series data.
        seed: The seed for random number generation to ensure reproducibility.

    Returns:
        An instance of TSBuilder containing the generated time series data.
    """
    ts_builder = TSBuilder(
        timesteps=timesteps_per_series,
        maxlags=series_time_lag,
        n_variables=n_variables,
        n_iterations=n_series_per_generative_process,
        noise_std=noise_std,
        seed=seed,
    )
    ts_builder.build()
    ts_builder.save(output_folder=output_folder, single_file=True)
    return ts_builder


if __name__ == "__main__":

    # Iterating over different configurations of variables and noise standard deviations
    for n_vars in [3, 5, 10, 20, 50]:
        for noise in [1, 5, 10]:
            # Define the output directory for the current configuration
            output = f"data_N{n_vars}_std{noise}/"
            # Ensure the output directory exists
            os.makedirs(output, exist_ok=True)

            # Generate and save the time series data for the current configuration
            generate_time_series(
                output_folder = output,
                series_time_lag = 3,
                n_variables = n_vars,
                timesteps_per_series = 250,
                n_series_per_generative_process = 50,
                noise_std = noise,
                seed = 42
            )
