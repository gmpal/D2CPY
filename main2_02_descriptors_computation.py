"""Module for computing and saving descriptors from time series data.

This module uses the DescriptorsGenerator class to compute various descriptors based
on time series data and saves the results and ground truth causal dataframes.
"""
from typing import Tuple, Any, Dict
from src.descriptors.d2c_past_gen import DescriptorsGenerator


def compute_descriptors(
    ts_builder: Any = None,
    data_path: str = None,
    n_jobs: int = 1,
    output_folder: str = "new_data/",
    mutual_information_proxy: str = "Ridge",
    proxy_params: Dict = None,
    family: Dict[str, bool] = None,
) -> Tuple[Any, Any, Any]:
    """
    Computes and saves descriptors based on time series data.

    Args:
        ts_builder: An instance of a time series builder, if available.
        data_path: Path to the directory containing time series data.
        n_jobs: Number of parallel jobs to use for computations.
        output_folder: Path to the directory where output will be saved.
        mutual_information_proxy: Name of the proxy method used for mutual information estimation.
        proxy_params: Parameters for the mutual information proxy method.
        family: Dictionary specifying which families of descriptors to compute.

    Returns:
        A tuple containing the DescriptorsGenerator instance, observed data, and ground truth causal dataframes.
    """
    # Default family of descriptors if not provided
    if family is None:
        family = {
            "basic": True,
            "var_var": True,
            "var_mb": True,
            "var_mb_given_var": True,
            "mb_mb_given_var": True,
            "structural": True,
        }

    descr_gen = DescriptorsGenerator(
        ts_builder=ts_builder,
        data_path=data_path,
        n_jobs=n_jobs,
        mutual_information_proxy=mutual_information_proxy,
        proxy_params=proxy_params,
        family=family,
    )
    descr_gen.generate()
    descr_gen.save(output_folder=output_folder)
    data = descr_gen.get_observations()
    ground_truth = descr_gen.get_causal_dfs()
    return descr_gen, data, ground_truth


if __name__ == "__main__":

    # Iterate through configurations of variables and noise standards
    for n_vars in [20,10,5,3]:
        for noise in [0.5, 0.75]:
            # Compute descriptors and get the data and ground truth
            compute_descriptors(
                data_path=f"data/synthetic/data_N{n_vars}_std{noise}/",
                n_jobs=50,
                output_folder=f"data/synthetic/data_N{n_vars}_std{noise}/",
            )
