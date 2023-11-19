# %%
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("..")
sys.path.append("../d2c/")
# Generating random data for the test
np.random.seed(42)  # Seed for reproducibility
from utils import *
from d2c import d2c



# %%
def jaccard_similarity(set_a, set_b):
    """Compute the Jaccard similarity between two sets."""
    intersection_len = len(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    return intersection_len / union_len


# %%
def generate_data(n_samples, n_features, top_k_features):
    # Step 1: Generate Y
    Y = np.random.rand(n_samples)

    # Step 2: Generate Columns of X
    X = []
    for i in range(n_features):
        if i < top_k_features:  # these columns will be related to Y
            factor = np.random.rand()  # random factor
            noise = np.random.rand(n_samples) * 0.05  # small noise to make it not perfectly correlated
            X.append(Y * factor + noise)
        else:
            X.append(np.random.rand(n_samples))
    X = np.array(X).TP

    return pd.DataFrame(X), pd.Series(Y)

# %%
nmax = 10
# Number of times the test will run
num_iterations = 100

# Expected result
expected_features = set(range(1, nmax + 1))

# Threshold for Jaccard similarity
similarity_threshold = 0.65

for i in range(num_iterations):
    X, Y = generate_data(100, 100, nmax)
    top_features = rankrho(X, Y, nmax=nmax, regr=False)
    
    # Check the Jaccard similarity between top features and expected features
    similarity = jaccard_similarity(set(top_features), expected_features)
    assert similarity >= similarity_threshold, f"Failed at iteration {i} with Jaccard similarity of {similarity:.2f}"


# %%
nmax = 10
# Number of times the test will run
num_iterations = 50

expected_features = set(range(1, nmax + 1))

# Threshold for Jaccard similarity
similarity_threshold = 0.8

for i in range(num_iterations):
    X, Y = generate_data(100, 100, nmax)
    top_features = rankrho(X, Y, nmax=nmax, regr=True)

     # Check the Jaccard similarity between top features and expected features
    similarity = jaccard_similarity(set(top_features), expected_features)
    assert similarity >= similarity_threshold, f"Failed at iteration {i} with Jaccard similarity of {similarity:.2f}"



