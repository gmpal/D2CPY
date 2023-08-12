"""
This script can be used to iterate over the datasets of a particular experiment.
Below you import your function "my_method" stored in the module causeme_my_method.

Importantly, you need to first register your method on CauseMe.
Then CauseMe will return a hash code that you use below to identify which method
you used. Of course, we cannot check how you generated your results, but we can
validate a result if you upload code. Users can filter the Ranking table to only
show validated results.
"""
import numpy as np
import json
import zipfile
import bz2
import time
import sys
sys.path.append("..")
sys.path.append("../d2c/")
import numpy as np
from multiprocessing import Pool
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


from causeme_my_method import my_method


def process_zip_file(name, clf, maxlags=1):
    print("Run on {}".format(name))
    data = np.loadtxt('experiments/'+name)

    # Runtimes for your own assessment
    start_time = time.time()

    # Run your method (adapt parameters if needed)
    val_matrix, p_matrix, lag_matrix = my_method(data, clf, maxlags)
    runtime = time.time() - start_time

    # Convert the matrices to the required format and return
    score = val_matrix.flatten()
    pvalue = p_matrix.flatten() if p_matrix is not None else None
    lag = lag_matrix.flatten() if lag_matrix is not None else None

    return score, pvalue, lag, runtime


if __name__ == '__main__':


 
    # Setup a python dictionary to store method hash, parameter values, and results
    results = {}

    ################################################
    # Identify method and used parameters
    ################################################

    # Method name just for file saving
    method_name = 'varmodel-python'

    # Insert method hash obtained from CauseMe after method registration
    results['method_sha'] = "e182a71f4e1645a1b9ede10f615df88a"

    # The only parameter here is the maximum time lag
    maxlags = 1

    # Parameter values: These are essential to validate your results
    # provided that you also uploaded code
    results['parameter_values'] = "maxlags=%d" % maxlags

    #################################################
    # Experiment details
    #################################################
    # Choose model and experiment as downloaded from causeme
    results['model'] = 'linear-VAR'

    # Here we choose the setup with N=3 variables and time series length T=150
    experimental_setup = 'N-3_T-150'
    results['experiment'] = results['model'] + '_' + experimental_setup

    # Adjust save name if needed
    save_name = '{}_{}_{}'.format(method_name,
                                results['parameter_values'],
                                results['experiment'])

    # Setup directories (adjust to your needs)
    experiment_zip = '../causeme/experiments/%s.zip' % results['experiment']
    results_file = '../causeme/results/%s.json.bz2' % (save_name)

    #################################################

    # Start of script
    scores = []
    pvalues = []
    lags = []
    runtimes = []

    # (Note that runtimes on causeme are only shown for validated results, this is more for
    # your own assessment here)

    # Loop over all datasets within an experiment
    # Important note: The datasets need to be stored in the order of their filename
    # extensions, hence they are sorted here
    print("Load data")
  

    # This will hold results from all processes
    results_from_mp = []

    with zipfile.ZipFile(experiment_zip, "r") as zip_ref:
        #unzip the files and make a list
        zip_ref.extractall("experiments")
        names = sorted(zip_ref.namelist())

    training_data = pd.read_csv('./_descriptors.csv')

    X_train = training_data.drop(['graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
    y_train = training_data['is_causal']

    clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    clf.fit(X_train, y_train)

    args_list = [(name, clf) for name in names]

    # Create a pool of worker processes
    with Pool(processes=10) as pool:
        results_from_mp = pool.starmap(process_zip_file, args_list)

    # Extract the results to the original lists
    scores, pvalues, lags, runtimes = [], [], [], []
    for result in results_from_mp:
        score, pvalue, lag, runtime = result
        scores.append(score)
        if pvalue is not None: pvalues.append(pvalue)
        if lag is not None: lags.append(lag)
        runtimes.append(runtime)

    # Store arrays as lists for json
    results['scores'] = np.array(scores).tolist()
    if len(pvalues) > 0: results['pvalues'] = np.array(pvalues).tolist()
    if len(lags) > 0: results['lags'] = np.array(lags).tolist()
    results['runtimes'] = np.array(runtimes).tolist()

    # Save data
    print('Writing results ...')
    results_json = bytes(json.dumps(results), encoding='latin1')
    with bz2.BZ2File(results_file, 'w') as mybz2:
        mybz2.write(results_json)
