# %%
import numpy as np
import pandas as pd
import json
import zipfile
import bz2
import time
import os
import sys

from multiprocessing import Pool

from imblearn.ensemble import BalancedRandomForestClassifier

sys.path.append('..')
from src.d2c.utils import create_lagged_multiple_ts
from src.d2c.d2c import D2C



# %%
def my_method(data, clf, maxlags, n_variables):
    T, N = data.shape

    data_df = pd.DataFrame(data)
    lagged_data = create_lagged_multiple_ts([data_df], maxlags) 
    d2c_test = D2C(None,lagged_data,maxlags=maxlags,n_variables=n_variables)
    
    X_test = d2c_test.compute_descriptors_no_dags()
    test_df = pd.DataFrame(X_test)
    test_df = test_df.drop(['graph_id','edge_source','edge_dest'], axis=1)

    y_pred = clf.predict_proba(test_df)[:,1]
    returned = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_pred, columns=['is_causal'])], axis=1)
    of_interest = returned[['edge_source','edge_dest','is_causal']]

    extended_val_matrix = np.zeros((n_variables * (maxlags + 1), n_variables), dtype='float32')
    
    for _, row in of_interest.iterrows():
        source =int(row['edge_source'])
        dest = int(row['edge_dest'])
        weight = row['is_causal']
        extended_val_matrix[source, dest] = weight

    val_matrix = np.zeros((N, N), dtype='float32')
    lag_matrix = np.zeros((N, N), dtype='float32')

    for i in range(n_variables):
        for j in range(n_variables):
            values = extended_val_matrix[i::n_variables, j] 
            val_matrix[i, j] = np.max(values)
            lag_matrix[i, j] = np.argmax(values)

    thresholded_val_matrix = val_matrix.copy()
    thresholded_val_matrix[thresholded_val_matrix < 0.5] = 0
    thresholded_val_matrix[thresholded_val_matrix >= 0.5] = 1

    return val_matrix, 1 - thresholded_val_matrix, lag_matrix

# %%
def process_zip_file(name, _, clf, maxlags=1, n_variables=3):
    print("\rRun on {}".format(name), end='', flush=True)
    data = np.loadtxt(name)
    
    # Runtimes for your own assessment
    start_time = time.time()
    # Run your method (adapt parameters if needed)
    val_matrix, p_matrix, lag_matrix = my_method(data, clf, maxlags,n_variables)
    runtime = time.time() - start_time

    # Convert the matrices to the required format and return
    score = val_matrix.flatten()
    pvalue = p_matrix.flatten() if p_matrix is not None else None
    lag = lag_matrix.flatten() if lag_matrix is not None else None

    return score, pvalue, lag, runtime



# %%

# n_variables_list = [3, 5, 10, 20]
# noise_std_list = [0.01,0.1,0.3, 0.5, 0.75]

# dfs = []
# for i, n_variables in enumerate(n_variables_list):
#     for j, noise_std in enumerate(noise_std_list):
#         descriptors_path = os.path.join('..','data','synthetic',f'data_N{n_variables}_std{noise_std}/descriptors_var.pkl')

#         df = pd.read_pickle(descriptors_path)
#         dfs.append(df)

# training_data = pd.concat(dfs, axis=0)

training_data = pd.read_pickle(os.path.join('..','data','descriptors','training_data.pkl'))


# %%
X_train = training_data.drop(['graph_id', 'edge_source', 'edge_dest', 'is_causal','value','pvalue'], axis=1) #TODO: add VAR to causeme! 
y_train = training_data['is_causal']
clf = BalancedRandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=1, random_state=0,sampling_strategy='all', replacement=True)
clf.fit(X_train, y_train)

# %%


# %%
for file in sorted(os.listdir(os.path.join('..','data','causeme')), reverse=True)[:25]:
    if not file.endswith('.zip'):
        continue

    results = {}
    results['method_sha'] = "0931a3e645e3436b89c56f5e1274dcb7"

    maxlags = 5 #general 
    n_variables = int(file.split('N-')[1].split('_')[0])
    results['parameter_values'] = "maxlags=%d" % maxlags
    results['model'] = file.split('_N-')[0]

    experimental_setup = file.split(results['model'])[1].split('.zip')[0][1:] #remove the first underscore

    results['experiment'] = results['model'] + '_' + experimental_setup

    save_name = '{}_{}_{}'.format('d2cpy',results['parameter_values'], results['experiment'])

    experiment_folder = os.path.join('..','results','causeme','experiments')
    results_folder = os.path.join('..','results','causeme','results')
    unzip_folder = os.path.join('..','results','causeme','unzipped')

    experiment_zip = os.path.join('..','data','causeme',file)
    experiment_results = os.path.join(results_folder,save_name+'.json.bz2')

    #################################################

    scores = []
    pvalues = []
    lags = []
    runtimes = []

    results_from_mp = []

    with zipfile.ZipFile(experiment_zip, "r") as zip_ref:
        #unzip the files and make a list
        zip_ref.extractall(unzip_folder)
        names = sorted(zip_ref.namelist())
    args_list = [(os.path.join(unzip_folder,name), 'd2cpy', clf, maxlags, n_variables) for name in names]

    with Pool(processes=40) as pool:
        results_from_mp = pool.starmap(process_zip_file, args_list)

    scores, pvalues, lags, runtimes = [], [], [], []
    for result in results_from_mp:
        score, pvalue, lag, runtime = result
        scores.append(score)
        if pvalue is not None: pvalues.append(pvalue)
        if lag is not None: lags.append(lag)
        runtimes.append(runtime)

    results['scores'] = np.array(scores).tolist()
    if len(pvalues) > 0: results['pvalues'] = np.array(pvalues).tolist()
    if len(lags) > 0: results['lags'] = np.array(lags).tolist()
    results['runtimes'] = np.array(runtimes).tolist()

    # Save data
    results_json = bytes(json.dumps(results), encoding='latin1')
    with bz2.BZ2File(experiment_results, 'w') as mybz2:
        mybz2.write(results_json)

    # Empty the folder unzip_folder
    for file in os.listdir(unzip_folder):
        os.remove(os.path.join(unzip_folder,file))
    
    print("")




# %%


# %%


# %%



