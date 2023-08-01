"""
This file must contain a function called my_method that triggers all the steps 
required in order to obtain

 *val_matrix: mandatory, (N, N) matrix of scores for links
 *p_matrix: optional, (N, N) matrix of p-values for links; if not available, 
            None must be returned
 *lag_matrix: optional, (N, N) matrix of time lags for links; if not available, 
              None must be returned

Zip this file (together with other necessary files if you have further handmade 
packages) to upload as a code.zip. You do NOT need to upload files for packages 
that can be imported via pip or conda repositories. Once you upload your code, 
we are able to validate results including runtime estimates on the same machine.
These results are then marked as "Validated" and users can use filters to only 
show validated results.

Shown here is a vector-autoregressive model estimator as a simple method.
"""
import sys
sys.path.append("..")
import numpy as np
from d2c.simulatedTimeSeries import SimulatedTimeSeries
from d2c.D2C import D2C
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Your method must be called 'my_method'
# Describe all parameters (except for 'data') in the method registration on CauseMe
def my_method(data, maxlags=1, correct_pvalues=True):

    # Input data is of shape (time, variables)
    T, N = data.shape

    data_df = pd.DataFrame(data)

    d2c_test = D2C([None],[data_df])
    X_test = d2c_test.compute_descriptors_no_dags()

    training_data = pd.read_csv('./timeseries_training.csv')

    X_train = training_data.drop(['graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
    y_train = training_data['is_causal']

    test_df = pd.DataFrame(X_test).drop(['graph_id', 'edge_source', 'edge_dest'], axis=1)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(test_df)

    returned = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_pred, columns=['is_causal'])], axis=1)
    of_interest = returned[['edge_source', 'edge_dest','is_causal']]
    

    val_matrix = np.zeros((N, N), dtype='float32')

    for index, row in of_interest.iterrows():
        source = row['edge_source']
        dest = row['edge_dest']
        weight = row['is_causal']
        val_matrix[source, dest] = weight

    return val_matrix, None, None
