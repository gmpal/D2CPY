import numpy as np
import pandas as pd

from numpy.linalg import inv, pinv, solve
from scipy.special import expit
from numpy.random import default_rng
from scipy.stats import entropy, zscore, skew, tstd, percentileofscore

from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np



def normalized_conditional_information(y, x1, x2=None, lin=True, verbose=True):
        """
        Normalized conditional information of x1 to y given x2
        I(x1;y| x2)= (H(y|x2)-H(y | x1,x2))/H(y|x2)
        """
        if verbose: print('normalized_conditional_information')

        if x2 is None:  # I(x1;y)= (H(y)-H(y | x1))/H(y)
            if verbose: print('x2 is None')
            entropy_y_given_x1 = normalized_prediction(x1, y, verbose=verbose)
            return max(0, 1 - entropy_y_given_x1)

        if verbose: print('x2 is not None')
        entropy_y_given_x2 = normalized_prediction(x2, y, verbose=verbose)
        x1_x2 = pd.concat([x1, x2],axis=1)
        entropy_y_given_x1_x2 = normalized_prediction(x1_x2, y, verbose=verbose)
        if verbose: print('entropy_y_given_x2: ', entropy_y_given_x2)
        return max(0, entropy_y_given_x2 - entropy_y_given_x1_x2 ) / (entropy_y_given_x2 + 0.01)

def normalized_prediction(X, Y, lin=True, verbose=True):
    """
    Normalized mean squared error of the dependency
    """

    if verbose: print('normalized_prediction')
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)
    if verbose: print(X.shape)    
    N, n = X.shape

    if n > 1: # TODO: check the case if all columns are constant, return 1
        if verbose: print('n > 1')
        X = np.delete(X, np.where(np.std(X, axis=0) < 0.01)[0], axis=1) # if there is any constant column, remove it
        X = np.delete(X, np.where(np.isnan(np.sum(X, axis=0)))[0], axis=1) # if there is any nan, remove it
    else:
        if verbose: print('n <= 1')
        if np.any(np.isnan(X)): return 1 # TODO: check this
            
    XX = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    if N < 5 or np.any(np.isnan(XX)): 
        if verbose: print('N < 5 or np.any(np.isnan(XX))')
        return np.var(Y)
    if lin: 
        if verbose: print('lin')
        return max(1e-3, ridge_regression(XX, Y)['MSE_loo'] / (1e-3 + np.var(Y)))

    #TODO: implement the nonlinear case



def ridge_regression(X_train, Y_train, X_test=None, lambda_val=1e-3):
    """
    Perform ridge regression and returns the trained model, predictions, and metrics.

    Args:
        X_train (np.ndarray): The training design matrix.
        Y_train (np.ndarray): The training response vector.
        X_test (np.ndarray, optional): The test design matrix. Defaults to None.
        lambda_val (float, optional): The regularization parameter. Defaults to 1e-3.

    Returns:
        dict: Dictionary containing the trained model, predictions, and computed metrics.
    """
    model = Ridge(alpha=lambda_val)
    model.fit(X_train, Y_train)

    Y_train_hat = model.predict(X_train)
    e_train = Y_train - Y_train_hat
    MSE_emp = mean_squared_error(Y_train, Y_train_hat)
    NMSE = MSE_emp / (np.var(Y_train)**2)

    e_loo = cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=len(X_train))
    MSE_loo = -np.mean(e_loo)

    Y_test_hat = None
    if X_test is not None:
        Y_test_hat = model.predict(X_test)

    return {
        'e_train': e_train,
        'MSE_emp': MSE_emp,
        'NMSE': NMSE,
        'MSE_loo': MSE_loo,
        'Y_train_hat': Y_train_hat,
        'Y_test_hat': Y_test_hat,
        'model': model,
    }



# def ridge_regression(X_train, Y_train, X_test=None, lambda_val=1e-3, verbose=True):
#     """
#         This function performs ridge regression, a variant of linear regression that 
#         includes a regularization term. This method is used to prevent overfitting and to 
#         handle multicollinearity (high correlation among predictors) in data. The regularization 
#         term is controlled by a parameter lambda, which shrinks the coefficients towards zero.

#         The function is based on the R code from the original D2C package.

#         Args:
#             X (np.ndarray): The design matrix.
#             Y (np.ndarray): The response vector.
#             X_ts (np.ndarray, optional): The test design matrix. Defaults to None.
#             lambda_val (float, optional): The regularization parameter. Defaults to 1e-3.

#         Returns:
#             dict: Dictionary containing the computed metrics.

#     """
#     if verbose: print('ridge_regression')

#     n = X_train.shape[1]  # Number of predictors
#     p = n + 1
#     N = X_train.shape[0]  # Number of observations

#     # Prepare the design matrix by adding a column of ones for the intercept
#     XX = np.c_[np.ones((N, 1)), X_train]

#     if lambda_val < 0:
#         if verbose: print('lambda_val < 0')
#         min_MSE_loo = np.inf
#         for lambda_current in np.arange(1e-3, 5, 0.5):
#             H1 = pinv(XX.T @ XX + lambda_current * np.eye(p))
#             beta_hat = H1 @ XX.T @ Y_train
#             H = XX @ H1 @ XX.T
#             Y_hat = XX @ beta_hat
#             e = Y_train - Y_hat
#             e_loo = e / (1 - np.diag(H))
#             MSE_loo = np.mean(e_loo**2)
#             if MSE_loo < min_MSE_loo:
#                 lambda_val = lambda_current
#                 min_MSE_loo = MSE_loo

#     H1 = pinv(XX.T @ XX + lambda_val * np.eye(p))
#     beta_hat = H1 @ XX.T @ Y_train
#     H = XX @ H1 @ XX.T
#     Y_hat = XX @ beta_hat
#     e = Y_train - Y_hat
#     var_hat_w = e.T @ e / (N - p)
#     MSE_emp = np.mean(e**2)
#     e_loo = e / (1 - np.diag(H))
#     MSE_loo = np.mean(e_loo**2)
#     NMSE = np.mean(e_loo**2) / (np.var(Y_train)**2)
#     if verbose: print('NMSE: ', NMSE)
#     Y_hat_ts = None
#     if X_test is not None:
#         if verbose: print('X_test is not None')
#         N_ts = X_test.shape[0]
#         if np.isscalar(X_test) and n > 1:
#             Y_hat_ts = np.r_[1, X_test] @ beta_hat
#         else:
#             XX_ts = np.c_[np.ones((N_ts, 1)), X_test]
#             Y_hat_ts = XX_ts @ beta_hat

#     return {
#         'e': e,
#         'beta_hat': beta_hat,
#         'MSE_emp': MSE_emp,
#         'sdse_emp': tstd(e**2),
#         'var_hat': var_hat_w,
#         'MSE_loo': MSE_loo,
#         'sdse_loo': tstd(e_loo**2),
#         'Y_hat': Y_hat,
#         'Y_hat_ts': Y_hat_ts,
#         'e_loo': e_loo,
#         'NMSE': NMSE
#     }

def column_based_correlation(X,Y,verbose=True):
    #TODO: multidimensional Y 
    if verbose: print('column_based_correlation')
    columns_of_X = X.shape[1]  # Number of columns in X

    correlation_vector = np.zeros(columns_of_X)  # Initialize correlation vector

    for i in range(columns_of_X):
        correlation_matrix = np.corrcoef(X.iloc[:, i], Y.iloc[:, 0])
        correlation_value = correlation_matrix[0, 1]
        correlation_vector[i] = correlation_value

    correlation_array = correlation_vector.reshape(1, -1)

    # Print the correlation vector
    return(correlation_array[0])

def co2i(X,Y, verbose=True):

    # check if Y is a pd.series and make it dataframe
    if isinstance(Y, pd.Series):
        Y = pd.DataFrame(Y)

    if verbose: print('co2i')

    correlation_vector = column_based_correlation(X,Y, verbose=verbose)
    corr_sq = np.square(correlation_vector)

    I = -0.5 * np.log(1 - corr_sq)
    if verbose: print('I: ', I)

    return I

def rankrho(X, Y, nmax=5, regr=False, verbose=True):
    """
    Perform mutual information ranking between two arrays.

    Parameters:
        X (array-like): Input array with shape (N, n), representing N samples and n features.
        Y (array-like): Input array with shape (N,). Target variable.
        nmax (int, optional): Number of top-ranked features to return. Defaults to 5.
        regr (bool, optional): Flag indicating whether to use ridge regression for ranking. Defaults to False.
        verbose (bool, optional): Flag indicating whether to display progress information. Defaults to True.

    Returns:
        list: Indices of the top-ranked features in X based on mutual information with Y.

    Notes:
        The function calculates the mutual information between each column of X and Y, and returns the indices of the
        columns in X that have the highest mutual information with Y. The number of indices returned is determined by the
        nmax parameter.

        If the variance of Y is less than 0.01, the function returns a list of indices ranging from 1 to nmax.

        If regr is False, the function uses the co2i function to calculate the mutual information. If regr is True, ridge
        regression is performed for each column of X with Y as the target variable, and the maximum coefficient value is
        used as the mutual information.

        The input arrays X and Y are expected to have compatible shapes, where X has shape (N, n) and Y has shape (N,).
        The function assumes that the columns of X and Y correspond to the same samples.

    Example:
        X = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]

        Y = [10, 20, 30]

        top_features = rankrho(X, Y, nmax=2, regr=True)

        # Output: [3, 2]
        # The third column of X has the highest mutual information with Y, followed by the second column.

    """
    if verbose: print('rankrho')
    # Number of columns in X and Y
    n = X.shape[1]
    # m = Y.shape[1] #TODO: handle the multivariate case
    N = X.shape[0]

    if np.var(Y) < 0.01:
        if verbose: print('np.var(Y) < 0.01')
        return list(range(1, nmax + 1))
    
    # Scaling X
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    Iy = np.zeros(n)

    if not regr:
        if verbose: print('not regr')
        Iy = co2i(X, Y, verbose=verbose)
        if verbose: print(Iy)
    else:
        if verbose: print('regr')
        for i in range(n):
            Iy[i] = abs(ridge_regression(X[:, i], Y)['beta_hat'][1])

    # if m > 1:
    #     Iy = np.mean(Iy, axis=1)

    return (np.argsort(Iy)[::-1] + 1)[:nmax]


def mRMR(X, Y, nmax, verbose=True):

    if verbose: print('mRMR')
    num_features = X.shape[1]
    
    # Calculate mutual information between each feature in X and Y
    mi_XY = mutual_info_regression(X, Y)
    if verbose: print("mi_XY: ", mi_XY)

    # Start with the feature with maximum MI with Y
    indices = [np.argmax(mi_XY)]
    
    for _ in range(nmax - 1):
        remaining_indices = list(set(range(num_features)) - set(indices))
        if verbose: print("remaining_indices: ", remaining_indices)
        mi_XX = np.zeros(len(remaining_indices))
        
        # Calculate mutual information between selected features and remaining features
        for i in range(len(remaining_indices)):
            mi_XX[i] = mutual_info_regression(X.iloc[:, indices], X.iloc[:, remaining_indices[i]])[0]
        
        # Calculate MRMR score for each remaining feature
        mrmr_scores = mi_XY[remaining_indices] - np.mean(mi_XX)
        if verbose: print("mrmr_scores: ", mrmr_scores)
        # Select feature with maximum MRMR score
        indices.append(remaining_indices[np.argmax(mrmr_scores)])
    
    return indices





# def mRMR(X, Y, nmax=5, first=None, all=False, back=False, lam=1, categ=False, verbose=True):
#     """
#     The mRMR (minimum Redundancy Maximum Relevance) filter is a feature selection method commonly 
#     used in machine learning and data analysis tasks. Its goal is to identify a subset of features 
#     that have the highest relevance to the target variable while minimizing redundancy among the 
#     selected features.
#     """
#     if verbose: print('mRMR')
#     n_columns = X.shape[1]
    
#     if categ and Y.dtype == 'object':
#         if verbose: print('Categorical Y')
#         relevance = np.zeros(n_columns)
#         redundancy = np.zeros((n_columns, n_columns))
        
#         for i in range(n_columns - 1):
#             for j in range(i + 1, n_columns):
#                 redundancy[i, j] = np.mean([mutual_info_score(X[:, i], X[:, j]),
#                                    mutual_info_score(X[:, j], X[:, i])])
#             relevance[i] = mutual_info_score(X[:, i], Y)
#     else:
#         if verbose: print('Continuous Y')
#         #scale X
#         X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        
#         relevance = co2i(X, Y) 
#         if verbose: print("relevance: ", relevance)
        
#         redundancy = np.zeros((n_columns, n_columns))
#         for i in range(n_columns):
#             redundancy[i] = co2i(X, X.iloc[:,i]) #TODO: check this
        
#         if verbose: print("redundancy: ", redundancy)
    
#     subs = [np.argmax(relevance)]
#     if verbose: print("subs: ", subs)

#     for j in range(len(subs), min(n_columns - 1, nmax)):
#         if verbose: print(" Looping through " , j)
#         mrmr = np.full(n_columns, -np.inf)
        
#         if len(subs) < (n_columns - 1):
#             if verbose: print("  len(subs) < (n - 1)")
#             if len(subs) > 1:
#                 mrmr[subs] = relevance[subs] + lam * np.mean(-redundancy[subs][:, np.setdiff1d(range(n_columns), subs)], axis=1)
#             else:
#                 mrmr[subs] = relevance[subs] + lam * (-redundancy[subs][:, np.setdiff1d(range(n_columns), subs)])
#         else:
#             if verbose: print("  len(subs) > (n - 1)")
#             mrmr[subs] = np.inf
        
#         s = np.argmax(mrmr)
#         sortmrmr = np.argsort(mrmr)[::-1][len(subs):]
#         allfs = np.concatenate((subs, sortmrmr))
#         subs = np.concatenate((subs, [s]))
#         if verbose: print("  subs: ", subs)
    
#     if back:
#         if verbose: print("Backward selection")
#         nsubs = []
#         while len(subs) > 1:
#             pd = np.zeros(len(subs))
#             for ii in range(len(subs)):
#                 X_temp = np.delete(X, np.where(subs == subs[ii])[0], axis=1)
#                 pd[ii] = ridge_regression(X_temp, Y)[1]
            
#             nsubs.insert(0, subs[np.argmin(pd)])
#             subs = np.delete(subs, np.argmin(pd))
        
#         subs = np.concatenate((subs, nsubs))
    
#     if all:
#         return allfs
#     else:
#         return subs[:nmax]
    




def ecdf(data, verbose=True):
    if verbose: print('ecdf')
    def _ecdf(x):
        return percentileofscore(data, x) / 100
    return _ecdf





def coeff(y, x1, x2=None, verbose=True):
    if verbose: print('coeff')
    if x2 is not None:
        X = np.column_stack((x1, x2))
    else:
        X = np.array(x1).reshape(-1, 1)

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]  # return the coefficient of x1
