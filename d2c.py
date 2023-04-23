import numpy as np
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.linalg import solve

import numpy as np
from typing import Union
from sklearn.preprocessing import scale

import numpy as np
from sklearn.preprocessing import scale

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import numpy as np
import statsmodels.api as sm

from typing import Optional

from scipy.stats import percentileofscore

import numpy as np
import pandas as pd
from functools import reduce
from typing import List, Union
from itertools import combinations





# TODO: from some_module import D2C_n, D2C_2, mimr, rankrho, epred, HOC, norminf, stab

def epred(X: np.ndarray, Y: np.ndarray, lin: bool = True, norm: bool = True) -> Union[np.ndarray, float]:
    """
    Predicts the error between X and Y using either linear regression or lazy prediction.

    Args:
    - X: A 2D numpy array of shape (N, n) containing the independent variables.
    - Y: A 1D numpy array of shape (N,) containing the dependent variable.
    - lin: A boolean value indicating whether to use linear regression (default True).
    - norm: A boolean value indicating whether to normalize X (default True).

    Returns:
    - If lin is True, returns a 1D numpy array of shape (n+1,) containing the coefficients of the linear regression model.
    - If lin is False, returns a float representing the error predicted using lazy prediction.
    """

    N, n = X.shape

    if n > 1:
        w_const = np.where(np.std(X, axis=0) < 0.01)[0]
        if w_const.size > 0:
            X = np.delete(X, w_const, axis=1)
        n = X.shape[1]

    XX = scale(X)
    
    if N < 5 or np.isnan(XX).any():
        raise ValueError("Error in pred")
    
    if lin:
        return regrlin(XX, Y)  # TODO: You need to implement the 'regrlin' function
    c1 = max(3, min(10, N - 2))
    c2 = max(c1, 5, min(N, 20))
    e = Y - lazy_pred(XX, Y, XX, conPar=[c1, c2], linPar=None, class_=False, cmbPar=10)  # TODO: You need to implement the 'lazy_pred' function
    
    return e




def stab(X: np.ndarray, Y: np.ndarray, lin: bool = True, R: int = 10) -> int:
    """
    Computes the stability of a linear regression model.

    Args:
    - X: A 1D numpy array of shape (N,) containing the independent variable.
    - Y: A 1D numpy array of shape (N,) containing the dependent variable.
    - lin: A boolean value indicating whether to use linear regression (default True).
    - R: An integer indicating the number of iterations to perform (default 10).

    Returns:
    - An integer value of -1, 0, or 1 indicating negative, neutral, or positive stability, respectively.
    """
    X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-4)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-4)

    Xhat = []
    Yhat = []

    for r in range(R):
        m1 = np.random.rand()
        m2 = np.random.rand()
        p = norm.pdf(X, loc=m1, scale=0.25) + norm.pdf(X, loc=m2, scale=0.25)
        rX = np.random.choice(X, 100, replace=True, p=p/np.sum(p))
        rY = Y[np.isin(X, rX)]

        D = pd.DataFrame({"X": rX, "Y": rY})
        L = LinearRegression().fit(D[["X"]], D["Y"])
        L2 = LinearRegression().fit(D[["Y"]], D["X"])

        Xts = np.arange(0, 1, 0.01)
        Yts = np.arange(0, 1, 0.01)
        pY = L.predict(Xts.reshape(-1, 1))
        pX = L2.predict(Yts.reshape(-1, 1))

        Yhat.append(pY)
        Xhat.append(pX)

    Yhat = np.array(Yhat)
    Xhat = np.array(Xhat)

    return np.sign(np.mean(np.std(Yhat, axis=0)) - np.mean(np.std(Xhat, axis=0)))


def npred(X: np.ndarray, Y: np.ndarray, lin: bool = True, norm: bool = True) -> float:
    """
    Computes the normalized mean squared error of the dependency between X and Y using either linear regression or lazy prediction.

    Args:
    - X: A 2D numpy array of shape (N, n) containing the independent variables.
    - Y: A 1D numpy array of shape (N,) containing the dependent variable.
    - lin: A boolean value indicating whether to use linear regression (default True).
    - norm: A boolean value indicating whether to normalize X (default True).

    Returns:
    - A float value representing the normalized mean squared error of the dependency.
    """

    # normalized mean squared error of the dependency
    N = X.shape[0]
    n = X.shape[1]

    if n > 1:
        w_const = np.where(np.std(X, axis=0) < 0.01)[0]
        if len(w_const) == n:
            return 1

        if len(w_const) > 0:
            X = np.delete(X, w_const, axis=1)
        if X.ndim == 1:
            X = X.reshape(N, 1)
        w_na = np.where(np.isnan(X.sum(axis=0)))[0]
        if len(w_na) > 0:
            X = np.delete(X, w_na, axis=1)
        n = X.shape[1]
    else:
        if np.any(np.isnan(X)):
            return 1

    XX = zscore(X)
    if N < 5 or np.any(np.isnan(XX)):
        return np.var(Y)

    if lin:
        lin_reg = LinearRegression().fit(XX, Y)
        mse_loo = -1 * np.mean((Y - lin_reg.predict(XX)) ** 2)
        return max(1e-3, mse_loo / (1e-3 + np.var(Y)))

    c1 = max(3, min(10, N - 2))
    c2 = max(c1, 5, min(N, 20))
    e = Y - lazy_pred(XX, Y, XX, conPar=[c1, c2], linPar=None, class_=False, cmbPar=10) # TODO: You need to implement the 'lazy_pred' function

    if norm:
        nmse = np.mean(e**2) / np.var(Y)
        return max(1e-3, nmse)

    return np.mean(e**2)




def coeff(y: np.ndarray, x1: np.ndarray, x2: np.ndarray = None) -> float:
    """
    Computes the coefficient of x1 in a linear regression model with y as the dependent variable and x1 and x2 as the independent variables.

    Args:
    - y: A 1D numpy array of shape (N,) containing the dependent variable.
    - x1: A 1D numpy array of shape (N,) containing the first independent variable.
    - x2: A 1D numpy array of shape (N,) containing the second independent variable (default None).

    Returns:
    - A float value representing the coefficient of x1 in the linear regression model.
    """
    Y = np.array(y)
    X = np.column_stack((x1, x2)) if x2 is not None else np.array(x1).reshape(-1, 1)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model.params[1]



def norminf(y: np.ndarray, x1: np.ndarray, x2: Optional[np.ndarray] = None, lin: bool = True) -> float:
    """
    Computes the normalized conditional information of x1 to y given x2: I(x1; y | x2) = (H(y | x2) - H(y | x1, x2)) / H(y | x2).

    Args:
    - y: A 1D numpy array of shape (N,) containing the dependent variable.
    - x1: A 1D numpy array of shape (N,) containing the first independent variable.
    - x2: A 1D numpy array of shape (N,) containing the second independent variable (default None).
    - lin: A boolean value indicating whether to use linear regression (default True).

    Returns:
    - A float value representing the normalized conditional information of x1 to y given x2.
    """
    
    if x2 is None:
        return max(0, 1 - npred(x1, y, lin=lin, norm=True))

    np = npred(x2, y, lin=lin, norm=False)
    x1x2 = np.column_stack((x1, x2))
    delta = max(0, np - npred(x1x2, y, lin=lin, norm=False)) / (np + 0.01)
    
    return delta



def descriptor(D, ca, ef, ns=None, lin=False, acc=True, struct=False, pq=None,
               bivariate=False, maxs=10, boot="mimr", errd=False,
               delta=False, stabD=False):
    """
    This function computes a set of descriptors for the D2C algorithm given two candidate nodes (ca, putative cause and ef, putative effect) and the observed data matrix D. The function uses the mimr algorithm to infer the Markov Blankets of the variables indexed by ca and ef (MBca and MBef). The descriptors are computed from a set of (conditional) mutual information terms describing the dependency between the variables ca and ef.

    Parameters:
    - D: observed data matrix of size [N, n], where N is the number of samples and n is the number of nodes
    - ca: node index (1 <= ca <= n) of the putative cause
    - ef: node index (1 <= ef <= n) of the putative effect
    - ns: size of the Markov Blanket
    - lin: boolean, if True it uses a linear model to assess a dependency, otherwise a local learning algorithm 
    - acc: boolean, if True it uses the accuracy of the regression as a descriptor
    - struct: boolean, if True it uses the ranking in the markov blanket as a descriptor
    - pq: a vector of quantiles used to compute the descriptor
    - bivariate: boolean, if True it includes the descriptors of the bivariate dependency
    - maxs: max number of pairs MB(i), MB(j) considered 
    - boot: feature selection algorithm

    Returns:
    - a vector of descriptors

    Details:
    The estimation of the information theoretic terms requires the estimation of the dependency between nodes. If lin=True, a linear assumption is made. Otherwise, the local learning estimator implemented by the R package lazy is used.

    References:
    - Gianluca Bontempi, Maxime Flauder (2014) From dependency to causality: a machine learning approach. Under submission
    - Bontempi G., Meyer P.E. (2010) Causal filter selection in microarray data. ICML'10
    - M. Birattari, G. Bontempi, and H. Bersini (1999) Lazy learning meets the recursive least squares algorithm. Advances in Neural Information Processing Systems 11, pp. 375-381. MIT Press.
    - G. Bontempi, M. Birattari, and H. Bersini (1999) Lazy learning for modeling and control design. International Journal of Control, 72(7/8), pp. 643-658.
    """


    if ns is None:
        ns = min(4, D.shape[1] - 2)

    if pq is None:
        pq = [0.1, 0.25, 0.5, 0.75, 0.9]

    D = (D - D.mean(axis=0)) / D.std(axis=0)

    if np.any(np.isnan(D)):
        raise ValueError("Error NA in descriptor")

    if np.any(np.isinf(D)):
        raise ValueError("Error Inf in descriptor")

    N, n = D.shape
    Icov = np.linalg.inv(np.cov(D.T) + np.eye(n) * 0.01)
    De = D2C_n(D, ca, ef, ns, lin, acc, struct, pq=pq, boot=boot, maxs=maxs)
    De = dict(zip(["M." + k for k in De.keys()], De.values()))
    wna = [k for k in De.keys() if np.isnan(De[k])]

    if len(wna) > 0:
        print(De)
        warnings.warn("NA in descriptor")

        for k in wna:
            De[k] = 0

    if errd:
        mfs = np.setdiff1d(np.arange(1, n + 1), ef)
        
        if boot == "mimr":
            fsef = mfs[mimr(D[:, mfs - 1], D[:, ef - 1], nmax=3)]
        elif boot == "rank":
            fsef = mfs[rankrho(D[:, mfs - 1], D[:, ef - 1], nmax=3)]
        
        eef = epred(D[:, fsef - 1], D[:, ef - 1], lin=lin)
        
        mfs = np.setdiff1d(np.arange(1, n + 1), ca)
        
        if boot == "mimr":
            fsca = mfs[mimr(D[:, mfs - 1], D[:, ca - 1], nmax=3)]
        elif boot == "rank":
            fsca = mfs[rankrho(D[:, mfs - 1], D[:, ca - 1], nmax=3)]
        
        eca = epred(D[:, fsca - 1], D[:, ca - 1], lin=lin)
        
        DD = D[:, np.unique(np.concatenate((ca, ef, fsef, fsca))) - 1]
        Icov2 = solve(np.cov(DD.T) + np.eye(DD.shape[1]) * 0.01)
        eDe = []
        
        eDe.append(norminf(eef, eca, D[:, ca - 1], lin=lin) - norminf(eef, eca, lin=lin))
        eDe.append(norminf(eef, eca, D[:, ef - 1], lin=lin) - norminf(eef, eca, lin=lin))
        eDe.append(norminf(eef, D[:, ca - 1], D[:, ef - 1], lin=lin) - norminf(eef, D[:, ca - 1], lin=lin))
        eDe.append(norminf(eca, D[:, ef - 1], D[:, ca - 1], lin=lin) - norminf(eca, D[:, ef - 1], lin=lin))
        eDe.append(norminf(eca, D[:, ef - 1], lin=lin))
        eDe.append(norminf(eef, D[:, ca - 1], lin=lin))
        
        eDe.extend([
            Icov[ca - 1, ef - 1], Icov2[0, 1],
            np.corrcoef(eef, D[:, ca - 1])[0, 1], np.corrcoef(eca, D[:, ef - 1])[0, 1],
            HOC(eef, eca, 1, 2), HOC(eef, eca, 2, 1), skew(eca), skew(eef)
        ])
        
        eDe = dict(zip(["M.e1", "M.e2", "M.e3", "M.e4", "M.e5", "M.e6",
                        "M.Icov", "M.Icov2",
                        "M.cor.e1", "M.cor.e2",
                        "B.HOC12.e", "B.HOC21.e", "B.skew.eca", "B.skew.eef"], eDe))


    if bivariate:
        De2 = D2C_2(D[:, ca], D[:, ef], pq=pq)
        De2 = dict(zip(["B." + k for k in De2.keys()], De2.values()))

    # rest of the code

    if errd:
        # rest of the code for errd case
        pass

    if bivariate:
        DD = {**DD, **De2}

    return DD




def E(x):
    """
    Computes the empirical distribution function of x at each observation.

    Args:
    - x: A 1D numpy array of shape (N,) containing the data.

    Returns:
    - A 1D numpy array of shape (N,) containing the empirical distribution function of x at each observation.
    """
    return percentileofscore(x, x, kind='rank') / 100





def D2C_n(D, ca, ef, ns=None, maxs=20, lin=False, acc=True, struct=True, pq=None, boot="mrmr"):
    """
    Compute the D2C_n score of the causal effect of X_c -> X_ef in the data set D, based on the conditional independence 
    criterion. 

    Parameters
    ----------
    D : pandas.DataFrame or numpy.ndarray
        The data set containing the variables X_ca, X_ef and other potential confounding variables.
    ca : int
        The column index (0-based) of the cause variable X_c in the data set.
    ef : int
        The column index (0-based) of the effect variable X_ef in the data set.
    ns : int, optional
        The maximum number of variables that can be included in the Markov blanket of X_ca and X_ef. Default is None, 
        which means the algorithm automatically determines ns based on the number of variables in the data set.
    maxs : int, optional
        The maximum number of variables to be considered in the search. Default is 20.
    lin : bool, optional
        Whether to use linear regression as the model to compute the normalized mean squared error (NMSE) of the 
        conditional dependence test. Default is False.
    acc : bool, optional
        Whether to use the accelerated version of the conditional dependence test. Default is True.
    struct : bool, optional
        Whether to include structural information as features in the D2C_n score. Default is True.
    pq : list of float, optional
        The quantiles used to compute the feature values based on the distribution of the empirical data. Default is 
        [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95].
    boot : str, optional
        The bootstrapping method used to estimate the Markov blanket. Can be one of "mrmr" (minimum redundancy maximum 
        relevance), "rank" (based on variable ranking), "mimr" (modified mrmr), or "mimr2" (another modified mrmr). 
        Default is "mrmr".

    Returns
    -------
    dict
        A dictionary containing the feature names and their corresponding D2C_n score. The features include the 
        structural information and the p-values from the conditional dependence tests at the univariate, bivariate, 
        and multivariate levels.

    Raises
    ------
    ValueError
        If the value of ca or ef is greater than the number of columns in the data set.

    """

    if pq is None:
        pq = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    if ns is None:
        ns = min(4, D.shape[1] - 2)

    n = D.shape[1]
    N = D.shape[0]

    if ca > n or ef > n:
        print("ca={}, ef={}, n={}".format(ca, ef, n))
        raise ValueError("error in D2C.n")

    namesx = []
    x = []

    MBca = set(range(1, n + 1)) - {ca}
    MBef = set(range(1, n + 1)) - {ef}
    MBca2 = MBca.copy()
    MBef2 = MBef.copy()

    if n > (ns + 1):
        ind = set(range(1, n + 1)) - {ca}
        ind = list(ind)
        ind.sort(key=lambda col: rankrho(D.loc[:, col], D.loc[:, ca], nmax=min(len(ind), 5 * ns)))

        if boot == "mrmr":
            MBca = mrmr(D.loc[:, ind], D.loc[:, ca], nmax=ns)

        if boot == "rank":
            MBca = ind[:ns]

        MBca2 = MBca

        if boot == "mimr":
            MBca2 = mimr(D.loc[:, ind], D.loc[:, ca], nmax=min(ns, round(n / 2)), caus=-1)
            MBca = mimr(D.loc[:, ind], D.loc[:, ca], nmax=2 * ns, init=True)
            MBca = list(set(MBca) - set(MBca2))
            MBca = MBca[:min(len(MBca), ns)]

        ind2 = set(range(1, n + 1)) - {ef}
        ind2 = list(ind2)
        ind2.sort(key=lambda col: rankrho(D.loc[:, col], D.loc[:, ef], nmax=min(len(ind2), 5 * ns)))

        if boot == "mrmr":
            MBef = mrmr(D.loc[:, ind2], D.loc[:, ef], nmax=ns)

        if boot == "rank":
            MBef = ind2[:ns]

        MBef2 = MBef
        if boot == "mimr":
            MBef2 = mimr(D.loc[:, ind2], D.loc[:, ef], nmax=min(ns, round(n / 2)), caus=-1)
            MBef = mimr(D.loc[:, ind2], D.loc[:, ef], nmax=2 * ns, init=True)
            MBef = list(set(MBef) - set(MBef2))
            MBef = MBef[:min(ns, len(MBef))]

        if boot == "mimr2":
            ind = list(set(ind).union(ind2) - {ca, ef})
            fs = mimr2(D[:, ind], D[:, ca], D[:, ef], nmax=ns, init=True)
            MBca = [ind[i] for i in fs["fs1"]]  # putative list of effects
            MBef = [ind[i] for i in fs["fs2"]]

        if struct:
            # position of effect in the MBca
            pos_ef = (MBca.index(ef) / ns) if ef in MBca else 2

            # position of ca in the MBef
            pos_ca = (MBef.index(ca) / ns) if ca in MBef else 2

            sx_ef = []
            # position of variables of MBef in MBca
            for i in range(len(MBef)):
                if MBef[i] in MBca:
                    sx_ef.append((MBca.index(MBef[i])) / ns)
                else:
                    sx_ef.append(2)

            sx_ca = []
            # position of variables of MBca in MBef
            for i in range(len(MBca)):
                if MBca[i] in MBef:
                    sx_ca.append((MBef.index(MBca[i])) / ns)
                else:
                    sx_ca.append(2)

            x = [pos_ca, pos_ef] + [np.quantile(sx_ca, p) for p in pq] + [np.quantile(sx_ef, p) for p in pq]
            namesx = ["pos.ca", "pos.ef"] + [f"sx.ca{i + 1}" for i in range(len(pq))] + [f"sx.ef{i + 1}" for i in range(len(pq))]


        # Univariate p-value
        x_uni = []

        for i in MBca:
            if i != ef:
                model = sm.OLS(D[:, ef], sm.add_constant(D[:, i])).fit()
                x_uni.append(model.pvalues[1])
            else:
                x_uni.append(0)

        x_uni = np.asarray(x_uni)

        # Bivariate p-value
        x_biv = []

        for i in range(len(MBca)):
            for j in range(i + 1, len(MBca)):
                model = sm.OLS(D[:, ef], sm.add_constant(D[:, [MBca[i], MBca[j]]])).fit()
                x_biv.append(model.pvalues[2])
                
        x_biv = np.asarray(x_biv)

        # Multivariate p-value
        model_full = sm.OLS(D[:, ef], sm.add_constant(D[:, [i for i in MBca if i != ef]])).fit()

        x_m = []

        for i in MBca:
            if i != ef:
                model_reduced = sm.OLS(D[:, ef], sm.add_constant(D[:, [j for j in MBca if j != ef and j != i]])).fit()
                ll_ratio = -2 * (model_reduced.llf - model_full.llf)
                p_val = chi2.sf(ll_ratio, 1)
                x_m.append(p_val)
            else:
                x_m.append(0)

        x_m = np.asarray(x_m)

        x = x + [np.quantile(x_uni, p) for p in pq] + [np.quantile(x_biv, p) for p in pq] + [np.quantile(x_m, p) for p in pq]
        namesx = namesx + [f"uni{i + 1}" for i in range(len(pq))] + [f"biv{i + 1}" for i in range(len(pq))] + [f"m{i + 1}" for i in range(len(pq))]

        return dict(zip(namesx, x))





def D2C_2():
    pass

import numpy as np

def varpred(x: np.ndarray, y: np.ndarray, R: int = 50) -> float:
    """
    Computes the variance of the predictor P(x) of the dependent variable y at each value of x.

    Args:
    - x: A 1D numpy array of shape (N,) containing the independent variable.
    - y: A 1D numpy array of shape (N,) containing the dependent variable.
    - R: An integer indicating the number of iterations to perform (default 50).

    Returns:
    - A float value representing the variance of the predictor P(x) of the dependent variable y at each value of x.
    """
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    xh = np.arange(-1.25, 1.25, 0.1)
    N = len(x)
    P = []
    beta = []

    for r in range(R):
        Ir = np.random.choice(N, round(4 * N / 5), replace=False)
        xr = x[Ir]
        yr = y[Ir]

        px = []
        for h in range(len(xh)):
            sx = np.argsort(np.abs(xr - xh[h]))[:min(10, len(xr))]
            px.append(np.mean(yr[sx]))

        P.append(px)

    P = np.array(P).T

    return np.mean(np.std(P, axis=1))
