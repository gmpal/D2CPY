import numpy as np
import pandas as pd
import rbnpy
import networkx as nx
import numpy as np
import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import pearsonr
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri

import numpy as np
from scipy.special import expit
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import pearsonr

def gendataDAG(N, n, sdw=0.1):
    if False:
        D = pd.DataFrame(np.random.normal(size=(N,n)))
        bn = gs(D)
        observedData = rbnpy.rbn(bn, N, D)
        trueDAG = nx.from_numpy_matrix(bn)
    else:
        trueDAG = nx.random_k_out_graph(n, k=2, alpha=0.3)
        errDist = np.random.choice(["mix", "cauchy", "t4", "mixt3"])
        observedData = (rbnpy.rmvDAG(N, trueDAG.to_numpy_matrix(), 
                                     errDist=errDist, mix=0.3)
                        + np.random.normal(size=(N,n), scale=sdw))
    if np.isnan(observedData.sum()):
        raise ValueError("Error in gendataDAG")
    if observedData.shape[1] != len(trueDAG.nodes()):
        raise ValueError("Error 2 in gendataDAG")
    return {"DAG": trueDAG, "data": observedData}


def HOC(x, y, i, j):
    return np.mean((x - np.mean(x))**i * (y - np.mean(y))**j) / ((np.std(x)**i) * (np.std(y)**j))

def is_parent(DAG, n1, n2):
    ## n1 : character name of node 1
    ## n2 : character name of node 2
    return nx.shortest_path_length(DAG, source=n1, target=n2) == 1

def is_child(DAG, n1, n2):
    s = nx.shortest_path_length(DAG, source=n2, target=n1)
    if np.isinf(s):
        return False
    return s == 1



def dagdistance(DAG, n1, n2):
    dout = nx.shortest_path_length(DAG, source=n1, target=n2)
    din = nx.shortest_path_length(DAG, source=n2, target=n1)
    if np.isinf(dout) and np.isinf(din):
        return np.random.choice([-5, 5])
    if np.isinf(dout) and np.isfinite(din):
        return -din
    if np.isfinite(dout) and np.isinf(din):
        return dout
    return (dout - din) / 2

def is_ancestor(DAG, n1, n2):
    s = nx.shortest_path_length(DAG, source=n1, target=n2)
    if np.isinf(s):
        return False
    return s >= 1

def is_descendant(DAG, n1, n2):
    s = nx.shortest_path_length(DAG, source=n2, target=n1)
    if np.isinf(s):
        return False
    return s >= 1

def is_mb(DAG, n1, n2):
    return is_child(DAG, n1, n2) or is_parent(DAG, n1, n2)

def is_what(iDAG, i, j, type):
    if type == "is.mb":
        return int(is_mb(iDAG, i, j))
    if type == "is.parent":
        return int(is_parent(iDAG, i, j))
    if type == "is.child":
        return int(is_child(iDAG, i, j))
    if type == "is.descendant":
        return int(is_descendant(iDAG, i, j))
    if type == "is.ancestor":
        return int(is_ancestor(iDAG, i, j))


def Sigm(x, W=1):
    E = np.exp(x)
    return 2*W*E/(1+E) - W

def rankrho(X, Y, nmax=5, regr=False, first=None):
    ## mutual information ranking
    ## 17/10/11
    n = X.shape[1]
    N = X.shape[0]
    m = Y.shape[1] if len(Y.shape) > 1 else 1
    
    if np.var(Y) < 0.01:
        return np.arange(1, nmax+1)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    Iy = np.zeros(n)
    if not regr:
        Iy = np.abs(np.array([spearmanr(X[:,i], Y[:,0])[0] for i in range(n)]))
    else:
        lr = LinearRegression()
        for i in range(n):
            lr.fit(X[:,i].reshape(-1,1), Y[:,0])
            Iy[i] = np.abs(lr.coef_[0])
    
    if m > 1:
        Iy = np.mean(Iy.reshape(-1, m), axis=1)
    
    return np.argsort(-Iy)[:nmax]

def quantization(x, nbin=1):
    if nbin == 1:
        return np.array(pd.cut(x, bins=[-np.inf, np.median(x), np.inf], labels=[1, 2]))
    else:
        return np.array(pd.cut(x, bins=[-np.inf, *np.quantile(x, [0.25, 0.5, 0.75]), np.inf], labels=[1, 2, 3, 4]))    



def H_sigmoid(n=2):
    a = np.random.uniform(-1, 1, size=n+1)
    def f(x):
        X = x**(np.arange(n+1))
        return expit(np.mean(X * a)) * 2 - 1
    return np.vectorize(f)

def H_Rn(n):
    a = np.random.uniform(-1, 1, size=n+1)
    def f(x):
        X = x**(np.arange(n+1))
        return np.sum(X * a)
    return np.vectorize(f)

def kernel_fct(X, knl=None, lambda_=0.1):
    N = X.shape[0]
    Y = np.random.normal(size=N)
    if knl is None:
        knl = Nystroem(kernel="rbf", gamma=np.random.uniform(0.5, 2), degree=np.random.choice([1,2]))
    K = knl.fit_transform(X)
    kr = KernelRidge(alpha=lambda_*N).fit(K, Y)
    Yhat = kr.predict(K)
    return Yhat

def H_kernel(knl=None):
    def f(x):
        return kernel_fct(x, knl=knl)
    return np.vectorize(f)





def pcor1(x, y, z):
    ## partial correlation cor(x,y|z)
    if isinstance(z, (int, float, np.number)):
        rho_xy = pearsonr(x, y)[0]
        rho_xz = pearsonr(x, z)[0]
        rho_yz = pearsonr(y, z)[0]
        if np.isnan(rho_xz + rho_yz + rho_xy):
            return 0
        if rho_xz == 1 or rho_yz == 1:
            return 0
        rho = (rho_xy - rho_xz * rho_yz) / (np.sqrt(1 - min(rho_xz**2, 0.99)) * np.sqrt(1 - min(rho_yz**2, 0.99)))
        return rho
    else:
        raise ValueError("z should be numeric")

def corDC(X, Y):
    ## correlation continuous matrix and discrete vector
    ## NB: the notion of sign has no meaning in this case. Mean of absolute values is taken
    ## 14/11/2011
    if not isinstance(Y, np.ndarray) or not np.issubdtype(Y.dtype, np.integer):
        raise ValueError("Y should be a discrete vector")
  
    N = X.shape[0]
    L = np.unique(Y)
  
    if len(L) == 2:
        lL = 1
    else:
        lL = len(L)
  
    cxy = np.zeros((X.shape[1], lL))
    for i in range(lL):
        yy = np.zeros(N)
        ind1 = np.where(Y == L[i])[0]
        ind2 = np.setdiff1d(np.arange(N), ind1)
        yy[ind1] = 1
        cxy[:,i] = np.abs(np.corrcoef(X.T, yy)[0,1:])
    
    return np.mean(cxy, axis=1)






def Icond(x, y=None, z=None, lambda_=0):
    ## conditional information cor(x,y|z)
  
    ## numeric z
    if isinstance(z, (int, float, np.number)):
        if np.ndim(x) == 1:
            return np.corrcoef(x, y)[0, 1]
        X = x
        n = X.shape[1]
        Ic = np.zeros((n, n))
        for i in range(n-1):
            for j in range(i+1, n):
                Ic[i,j] = Icond(X[:,i], X[:,j], z)
                Ic[j,i] = Ic[i,j]
        return Ic

    ## factor z and vectors x and y
    elif y is not None:
        if not isinstance(z, np.ndarray) or not np.issubdtype(z.dtype, np.integer):
            raise ValueError("z should be a discrete vector")
        L = np.unique(z)
        lL = len(L)
        w = np.zeros(lL)
        for i in range(lL):
            w[i] = np.sum(z == L[i])
        w /= np.sum(w)

        Ic = np.zeros(lL)
        for i in range(lL):
            ind1 = np.where(z == L[i])[0]
            Ic[i] = np.corrcoef(x[ind1], y[ind1])[0, 1]
        return np.sum(w * Ic)

    ## factor z and matrix x
    else:
        if not isinstance(z, np.ndarray) or not np.issubdtype(z.dtype, np.integer):
            raise ValueError("z should be a discrete vector")
        X = x
        n = X.shape[1]
        L = np.unique(z)
        lL = len(L)
        w = np.zeros(lL)
        for i in range(lL):
            w[i] = np.sum(z == L[i])
        w /= np.sum(w)

        Ic = np.zeros((n, n))
        W = 0
        for i in range(lL):
            ind1 = np.where(z == L[i])[0]
            if len(ind1) > 8:
                corr = np.corrcoef(X[ind1,:], rowvar=False)
                shrunk_corr = corr_shrinkage(corr, lambda_)
                Ic += w[i] * np.abs(shrunk_corr)
                W += w[i]
        return Ic / W


def corr_shrinkage(corr, lambda_):
    n = corr.shape[0]
    shrink = np.maximum(0, np.diag(corr)**2 - 1) / n
    shrunk_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            r = corr[i,j]
            r_shrink = (1 - lambda_) * r + lambda_ * shrink[i] * shrink[j] / np.maximum(shrink[i] * shrink[j], 1e-10)
            shrunk_corr[i,j] = r_shrink
            shrunk_corr[j,i] = r_shrink
    return shrunk_corr



import numpy as np
from scipy.stats import norm

def ppears(r_hat, N, S=0):
    n = len(r_hat)
    p = np.zeros(n)

    for i in range(n):
        z = abs(0.5*(np.log(1+r_hat[i])-np.log(1-r_hat[i])))*np.sqrt(N[i]-S-3)
        p[i] = norm.sf(z)

    return p

def corXY(X, Y):
    # correlation continuous matrix and continuous/discrete vector matrix
    n = X.shape[1]
    N = X.shape[0]
    m = Y.shape[1] if Y.ndim == 2 else 1
    cXY = np.zeros((n,m))
  
    for i in range(m):
        YY = Y[:,i] if Y.ndim == 2 else Y
        if np.issubdtype(YY.dtype, np.number):
            cXY[:,i] = np.corrcoef(X,YY,rowvar=False)[0,:-1]
        else:
            cXY[:,i] = corDC(X,YY)

    return cXY

def cor2I2(rho):
    rho = np.clip(rho, -1+1e-5, 1-1e-5)
    return -0.5*np.log(1-rho**2)

def corDC(X, Y):
    # correlation continuous matrix and discrete vector
    # NB: the notion of sign has no meaning in this case. Mean of absolute values is taken
    N = X.shape[0]
    L = np.unique(Y)
    lL = len(L)
  
    if lL == 2:
        lL = 1
      
    cxy = np.zeros(lL)
    for i in range(lL):
        yy = np.zeros(N)
        ind1 = np.where(Y == L[i])[0]
        ind2 = np.setdiff1d(np.arange(N), ind1)
        yy[ind1] = 1
        cxy[i] = np.mean(np.abs(np.corrcoef(X, yy, rowvar=False)[0,:-1]))

    return cxy



import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor

#TODO: adapt to use lazy 
def lazy_pred(X, Y, X_ts, class_=False, return_more=False, conPar=3, linPar=5, cmbPar=10):
    
    n = X.shape[1]
    N = X.shape[0]
    
    if class_:
        l_Y = np.unique(Y)
        L = len(l_Y)
        u = np.unique(Y)
        
        if len(u) == 1:
            P = np.zeros((X_ts.shape[0], L))
            col_names = [str(l) for l in l_Y]
            P[:, u] = 1
            out_hat = pd.Categorical(np.repeat(str(u), len(X_ts)), categories=col_names)
            return {'pred': out_hat, 'prob': P}
        
        if L == 2:
            raise ValueError("not supported")
        else:
            algo = 'lazy'
            raise ValueError("not supported")
    else: # regression
        d = pd.concat([pd.Series(Y, name='Y'), pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n)])], axis=1)
        mod = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None, 
                            predictions=True, unique_id=None, random_state=None,
                            sample_size=256, folds=10)
        mod.fit(d.iloc[:, 1:], d['Y'])
        y_pred = mod.predict(X_ts)
        if return_more:
            return mod
        else:
            return y_pred
        


import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def rf_pred(X, Y, X_ts, clf=False, **kwargs):
    n = X.shape[1]
    N = X.shape[0]
    
    if isinstance(X_ts, list):
        X_ts = pd.DataFrame(X_ts).T
    if isinstance(X_ts, pd.Series):
        X_ts = pd.DataFrame(X_ts).T
    if isinstance(X_ts, pd.DataFrame):
        N_ts = X_ts.shape[0]
        if n == 1:
            X = pd.DataFrame(X, columns=['x0'])
            X_ts = pd.DataFrame(X_ts, columns=['x0'])
    elif isinstance(X_ts, np.ndarray):
        if n == 1:
            X = pd.DataFrame(X, columns=['x0'])
            X_ts = pd.DataFrame(X_ts, columns=['x0'])
            N_ts = X_ts.shape[0]
        else:
            X_ts = pd.DataFrame(X_ts)
            N_ts = X_ts.shape[0]
            X_ts.columns = ['x'+str(i) for i in range(n)]
            
    if clf:
        # classification
        l_Y = np.unique(Y)
        L = len(l_Y)
        if len(l_Y) == 1:
            P = pd.DataFrame(np.zeros((N_ts, L)), columns=l_Y)
            P.loc[:, l_Y[0]] = 1
            out_hat = pd.Series([l_Y[0]]*N_ts)
            return {'pred': out_hat, 'prob': P}
        if L == 2:
            clf = RandomForestClassifier(**kwargs)
            clf.fit(X, Y)
            out_hat = clf.predict(X_ts)
            P = clf.predict_proba(X_ts)
            return {'pred': out_hat, 'prob': P}
        else:
            clf = RandomForestClassifier(**kwargs)
            clf.fit(X, Y)
            out_hat = clf.predict(X_ts)
            P = clf.predict_proba(X_ts)
            return {'pred': out_hat, 'prob': P}
    else:
        # regression
        reg = RandomForestRegressor(**kwargs)
        reg.fit(X, Y)
        out_hat = reg.predict(X_ts)
        return out_hat



import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import scale
from scipy.stats import norm

def mimr(X, Y, nmax=5, init=False, lambda_=0.5, spouse_removal=True, caus=1):
    if np.var(Y) < 0.01:
        return np.arange(1, nmax + 1)
    
    m = Y.shape[1]
    n = X.shape[1]
    orign = n
    N = X.shape[0]
    H = np.var(X, axis=0)
    HY = np.var(Y)
    CY = corXY(X, Y)
    Iy = cor2I2(CY)
    subset = np.arange(1, n + 1)
    
    pv_rho = pears(np.concatenate([CY]), N + np.zeros(n))
    
    if spouse_removal:
        pv = pears(np.concatenate([CY]), N + np.zeros(n))
        s = np.argsort(pv)[::-1]
        hw = max(1, min(n - nmax, np.sum(s > 0.05)))
        spouse = s[:hw]
        subset = np.setdiff1d(np.arange(1, n + 1), s[:hw])
        X = X[:, subset - 1]
        Iy = Iy[subset - 1]
        n = X.shape[1]
    
    CCx = np.corrcoef(X.T)
    Ix = cor2I2(CCx)
    Ixx = Icond(X, z=Y, lambda_=0.02)
    
    Inter = np.empty((n, n))
    if init:
        max_kj = -np.inf
        for kk in range(n - 1):
            for jj in range(kk + 1, n):
                Inter[kk, jj] = (1 - lambda_) * (Iy[kk] + Iy[jj]) + caus * lambda_ * (Ixx[kk, jj] - Ix[kk, jj])
                Inter[jj, kk] = Inter[kk, jj]
                if Inter[kk, jj] > max_kj:
                    max_kj = Inter[kk, jj]
                    subs = np.array([kk, jj])
    else:
        subs = np.argmax(Iy)
    
    if nmax > len(subs):
        last_subs = np.array([])
        for j in range(len(subs), min(n-1, NMAX-1)):
            mrmr = np.repeat(-np.inf, n)
            if len(subs) < (n-1):
                if len(subs) > 1:
                    inter = mimr_utils.compute_interactions(X[:, subs], Y, lambda_inter=lambda_inter)
                    mrmr[~np.in1d(np.arange(n), subs)] = (1-lambda_mr)*mutual_info_classif(X[:, ~np.in1d(np.arange(n), subs)], Y) + lambda_mr*causality_utils.causality(X[:, ~np.in1d(np.arange(n), subs)], Y, inter=inter, alpha=alpha_causality, caus=caus)
                else:
                    inter = mimr_utils.compute_interactions(X[:, subs], Y, lambda_inter=lambda_inter)
                    mrmr[~np.in1d(np.arange(n), subs)] = (1-lambda_mr)*mutual_info_classif(X[:, ~np.in1d(np.arange(n), subs)], Y) + lambda_mr*(-np.diag(corrcoef(X[:, ~np.in1d(np.arange(n), subs)])) + inter)
            s = np.argmax(mrmr)
            subs = np.append(subs, s)
            if np.array_equal(last_subs, subs):
                break
            last_subs = subs
    ra = subset[subs]
    if nmax > len(ra):
        ra = np.append(ra, np.setdiff1d(np.arange(n), ra))

    return ra






import numpy as np
from scipy.stats import pearsonr
from .mim import cor2I2, corXY, ppears
from .Icond import Icond


def mimr2(X, Y1, Y2, nmax=5, init=False, lambda_=0.5, spouse_removal=True, caus=1):
    pass 
    #TODO: implement this function

def mimrmat(X, Y, nmax=5, init=False, lambda_=0.5, spouse_removal=True, caus=1):
    pass
    #TODO: implement this function

def mimreff(X, Y, nmax=5, init=False, lambda_=0.5, spouse_removal=True, caus=1):
    pass
    #TODO: implement this function

def mrmr(X, Y, nmax=5, init=False, lambda_=0.5, spouse_removal=True, caus=1):
    pass
    #TODO: implement this function

def mrmr2(X, Y, nmax=5, init=False, lambda_=0.5, spouse_removal=True, caus=1):
    pass
    #TODO: implement this function

from scipy.stats import pearsonr

def assoc(x, y):
    corr, pval = pearsonr(x, y)
    return [abs(corr), pval]


import numpy as np
from sklearn.metrics import roc_auc_score

def BER(Ytrue, Yhat):
    if not isinstance(Ytrue, np.ndarray) or not isinstance(Yhat, np.ndarray):
        raise TypeError("BER accepts only numpy arrays")
    TN = np.sum(np.logical_and(Yhat == 0, Ytrue == 0))
    FN = np.sum(np.logical_and(Yhat == 0, Ytrue == 1))
    TP = np.sum(np.logical_and(Yhat == 1, Ytrue == 1))
    FP = np.sum(np.logical_and(Yhat == 1, Ytrue == 0))
    b1 = FP / (TN + FP)
    b2 = FN / (FN + TP)
    if np.isnan(b1):
        b1 = 0
    if np.isnan(b2):
        b2 = 0
    return 0.5 * (b1 + b2)


def AUC(y, yhat):
    return roc_auc_score(y, yhat)



import numpy as np


def MakeEmbedded(ts, n, delay, hor=[1], w=None):
    if w is None:
        w = np.arange(ts.shape[1])
    no_data = ts.shape[0]
    no_var = ts.shape[1]
    if n.shape[0] != no_var:
        raise ValueError("Error in the size of embedding n")
    if delay.shape[0] != no_var:
        raise ValueError("Error in the size of delay")
    if len(hor) != len(w):
        raise ValueError("Error in the size of horizon hor")
    N = no_data - np.max(n) - np.max(delay)
    Input = np.zeros((N, np.sum(n)))
    Output = np.zeros((N, np.sum(hor)))
    for i in range(N):
        for j in range(no_var):
            k = np.arange(1, n[j]+1)
            Input[i, np.sum(n[0:j])+np.arange(n[j])] = ts[i+n[j]-k+np.max(n)-n[j]+np.max(delay)-delay[j], j]
            for ww in range(len(w)):
                if ww == 0:
                    iw = 0
                else:
                    iw = np.sum(hor[0:ww])
                Output[i, np.arange(iw, iw+hor[ww])] = np.nan
                M = min(no_data, i+np.max(n)+np.max(delay)+hor[ww]-1)
                Output[i, np.arange(iw, iw+M-(i+np.max(n)+np.max(delay))+1)] = ts[np.arange(i+np.max(n)+np.max(delay), M), w[ww]]
    return {'inp': Input, 'out': Output}




import numpy as np
import networkx as nx

def genTS(nn, NN, sd=0.5, num=1):
    n = 4  # max embedding order
    Y = np.random.normal(scale=0.1, size=nn)
    ep = 0
    th0 = np.random.normal()
    fs = np.random.choice(range(n + 1), size=4, replace=True)
    state = 0
    
    if num > 0:
        for i in range(NN):
            N = len(Y)

            if num == 1:
                nfs = 2
                e = np.random.normal()
                Y = np.concatenate([Y, [-0.4*(3 - Y[N - fs[0]]**2)/(1 + Y[N - fs[0]]**2) + 0.6*(3 - (Y[N - fs[1]] - 0.5)**3)/(1 + (Y[N - fs[1]] - 0.5)**4) + sd*(e + th0*ep)]])
                ep = e
            
            elif num == 2:
                nfs = 2
                e = np.random.normal()
                Y = np.concatenate([Y, [(0.4 - 2*np.exp(-50*Y[N - fs[0]]**2))*Y[N - fs[0]] + (0.5 - 0.5*np.exp(-50*Y[N - fs[1]]**2))*Y[N - fs[1]] + sd*(e + th0*ep)]])
                ep = e
            
            elif num == 3:
                nfs = 3
                e = np.random.normal()
                Y = np.concatenate([Y, [1.5*np.sin(np.pi/2*Y[N - fs[0]]) - np.sin(np.pi/2*Y[N - fs[1]]) + np.sin(np.pi/2*Y[N - fs[2]]) + sd*(e + th0*ep)]])
                ep = e
            
            elif num == 4:
                nfs = 2
                e = np.random.normal()
                Y = np.concatenate([Y, [2*np.exp(-0.1*Y[N - fs[0]]**2)*Y[N - fs[0]] - np.exp(-0.1*Y[N - fs[1]]**2)*Y[N - fs[1]] + sd*(e + th0*ep)]])
                ep = e
            
            elif num == 5:
                nfs = 1
                Y = np.concatenate([Y, [-2*Y[N - fs[0]]*max(0, np.sign(-Y[N - fs[0]])) + 0.4*Y[N - fs[0]]*max(0, np.sign(Y[N - fs[0]])) + sd*np.random.normal()]])
            
            elif num == 6:
                nfs = 2
                Y = np.concatenate([Y, [0.8*np.log(1 + 3*Y[N - fs[0]]**2) - 0.6*np.log(1 + 3*Y[N - fs[1]]**2) + sd*np.random.normal()]])
            
            elif num == 7:
                nfs = 2
                Y = np.concatenate([Y, [1.5*np.sin(np.pi/2*Y[N - fs[0]]) - np.sin(np.pi/2*Y[N - fs[1]]) + sd*np.random.normal()]])
            
            elif num == 8:
                nfs = 2
                Y = np.concatenate([Y, [(0.5 - 1.1*np.exp(-50*Y[N - fs[0]]**2))*Y[N - fs[0]] + (0.3 - 0.5*np.exp(-50*Y[N - fs[1]]**2))*Y[N - fs[1]] + sd*np.random.normal()]])
            
            elif num == 9:
                nfs = 2
                Y = np.concatenate([Y, [0.3*Y[N - fs[0]] + 0.6*Y[N - fs[1]] + (0.1 - 0.9*Y[N - fs[0]] + 0.8*Y[N - fs[1]])/(1 + np.exp(-10*Y[N - fs[0]])) + sd*np.random.normal()]])
            
            elif num == 10:
                nfs = 1
                Y = np.concatenate([Y, [np.sign(Y[N - fs[0]]) + sd*np.random.normal()]])
            
            elif num == 11:
                nfs = 1
                Y = np.concatenate([Y, [0.8*Y[N - fs[0]] - 0.8*Y[N - fs[0]]/(1 + np.exp(-10*Y[N - fs[0]])) + sd*np.random.normal()]])
            
            elif num == 12:
                nfs = 2
                Y = np.concatenate([Y, [0.3*Y[N - fs[0]] + 0.6*Y[N - fs[1]] + (0.1 - 0.9*Y[N - fs[0]] + 0.8*Y[N - fs[1]])/(1 + np.exp(-10*Y[N - fs[0]])) + sd*np.random.normal()]])
            
            elif num == 13:
                nfs = 1
                fs = np.arange(1, -num - 1, -1)
                Y = np.concatenate([Y, [min(1, max(0, 3.8*Y[N - fs[0]]*(1 - Y[N - fs[0]]) + sd*np.random.normal()))]])
            
            elif num == 14:
                nfs = 2
                fs = np.arange(0, -num, -1)
                Y = np.concatenate([Y, [1 - 1.4*Y[N - fs[0]]**2 + 0.3*Y[N - fs[1]] + 0.001*sd*np.random.normal()]])

            elif num == 15:
                nfs = 1
                if Y[N - fs[0]] < 1:
                    Y = np.concatenate([Y, [-0.5*Y[N - fs[0]] + sd*np.random.normal()]])
                else:
                    Y = np.concatenate([Y, [0.4*Y[N - fs[0]] + sd*np.random.normal()]])
            
            elif num == 16:
                nfs = 1
                if state == 1:
                    Y = np.concatenate([Y, [-0.5*Y[N - fs[0]] + sd*np.random.normal()]])
                else:
                    Y = np.concatenate([Y, [0.4*Y[N - fs[0]] + sd*np.random.normal()]])
                if np.random.uniform() > 0.9:
                    state = 1 - state
            
            elif num == 17:
                nfs = 4
                Y = np.concatenate([Y, [np.sqrt(0.000019 + 0.846*((Y[N - fs[0]])**2 + 0.3*(Y[N - fs[1]])**2 + 0.2*(Y[N - fs[2]])**2 + 0.1*(Y[N - fs[3]])**2))*sd*np.random.normal()]])
                
                if np.any(np.isnan(Y)) or np.any(np.abs(Y) > 1000):
                    raise ValueError("error")
        
        if num < 0:
            fs = np.arange(1, -num - 1, -1)
            nfs = len(fs)
            ord_ = -num
            Cf = np.random.normal(size=ord_)
            ma = np.random.normal(size=2)
            while np.any(np.abs(np.roots(np.concatenate([[1], -Cf]))) <= 1):
                Cf = np.random.normal(size=ord_)
            Y = arma_generate_sample(ar=np.concatenate([[1], -Cf]), ma=ma, nsample=NN, scale=sd)
        
        fs = fs[:nfs]
        Y = preprocessing.scale(Y[nn-1:])
        if np.any(np.isnan(Y)):
            raise ValueError("error")
        M = make_embedded(array(Y.reshape(-1, 1)), n=nn, delay=0, hor=np.ones(1), w=np.arange(1))
        netw_dag = nx.DiGraph()
        
        for j in range(nn - max(fs) - 1):
            for f in fs:
                netw_dag.add_edge(j + f + 1, j)
        
        return M[0], netw_dag

               

