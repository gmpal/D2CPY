import pandas as pd
import numpy as np 
from multiprocessing import Pool
import networkx as nx

from typing import Union, Tuple, Any

from scipy.stats import skew, tstd
from numpy.linalg import inv, pinv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold

# from simulatedDAGs import SimulatedDAGs
from d2c.utils import *

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import pickle

import time
from tqdm import tqdm

#TODO: improve multiprocessing handling and clean code
def _process_idx(idx, pairs, compute_descriptors):
    print('\rProcessing ',idx, ' of ', len(pairs), end='', flush=True)
    start = time.time()
    X_list = []
    for pair in pairs:
        X_pair = compute_descriptors(idx, pair[0], pair[1])
        X_list.append(X_pair)
    X = pd.concat([pd.DataFrame(X) for X in X_list], axis=0)
    return X


class D2C:
    def __init__(self, dags, observations, rev: bool = True, boot: str = "rank", verbose=False, random_state: int = 42, n_jobs: int = 1, dynamic: bool = True, n_variables: int = 3, maxlags: int = 3 , use_real_MB: bool = False, balanced: bool = True ) -> None:
        """
        Class for D2C analysis.

        D2C (Dependency to Causalilty) analysis is a method for inferring causal relationships
        from observational data using simulated directed acyclic graphs (DAGs) and computing 
        asymmetric descriptors from the observations associated with each DAG.

        Args:
        #TODO: CORRECT
            simulatedDAGs (SimulatedDAGs): An instance of the SimulatedDAGs class.
            rev (bool, optional): Whether to consider reverse edges. Defaults to True.
            n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.DAGs_index = np.arange(len(observations))
        self.DAGs = dags #it's a LIST
        self.observations_from_DAGs = observations #it's a LIST  
        self.rev = rev
        self.boot = boot
        self.X = None
        self.Y = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.dynamic = dynamic #flag for handling time series data
        self.n_variables = n_variables #number of variables in the time series #TODO: handle for nontimeseries as well
        self.maxlags = maxlags

        self.use_real_MB = use_real_MB
        self.balanced = balanced


        self.test_couples = []

   
    def compute_descriptors_no_dags(self):
        if not self.dynamic:
            pairs = [(i, j) for i in range(self.n_variables) for j in range(self.n_variables) if i != j]
        else: 
            pairs = [(i, j) for i in range(self.n_variables,  self.n_variables + self.n_variables * self.maxlags) for j in range(self.n_variables) if i != j]
        
        if self.n_jobs == 1:
            results = [_process_idx(idx, pairs, self.compute_descriptors) for idx in self.DAGs_index]
        else:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(_process_idx, [(idx, pairs, self.compute_descriptors) for idx in self.DAGs_index])

        X = pd.concat([pd.DataFrame(X) for X in results], axis=0)
        return X


    def initialize(self) -> None:
        """
        Initialize the D2C object by computing descriptors in parallel for all observations.

        """
        if self.n_jobs == 1:
            results = [self._compute_descriptors_for_edge_pairs(DAG_index) for DAG_index in self.DAGs_index]
        else:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(
                    self._compute_descriptors_for_edge_pairs,
                    zip(self.DAGs_index)
                )

        X_list, Y_list = zip(*results)
        self.X = pd.concat([pd.DataFrame(X) for X in X_list], axis=0)
        self.Y = pd.concat([pd.DataFrame(Y) for Y in Y_list], axis=0)

    def _compute_descriptors_for_edge_pairs(self, DAG_index: Any) -> Tuple[list, list]:
        """
        Compute descriptors in parallel for a given observation.
        """
        X = []
        Y = []
        print("DAG_index start", DAG_index)
        nodes = len(self.DAGs[DAG_index].nodes)

        child_edge_pairs = []
        
        if not self.dynamic:
            all_possible_edges = [(i, j) for i in range(self.n_variables) for j in range(self.n_variables) if i != j]
        else: 
            all_possible_edges = [(i, j) for i in range(self.n_variables,  self.n_variables + self.n_variables * self.maxlags) for j in range(self.n_variables) if i != j]
        

        #if dependency_type == "is.child": #TODO implement other dependencies
        for parent_node, child_node in self.DAGs[DAG_index].edges:
            child_edge_pairs.append((int(parent_node), int(child_node)))


        #TODO: the selection of edges is too much hardcoded ! 
        num_rows = len(child_edge_pairs)  # Number of rows in the 2D array
        num_samples = min(num_rows, 20)  # Number of samples to pick, ensuring it's not more than the number of rows

        # Generate random indices
        # np.seed(42)
        random_indices = np.random.choice(num_rows, size=num_samples, replace=False)
        # Select the rows corresponding to these indices
        selected_rows = [child_edge_pairs[i] for i in random_indices]

        child_edge_pairs = selected_rows

        self.test_couples.extend(child_edge_pairs)
        
        if self.verbose: print("child_edge_pairs", child_edge_pairs)
        # print(child_edge_pairs)
        for edge_pair in child_edge_pairs:
            parent, child = edge_pair[0], edge_pair[1]
            # print("Coppia giusta ",parent,child)
            descriptor = self.compute_descriptors(DAG_index, parent, child)
            X.append(descriptor)
            Y.append(1)  # Label edge as "is.child"

        for edge_pair in child_edge_pairs:
            parent, child = edge_pair[0], edge_pair[1]
            # print("Coppia giusta ",parent,child)
            descriptor = self.compute_descriptors(DAG_index, child, parent)
            X.append(descriptor)
            Y.append(0)  # Label edge as "is

        # For all remaining edges that are not in the DAG, compute descriptors and label them as "not a child"
                                        
        # selected_indices = np.random.choice(len(all_possible_edges), size=len(child_edge_pairs), replace=False)

        # Use list comprehension to select edges
        # selected_edges = [all_possible_edges[i] for i in selected_indices]
        counter=0
        for edge_pair in all_possible_edges:
            if edge_pair not in child_edge_pairs:

                if counter == num_samples:
                    break #TODO: for A -> B, we keep B -> A and another one, it's slightly unbalanced
                
                # print("not in children:",edge_pair)
                parent, child = edge_pair[0], edge_pair[1]
                self.test_couples.extend([edge_pair])
                descriptor = self.compute_descriptors(DAG_index, parent, child)
                X.append(descriptor)
                Y.append(0)  # Label edge as "not a child"
                
                
                counter+=1
                
        #pickle the couple X,Y  
        print("DAG_index end", DAG_index)
        return X, Y

    def compute_markov_blanket(self, DAG_index, D, variable, MB_size, verbose=False):
        '''
        Compute the Markov Blanket of a variable in a dataset under the assumption that one variable is the cause and the other the effect.

        Parameters:
        DAG_index (int): Index of the DAG in the list of DAGs. (Used only if we compute the real MB)
        D (pd.DataFrame): The dataset.
        variable (int): Node index of the putative cause. Must be in the range [0, n).
        ns (int, optional): Size of the Markov Blanket. Defaults to min(4, n - 2).
        

        '''
        # if self.DAGs is None:
        if not self.use_real_MB: 
            ind = list(set(np.arange(D.shape[1])) - {variable})
            if self.boot == "mrmr": 
                order = mRMR(D.iloc[:,ind],D.iloc[:,variable],nmax=min(len(ind),5*MB_size),verbose=self.verbose)
            elif self.boot == "rank":
                order = rankrho(D.iloc[:,ind],D.iloc[:,variable],nmax=min(len(ind),5*MB_size),verbose=self.verbose)
            sorted_ind = [ind[i] for i in order]
            return sorted_ind[:MB_size]
            #TODO: at the moment we do not use mRMR, just rankrho.
            # return ind[mRMR(D.iloc[:,ind],D.iloc[:,variable],nmax=MB_size,verbose=self.verbose)]  
        else: 
            dag = self.DAGs[DAG_index]
            # node = str(variable)
            #TODO: assess if this is the best way to handle the passage between strings and integers
            node = variable
            parents = list(dag.predecessors(node))
            children = list(dag.successors(node))
            parents_of_children = []
            for child in children:
                parents_of_children.extend(list(dag.predecessors(child)))
            parents_of_children = list(set(parents_of_children))
            
            
            MB = list(set(parents + parents_of_children + children))
            #remove node
            MB = list(set(MB) - {node})
            if len(MB) > MB_size:
                #TODO: fix this in order to select the top most relevant
                #random selection
                MB = np.random.choice(MB, size=MB_size, replace=False)

            MB = [int(i) for i in MB] #TODO: handle the passage between strings and integers in a better way
            return list(MB)


    def compute_descriptors(self, DAG_index, ca, ef, MB_size=None, maxs=20,
            lin=False, acc=True, compute_error_descriptors=False,
            pq= [0.05,0.1,0.25,0.5,0.75,0.9,0.95]):
        
        """
        Compute descriptor of two variables in a dataset under the assumption that one variable is the cause and the other the effect.

        Parameters:
        ca (int): Node index of the putative cause. Must be in the range [0, n).
        ef (int): Node index of the putative effect. Must be in the range [0, n).
        ns (int, optional): Size of the Markov Blanket. Defaults to min(4, n - 2).
        lin (bool, optional): If True, uses a linear model to assess dependency. Defaults to False.
        acc (bool, optional): If True, uses the accuracy of the regression as a descriptor. Defaults to True.
        pq (list of float, optional): A list of quantiles used to compute the descriptor. Defaults to [0.1,0.25,0.5,0.75,0.9].
        maxs (int, optional): Max number of pairs MB(i), MB(j) considered. Defaults to 10.
        errd (bool, optional): If True, includes the descriptors of the error. Defaults to False.
        delta (bool, optional): Not used in current implementation. Defaults to False.
        stabD (bool, optional): Not used in current implementation. Defaults to False.

        Returns:
        dict: A dictionary with the computed descriptors.

        Raises:
        ValueError: If there are missing or infinite values in D.
        """

        ca = int(ca)
        ef = int(ef)

        D = self.observations_from_DAGs[DAG_index]
        # print(D.columns)
        D = (D - D.mean()) / D.std()
        n_observations, n_features = D.shape
        
        if self.verbose: print(f"Computing descriptors of DAG {DAG_index} edge pair {ca} {ef}")
        
        if np.any(np.isnan(D)) or np.any(np.isinf(D)): raise ValueError("Error: NA or Inf in data")

        # Set default value for ns if not provided
        if MB_size is None:
            MB_size = min(4, n_features-2)
            if self.verbose: print(MB_size)

        # Initializations of Markov Blankets
        MBca = list(set(np.arange(n_features)) - {ca})
        MBef = list(set(np.arange(n_features)) - {ef})

        # Creation of the Markov Blanket of ca (denoted MBca) and ef (MBef)
        if n_features > (MB_size+1):
            MBca = self.compute_markov_blanket(DAG_index, D, ca, MB_size)
            MBef = self.compute_markov_blanket(DAG_index, D, ef, MB_size)

        # print("MBca", MBca)
        # print("MBef", MBef)

        common_causes = list(set(MBca).intersection(set(MBef)))
        # common_causes_columns = None 
        # if len(common_causes) > 0:
        #     common_causes_columns = D.iloc[:, common_causes]

        # I(cause; effect | common_causes) 
      
        com_cau = normalized_conditional_information(D.iloc[:, [ef]], D.iloc[:, [ca]], D.iloc[:, common_causes]) 


        # b: ef = b * (ca + mbef)
        coeff_cause = coeff(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, MBef])
        
        # b: ca = b * (ef + mbca)
        coeff_eff = coeff(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:, MBca])
        
        #I(cause; effect) 
        cau_eff = normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef]) 

        #I(effect; cause)
        eff_cau = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca]) 

        #I(effect; cause | MBeffect) 
        eff_cau_mbeff = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, MBef])

        #I(cause; effect | MBcause)
        cau_eff_mbcau = normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:, MBca]) 

        #I(effect; cause | arrays_m_plus_MBca)
        eff_cau_mbcau_plus = [0] if not MBef else [normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, np.unique(np.concatenate(([m], MBca))).tolist()]) for m in MBef]

        #I(cause; effect | arrays_m_plus_MBef)
        cau_eff_mbeff_plus = [0] if not MBca else [normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:, np.unique(np.concatenate(([m], MBef))).tolist()]) for m in MBca]
        
        #I(m; cause) for m in MBef
        m_cau = [0] if not MBef else [normalized_conditional_information(D.iloc[:, MBef[j]], D.iloc[:, ca]) for j in range(len(MBef))]

        #I(m; effect) for m in MBca
        m_eff = [0] if not MBca else [normalized_conditional_information(D.iloc[:, MBca[j]], D.iloc[:, ef]) for j in range(len(MBca))]

        #I(cause; m | effect) for m in MBef
        cau_m_eff = [0] if not MBef else [normalized_conditional_information(D.iloc[:, ca], D.iloc[:, MBef[j]], D.iloc[:, ef]) for j in range(len(MBef))]
        
        #I(effect; m | cause) for m in MBef
        eff_m_cau = [0] if not MBca else [normalized_conditional_information(D.iloc[:, ef], D.iloc[:, MBca[j]], D.iloc[:, ca]) for j in range(len(MBca))]

        #create all possible couples of MBca and MBef
        mbca_mbef_couples = list(np.array(np.meshgrid(np.arange(len(MBca)), np.arange(len(MBef)))).T.reshape(-1,2))

        #I(mca ; mef | cause) for (mca,mef) in mbca_mbef_couples
        mca_mef_cau = [0] if not mbca_mbef_couples else [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBef[j]], D.iloc[:, ca]) for i, j in mbca_mbef_couples]

        #I(mca ; mef| effect) for (mca,mef) in mbca_mbef_couples
        mca_mef_eff = [0] if not mbca_mbef_couples else [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBef[j]], D.iloc[:, ef]) for i, j in mbca_mbef_couples]
        
        mbca_couples = list(np.array([(i, j) for i in range(len(MBca)) for j in range(i+1, len(MBca))]))
        
        # #I(mca ; mca| cause) - I(mca ; mca) for (mca,mca) in mbca_couples
        # # mca_mca_cau = [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBca[j]], D.iloc[:, ca]) - normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBca[j]]) for i, j in mbca_couples]
        # #problem is here
        mca_mca_cau = [0] if not mbca_couples else [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBca[j]], D.iloc[:, ca]) for i, j in mbca_couples]

        mbef_couples = list(np.array([(i, j) for i in range(len(MBef)) for j in range(i+1, len(MBef))]))

        #I(mbe ; mbe| effect) - I(mbe ; mbe) for (mbe,mbe) in mbef_couples
        # mbe_mbe_eff = [normalized_conditional_information(D.iloc[:, MBef[i]], D.iloc[:, MBef[j]], D.iloc[:, ef]) - normalized_conditional_information(D.iloc[:, MBef[i]], D.iloc[:, MBef[j]]) for i, j in mbef_couples]
        mbe_mbe_eff = [0] if not mbef_couples else [normalized_conditional_information(D.iloc[:, MBef[i]], D.iloc[:, MBef[j]], D.iloc[:, ef]) for i, j in mbef_couples]
        
        # E_ef = pd.DataFrame(ecdf(D.iloc[:, ef])(D.iloc[:, ef])) 
        # E_ca = pd.DataFrame(ecdf(D.iloc[:, ca])(D.iloc[:, ca]))

        # #I(cause ; ecdf_effect) 
        # gini_ca_ef = normalized_conditional_information(D.iloc[:, ca], E_ef)

        # #I(cause ; ecdf_effect) 
        # gini_ef_ca = normalized_conditional_information(D.iloc[:, ef], E_ca)

        # #I(effect ; ecdf_cause | MBeffect)
        # gini_delta = normalized_conditional_information(D.iloc[:, ef], E_ca, D.iloc[:, MBef])  
        
        # #I(cause ; ecdf_effect | MBcause)
        # gini_delta2 = normalized_conditional_information(D.iloc[:, ca], E_ef, D.iloc[:, MBca]) 


        # if compute_error_descriptors:
        #     mfs = [i for i in range(n_features) if i not in [ef]]
        #     # if boot == "mimr":
        #         # fsef = [mfs[i] for i in mimr(D.iloc[:, mfs], D.iloc[:, ef], nmax=3)]
        #     # if boot == "rank":
        #     #     fsef = [mfs[i] for i in rankrho(D.iloc[:, mfs], D.iloc[:, ef], nmax=3)]
        #     ranking = rankrho(D.iloc[:, mfs], D.iloc[:, ef], nmax=min(n_features-1,3)) 
        #     fsef = [mfs[i] for i in ranking]
        #     eef = epred(D.iloc[:, fsef], D.iloc[:, ef])
        #     if self.verbose: print("eef done")
        #     mfs = [i for i in range(n_features) if i not in [ca]]
        #     # if boot == "mimr":
        #     #     fsca = [mfs[i] for i in mimr(D.iloc[:, mfs], D.iloc[:, ca], nmax=3)]
        #     # if boot == "rank":
        #     #     fsca = [mfs[i] for i in rankrho(D.iloc[:, mfs], D.iloc[:, ca], nmax=3)]
        #     ranking = rankrho(D.iloc[:, mfs], D.iloc[:, ca], nmax=min(n_features-1,3))
        #     fsca = [mfs[i] for i in ranking]
        #     eca = epred(D.iloc[:, fsca], D.iloc[:, ca])
        #     if self.verbose: print("eca done")
        #     merged_list = [ca, ef] + fsef + fsca
        #     unique_indices = np.unique(merged_list)
        #     DD = D.iloc[:, unique_indices]

        #     # Icov2 = np.linalg.pinv(np.cov(DD) + np.diag([0.01] * DD.shape[1]))

        #     eDe = []

        #     eef = pd.Series(eef)
        #     eca = pd.Series(eca)

        #     eDe.append(normalized_conditional_information(eef, eca, D.iloc[:, ca]) - normalized_conditional_information(eef, eca))
        #     eDe.append(normalized_conditional_information(eef, eca, D.iloc[:, ef]) - normalized_conditional_information(eef, eca))
        #     eDe.append(normalized_conditional_information(eef, D.iloc[:, ca], D.iloc[:, ef]) - normalized_conditional_information(eef, D.iloc[:, ca]))
        #     eDe.append(normalized_conditional_information(eca, D.iloc[:, ef], D.iloc[:, ca]) - normalized_conditional_information(eca, D.iloc[:, ef]))
        #     eDe.append(normalized_conditional_information(eca, D.iloc[:, ef]))
        #     eDe.append(normalized_conditional_information(eef, D.iloc[:, ca]))
            
        #     #TODO: understand the need for Icov
        #     cov_D = np.cov(D, rowvar=False)  # Get covariance matrix. Set rowvar to False to ensure columns are treated as variables.
        #     shifted_cov_D = cov_D + np.eye(D.shape[1]) * 0.01  # Add 0.01 to the diagonal
        #     Icov = np.linalg.inv(shifted_cov_D)
        #     # Assuming Icov is already defined
        #     eDe.extend([
        #         Icov[ca, ef],
        #         # Icov2[0, 1],
        #         np.corrcoef(eef, D.iloc[:, ca])[0, 1],
        #         np.corrcoef(eca, D.iloc[:, ef])[0, 1],
        #         HOC(eef, eca, 0, 1),
        #         HOC(eef, eca, 1, 0),
        #         skew(eca),
        #         skew(eef)
        #     ])

        #     eDe_names = [
        #         "M.e1", "M.e2", "M.e3", "M.e4", "M.e5", "M.e6",
        #         "M.Icov", "M.Icov2",
        #         "M.cor.e1", "M.cor.e2",
        #         "B.HOC12.e", "B.HOC21.e", "B.skew.eca", "B.skew.eef"
        #     ]



        # ###
        # namesx = ["effca","effef","comcau","delta","delta2"]
        # namesx += ["delta.i" + str(i+1) for i in range(len(pq))]
        # namesx += ["delta2.i" + str(i+1) for i in range(len(pq))]
        # namesx += ["ca.ef","ef.ca"]
        # namesx += ["I1.i" + str(i+1) for i in range(len(pq))]
        # namesx += ["I1.j" + str(i+1) for i in range(len(pq))]
        # namesx += ["I2.i" + str(i+1) for i in range(len(pq))]
        # namesx += ["I2.j" + str(i+1) for i in range(len(pq))]
        # namesx += ["I3.i" + str(i+1) for i in range(len(pq))]
        # namesx += ["I3.j" + str(i+1) for i in range(len(pq))]
        # namesx += ["Int3.i" + str(i+1) for i in range(len(pq))]
        # namesx += ["Int3.j" + str(i+1) for i in range(len(pq))]
        # namesx += ["gini.delta","gini.delta2","gini.ca.ef","gini.ef.ca"]
        # namesx += ['N', 'n', 'n/N', 'B.kurtosis1', 'B.kurtosis2', 'B.skewness1', 'B.skewness2',
        #                 'B.hoc12', 'B.hoc21', 'B.hoc13', 'B.hoc31']

        # keys = ['graph_id','edge_source','edge_dest'] + namesx

        
        values = [DAG_index, ca, ef]
        values.extend([coeff_cause, coeff_eff, com_cau, eff_cau_mbeff, cau_eff_mbcau])
        values.extend(np.quantile(eff_cau_mbcau_plus, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(cau_eff_mbeff_plus, q=pq, axis=0).flatten()) 
        values.extend([cau_eff, eff_cau])
        values.extend(np.quantile(m_cau, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(m_eff, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(cau_m_eff, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(eff_m_cau, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(mca_mef_cau, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(mca_mef_eff, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(mca_mca_cau, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(mbe_mbe_eff, q=pq, axis=0).flatten()) 
        # values.extend([gini_delta, gini_delta2,gini_ca_ef, gini_ef_ca])
        values.extend([
                n_observations,                
                n_features,
                n_features/n_observations,
                kurtosis(D.iloc[:, ca]),
                kurtosis(D.iloc[:, ef]),
                skew(D.iloc[:, ca]),
                skew(D.iloc[:, ef]),
                HOC(D.iloc[:, ca], D.iloc[:, ef], 1, 2),
                HOC(D.iloc[:, ca], D.iloc[:, ef], 2, 1),
                HOC(D.iloc[:, ca], D.iloc[:, ef], 1, 3),
                HOC(D.iloc[:, ca], D.iloc[:, ef], 3, 1),
                # stab(D.iloc[:, ca], D.iloc[:, ef]),
                # stab(D.iloc[:, ef], D.iloc[:, ca])
                ]) 
        
        keys = ['graph_id','edge_source','edge_dest']
        keys = keys + [f"Feature{i}" for i in range(len(values) - len(keys))]
        if compute_error_descriptors:
            values.extend(eDe)
            keys.extend(eDe_names)

        
        # Replace NA values with 0
        dictionary = dict(zip(keys, values))
        # for key in dictionary:
        #     if np.isnan(dictionary[key]):
        #         dictionary[key] = 0
        
        # if self.verbose: print(datetime.now().strftime('%H:%M:%S'), "Descriptors for DAG", DAG_index, "edge pair", ca, ef, "computed")
        return dictionary

    def load_descriptors_df(self, path: str) -> None:
        """
        Load descriptors from a CSV file.

        Parameters:
            path (str): The path to the CSV file.

        """
        df = pd.read_csv(path)
        self.X = df.iloc[:, :-1]
        self.Y = df.iloc[:, -1]

    def get_descriptors_df(self) -> pd.DataFrame:
        """
        Get the concatenated DataFrame of X and Y.

        Returns:
            pd.DataFrame: The concatenated DataFrame of X and Y.

        """
        concatenated_df = pd.concat([self.X,self.Y], axis=1)
        concatenated_df.columns = concatenated_df.columns[:len(concatenated_df.columns)-1].tolist() + ["is_causal"]
        return concatenated_df
    
    def save_descriptors_df(self, path: str) -> None:
        """
        Save the concatenated DataFrame of X and Y to a CSV file.

        Parameters:
            path (str): The path to the CSV file.

        """
        self.get_descriptors_df().to_csv(path, index=False)

    def get_test_couples(self):
        return self.test_couples
        