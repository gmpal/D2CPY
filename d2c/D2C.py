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

from d2c.simulatedDAGs import SimulatedDAGs
from d2c.utils import *

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class D2C:
    def __init__(self, dags, observations, rev: bool = True, verbose=False, random_state: int = 42, n_jobs: int = 1) -> None:
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
        self.DAGs_index = np.arange(len(dags))
        self.DAGs = dags #it's a LIST
        self.observations_from_DAGs = observations #it's a LIST  
        self.rev = rev
        self.X = None
        self.Y = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state


    def compute_descriptors_no_dags(self):

        #TODO: this assumes to receive a list, but it should handle single observations better
        num_nodes = self.observations_from_DAGs[0].shape[1]
        pairs = []
        X = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    pairs.append((i,j))
                    X.append(self._compute_descriptors(0, i, j))
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
        print(DAG_index)
        X = []
        Y = []

        nodes = len(self.DAGs[DAG_index].nodes)

        child_edge_pairs = []
        all_possible_edges = [(i, j) for i in range(nodes) for j in range(nodes) if i != j]
        #if dependency_type == "is.child": #TODO implement other dependencies
        for parent_node, child_node in self.DAGs[DAG_index].edges:
            child_edge_pairs.append((parent_node, child_node))

        for edge_pair in child_edge_pairs:
            parent, child = edge_pair[0], edge_pair[1]
            descriptor = self._compute_descriptors(DAG_index, parent, child)
            X.append(descriptor)
            Y.append(1)  # Label edge as "is.child"

        # For all remaining edges that are not in the DAG, compute descriptors and label them as "not a child"
        for edge_pair in all_possible_edges:
            if edge_pair not in child_edge_pairs:
                parent, child = edge_pair[0], edge_pair[1]
                descriptor = self._compute_descriptors(DAG_index, parent, child)
                X.append(descriptor)
                Y.append(0)  # Label edge as "not a child"
        return X, Y
            

    def _compute_descriptors(self, DAG_index, ca, ef, MB_size=None, maxs=20,
            lin=False, acc=True, compute_error_descriptors=True,
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
        D = self.observations_from_DAGs[DAG_index]
        D = (D - D.mean()) / D.std()
        n_observations, n_features = D.shape
        
        if np.any(np.isnan(D)) or np.any(np.isinf(D)): raise ValueError("Error: NA or Inf in data")

        # Set default value for ns if not provided
        if MB_size is None:
            MB_size = min(4, n_features-2)

        # Initializations of Markov Blankets
        MBca = set(np.arange(n_features)) - {ca}
        MBef = set(np.arange(n_features)) - {ef}
        # intersection of the Markov Blanket of ca and ef
        common_causes = MBca.intersection(MBef)
        # Creation of the Markov Blanket of ca (denoted MBca) and ef (MBef)
        if n_features > (MB_size+1):
            # MBca
            ind = list(set(np.arange(n_features)) - {ca})
            ind = rankrho(D.iloc[:,ind],D.iloc[:,ca],nmax=min(len(ind),5*MB_size),verbose=self.verbose) - 1 #python starts from 0
            MBca = ind[mRMR(D.iloc[:,ind],D.iloc[:,ca],nmax=MB_size,verbose=self.verbose)]  

            # MBef
            ind2 = list(set(np.arange(n_features)) - {ef})
            ind2 = rankrho(D.iloc[:,ind2],D.iloc[:,ef],nmax=min(len(ind2),5*MB_size),verbose=self.verbose)  
            MBef = ind2[mRMR(D.iloc[:,ind2],D.iloc[:,ef],nmax=MB_size,verbose=self.verbose)]  
    
        # I(cause; effect | common_causes) 
        comcau = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, list(common_causes)]) if len(common_causes) > 0 else 1

        effca = coeff(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, MBef], verbose=self.verbose)
        effef = coeff(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:, MBca], verbose=self.verbose)

        #I(cause; effect) 
        ca_ef = normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef]) 
        #I(effect; cause)
        ef_ca = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca]) 

        #I(effect; cause | MBeffect) 
        delta = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, MBef]) 
        #I(cause; effect | MBcause)
        delta2 = normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:, MBca]) 


        delta_i = []  
        delta2_i = []
        arrays_m_plus_MBca = [np.unique(array).tolist() for array in [np.concatenate(([m], MBca)) for m in MBef]]
        arrays_m_plus_MBef = [np.unique(array).tolist() for array in [np.concatenate(([m], MBef)) for m in MBca]]

        for array in arrays_m_plus_MBca:
            delta_i.append( normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:,  array]))
        for array in arrays_m_plus_MBef:
            delta2_i.append(normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:,  array]))

        I1_i = [normalized_conditional_information(D.iloc[:, MBef[j]], D.iloc[:, ca]) for j in range(len(MBef))]
        I1_j = [normalized_conditional_information(D.iloc[:, MBca[j]], D.iloc[:, ef]) for j in range(len(MBca))]

        I2_i = [normalized_conditional_information(D.iloc[:, ca], D.iloc[:, MBef[j]], D.iloc[:, ef]) for j in range(len(MBef))]
        I2_j = [normalized_conditional_information(D.iloc[:, ef], D.iloc[:, MBca[j]], D.iloc[:, ca]) for j in range(len(MBca))]

        # Randomly select maxs pairs
        IJ = np.array(np.meshgrid(np.arange(len(MBca)), np.arange(len(MBef)))).T.reshape(-1,2)
        np.random.shuffle(IJ)
        IJ = IJ[:min(maxs, len(IJ))]

        I3_i = [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBef[j]], D.iloc[:, ca]) for i, j in IJ]
        I3_j = [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBef[j]], D.iloc[:, ef]) for i, j in IJ]

        IJ = np.array([(i, j) for i in range(len(MBca)) for j in range(i+1, len(MBca))])
        np.random.shuffle(IJ)
        IJ = IJ[:min(maxs, len(IJ))]

        Int3_i = [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBca[j]], D.iloc[:, ca]) - normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBca[j]]) for i, j in IJ]
        if len(Int3_i) == 0:
            Int3_i = [0]
            #TODO: verify why this happens

        IJ = np.array([(i, j) for i in range(len(MBef)) for j in range(i+1, len(MBef))])
        np.random.shuffle(IJ)
        IJ = IJ[:min(maxs, len(IJ))]

        Int3_j = [normalized_conditional_information(D.iloc[:, MBef[i]], D.iloc[:, MBef[j]], D.iloc[:, ef]) - normalized_conditional_information(D.iloc[:, MBef[i]], D.iloc[:, MBef[j]]) for i, j in IJ]
        if len(Int3_j) == 0:
            Int3_j = [0]
            #TODO: verify why this happens
        E_ef = ecdf(D.iloc[:, ef],verbose=self.verbose)(D.iloc[:, ef]) 
        E_ca = ecdf(D.iloc[:, ca],verbose=self.verbose)(D.iloc[:, ca])

        gini_ca_ef = normalized_conditional_information(D.iloc[:, ca], pd.DataFrame(E_ef))
        gini_ef_ca = normalized_conditional_information(D.iloc[:, ef], pd.DataFrame(E_ca))

        gini_delta = normalized_conditional_information(D.iloc[:, ef], pd.DataFrame(E_ca), D.iloc[:, MBef])
        gini_delta2 = normalized_conditional_information(D.iloc[:, ca], pd.DataFrame(E_ef), D.iloc[:, MBca])

        if compute_error_descriptors:
            mfs = [i for i in range(n_features) if i not in [ef]]
            # if boot == "mimr":
                # fsef = [mfs[i] for i in mimr(D.iloc[:, mfs], D.iloc[:, ef], nmax=3)]
            # if boot == "rank":
            #     fsef = [mfs[i] for i in rankrho(D.iloc[:, mfs], D.iloc[:, ef], nmax=3)]
            ranking = rankrho(D.iloc[:, mfs], D.iloc[:, ef], nmax=min(n_features-1,3))  -1
            fsef = [mfs[i] for i in ranking]
            eef = epred(D.iloc[:, fsef], D.iloc[:, ef])

            mfs = [i for i in range(n_features) if i not in [ca]]
            # if boot == "mimr":
            #     fsca = [mfs[i] for i in mimr(D.iloc[:, mfs], D.iloc[:, ca], nmax=3)]
            # if boot == "rank":
            #     fsca = [mfs[i] for i in rankrho(D.iloc[:, mfs], D.iloc[:, ca], nmax=3)]
            ranking = rankrho(D.iloc[:, mfs], D.iloc[:, ca], nmax=min(n_features-1,3)) -1
            fsca = [mfs[i] for i in ranking]
            eca = epred(D.iloc[:, fsca], D.iloc[:, ca])

            merged_list = [ca, ef] + fsef + fsca
            unique_indices = np.unique(merged_list)
            DD = D.iloc[:, unique_indices]

            # Icov2 = np.linalg.pinv(np.cov(DD) + np.diag([0.01] * DD.shape[1]))

            eDe = []

            eef = pd.Series(eef)
            eca = pd.Series(eca)

            eDe.append(normalized_conditional_information(eef, eca, D.iloc[:, ca]) - normalized_conditional_information(eef, eca))
            eDe.append(normalized_conditional_information(eef, eca, D.iloc[:, ef]) - normalized_conditional_information(eef, eca))
            eDe.append(normalized_conditional_information(eef, D.iloc[:, ca], D.iloc[:, ef]) - normalized_conditional_information(eef, D.iloc[:, ca]))
            eDe.append(normalized_conditional_information(eca, D.iloc[:, ef], D.iloc[:, ca]) - normalized_conditional_information(eca, D.iloc[:, ef]))
            eDe.append(normalized_conditional_information(eca, D.iloc[:, ef]))
            eDe.append(normalized_conditional_information(eef, D.iloc[:, ca]))
            
            #TODO: understand the need for Icov
            cov_D = np.cov(D, rowvar=False)  # Get covariance matrix. Set rowvar to False to ensure columns are treated as variables.
            shifted_cov_D = cov_D + np.eye(D.shape[1]) * 0.01  # Add 0.01 to the diagonal
            Icov = np.linalg.inv(shifted_cov_D)
            # Assuming Icov is already defined
            eDe.extend([
                Icov[ca, ef],
                # Icov2[0, 1],
                np.corrcoef(eef, D.iloc[:, ca])[0, 1],
                np.corrcoef(eca, D.iloc[:, ef])[0, 1],
                HOC(eef, eca, 0, 1),
                HOC(eef, eca, 1, 0),
                skew(eca),
                skew(eef)
            ])

            eDe_names = [
                "M.e1", "M.e2", "M.e3", "M.e4", "M.e5", "M.e6",
                "M.Icov", "M.Icov2",
                "M.cor.e1", "M.cor.e2",
                "B.HOC12.e", "B.HOC21.e", "B.skew.eca", "B.skew.eef"
            ]



        ###
        namesx = ["effca","effef","comcau","delta","delta2"]
        namesx += ["delta.i" + str(i+1) for i in range(len(pq))]
        namesx += ["delta2.i" + str(i+1) for i in range(len(pq))]
        namesx += ["ca.ef","ef.ca"]
        namesx += ["I1.i" + str(i+1) for i in range(len(pq))]
        namesx += ["I1.j" + str(i+1) for i in range(len(pq))]
        namesx += ["I2.i" + str(i+1) for i in range(len(pq))]
        namesx += ["I2.j" + str(i+1) for i in range(len(pq))]
        namesx += ["I3.i" + str(i+1) for i in range(len(pq))]
        namesx += ["I3.j" + str(i+1) for i in range(len(pq))]
        namesx += ["Int3.i" + str(i+1) for i in range(len(pq))]
        namesx += ["Int3.j" + str(i+1) for i in range(len(pq))]
        namesx += ["gini.delta","gini.delta2","gini.ca.ef","gini.ef.ca"]
        namesx += ['N', 'n', 'n/N', 'B.kurtosis1', 'B.kurtosis2', 'B.skewness1', 'B.skewness2',
                        'B.hoc12', 'B.hoc21', 'B.hoc13', 'B.hoc31']

        keys = ['graph_id','edge_source','edge_dest'] + namesx

        
        values = [DAG_index, ca, ef]
        values.extend([effca, effef, comcau, delta, delta2])
        values.extend(np.quantile(delta_i, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(delta2_i, q=pq, axis=0).flatten()) 
        values.extend([ca_ef, ef_ca])
        values.extend(np.quantile(I1_i, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(I1_j, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(I2_i, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(I2_j, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(I3_i, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(I3_j, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(Int3_i, q=pq, axis=0).flatten()) 
        values.extend(np.quantile(Int3_j, q=pq, axis=0).flatten()) 
        values.extend([gini_delta, gini_delta2,gini_ca_ef, gini_ef_ca])
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
        
        if compute_error_descriptors:
            values.extend(eDe)
            keys.extend(eDe_names)

        
        # Replace NA values with 0
        dictionary = dict(zip(keys, values))
        # for key in dictionary:
        #     if np.isnan(dictionary[key]):
        #         dictionary[key] = 0
        
        # print(datetime.now().strftime('%H:%M:%S'), "Descriptors for DAG", DAG_index, "edge pair", ca, ef, "computed")
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
        concatenated_df = self.get_descriptors_df()
        concatenated_df.to_csv(path, index=False)

    def get_score(self, model: RandomForestClassifier = RandomForestClassifier(), n_splits: int = 10, metric: str = "accuracy") -> Union[float, None]:

        
        dataframe = self.get_descriptors_df()
        X = dataframe.drop(columns=['graph_id', 'is_causal'])
        y = dataframe['is_causal']
        groups = dataframe['graph_id']

        rf_classifier = model

        group_kfold = GroupKFold(n_splits=n_splits)  # You can change the number of splits (e.g., 5-fold cross-validation)

        # Perform cross-validation
        cv_scores = cross_val_score(rf_classifier, X, y, cv=group_kfold, groups=groups, scoring=metric)

        mean_f1 = cv_scores.mean()
        return mean_f1

        
    def test(self, model: RandomForestClassifier = RandomForestClassifier(), metric:str = 'accuracy', test_df: pd.DataFrame = None, reconstruct: bool = False): 
        """
        Test the performance of a D2C model on a Pandas Dataframe of descriptors from another D2C object. 
        Optionally reconctructs the DAG according to the predictions of the model.

        Parameters:
            d2c (D2C): The D2C object to test.
            metric (str): The metric to use for evaluation. Defaults to 'accuracy'.
            test_df (pd.DataFrame): The DataFrame of descriptors to use for testing. Defaults to None.
            reconstruct (bool): Whether to reconstruct the DAG. Defaults to False.

        Returns:
            float: The score of the model.
            nx.DiGraph: The reconstructed DAG.

        """
        dataframe = self.get_descriptors_df()
        X_train = dataframe.drop(['graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
        y_train = dataframe['is_causal']
        
        X_test = test_df.drop(['graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
        y_test = test_df['is_causal']

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        G = None
        if reconstruct: 
            # Assuming df is your DataFrame
            G = nx.DiGraph()  # Creates a directed graph
            test_df['is_causal'] = y_pred
            for index, row in test_df.iterrows():
                if row['is_causal']:
                    G.add_edge(row['edge_source'], row['edge_dest'])
            
        if metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif metric == 'f1':
            score =  f1_score(y_test, y_pred)
        elif metric == 'precision':
            score =  precision_score(y_test, y_pred)
        elif metric == 'recall':
            score =  recall_score(y_test, y_pred)
        elif metric == 'roc_auc':
            score =  roc_auc_score(y_test, y_pred)
        else:
            raise ValueError("Metric not supported")
        
        return score, G
