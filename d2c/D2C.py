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


class D2C:
    def __init__(self, simulatedDAGs: SimulatedDAGs, rev: bool = True, verbose=False, random_state: int = 42, n_jobs: int = 1) -> None:
        """
        Class for D2C analysis.

        D2C (Dependency to Causalilty) analysis is a method for inferring causal relationships
        from observational data using simulated directed acyclic graphs (DAGs) and computing 
        asymmetric descriptors from the observations associated with each DAG.

        Args:
            simulatedDAGs (SimulatedDAGs): An instance of the SimulatedDAGs class.
            rev (bool, optional): Whether to consider reverse edges. Defaults to True.
            n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.DAGs_index = np.arange(len(simulatedDAGs.list_DAGs))
        self.DAGs = simulatedDAGs.list_DAGs
        self.observations_from_DAGs = simulatedDAGs.list_observations
        self.rev = rev
        self.X = None
        self.Y = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

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

        edge_pairs = self._generate_edge_pairs(DAG_index,"is.child")

        for edge_pair in edge_pairs:
            parent, child = edge_pair[0], edge_pair[1]
            print("Computing descriptors for DAG", DAG_index, "edge pair", parent, child)
            descriptor = self._compute_descriptors(DAG_index, parent, child)
            X.append(descriptor)
            Y.append(1)  # Label edge as "is.child"

            if self.rev:
                # Reverse edge direction
                descriptor_reverse = self._compute_descriptors(DAG_index, child, parent)
                X.append(descriptor_reverse)
                Y.append(0)  # Label reverse edge as NOT "is.child"

        return X, Y
    
    def _generate_edge_pairs(self, DAG_index, dependency_type: str) -> list:
        """
        Generate pairs of edges based on the dependency type.

        Args:
            dependency_type (str): The type of dependency.

        Returns:
            list: List of edge pairs.

        """
        edge_pairs = []
        if dependency_type == "is.child":
            for parent_node, child_node in self.DAGs[DAG_index].edges:
                edge_pairs.append((parent_node, child_node))
        print("Edge pairs for DAG", DAG_index, "computed:", edge_pairs)
        return edge_pairs

    


    def _compute_descriptors(self, DAG_index, ca, ef, ns=None, maxs=20,
            lin=False, acc=True, struct=True,
            pq= [0.05,0.1,0.25,0.5,0.75,0.9,0.95], boot="mrmr"):
        
        """
        Compute descriptor of two variables in a dataset under the assumption that one variable is the cause and the other the effect.

        Parameters:
        ca (int): Node index of the putative cause. Must be in the range [0, n).
        ef (int): Node index of the putative effect. Must be in the range [0, n).
        ns (int, optional): Size of the Markov Blanket. Defaults to min(4, n - 2).
        lin (bool, optional): If True, uses a linear model to assess dependency. Defaults to False.
        acc (bool, optional): If True, uses the accuracy of the regression as a descriptor. Defaults to True.
        struct (bool, optional): If True, uses the ranking in the Markov blanket as a descriptor. Defaults to False.
        pq (list of float, optional): A list of quantiles used to compute the descriptor. Defaults to [0.1,0.25,0.5,0.75,0.9].
        maxs (int, optional): Max number of pairs MB(i), MB(j) considered. Defaults to 10.
        boot (str, optional): Feature selection algorithm. Defaults to "mimr".
        errd (bool, optional): If True, includes the descriptors of the error. Defaults to False.
        delta (bool, optional): Not used in current implementation. Defaults to False.
        stabD (bool, optional): Not used in current implementation. Defaults to False.

        Returns:
        dict: A dictionary with the computed descriptors.

        Raises:
        ValueError: If there are missing or infinite values in D.
        """
        D = self.observations_from_DAGs[DAG_index]

        print('Computing descriptors for edge pair: ', ca, ef, 'in DAG', DAG_index)

        #scale using pandas
        D = (D - D.mean()) / D.std()

        if np.any(np.isnan(D)) or np.any(np.isinf(D)): raise ValueError("Error: NA or Inf in descriptor") # Check if there are any missing or infinite values in the data
    
        # Number of variables
        n = D.shape[1]
        # Number of observations
        N = D.shape[0]


        # Set default value for ns if not provided
        if ns is None:
            ns = min(4, n-2)

        # Check that ca and ef are within the valid range
        if ca >= n or ef >= n:
            raise ValueError(f"ca={ca}, ef={ef}, n={n}\nerror in D2C_n")
    

        # Initial sets for Markov Blanket
        MBca = set(np.arange(n)) - {ca}
        MBef = set(np.arange(n)) - {ef}
        # intersection of the Markov Blanket of ca and ef
        common_causes = MBca.intersection(MBef)
        if self.verbose: 
            print("common_causes: ", common_causes)
        # Creation of the Markov Blanket of ca (denoted MBca) and ef (MBef)
        if n > (ns+1):

            if self.verbose: print("Computing Markov Blanket")
            # MBca
            ind = list(set(np.arange(n)) - {ca})

            if self.verbose: 
                print("About to rankrho")

            ind = rankrho(D.iloc[:,ind],D.iloc[:,ca],nmax=min(len(ind),5*ns),verbose=self.verbose) - 1 #python starts from 0
            if self.verbose: print('Ind:',ind)
            if self.verbose: print('Exited rankrho')

            if boot == "mrmr":
                mrmr = mRMR(D.iloc[:,ind],D.iloc[:,ca],nmax=ns,verbose=self.verbose)
                MBca = ind[mrmr]  
                if self.verbose: print("MBca: ", MBca)
            # MBef
            ind2 = list(set(np.arange(n)) - {ef})
            ind2 = rankrho(D.iloc[:,ind2],D.iloc[:,ef],nmax=min(len(ind2),5*ns),verbose=self.verbose)  
            if boot == "mrmr":
                MBef = ind2[mRMR(D.iloc[:,ind2],D.iloc[:,ef],nmax=ns,verbose=self.verbose)]  
                if self.verbose: print("MBef: ", MBef)
    
        if acc:
            comcau = 1

            if len(common_causes) > 0:
                if self.verbose: print("common_causes: ", common_causes)
                comcau = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, list(common_causes)], lin=lin,verbose=self.verbose) 

            effca = coeff(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, MBef], verbose=self.verbose)
            effef = coeff(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:, MBca], verbose=self.verbose)

            if self.verbose: print("effca: ", effca, "effef: ", effef)

            ca_ef = normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef], lin=lin,verbose=self.verbose) 
            ef_ca = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], lin=lin,verbose=self.verbose) 

            if self.verbose: print("ca_ef: ", ca_ef, "ef_ca: ", ef_ca)

            delta = normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:, MBef], lin=lin,verbose=self.verbose) 
            delta2 = normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:, MBca], lin=lin,verbose=self.verbose) 

            if self.verbose: print("delta: ", delta, "delta2: ", delta2)    


            delta_i = []  
            delta2_i = []
            arrays_m_plus_MBca = [np.unique(array).tolist() for array in [np.concatenate(([m], MBca)) for m in MBef]]
            arrays_m_plus_MBef = [np.unique(array).tolist() for array in [np.concatenate(([m], MBef)) for m in MBca]]

            for array in arrays_m_plus_MBca:
                delta_i.append( normalized_conditional_information(D.iloc[:, ef], D.iloc[:, ca], D.iloc[:,  array], lin=lin,verbose=self.verbose))
            for array in arrays_m_plus_MBef:
                delta2_i.append(normalized_conditional_information(D.iloc[:, ca], D.iloc[:, ef], D.iloc[:,  array], lin=lin,verbose=self.verbose))

            if self.verbose: print("delta_i: ", delta_i, "delta2_i: ", delta2_i)

            I1_i = [normalized_conditional_information(D.iloc[:, MBef[j]], D.iloc[:, ca], lin=lin,verbose=self.verbose) for j in range(len(MBef))]
            I1_j = [normalized_conditional_information(D.iloc[:, MBca[j]], D.iloc[:, ef], lin=lin,verbose=self.verbose) for j in range(len(MBca))]

            if self.verbose: print("I1_i: ", I1_i, "I1_j: ", I1_j)

            I2_i = [normalized_conditional_information(D.iloc[:, ca], D.iloc[:, MBef[j]], D.iloc[:, ef], lin=lin,verbose=self.verbose) for j in range(len(MBef))]
            I2_j = [normalized_conditional_information(D.iloc[:, ef], D.iloc[:, MBca[j]], D.iloc[:, ca], lin=lin,verbose=self.verbose) for j in range(len(MBca))]

            if self.verbose: print("I2_i: ", I2_i, "I2_j: ", I2_j)

            # Randomly select maxs pairs
            IJ = np.array(np.meshgrid(np.arange(len(MBca)), np.arange(len(MBef)))).T.reshape(-1,2)
            np.random.shuffle(IJ)
            IJ = IJ[:min(maxs, len(IJ))]

            if self.verbose: print("IJ: ", IJ)

            I3_i = [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBef[j]], D.iloc[:, ca], lin=lin,verbose=self.verbose) for i, j in IJ]
            I3_j = [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBef[j]], D.iloc[:, ef], lin=lin,verbose=self.verbose) for i, j in IJ]

            if self.verbose: print("I3_i: ", I3_i, "I3_j: ", I3_j)

            IJ = np.array([(i, j) for i in range(len(MBca)) for j in range(i+1, len(MBca))])
            np.random.shuffle(IJ)
            IJ = IJ[:min(maxs, len(IJ))]

            Int3_i = [normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBca[j]], D.iloc[:, ca], lin=lin,verbose=self.verbose) - normalized_conditional_information(D.iloc[:, MBca[i]], D.iloc[:, MBca[j]], lin=lin,verbose=self.verbose) for i, j in IJ]

            if self.verbose: print("Int3_i: ", Int3_i)

            IJ = np.array([(i, j) for i in range(len(MBef)) for j in range(i+1, len(MBef))])
            np.random.shuffle(IJ)
            IJ = IJ[:min(maxs, len(IJ))]

            Int3_j = [normalized_conditional_information(D.iloc[:, MBef[i]], D.iloc[:, MBef[j]], D.iloc[:, ef], lin=lin,verbose=self.verbose) - normalized_conditional_information(D.iloc[:, MBef[i]], D.iloc[:, MBef[j]], lin=lin,verbose=self.verbose) for i, j in IJ]

            if self.verbose: print("Int3_j: ", Int3_j)

            E_ef = ecdf(D.iloc[:, ef],verbose=self.verbose)(D.iloc[:, ef]) 
            E_ca = ecdf(D.iloc[:, ca],verbose=self.verbose)(D.iloc[:, ca])

            if self.verbose: print("E_ef: ", E_ef, "E_ca: ", E_ca)

            gini_ca_ef = normalized_conditional_information(D.iloc[:, ca], pd.DataFrame(E_ef), lin=lin,verbose=self.verbose)
            gini_ef_ca = normalized_conditional_information(D.iloc[:, ef], pd.DataFrame(E_ca), lin=lin,verbose=self.verbose)

            if self.verbose: print("gini_ca_ef: ", gini_ca_ef, "gini_ef_ca: ", gini_ef_ca)

            gini_delta = normalized_conditional_information(D.iloc[:, ef], pd.DataFrame(E_ca), D.iloc[:, MBef], lin=lin,verbose=self.verbose)
            gini_delta2 = normalized_conditional_information(D.iloc[:, ca], pd.DataFrame(E_ef), D.iloc[:, MBca], lin=lin,verbose=self.verbose)

            if self.verbose: print("gini_delta: ", gini_delta, "gini_delta2: ", gini_delta2)

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

            
            # Replace NA values with 0
            dictionary = dict(zip(keys, values))
            # for key in dictionary:
            #     if np.isnan(dictionary[key]):
            #         dictionary[key] = 0
            
            print("Descriptors for DAG", DAG_index, "edge pair", ca, ef, "computed")
            return dictionary

    def load_descriptors(self, path: str) -> None:
        """
        Load descriptors from a CSV file.

        Parameters:
            path (str): The path to the CSV file.

        """
        df = pd.read_csv(path)
        self.X = df.iloc[:, :-1]
        self.Y = df.iloc[:, -1]

    def get_df(self) -> pd.DataFrame:
        """
        Get the concatenated DataFrame of X and Y.

        Returns:
            pd.DataFrame: The concatenated DataFrame of X and Y.

        """
        concatenated_df = pd.concat([self.X,self.Y], axis=1)
        concatenated_df.columns = concatenated_df.columns[:len(concatenated_df.columns)-1].tolist() + ["is_causal"]
        return concatenated_df
    
    def save_df(self, path: str) -> None:
        """
        Save the concatenated DataFrame of X and Y to a CSV file.

        Parameters:
            path (str): The path to the CSV file.

        """
        concatenated_df = self.get_df()
        concatenated_df.to_csv(path, index=False)

    def get_score(self, model: RandomForestClassifier = RandomForestClassifier(), n_splits: int = 10, metric: str = "accuracy") -> Union[float, None]:

        
        dataframe = self.get_df()
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
        Test the performance of a D2C model on a Pandas Datafrane of descriptors from another D2C object.

        Parameters:
            d2c (D2C): The D2C object to test.
            metric (str): The metric to use for evaluation. Defaults to 'accuracy'.

        Returns:
            float: The score of the model.

        """
        dataframe = self.get_df()
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

            for index, row in pd.concat([X_test, y_test], axis=1).iterrows():
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
