from typing import List, Tuple, Union
import numpy as np
import networkx as nx
# from scipy.stats import truncnorm
import random
import multiprocessing as mp

class SimulatedDAG:
    """
    An class to store a list of DAGs and associated observations.

    Attributes:
    list_DAGs: list of stored DAGs
    list_observationsDAGs: list of observed datasets, each sampled from the corresponding member of list_DAGs
    NDAG: number of DAGs.

    functionType: type of the dependency. It is either one of [0,1,2] or a two values range [0,2]. It corresponds to ("linear", "quadratic","sigmoid")
    
    seed: random seed

    Methods:
    create_simulated_DAG: creates and simulates a specified number of DAGs and associated observations

    References:
    Gianluca Bontempi, Maxime Flauder (2015) From dependency to causality: a machine learning approach. JMLR, 2015.

    Examples:
    # Creating a simulatedDAG object
    sd = simulatedDAG(list.DAGs = NULL, list.observationsDAGs = NULL, NDAG = 10, functionType = "linear", seed = 0)

    # Creating and simulating DAGs
    sd.create_simulated_DAG(NDAG = 10, noNodes = c(15,40), N = c(50,100), sdn = c(0.45,0.75), verbose = FALSE, 
                            functionType = "linear", quantize = TRUE, maxpar_pc = 0.6, goParallel = TRUE, additive = TRUE, 
                            weights = c(-1, 1), maxV = 100)
    """
    
    def __init__(self, list_DAGs=None, list_observationsDAGs=None, NDAG=None, functionType=None, seed=None):
        self.list_DAGs = list_DAGs
        self.list_observationsDAGs = list_observationsDAGs
        self.NDAG = NDAG
        self.functionType = functionType
        self.seed = seed

        self.FUNCTION_TYPE_DICT = {0: "linear", 1: "quadratic", 2: "sigmoid"}

    def initialize(self, nodes_number, observations_number, noise_std, quantize, maxpar_pc, additive, weights, maxV, verbose, n_jobs):
        """
        Creates and simulates a specified number of DAGs and associated observations.

        Args:
        nodes_number: number of Nodes of the DAGs. If it is a two-valued vector [a,b], the value of Nodes is randomly sampled in the interval
        observations_number: number of sampled observations for each DAG. If it is a two-valued vector [a,b], the value of N is randomly sampled in the interval [a,b]
        noise_std: standard deviation of additive noise. If it is a two-valued vector [a,b], the value of N is randomly sampled in the interval
        
        
        quantize: if True it discretizes the observations into two bins. If it is a two-valued vector [a,b], the value of quantize is randomly sampled in the interval [a,b] #TODO: right now it is only boolean. check the case of two valued vectors, what does it mean? 
        maxpar_pc: maximum number of parents expressed as a percentage of the number of nodes
        additive: if True the output is the sum of the H transformation of the inputs #TODO: right now it is only boolean. check the case of two valued vectors, what does it mean? 

        weights: [lower,upper], lower and upper bound of the values of the linear weights
        maxV: maximum value #TODO: of the linear weights? 

        verbose: if True it prints out the state of progress
        n_jobs: if >1 it uses parallelism

        """
        
        self.nodes_number = nodes_number
        self.quantize = quantize
        self.maxpar_pc = maxpar_pc
        self.observations_number = observations_number
        self.noise_std = noise_std
        self.additive = additive

        self.weights = weights
        self.maxV = maxV

        self.verbose = verbose
        self.n_jobs = n_jobs


        self.generate()
    

    def sample_in_range(self, values): 
        if isinstance(values, list) and len(values) == 2:
            return random.sample(range(values[0], values[1]+1), 1)[0]
        #elif numeric 
        elif isinstance(values, int) or isinstance(values, float):
            return values
        else:
            raise ValueError("Invalid values")
    
    def sample_from_list(self, values):
        if isinstance(values, list):
            return random.sample(values, 1)[0]
        elif isinstance(values, int) or isinstance(values, float):
            return values
        else:
            raise ValueError("Invalid values")
        
    def generate(self):
        if self.n_jobs > 1: 
            # TODO: parallelize
            pass 

        X = None 
        Y = None 
        list_of_dags = None
        list_of_observations_dags = None

        if self.NDAG <= 0: 
            raise ValueError("NDAG must be greater than 0")
        
        for current_DAG in range(self.NDAG):
            random.seed(self.seed + current_DAG)

            
            quantize_current_DAG = self.sample_from_list(self.quantize)
            observations_number_current_DAG = self.sample_in_range(self.observations_number)
            nodes_number_current_DAG = self.sample_in_range(self.nodes_number) #TODO: check why there is a max 3, and if it is necessary 
            noise_std_current_DAG = self.sample_in_range(self.noise_std)

            if random.uniform(0, 1) < 0.5:
                functionType_current_DAG = self.FUNCTION_TYPE_DICT[self.sample_from_list(self.functionType)]
                additive_current_DAG = self.sample_from_list(self.additive)
                
                sdn_ii = self.noise_std #TODO: why ii ?

                weights_current_DAG = self.weights
                maxV_current_DAG = self.maxV

                HH = []

                #TODO: incoherence with respect to the docstring! What happens with the previous values of functionType? 
                for functionType_i in self.functionType:
                    if functionType_i == "linear":
                        H = lambda: H_Rn(1)
                    elif functionType_i == "quadratic":
                        H = lambda: H_Rn(2)
                    elif functionType_i == "sigmoid":
                        H = lambda: H_sigmoid(1)
                    elif functionType_i == "kernel":
                        H = lambda: H_kernel()
                    HH.append(H)

                counter_2 = 0
                while True:
                    V = range(1, max(4, nodes_number_current_DAG - counter_2))

                    maxpar_pc_current_DAG = self.sample_from_list(self.maxpar_pc) # TODO: parallel maxima and minima: check why and how 

                    maxpar = round(maxpar_pc_i * noNodes)

                    wgt = np.random.uniform(low=0.85, high=1, size=1)[0]

                    netwDAG = random_dag(V, max_parents=maxpar, weights=wgt)

                


            else: 
                g = self.gendataDAG(observations_number_current_DAG, nodes_number_current_DAG, noise_std_current_DAG)
                netwDAG = g['DAG']
                observationsDAG = g['data']
                #TODO: check edges and uncomment
                #if self.verbose: print(f"simulatedDAG gendataDAG: DAG number: {i} generated: #nodes= {len(graph.edges(netwDAG))}, # edges= {sum([len(edge) for edge in graph.edges(netwDAG)])}, # samples= {observationsDAG.shape[0]}, # cols= {observationsDAG.shape[1]}")



    def gendataDAG(self, N, noNodes, sdn):
        #TODO: implement either here or as a class 
        pass

    # def _generateDAG(self, noNodes: int, seed: int = 1234) -> nx.DiGraph:
    #     """
    #     Generates a random DAG with noNodes nodes
    #     """
    #     np.random.seed(seed)
    #     G = nx.gnp_random_graph(noNodes, 0.5, directed=True)
    #     G = nx.DiGraph(G)
    #     G.remove_edges_from(nx.selfloop_edges(G))
    #     G = nx.DiGraph(nx.algorithms.dag.transitive_reduction(G))
    #     return G