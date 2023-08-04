import unittest
import sys
sys.path.append("..")
from d2c.simulatedDAGs import SimulatedDAGs
import numpy as np
import networkx as nx
import pandas as pd


class TestSimulatedDAGs(unittest.TestCase):

    def setUp(self):
        self.simulated_DAGs = SimulatedDAGs(n_dags=2, n_observations=5, n_nodes=3, function_types=["linear"], sdn=0.1, 
                                            random_state=42, n_jobs=1)

    # tests if generate_dags() generates the correct number of DAGs and that they are of the correct type.
    def test_generate_dags(self):
        self.simulated_DAGs.generate()
        self.assertEqual(len(self.simulated_DAGs.list_DAGs), 2)
        for dag in self.simulated_DAGs.list_DAGs:
            self.assertIsInstance(dag, nx.DiGraph)
    # tests if simulate_observations() generates the correct number of dataframes and that they are of the correct type.
    def test_simulate_observations(self):
        self.simulated_DAGs.generate()
        self.simulated_DAGs.simulate_observations()
        self.assertEqual(len(self.simulated_DAGs.list_observations), 2)
        for obs in self.simulated_DAGs.list_observations:
            self.assertIsInstance(obs, pd.DataFrame)
    
    # test if _H_sigmoid(1) returns a function that, when applied to an argument, produces a result of the correct type.
    def test_sigmoid_function(self):
        f = self.simulated_DAGs._H_sigmoid(1)
        result = f(1)
        self.assertIsInstance(result, np.ndarray)

    # test if _H_Rn(1) returns a function that, when applied to an argument, produces a result of the correct type.
    def test_Rn_function(self):
        f = self.simulated_DAGs._H_Rn(1)
        result = f(1)
        self.assertIsInstance(result, np.ndarray)

    # test if _H_kernel() returns a function that, when applied to an argument, produces a result of the correct type.
    # def test_kernel_function(self):
    #     f = self.simulated_DAGs._H_kernel()
    #     result = f(np.array([1,2,3,4,5]))
    #     self.assertIsInstance(result, np.ndarray)
    
    # test if _kernel_fct() produces a result of the correct type when applied to an array argument.
    def test_kernel_fct(self):
        result = self.simulated_DAGs._kernel_fct(np.array([1,2,3,4,5]))
        self.assertIsInstance(result, np.ndarray)
        

if __name__ == '__main__':
    unittest.main()
