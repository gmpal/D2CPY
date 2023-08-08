import pandas as pd
import networkx as nx
import numpy as np
import unittest

from simulated import Simulated
from simulatedTimeSeries import SimulatedTimeSeries

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Create the sample DAG
        self.dag = nx.DiGraph()
        # For simplicity, we are using just a few nodes
        self.dag.add_node("1_t-1", value=None, bias=0, sigma=1)
        self.dag.add_node("2", value=None, bias=1, sigma=1)
        self.dag.add_edge("1_t-1", "y", weight=2, H="linear")

        # Sample data
        self.data = pd.DataFrame({1: [1, 2, 3], 2: [0, 0, 0]})
        
        self.obj = SimulatedTimeSeries(1,10,2)  # Replace with your actual class name

    def test_generate_timestep_observation(self):
        # Run the function
        self.obj._generate_timestep_observation(self.dag, self.data)

        # Check if values in the data match expected computation
        # For this simple example, y's new value is y's bias (1) + weight(2) * x's previous value
        expected_y_value = 1 + 2 * self.data.loc[len(self.data) - 2, 1]
        actual_y_value = self.data.loc[len(self.data) - 1, 2]

        self.assertEqual(np.round(expected_y_value, 2), np.round(actual_y_value, 2))

if __name__ == "__main__":
    unittest.main()
