import unittest
from unittest.mock import Mock, patch
from d2c.simulatedDAGs import SimulatedDAGs
from d2c.D2C import D2C
import pandas as pd
import numpy as np

class TestD2C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulatedDAGs = SimulatedDAGs()
        cls.D2C = D2C(cls.simulatedDAGs)

    # use unittest.mock to mock the multiprocessing.Pool object, and simply test that this object is being called
    def test_initialize(self):
        with patch('multiprocessing.Pool') as mock_pool:
            self.D2C.initialize()
            mock_pool.assert_called()

    # test that the function returns a tuple of two lists, as expected.
    def test_compute_descriptor_parallel(self):
        obs = Mock()
        res = self.D2C._compute_descriptor_parallel(obs)
        self.assertIsInstance(res, tuple)
        self.assertIsInstance(res[0], list)
        self.assertIsInstance(res[1], list)

    #  test that the function returns a list, and that if the list is non-empty, its elements are tuples.
    def test_generate_edge_pairs(self):
        edge_pairs = self.D2C._generate_edge_pairs("is.child")
        self.assertIsInstance(edge_pairs, list)
        if len(edge_pairs) > 0:
            self.assertIsInstance(edge_pairs[0], tuple)

    # create a random pandas DataFrame and test that the function returns a dictionary with the correct keys.
    def test_compute_descriptor(self):
        D = pd.DataFrame(np.random.rand(100, 2), columns=list('ab'))
        descriptors = self.D2C._compute_descriptor(D, 0, 1)
        self.assertIsInstance(descriptors, dict)
        self.assertIn('correlation', descriptors)
        self.assertIn('skewness_ca', descriptors)
        self.assertIn('skewness_ef', descriptors)
        self.assertIn('mean_diff', descriptors)

    # test that the function returns a pandas DataFrame.
    def test_get_df(self):
        df = self.D2C.get_decsriptors_df()
        self.assertIsInstance(df, pd.DataFrame)

    # mock the RandomForestClassifier object and ensure that the function returns a score between 0 and 1.
    def test_get_score(self):
        with patch('sklearn.ensemble.RandomForestClassifier') as mock_rfc:
            mock_rfc.return_value.score.return_value = 0.9
            score = self.D2C.get_score()
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1)

    #  test that the function raises a ValueError when an invalid metric is provided.
    def test_get_score_invalid_metric(self):
        with self.assertRaises(ValueError):
            self.D2C.get_score(metric="invalid_metric")

if __name__ == '__main__':
    unittest.main()
