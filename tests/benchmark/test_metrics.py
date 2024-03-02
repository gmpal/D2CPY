import pandas as pd
import unittest
from benchmark.metrics import card_precision_top_k_day

class TestMetrics(unittest.TestCase):
    def test_card_precision_top_k_day(self):
        # Create a sample DataFrame for testing
        df = pd.DataFrame({
            'graph_id': [1, 1, 2, 2, 3, 3],
            'predicted_proba': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            'is_causal': [1, 0, 1, 0, 1, 0]
        })

        # Define the expected precision top k value
        expected_precision_top_k = 2 / 3

        # Call the function to compute the precision top k
        precision_top_k = card_precision_top_k_day(df, top_k=3)

        # Assert that the computed precision top k matches the expected value
        self.assertEqual(precision_top_k, expected_precision_top_k)

if __name__ == '__main__':
    unittest.main()