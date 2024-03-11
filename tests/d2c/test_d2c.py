import pytest
import networkx as nx
import numpy as np

from src.d2c.d2c import get_markov_blanket, D2C

@pytest.fixture
def create_dag():
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    return dag

def test_node_with_no_parents_or_children(create_dag):
    node = 5
    expected_result = [4]
    assert get_markov_blanket(create_dag, node) == expected_result, f"Test case 1 failed. Expected: {expected_result}"

def test_node_with_parents_and_children(create_dag):
    node = 2
    expected_result = [1, 4]
    assert get_markov_blanket(create_dag, node) == expected_result, f"Test case 2 failed. Expected: {expected_result}"

def test_node_with_parents_of_children(create_dag):
    node = 4
    expected_result = [1, 2, 3, 5]
    assert get_markov_blanket(create_dag, node) == expected_result, f"Test case 3 failed. Expected: {expected_result}"

def test_node_with_no_parents_or_children_and_non_existent(create_dag):
    node = 6
    expected_result = []
    assert get_markov_blanket(create_dag, node) == expected_result, f"Test case 4 failed. Expected: {expected_result}"

def test_compute_descriptors_for_dag(create_dag):
    d2c = D2C()
    dag = create_dag()
    n_variables = 5
    dynamic = False
    maxlags = 1
    seed = 42
    num_samples = 20

    X, Y = d2c.compute_descriptors_with_dag(dag, n_variables, dynamic, maxlags, seed, num_samples)

    # Check if the number of samples is correct
    assert len(X) == num_samples * 2, f"Test case 5 failed. Expected: {num_samples * 2} samples, but got {len(X)} samples"

    # Check if the labels are correct
    assert len(Y) == num_samples * 2, f"Test case 5 failed. Expected: {num_samples * 2} labels, but got {len(Y)} labels"

    # Check if the selected rows are in the test couples
    for i in range(num_samples):
        parent, child = d2c.test_couples[i]
        assert (parent, child) in d2c.test_couples, f"Test case 5 failed. Selected row ({parent}, {child}) not in test couples"

    # Check if the non-existing edges are added correctly
    non_existing_edges = [(i, j) for i in range(n_variables) for j in range(n_variables) if i != j and (i, j) not in dag.edges]
    for i in range(num_samples):
        parent, child = non_existing_edges[i]
        assert (parent, child) in d2c.test_couples, f"Test case 5 failed. Non-existing edge ({parent}, {child}) not in test couples"

    print("All test cases passed!")