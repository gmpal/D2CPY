import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from src.d2c.estimators import MutualInformationEstimator
from src.d2c.lowess import LOWESS  

@pytest.fixture
def default_mi_estimator():
    return MutualInformationEstimator()

@pytest.fixture
def custom_mi_estimator():
    proxy_params = {'alpha': 0.1}
    return MutualInformationEstimator(proxy='Ridge', proxy_params=proxy_params)

def test_initialization_default(default_mi_estimator):
    assert default_mi_estimator.proxy == 'Ridge'
    assert default_mi_estimator.proxy_params == {}

def test_initialization_custom(custom_mi_estimator):
    assert custom_mi_estimator.proxy == 'Ridge'
    assert custom_mi_estimator.proxy_params == {'alpha': 0.1}

def test_get_regression_model_ridge(default_mi_estimator):
    model = default_mi_estimator.get_regression_model()
    assert isinstance(model, Ridge)

def test_get_regression_model_lowess():
    mi_estimator = MutualInformationEstimator(proxy='LOWESS')
    model = mi_estimator.get_regression_model()
    assert isinstance(model, LOWESS)

def test_mse_method(default_mi_estimator):
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Y = np.array([10, 20, 30])
    cv = 3
    mse = default_mi_estimator.mse(X, Y, cv)
    model = Ridge().fit(X, Y)  # Assuming Ridge is the proxy model being used
    expected_mse = mean_squared_error(Y, model.predict(X))
    assert np.isclose(mse, expected_mse)

def test_estimate_method_x2_none(default_mi_estimator):
    y = pd.Series([1, 2, 3, 4, 5])
    x1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    mi = default_mi_estimator.estimate(y, x1)
    mse = default_mi_estimator.mse(x1, y, cv=5)  # Ensure this matches the `estimate` method's internal cv if set
    assert mi == 1 - mse / np.var(y)

def test_estimate_method_x2_not_none(default_mi_estimator):
    y = pd.Series([1, 2, 3, 4, 5])
    x1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    x2 = pd.DataFrame({'B': [6, 7, 8, 9, 10]})
    mi = default_mi_estimator.estimate(y, x1, x2)
    x1x2 = pd.concat([x1, x2], axis=1)
    mi_expected = 1 - default_mi_estimator.mse(x1x2, y, cv=5) / default_mi_estimator.mse(x2, y, cv=5)  # Adjust `cv` as needed
    assert np.isclose(mi, mi_expected)
