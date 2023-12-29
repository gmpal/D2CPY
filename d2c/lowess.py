import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import time
class LOWESS(BaseEstimator, RegressorMixin):
    def __init__(self, tau):
        self.tau = tau
        self.X_ = None
        self.y_ = None
        self.theta_ = None

    def wm(self, point, X):
        # Calculate the squared differences in a vectorized way
        # point is reshaped to (1, -1) for broadcasting to match the shape of X
        differences = X - point.reshape(1, -1)
        squared_distances = np.sum(differences ** 2, axis=1)

        # Calculate the weights
        tau_squared = -2 * self.tau * self.tau
        weights = np.exp(squared_distances / tau_squared)

        # Create a diagonal matrix from the weights
        weight_matrix = np.diag(weights)

        return weight_matrix


    def fit(self, X, y):
        # Fit the model to the data
        self.X_ = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0],1), axis=1)
        self.y_ = np.array(y).reshape(-1, 1)
        return self

    def predict(self, X):
        # Predict using the fitted model

        #allocate array of size X.shape[0]
        preds = np.empty(X.shape[0])
        X_ = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0],1), axis=1)

        start = time.time()

        for i in range(X.shape[0]):
            point_ = X_[i] 
            w = self.wm(point_, self.X_)
            self.theta_ =  np.linalg.pinv(self.X_.T@(w @ self.X_))@self.X_.T@(w @ self.y_)
            pred = np.dot(point_, self.theta_)
            preds[i] = pred

        return preds.reshape(-1, 1)

import numpy as np
from sklearn.metrics import r2_score


def main():

    # Generating synthetic nonlinear data
    np.random.seed(0)  # For reproducibility
    X = np.linspace(0, 3, 100)
    y = np.sin(4 * X) + np.random.normal(0, 0.2, X.shape[0])  # Sinusoidal pattern with noise

    # Reshape X for compatibility with our LOWESS model
    X = X.reshape(-1, 1)

    # Initialize LOWESS model
    tau = 0.1  # Bandwidth for the LOWESS model, adjust as needed
    model = LOWESS(tau)

    # Fit the model
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X).flatten()

    # Calculate R² score
    r2 = r2_score(y, y_pred)
    print("R² score:", r2)


if __name__ == "__main__":
    main()