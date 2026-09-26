# Libraries
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin


class NesterovRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.01, momentum=0.9, max_iter=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, y):
        # Vector đặc trưng
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        # Thêm bias
        X_bias = np.c_[np.ones(n_samples), X]

        self.weights_ = np.zeros(n_features + 1)
        velocity = np.zeros(n_features + 1)

        previous_loss = float("inf")

        for _ in range(self.max_iter):
            # Nesterov look-ahead
            lookahead_weights = self.weights_ + self.momentum * velocity

            # Prediction tại vị trí look-ahead
            predictions = X_bias @ lookahead_weights

            # Gradient
            error = predictions - y

            gradient = (
                X_bias.T @ error
            ) / n_samples

            # Update velocity
            velocity = self.momentum * velocity - self.learning_rate * gradient

            # Update weights
            self.weights_ += velocity

            # Loss
            loss = np.mean(error ** 2)

            if abs(previous_loss - loss) < self.tolerance:
                break

            previous_loss = loss

        return self

    def predict(self, X):
        # Predict Model
        X = np.asarray(X, dtype=float)
        X_bias = np.c_[np.ones(X.shape[0]), X]

        return X_bias @ self.weights_
    