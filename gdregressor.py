#this is for part 2

import numpy as np

class Gdregressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
       
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        # Standardize X
        self.X_mean = np.mean(X_train, axis=0)
        self.X_std = np.std(X_train, axis=0)
        X_train = (X_train - self.X_mean) / (self.X_std + 1e-8)

        # Standardize y
        self.y_mean = np.mean(y_train)
        self.y_std = np.std(y_train)
        y_train = (y_train - self.y_mean) / (self.y_std + 1e-8)

        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            y_hat = np.dot(X_train, self.coef_) + self.intercept_

            intercept_der = -2 * np.mean(y_train - y_hat)
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]

            self.intercept_ -= self.lr * intercept_der
            self.coef_ -= self.lr * coef_der

        # Rescale coefficients to original scale
        self.coef_ = self.coef_ * (self.y_std / (self.X_std + 1e-8))
        self.intercept_ = self.y_mean - np.dot(self.X_mean, self.coef_)

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        return np.dot(X_test, self.coef_) + self.intercept_


#this is for part 1
class Gdregressor:
    def mean(values):
        return sum(values) / len(values)

    def std(values):
        mu = Gdregressor.mean(values)
        variance = sum((x - mu) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        # Convert data to list of floats
        X_train = [[float(x) for x in row] for row in X_train]
        y_train = [float(y) for y in y_train]

        def transpose(matrix):
            return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

        X_train_T = transpose(X_train)

        # Compute mean and std
        self.X_mean = [Gdregressor.mean(col) for col in X_train_T]
        self.X_std = [Gdregressor.std(col) for col in X_train_T]

        # Standardize training data
        X_train_scaled = []
        for row in X_train:
            scaled_row = []
            for i in range(len(row)):
                scaled_value = (row[i] - self.X_mean[i]) / (self.X_std[i] + 1e-8)
                scaled_row.append(scaled_value)
            X_train_scaled.append(scaled_row)

        X_train = X_train_scaled
        self.intercept_ = 0
        n_features = len(X_train[0])
        n_samples = len(X_train)
        self.coef_ = [1.0] * n_features

        for epoch in range(self.epochs):
            y_hat = []
            for row in X_train:
                dot = sum(row[i] * self.coef_[i] for i in range(n_features))
                y_hat.append(dot + self.intercept_)

            # Calculate residuals
            residuals = [y_train[i] - y_hat[i] for i in range(n_samples)]

            # Gradients
            intercept_der = -2 * sum(residuals) / n_samples

            coef_der = []
            for j in range(n_features):
                grad = -2 * sum(residuals[i] * X_train[i][j] for i in range(n_samples)) / n_samples
                coef_der.append(grad)

            # Update
            self.intercept_ -= self.lr * intercept_der
            for j in range(n_features):
                self.coef_[j] -= self.lr * coef_der[j]

    def predict(self, X_test):
        X_test = [[float(x) for x in row] for row in X_test]

        # Standardize using training mean/std
        X_test_scaled = []
        for row in X_test:
            scaled_row = []
            for i in range(len(row)):
                scaled_value = (row[i] - self.X_mean[i]) / (self.X_std[i] + 1e-8)
                scaled_row.append(scaled_value)
            X_test_scaled.append(scaled_row)

        predictions = []
        for row in X_test_scaled:
            dot = sum(row[i] * self.coef_[i] for i in range(len(row)))
            predictions.append(dot + self.intercept_)

        return predictions


#this is for part3
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
