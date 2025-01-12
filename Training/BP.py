import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=None, hidden_size=4, output_size=1, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_input_hidden = None
        self.weights_hidden_output = None

        self.bias_hidden = None
        self.bias_output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        self.input_size = X.shape[1]

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

        for epoch in range(self.epochs):
            output = self.feedforward(X)
            self.backward(X, y, self.learning_rate)

    def feedforward(self, X):
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)

        # Hidden to output
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def predict(self, X):
        predictions = self.feedforward(X)
        return (predictions > 0.5).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

dataset = pd.read_csv('../Normalized_Datasets/Train/raw_z_score_scaled.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values.reshape(-1, 1)  # Target

param_grid = {
    'hidden_size': [1],
    'learning_rate': [0.1],
    'epochs': [2000]
}

nn = NeuralNetwork(input_size=X.shape[1])

grid_search = GridSearchCV(nn, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X, y)

print("Best parameters found:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

best_nn = grid_search.best_estimator_
output = best_nn.predict(X)
accuracy = best_nn.score(X, y)
print("Predictions after training:")
print(output)
print(f"Accuracy: {accuracy:.2f}")
