import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('/content/train_z_score_scaled.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Manual Backpropagation Implementation ----
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return z > 0

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2  # Linear output for regression
        return self.A2

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true):
        m = X.shape[0]  # Number of samples
        y_pred = self.A2

        # Gradients for output layer
        dZ2 = 2 * (y_pred - y_true.reshape(-1, 1)) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Gradients for hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

bp_model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=32, output_dim=1, learning_rate=0.01)
bp_model.train(X_train, y_train, epochs=100)

# ---- MLP Training Using TensorFlow ----

mlp_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # First hidden layer
    Dense(32, activation='relu'),                              # Second hidden layer
    Dense(16, activation='relu'),                              # Third hidden layer
    Dense(1, activation='linear')                              # Output layer (for regression)
])

mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

eval_loss, eval_mae = mlp_model.evaluate(X_test, y_test, verbose=1)
print(f"MLP Test Loss: {eval_loss:.4f}, MLP Test MAE: {eval_mae:.4f}")
