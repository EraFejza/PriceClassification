import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Initialize parameters
np.random.seed(0)  # For reproducibility
momentum = 0.9
learning_rate_start = 0.001
learning_rate_end = 0.00005
n_epochs = 100
batch_size = 32

# Load CSV file (replace 'training_data.csv' with your actual CSV filename)
data = pd.read_csv(f'../Normalized_Datasets/Train/raw_z_score_scaled.csv')

# Split data into features and target
X_train = data.iloc[:, :-1].values  # All columns except the last (target)
y_train = data.iloc[:, -1].values  # Only the last column (target)

# Initialize weights and bias
n_features = X_train.shape[1]  # Number of features
weights = np.random.rand(n_features, 1)
bias = np.random.rand(1)

# Initialize momentum terms
velocity_w = np.zeros_like(weights)
velocity_b = np.zeros_like(bias)


# Mock gradient calculation function (replace with your actual gradient calculation)
def compute_gradients(X_batch, y_batch, weights, bias):
    # Placeholder for gradient calculation logic based on X_batch and y_batch
    grad_w = np.random.rand(weights.shape[0], 1)  # Placeholder gradients
    grad_b = np.random.rand(1)  # Placeholder gradients
    return grad_w, grad_b


# Adaptive learning rate schedule
def adaptive_lr(epoch):
    # Adaptive learning rate decays from 0.001 to 0.00005 over epochs
    lr = learning_rate_start - (epoch * (learning_rate_start - learning_rate_end) / n_epochs)
    return lr


# Training loop
for epoch in range(n_epochs):
    # Shuffle data and create mini-batches
    permutation = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[permutation]
    y_shuffled = y_train[permutation]

    for batch_start in range(0, X_train.shape[0], batch_size):
        X_batch = X_shuffled[batch_start:batch_start + batch_size]
        y_batch = y_shuffled[batch_start:batch_start + batch_size]

        # Compute gradients
        grad_w, grad_b = compute_gradients(X_batch, y_batch, weights, bias)

        # Update weights with momentum and adaptive learning rate
        velocity_w = momentum * velocity_w - adaptive_lr(epoch) * grad_w
        velocity_b = momentum * velocity_b - adaptive_lr(epoch) * grad_b
        weights += velocity_w
        bias += velocity_b

    print(f'Epoch {epoch + 1}/{n_epochs}, Learning Rate: {adaptive_lr(epoch)}')

# After training, weights and bias will be updated
print("Final weights:", weights)
print("Final bias:", bias)

# Load Test Data
test_data = pd.read_csv(f'../Normalized_Datasets/Test/raw_z_score_scaled.csv')
X_test = test_data.iloc[:, :-1].values  # All columns except the last (target)
y_test = test_data.iloc[:, -1].values  # Only the last column (target)

# Make predictions on test data
predictions = np.dot(X_test, weights) + bias  # Linear combination
predictions = (predictions >= 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
