import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(f'../Normalized_Datasets/Train/raw_z_score_scaled.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- MLP Training ----
mlp_model = MLPClassifier(
        hidden_layer_sizes=[64],
        max_iter=50,
        batch_size=16,
        solver="adam",
        activation="logistic",
        learning_rate_init=0.001,
        random_state=42
)

mlp_model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred)
print(f"MLP Model Accuracy: {mlp_accuracy}, using: solver-> {mlp_model.solver}, activation-> {mlp_model.activation}")