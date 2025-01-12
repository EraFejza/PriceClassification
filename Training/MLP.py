import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

datasets = [
    '../Normalized_Datasets/Train/raw_z_score_scaled.csv',
    '../Normalized_Datasets/Train/raw_min_max_scaled.csv',
    '../Normalized_Datasets/Train/raw_decimal_scaled.csv',
    '../Normalized_Datasets/Train/train_z_score_scaled.csv',
    '../Normalized_Datasets/Train/train_min_max_scaled.csv',
    '../Normalized_Datasets/Train/train_decimal_scaled.csv',
]

param_grid = {
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.0005],
    'activation': ['identity', 'logistic', 'relu', 'tanh']
}

for dataset_path in datasets:

    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    mlp_model = MLPClassifier(hidden_layer_sizes=[64], max_iter=50, batch_size=16, random_state=42)

    grid_search = GridSearchCV(estimator=mlp_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_mlp_model = grid_search.best_estimator_

    y_pred = best_mlp_model.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, y_pred)
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Best MLP Model Accuracy: {mlp_accuracy}, using: solver-> {best_mlp_model.solver}, activation-> {best_mlp_model.activation}, learning_rate_init-> {best_mlp_model.learning_rate_init}")
    print("-----")

    #The best comnbination is Original dataset with Z-score normalization, Adam Optimizer, lerning rate 0.001, activation Logistic
