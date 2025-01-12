import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Hybrid model combining Random Forest and MLP
def hybrid_model(dataset, normalization, n_estimators, criterion, epochs, batch_size, learning_rate):
    # File paths for datasets
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw Datasets/train.csv"
    }

    if dataset not in file_mapping:
        raise Exception("No such dataset!")

    # Load the dataset
    data = pd.read_csv(file_mapping[dataset])
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Random Forest for feature selection
    rf_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=42)
    rf_model.fit(x_train, y_train)

    # MLP model with the best configuration (tested out separately to avoid excessive number of combinations)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=[64],             # hidden layer configuration
        max_iter=epochs,                     # number of epochs
        batch_size=batch_size,               # batch size
        solver="adam",                       # optimizer (Adam)
        activation="logistic",               # activation function (logistic)
        learning_rate_init=learning_rate,    # learning rate
        random_state=42                      # Random state for reproducibility
    )
    mlp_model.fit(x_train, y_train)

    # Final hybrid prediction (simple averaging of probabilities)
    rf_probs = rf_model.predict_proba(x_test)
    mlp_probs = mlp_model.predict_proba(x_test)
    hybrid_probs = (rf_probs + mlp_probs) / 2
    hybrid_predictions = hybrid_probs.argmax(axis=1)

    # Calculate and print accuracy
    hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
    print(f"Hybrid Model Accuracy with criterion={criterion}: {hybrid_accuracy}")


# Main function
if __name__ == "__main__":
    datasets = ["feature_engineered", "original"]
    normalizations = ["min_max", "z_score", "decimal"]
    n_estimators_list = [50, 100]
    learning_rate = 0.001
    epochs = 50
    batch_size = 16  # Batch size for MLP
    criteria = ["gini", "entropy"]

    # Iterate through all combinations for Random Forest and with fixed best MLP model
    for dataset in datasets:
        for normalization in normalizations:
            for n_estimators in n_estimators_list:
                for criterion in criteria:
                    print(f"Running hybrid model with dataset={dataset}, normalization={normalization}, "
                          f"n_estimators={n_estimators}, criterion={criterion}")
                    hybrid_model(
                        dataset=dataset,
                        normalization=normalization,
                        n_estimators=n_estimators,
                        criterion=criterion,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size=batch_size
                    )
                    print("---")

            '''
            Best combination: 
            
            - multi layer perceptron set up:
              Original Dataset with z-score normalization
              1 layer of size 64
              50 epochs
              batch size 16
              Adam solver with learning rate 0.001
              logistic activation function
              
            - random forest set up:
              Original Dataset with z-score normalization
              Entropy criterion
              100 estimators
              
            Accuracy achieved -> 0.9825 = 98.25%
            '''