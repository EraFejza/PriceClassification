import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hybrid model combining Random Forest and MLP
def hybrid_model(file_path, n_estimators, criterion, epochs, batch_size, learning_rate):
    # Load the dataset
    data = pd.read_csv(file_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Random Forest for feature selection
    rf_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=42)
    rf_model.fit(x_train, y_train)

    # MLP model with the best configuration
    mlp_model = MLPClassifier(
        hidden_layer_sizes=[64],
        max_iter=epochs,
        batch_size=batch_size,
        solver="adam",
        activation="logistic",
        learning_rate_init=learning_rate,
        random_state=42
    )
    mlp_model.fit(x_train, y_train)

    # Final hybrid prediction (simple averaging of probabilities)
    rf_probs = rf_model.predict_proba(x_test)
    mlp_probs = mlp_model.predict_proba(x_test)
    hybrid_probs = (rf_probs + mlp_probs) / 2
    hybrid_predictions = hybrid_probs.argmax(axis=1)

    # here im calculating and printing accuracy
    hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
    print(f"Hybrid Model Accuracy with criterion={criterion}: {hybrid_accuracy}")


def run_hybrid_combinations(datasets, normalizations, n_estimators_list, criteria, epochs, batch_size, learning_rate):
    file_mapping = {
        "feature_engineered": "../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": "../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": "../Raw Datasets/train.csv"
    }

    for dataset in datasets:
        for criterion in criteria:
            for n_estimators in n_estimators_list:
                if dataset != "original_no_normalization":
                    for normalization in normalizations:
                        file_path = file_mapping[dataset].format(normalization=normalization)
                        print(f"Running hybrid model with dataset={dataset}, normalization={normalization}, "
                              f"n_estimators={n_estimators}, criterion={criterion}")
                        hybrid_model(file_path, n_estimators, criterion, epochs, batch_size, learning_rate)
                        print("---")
                else:
                    file_path = file_mapping[dataset]
                    print(f"Running hybrid model with dataset={dataset}, n_estimators={n_estimators}, criterion={criterion}")
                    hybrid_model(file_path, n_estimators, criterion, epochs, batch_size, learning_rate)
                    print("---")


if __name__ == "__main__":
    datasets = ["feature_engineered", "original"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["min_max", "z_score", "decimal"]
    n_estimators_list = [50, 100]
    learning_rate = 0.001
    epochs = 50
    batch_size = 16  # Batch size for MLP
    criteria = ["gini", "entropy"]

    # Run combinations for datasets with normalization
    run_hybrid_combinations(datasets, normalizations, n_estimators_list, criteria, epochs, batch_size, learning_rate)

    # Run combinations for raw dataset without normalization
    run_hybrid_combinations(dataset_raw, None, n_estimators_list, criteria, epochs, batch_size, learning_rate)

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
