import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_decision_tree_model(file_path, train_split, criterion, max_depth):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_split)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def decision_tree_model(dataset="raw", normalization="raw", train_split=0.8, criterion="gini", max_depth=None):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw Datasets/train.csv"
    }

    if dataset in file_mapping:
        file_path = file_mapping[dataset]
        return train_decision_tree_model(file_path, train_split, criterion, max_depth)
    else:
        raise Exception("No such dataset!")


def run_decision_tree_combinations(datasets, criteria, max_depths, normalizations=None):
    for dataset in datasets:
        for criterion in criteria:
            for max_depth in max_depths:
                if normalizations:
                    for normalization in normalizations:
                        print(
                            f"Running {dataset} with normalization={normalization}, criterion={criterion}, max_depth={max_depth}")
                        decision_tree_model(dataset=dataset, normalization=normalization, criterion=criterion,
                                            max_depth=max_depth)
                        print("---")
                else:
                    print(f"Running {dataset} with criterion={criterion}, max_depth={max_depth}")
                    decision_tree_model(dataset=dataset, criterion=criterion, max_depth=max_depth)
                    print("---")


if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal", "z_score", "min_max"]
    criteria = ["gini", "entropy"]
    max_depths = [5, 15, 20, 25]

    run_decision_tree_combinations(datasets, criteria, max_depths, normalizations)

    run_decision_tree_combinations(dataset_raw, criteria, max_depths)

    '''
    Best combination: 

    Original dataset with z-score normalization
    criterion=entropy
    max_depth=15
    Highest accuracy achieved -> 0.8825 = 88.25%

    '''
