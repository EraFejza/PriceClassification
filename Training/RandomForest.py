import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_random_forest(file_path, train_split, n_estimators, criterion, max_depth):
    data = pd.read_csv(file_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_split, random_state=123)
    classifier_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
    classifier_model.fit(x_train, y_train)

    y_predicted = classifier_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print(accuracy)
    return accuracy


def random_forest_model(dataset="raw", normalization="raw", train_split=0.8, n_estimators=100, criterion="gini", max_depth=None):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw Datasets/train.csv"
    }

    if dataset in file_mapping:
        file_path = file_mapping[dataset]
        return train_random_forest(file_path, train_split, n_estimators, criterion, max_depth)
    else:
        raise Exception("Dataset doesn't exist")


def run_random_forest_combinations(datasets, n_estimators_list, criteria, max_depths, normalizations=None):
    for dataset in datasets:
        for criterion in criteria:
            for max_depth in max_depths:
                if normalizations:
                    for normalization in normalizations:
                        for n_estimators in n_estimators_list:
                            print(f"Running with dataset {dataset} with normalization={normalization}, criterion={criterion}, max_depth={max_depth}, n_estimators={n_estimators}")
                            random_forest_model(dataset=dataset, normalization=normalization, criterion=criterion,
                                                max_depth=max_depth, n_estimators=n_estimators)
                            print("---")
                else:
                    for n_estimators in n_estimators_list:
                        print(f"Running with dataset {dataset} with criterion={criterion}, max_depth={max_depth}, n_estimators={n_estimators}")
                        random_forest_model(dataset=dataset, criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
                        print("---")


if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal", "z_score", "min_max"]
    criteria = ["gini", "entropy"]
    max_depths = [10, 15, 20, 25]
    n_estimators_list = [100, 200]

    run_random_forest_combinations(datasets, n_estimators_list, criteria, max_depths, normalizations)
    run_random_forest_combinations(dataset_raw, n_estimators_list, criteria, max_depths)
