import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def train_dbscan_model(file_path, train_split, eps, min_samples):
    data = pd.read_csv(file_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=38)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(x_train)
    y_predict = dbscan.fit_predict(x_test)

    # Check if there are at least two clusters
    if len(set(y_predict)) > 1:
        score = silhouette_score(x_test, y_predict)
        print(f"Silhouette Score: {score}")
    else:
        print("Silhouette Score: Not applicable (only one cluster)")

    ari = adjusted_rand_score(y_test, y_predict)
    nmi = normalized_mutual_info_score(y_test, y_predict)
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")


def dbscan_model(dataset="raw", normalization="raw", train_split=0.7, eps=0.5, min_samples=5):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw Datasets/train.csv"
    }

    if dataset in file_mapping:
        train_dbscan_model(file_mapping[dataset], train_split, eps, min_samples)
    else:
        raise Exception("No such dataset!")


def run_dbscan_combinations(datasets, eps_list, min_samples_list, normalizations=None):
    for dataset in datasets:
        for eps in eps_list:
            for min_samples in min_samples_list:
                if normalizations:
                    for normalization in normalizations:
                        print(f"Running {dataset} with {normalization}, eps={eps}, min_samples={min_samples}")
                        dbscan_model(dataset=dataset, normalization=normalization, eps=eps, min_samples=min_samples)
                        print("---")
                else:
                    print(f"Running {dataset} with eps={eps}, min_samples={min_samples}")
                    dbscan_model(dataset=dataset, eps=eps, min_samples=min_samples)
                    print("---")


if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal", "z_score", "min_max"]
    eps_list = [0.3, 0.5, 0.7]
    min_samples_list = [3, 5, 7]

    # Run combinations for datasets with normalization
    run_dbscan_combinations(datasets, eps_list, min_samples_list, normalizations)

    run_dbscan_combinations(dataset_raw, eps_list, min_samples_list)