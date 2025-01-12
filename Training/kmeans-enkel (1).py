
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score

def train_kmeans_model(file_path, train_split, n_clusters):
    data = pd.read_csv(file_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=38)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x_train)
    y_predict = kmeans.predict(x_test)
    score = silhouette_score(x_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Silhouette Score: {score}")
    print(f"Accuracy: {accuracy}")

def kmeans_model(dataset="raw", normalization="raw", train_split=0.7, n_clusters=3):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw Datasets/train.csv"
    }

    if dataset in file_mapping:
        train_kmeans_model(file_mapping[dataset], train_split, n_clusters)
    else:
        raise Exception("No such dataset!")

def run_kmeans_combinations(datasets, n_clusters_list, normalizations=None):
    for dataset in datasets:
        for n_clusters in n_clusters_list:
            if normalizations:
                for normalization in normalizations:
                    print(f"Running {dataset} with {normalization}, n_clusters={n_clusters}")
                    kmeans_model(dataset=dataset, normalization=normalization, n_clusters=n_clusters)
                    print("---")
            else:
                print(f"Running {dataset} with n_clusters={n_clusters}")
                kmeans_model(dataset=dataset, n_clusters=n_clusters)
                print("---")

if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal", "z_score", "min_max"]
    n_clusters_list = [2, 3, 4, 5]

    # Run combinations for datasets with normalization
    run_kmeans_combinations(datasets, n_clusters_list, normalizations)

    run_kmeans_combinations(dataset_raw, n_clusters_list)
