import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os


class KNNModel:
    def __init__(self, normalized_dataset_path, raw_dataset_path, n_neighbors=5, metric='euclidean', weights='uniform'):
        self.normalized_dataset_path = normalized_dataset_path
        self.raw_dataset_path = raw_dataset_path
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.model = None
        self.X = None
        self.y = None

    def load_data(self, dataset_type):
        if dataset_type == 'minmax':
            dataset = 'train_min_max_scaled.csv'
            data_path = os.path.join(self.normalized_dataset_path, 'Train', dataset)
        elif dataset_type == 'zscore':
            dataset = 'train_z_score_scaled.csv'
            data_path = os.path.join(self.normalized_dataset_path, 'Train', dataset)
        elif dataset_type == 'decimal':
            dataset = 'train_decimal_scaled.csv'
            data_path = os.path.join(self.normalized_dataset_path, 'Train', dataset)
        elif dataset_type == 'raw':
            dataset = 'train.csv'
            data_path = os.path.join(self.raw_dataset_path, dataset)
        else:
            raise ValueError("Invalid dataset type")

        data = pd.read_csv(data_path)
        self.X = data.iloc[:, :-1]
        self.y = data.iloc[:, -1]

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric, weights=self.weights)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def run_pipeline(self):
        for dataset_type in ['minmax', 'decimal', 'zscore', 'raw']:
            for n_neighbors in [3, 5, 7, 9, 11, 15, 21]:
                for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
                    for weights in ['uniform', 'distance']:
                        self.n_neighbors = n_neighbors
                        self.metric = metric
                        self.weights = weights
                        print(
                            f"Running KNN with {dataset_type}, n_neighbors={n_neighbors}, metric={metric}, weights={weights}")
                        self.load_data(dataset_type)
                        accuracy = self.train_model()
                        print(f"Accuracy: {accuracy * 100:.2f}%")
                        print('---')


if __name__ == "__main__":
    normalized_dataset_path = '../Normalized_Datasets'
    raw_dataset_path = '../Raw Datasets'

    knn_model = KNNModel(normalized_dataset_path, raw_dataset_path)
    knn_model.run_pipeline()
