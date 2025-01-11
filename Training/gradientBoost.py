import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os

class GradientBoost:

    def __init__(self, normalized_dataset_path, raw_dataset_path, max_depth=None, n_estimators=100):
        self.normalized_dataset_path = normalized_dataset_path
        self.raw_dataset_path = raw_dataset_path
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = None
        self.X_scaled = None
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

        if dataset_type == 'minmax':
            scaler = MinMaxScaler()
            self.X_scaled = scaler.fit_transform(self.X)
        elif dataset_type == 'zscore':
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.X)
        else:
            self.X_scaled = self.X

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=0)
        self.model = GradientBoostingClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def run_pipeline(self):
        for dataset_type in ['minmax', 'decimal', 'zscore', 'raw']:
            for n_estimators in [50, 100, 200]:
                self.n_estimators = n_estimators
                print(f"Running GradientBoosting with {dataset_type}, {n_estimators} estimators")
                self.load_data(dataset_type)
                accuracy = self.train_model()
                print(f"Accuracy: {accuracy * 100:.2f}%")
                print('---')

if __name__ == "__main__":
    normalized_dataset_path = '../Normalized_Datasets'
    raw_dataset_path = '../Raw Datasets'
    gradient_boosting_model = GradientBoost(normalized_dataset_path, raw_dataset_path, max_depth=7)
    gradient_boosting_model.run_pipeline()
