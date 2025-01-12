import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings("ignore", message=".force_all_finite.")


class LightGBMModel:
    def __init__(self, normalized_dataset_path, raw_dataset_path, max_depth=10, learning_rate=0.1, n_estimators=100):
        self.normalized_dataset_path = normalized_dataset_path
        self.raw_dataset_path = raw_dataset_path
        self.max_depth = max_depth
        self.learning_rate = learning_rate
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

    def apply_scaling(self, dataset_type):
        if dataset_type == 'minmax':
            scaler = MinMaxScaler()
            self.X_scaled = scaler.fit_transform(self.X)
        elif dataset_type == 'zscore':
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.X)
        else:
            self.X_scaled = self.X  # No scaling for 'decimal' and 'raw'

    def balance_classes(self):
        smote = SMOTE(random_state=42)
        self.X_scaled, self.y = smote.fit_resample(self.X_scaled, self.y)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)

        # Initialize LightGBM
        self.model = LGBMClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            num_leaves=2 ** self.max_depth - 1,  # Ensure sufficient splits
            min_data_in_leaf=10,  # Avoid empty leaves
            verbose=-1  # Suppress LightGBM internal logs
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def run_pipeline(self):
        best_accuracy = 0
        best_config = {}
        for dataset_type in ['minmax', 'zscore', 'decimal', 'raw']:
            self.load_data(dataset_type)
            self.apply_scaling(dataset_type)
            self.balance_classes()
            for n_estimators in [100, 200]:
                for learning_rate in [0.05, 0.1]:
                    for max_depth in [5, 10]:
                        self.n_estimators = n_estimators
                        self.learning_rate = learning_rate
                        self.max_depth = max_depth
                        print(
                            f"Running LightGBM with {dataset_type}, n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
                        accuracy = self.train_model()
                        print(f"Accuracy: {accuracy * 100:.2f}%")
                        print("---")
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_config = {
                                "dataset": dataset_type,
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth
                            }

        print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
        print("Best Configuration:", best_config)


if __name__ == "__main__":
    normalized_dataset_path = '../Normalized_Datasets'
    raw_dataset_path = '../Raw Datasets'
    lightgbm_model = LightGBMModel(normalized_dataset_path, raw_dataset_path)
    lightgbm_model.run_pipeline()
