import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

def train_naive_bayes_model(file_path, train_split):
    data = pd.read_csv(file_path)
    x = data.iloc[:, 1:-1]
    y = pd.Series(data['price_range'])

    # Apply Min-Max scaling to ensure all values are non-negative
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_split, random_state=42)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")

    # Calculate and display the confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="coolwarm")
    plt.title("Confusion Matrix - Train Data")
    # plt.show()

def naive_bayes_model(dataset="raw", normalization="raw", train_split=0.5):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw Datasets/train.csv"
    }

    if dataset in file_mapping:
        train_naive_bayes_model(file_mapping[dataset], train_split)
    else:
        raise Exception("No such dataset!")

def run_naive_bayes_combinations(datasets, normalizations=None):
    for dataset in datasets:
        if normalizations:
            for normalization in normalizations:
                print(f"Running {dataset} with {normalization}")
                naive_bayes_model(dataset=dataset, normalization=normalization)
                print("---")
        else:
            print(f"Running {dataset}")
            naive_bayes_model(dataset=dataset)
            print("---")

if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal", "z_score", "min_max"]

    # Run combinations for datasets with normalization
    run_naive_bayes_combinations(datasets, normalizations)

    run_naive_bayes_combinations(dataset_raw)