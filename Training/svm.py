from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#implementing SVM
def train_svm_model(file_path, train_split, kernel, degree):
    data = pd.read_csv(file_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = svm.SVC(kernel=kernel, degree=degree if degree != 0 else 3)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")

#using 7 datsets, 3 feature engineereed, 3 original with normalization and 1 raw with no normalization
def svm_model(dataset="raw", normalization="raw", train_split=0.8, kernel="rbf", degree=0):
    file_mapping = {
        "feature_engineered": f"../Normalized_Datasets/Train/train_{normalization}_scaled.csv",
        "original": f"../Normalized_Datasets/Train/raw_{normalization}_scaled.csv",
        "original_no_normalization": f"../Raw Datasets/train.csv"
    }

    if dataset in file_mapping:
        train_svm_model(file_mapping[dataset], train_split, kernel, degree)
    else:
        raise Exception("No such dataset!")

#a loop to try every combination with datasets, degrees, kernels in order to have a better chance of getting
#the best accuracy
def run_svm_combinations(datasets, kernels, degrees, normalizations=None):
    for dataset in datasets:
        for kernel in kernels:
            for degree in degrees:
                if normalizations:
                    for normalization in normalizations:
                        print(f"Running {dataset} with {normalization}, kernel={kernel}, degree={degree}")
                        svm_model(dataset=dataset, normalization=normalization, kernel=kernel, degree=degree)
                        print("---")
                else:
                    print(f"Running {dataset} with kernel={kernel}, degree={degree}")
                    svm_model(dataset=dataset, kernel=kernel, degree=degree)
                    print("---")


if __name__ == "__main__":
    datasets = ["original", "feature_engineered"]
    dataset_raw = ["original_no_normalization"]
    normalizations = ["decimal", "z_score", "min_max"]
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    degrees = [0, 2, 3, 4]  # Degree 0 implies default handling for non-poly kernels

    # Running combinations for datasets
    run_svm_combinations(datasets, kernels, degrees, normalizations)

    run_svm_combinations(dataset_raw, kernels, degrees)

    ''' Best Combination:
        Dataset: original
        Normalization: z_score
        Kernel: linear
        Degree: 0
        Accuracy: 0.9725'''
