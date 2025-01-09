from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Data_Manipulation.load_preprocessing import data_loader


def apply_svm_feature_engineered_dataset(normalization_type, train_split, kernel, degree):
    x = data_loader(f"C:/Users/erafe/OneDrive/Desktop/Phone-price-classification/Datasets/feature_engineered_{normalization_type}_scaled.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = svm.SVC(kernel=kernel, degree=degree if degree != 0 else 3)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_svm_normal_dataset(normalization_type, train_split, kernel, degree):
    x = data_loader(f"C:/Users/erafe/OneDrive/Desktop/Phone-price-classification/Datasets/original_{normalization_type}_scaled.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = svm.SVC(kernel=kernel, degree=degree if degree != 0 else 3)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_svm_raw_dataset(train_split, kernel, degree):
    x = data_loader("C:/Users/erafe/OneDrive/Desktop/Phone-price-classification/Datasets/raw.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = svm.SVC(kernel=kernel, degree=degree if degree != 0 else 3)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def svm_model(dataset="raw", normalization="raw", train_split=0.8, kernel="rbf", degree=0):
    if dataset == "feature_engineered":
        apply_svm_feature_engineered_dataset(normalization, train_split, kernel, degree)
    elif dataset == "normal":
        apply_svm_normal_dataset(normalization, train_split, kernel, degree)
    elif dataset == "raw":
        apply_svm_raw_dataset(train_split, kernel, degree)
    else:
        raise Exception("No such dataset!")


if __name__ == "__main__":
    # Automate combinations of parameters
    datasets = ["normal","feature_engineered"]
    normalizations = ["decimal","z_score", "min_max"]
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    degrees = [0, 2, 3, 4]  # Degree 0 implies default handling for non-poly kernels


    for dataset in datasets:
        for normalization in normalizations:
            for kernel in kernels:
                for degree in degrees:
                    print(f"Running {dataset} with {normalization}, kernel={kernel}, degree={degree}")
                    svm_model(dataset=dataset, normalization=normalization, kernel=kernel, degree=degree)
                    print("---")

    for kernel in kernels:
        for degree in degrees:
            print(f"Running raw with kernel={kernel}, degree={degree}")
            svm_model(dataset='raw', kernel=kernel, degree=degree)
            print("---")