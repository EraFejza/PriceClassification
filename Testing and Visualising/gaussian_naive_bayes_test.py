import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib


matplotlib.use('TkAgg')

# Function to test with the Gaussian Naive Bayes model and visualize the results
def test_with_gaussian_nb_model(train_file, test_file, output_file):
    """
    Trains, tests, and visualizes the Gaussian Naive Bayes model.
    Includes visualizations for correlation and pairplots.

    :param train_file: Path to the train dataset file (with price_range column).
    :param test_file: Path to the test dataset file (without price_range column).
    :param output_file: Path to save the test dataset with predictions.
    """
    # Load train and test datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Split features and target for train data
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # Drop the 'id' column from test data if it exists
    if 'id' in test_data.columns:
        test_data = test_data.drop(columns=['id'])

    # Features for test data (no target column in test set)
    x_test = test_data

    # Train Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(x_train, y_train)

    # Predict on test dataset
    predictions = model.predict(x_test)

    # Add predictions to the test dataset (new column)
    test_data['price_range'] = predictions

    # Save results with predictions
    test_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Visualization 1: Correlation heatmap for train data
    plt.figure(figsize=(12, 8))
    train_corr = train_data.corr()
    sns.heatmap(train_corr, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap - Training Data")
    plt.show()

    # Visualization 2: Correlation heatmap for test data (with predictions)
    plt.figure(figsize=(12, 8))
    test_corr = test_data.corr()
    sns.heatmap(test_corr, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap - Test Data with Predictions")
    plt.show()

    # Visualization 3: Pair plot for selected attributes (sc_h, sc_w) vs price_range in train data
    sns.pairplot(train_data[['sc_h', 'sc_w', 'price_range']], hue="price_range", palette="coolwarm")
    plt.suptitle("Pair plot - Training Data (sc_h, sc_w vs price_range)", y=1.02)
    plt.show()

    # Visualization 4: Pairplot for selected attributes (sc_h, sc_w) vs price_range in test data (with predictions)
    sns.pairplot(test_data[['sc_h', 'sc_w', 'price_range']], hue="price_range", palette="coolwarm")
    plt.suptitle("Pair plot - Test Data with Predictions (sc_h, sc_w vs price_range)", y=1.02)
    plt.show()

    # Visualization 5: Pair plot for selected attribute (ram) vs price_range in train data
    sns.pairplot(train_data[['ram', 'price_range']], hue="price_range", palette="coolwarm")
    plt.suptitle("Pair plot - Training Data (ram vs price_range)", y=1.02)
    plt.show()

    # Visualization 6: Pair plot for selected attributes (ram) vs price_range in test data (with predictions)
    sns.pairplot(test_data[['ram', 'price_range']], hue="price_range", palette="coolwarm")
    plt.suptitle("Pair plot - Test Data with Predictions (ram vs price_range)", y=1.02)
    plt.show()

if __name__ == "__main__":
    # Paths to datasets
    train_file = "../Normalized_Datasets/Train/raw_z_score_scaled.csv"
    test_file = "../Normalized_Datasets/Test/raw_z_score_scaled.csv"
    output_file = "../Raw Datasets/test_with_predictions_gaussian_nb.csv"

    # Test with the Gaussian Naive Bayes model and visualize
    test_with_gaussian_nb_model(train_file, test_file, output_file)