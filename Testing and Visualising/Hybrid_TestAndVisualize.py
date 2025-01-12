import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Function to test with the hybrid model and visualize the results
def test_with_hybrid_model(model_rf, model_mlp, train_file, test_file, output_file):
    """
    Uses pre-trained models to predict the price_range for the test dataset,
    and visualizes the correlation and pairwise relationships for both train and test datasets.

    :param model_rf: Pre-trained RandomForest model.
    :param model_mlp: Pre-trained MLP model.
    :param train_file: Path to the train dataset file (with price_range column).
    :param test_file: Path to the test dataset file (without price_range column).
    :param output_file: Path to save the test dataset with predictions.
    """
    # Load train and test datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Predict probabilities with both models
    rf_probs = model_rf.predict_proba(test_data)
    mlp_probs = model_mlp.predict_proba(test_data)

    # Final hybrid prediction (average of probabilities)
    hybrid_probs = (rf_probs + mlp_probs) / 2
    hybrid_predictions = hybrid_probs.argmax(axis=1)

    # Add predictions to the test dataset
    test_data['price_range'] = hybrid_predictions

    # Save results
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

    # Visualization 3: Pairplot for selected attributes (sc_h, sc_w) vs price_range in train data
    sns.pairplot(train_data[['sc_h', 'sc_w', 'price_range']], hue="price_range", palette="coolwarm")
    plt.suptitle("Pairplot - Training Data (sc_h, sc_w vs price_range)", y=1.02)
    plt.show()

    # Visualization 4: Pairplot for selected attributes (sc_h, sc_w) vs price_range in test data (with predictions)
    sns.pairplot(test_data[['sc_h', 'sc_w', 'price_range']], hue="price_range", palette="coolwarm")
    plt.suptitle("Pairplot - Test Data with Predictions (sc_h, sc_w vs price_range)", y=1.02)
    plt.show()


if __name__ == "__main__":
    # Load train dataset
    train_file = "../Normalized_Datasets/Train/raw_z_score_scaled.csv"
    test_file = "../Normalized_Datasets/Test/raw_z_score_scaled.csv"
    output_file = "../Raw Datasets/test_with_predictions_hybrid.csv"

    train_data = pd.read_csv(train_file)

    # Exclude the target column for training
    x_train = train_data.iloc[:, :-1]  # All columns except the last (price_range)
    y_train = train_data.iloc[:, -1]  # Last column (price_range)

    # Train Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
    model_rf.fit(x_train, y_train)

    # Train MLP
    model_mlp = MLPClassifier(
        hidden_layer_sizes=[64],
        max_iter=50,
        batch_size=16,
        solver="adam",
        activation="logistic",
        learning_rate_init=0.001,
        random_state=42
    )
    model_mlp.fit(x_train, y_train)

    # Test with the hybrid model and visualize for both train and test data
    test_with_hybrid_model(model_rf, model_mlp, train_file, test_file, output_file)
