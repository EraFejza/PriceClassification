import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

train_data = pd.read_csv(f'../Normalized_Datasets/Train/raw_z_score_scaled.csv')
test_data = pd.read_csv(f'../Normalized_Datasets/Test/raw_z_score_scaled.csv')

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

x_test = test_data.drop(columns=['id'])


# ---- MLP Training ----
mlp_model = MLPClassifier(
        hidden_layer_sizes=[64],
        max_iter=2000,
        batch_size=16,
        solver="adam",
        activation="logistic",
        learning_rate_init=0.001,
        random_state=42

)

mlp_model.fit(X_train, y_train)

predictions = mlp_model.predict(x_test)
test_data['price_range'] = predictions

test_data.to_csv(f"../Raw Datasets/test_with_predictions_z_score_mlp.csv", index=False)
print(f"Predictions saved to /Raw Datasets/test_with_predictions_z_score_mlp.csv")

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