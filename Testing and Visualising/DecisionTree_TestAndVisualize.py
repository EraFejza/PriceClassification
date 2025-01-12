import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv("../Normalized_Datasets/Train/train_z_score_scaled.csv")
test_data = pd.read_csv("../Normalized_Datasets/Test/test_z_score_scaled.csv")

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.drop(columns=["id"])

criterion = "entropy"
max_depth = 15
model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_data_with_predictions = test_data.copy()
test_data_with_predictions["price_range"] = y_pred

plt.figure(figsize=(12, 8))
train_corr = train_data.corr()
sns.heatmap(train_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Training Data")
plt.show()

plt.figure(figsize=(12, 8))
test_corr = test_data_with_predictions.corr()
sns.heatmap(test_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Test Data with Predictions")
plt.show()

selected_features = ["battery_power", "ram", "resolution"]
sns.pairplot(train_data[selected_features + ["price_range"]], hue="price_range", palette="coolwarm")
plt.suptitle("Pairplot - Training Data", y=1.02)
plt.show()

sns.pairplot(test_data_with_predictions[selected_features + ["price_range"]], hue="price_range", palette="coolwarm")
plt.suptitle("Pairplot - Test Data with Predictions", y=1.02)
plt.show()
