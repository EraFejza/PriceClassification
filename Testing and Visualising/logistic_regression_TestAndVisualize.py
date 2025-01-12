import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv(f'../Normalized_Datasets/Train/raw_z_score_scaled.csv')
test_data = pd.read_csv(f'../Normalized_Datasets/Test/raw_z_score_scaled.csv')

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

x_test = test_data.drop(columns=['id'])


log_reg = LogisticRegression(C=1, max_iter=2000, penalty='l1', solver='saga')
log_reg.fit(X_train, y_train)

predictions = log_reg.predict(x_test)
test_data['price_range'] = predictions

test_data.to_csv(f"../Raw Datasets/test_with_predictions_z_score_logistic.csv", index=False)
print(f"Predictions saved to /Raw Datasets/test_with_predictions_z_score_logistic.csv")

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