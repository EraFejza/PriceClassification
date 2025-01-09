import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
file_path = 'C:/Users/erafe/OneDrive/Desktop/Phone-price-classification/train.csv'
data = pd.read_csv(file_path)

# Define categorical and target columns
categorical_features = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
target_column = "price_range"

# Identify numerical features
numerical_features = [col for col in data.columns if col not in categorical_features + [target_column]]



# Function for Decimal Scaling normalization
def decimal_scaling(data, numerical_features):
    scaled_data = data.copy()
    for feature in numerical_features:
        magnitude = 10 ** (np.ceil(np.log10(np.abs(data[feature]).max())))
        scaled_data[feature] = data[feature] / magnitude
    return scaled_data

# Function for Min-Max normalization
def min_max_normalizer(data, numerical_features):
    min_max_scaler = MinMaxScaler()
    scaled_data = data.copy()
    scaled_data[numerical_features] = min_max_scaler.fit_transform(data[numerical_features])
    return scaled_data

# Function for Z-Score normalization
def z_score_normalizer(data, numerical_features):
    standard_scaler = StandardScaler()
    scaled_data = data.copy()
    scaled_data[numerical_features] = standard_scaler.fit_transform(data[numerical_features])
    return scaled_data

# Feature Engineering Function
def feature_engineering(data):
    epsilon = 1e-9  # Small value to avoid division by zero
    data = data.copy()

    # Create 'resolution' feature
    data['resolution'] = (data['px_width'] / (data['sc_w'] + epsilon)) * (data['px_height'] / (data['sc_h'] + epsilon))

    # Create 'screen_size' feature
    data['screen_size'] = data['sc_w'] * data['sc_h']

    # Drop redundant columns
    features_to_drop = ['px_width', 'px_height', 'sc_w', 'sc_h']
    data = data.drop(columns=features_to_drop)
    return data


# Apply feature engineering to create a new dataset
feature_engineered_data = feature_engineering(data)

# Update numerical features after feature engineering
numerical_features_engineered = [col for col in feature_engineered_data.columns if
                                 col not in categorical_features + [target_column]]

# Apply normalizations to both original and feature-engineered datasets
datasets = {
    "original": (data, numerical_features),
    "feature_engineered": (feature_engineered_data, numerical_features_engineered)
}

normalization_methods = {
    "decimal_scaled": decimal_scaling,
    "min_max_scaled": min_max_normalizer,
    "z_score_scaled": z_score_normalizer
}

# Save normalized datasets
output_path = "C:/Users/erafe/OneDrive/Desktop/Phone-price-classification/Datasets"
for dataset_name, (dataset, features) in datasets.items():
    for norm_name, norm_function in normalization_methods.items():
        normalized_dataset = norm_function(dataset, features)
        file_name = f"{output_path}/{dataset_name}_{norm_name}.csv"
        normalized_dataset.to_csv(file_name, index=False)

# Display a summary of saved files
print("Feature engineering and normalization completed. Files saved:")
for dataset_name in datasets.keys():
    for norm_name in normalization_methods.keys():
        print(f"- {dataset_name}_{norm_name}.csv")

