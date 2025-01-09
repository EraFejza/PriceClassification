import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
train = pd.read_csv("../Raw Datasets/train.csv")
test = pd.read_csv("../Raw Datasets/test.csv")

# Define categorical features and target column
categorical_features = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
target_column = 'price_range'
id_column = 'id'  # Column to exclude from normalization in test dataset

# Identify numerical features
train_features = [col for col in train.columns if col not in categorical_features + [target_column]]
test_features = [col for col in test.columns if col not in categorical_features + [id_column]]

# Function for Decimal Scaling normalization
def decimal_scaling(train_data, test_data, numerical_features):
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    for feature in numerical_features:
        magnitude = 10 ** (np.ceil(np.log10(np.abs(train_data[feature]).max())))
        train_scaled[feature] = train_data[feature] / magnitude
        if feature in test_data.columns:
            test_scaled[feature] = test_data[feature] / magnitude
    return train_scaled, test_scaled

# Function for Min-Max normalization
def min_max_normalizer(train_data, test_data, numerical_features):
    min_max_scaler = MinMaxScaler()
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[numerical_features] = min_max_scaler.fit_transform(train_data[numerical_features])
    test_scaled[numerical_features] = min_max_scaler.transform(test_data[numerical_features])
    return train_scaled, test_scaled

# Function for Z-Score normalization
def z_score_normalizer(train_data, test_data, numerical_features):
    standard_scaler = StandardScaler()
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[numerical_features] = standard_scaler.fit_transform(train_data[numerical_features])
    test_scaled[numerical_features] = standard_scaler.transform(test_data[numerical_features])
    return train_scaled, test_scaled

# Feature Engineering Function
def feature_engineering(data, is_test=False):
    epsilon = 1e-9  # Small value to avoid division by zero
    data = data.copy()

    # Create 'resolution' feature
    data['resolution'] = (data['px_width'] / (data['sc_w'] + epsilon)) * (data['px_height'] / (data['sc_h'] + epsilon))

    # Create 'screen_size' feature
    data['screen_size'] = data['sc_w'] * data['sc_h']

    # Drop redundant columns
    features_to_drop = ['px_width', 'px_height', 'sc_w', 'sc_h']
    data = data.drop(columns=features_to_drop)

    # Handle 'id' column for test dataset
    if is_test and id_column in data.columns:
        id_column_data = data[id_column]
        data = data.drop(columns=[id_column])
        data.insert(0, id_column, id_column_data)

    # Move 'price_range' to the last column in train dataset
    if not is_test and target_column in data.columns:
        columns = [col for col in data.columns if col != target_column]  # Exclude 'price_range'
        columns.append(target_column)  # Add 'price_range' as the last column
        data = data[columns]

    return data

# Save normalized datasets for raw (non-feature-engineered) datasets
raw_normalized_datasets = {}
raw_numerical_features = [col for col in train.columns if col not in categorical_features + [target_column]]
raw_test_numerical_features = [col for col in test.columns if col not in categorical_features + [id_column]]

# Decimal Scaling for raw datasets
raw_train_decimal_scaled, raw_test_decimal_scaled = decimal_scaling(train, test, raw_numerical_features)
raw_normalized_datasets['decimal_scaled'] = (raw_train_decimal_scaled, raw_test_decimal_scaled)

# Min-Max Normalization for raw datasets
raw_train_min_max_scaled, raw_test_min_max_scaled = min_max_normalizer(train, test, raw_numerical_features)
raw_normalized_datasets['min_max_scaled'] = (raw_train_min_max_scaled, raw_test_min_max_scaled)

# Z-Score Normalization for raw datasets
raw_train_z_score_scaled, raw_test_z_score_scaled = z_score_normalizer(train, test, raw_numerical_features)
raw_normalized_datasets['z_score_scaled'] = (raw_train_z_score_scaled, raw_test_z_score_scaled)

# Save raw normalized datasets
raw_output_path = "../Normalized_Datasets"
for norm_name, (raw_train_set, raw_test_set) in raw_normalized_datasets.items():
    raw_train_file_name = f"{raw_output_path}/Train/raw_{norm_name}.csv"
    raw_test_file_name = f"{raw_output_path}/Test/raw_{norm_name}.csv"
    raw_train_set.to_csv(raw_train_file_name, index=False)
    raw_test_set.to_csv(raw_test_file_name, index=False)

# Apply feature engineering to both train and test datasets
train = feature_engineering(train, is_test=False)
test = feature_engineering(test, is_test=True)

# Update numerical features after feature engineering
numerical_features = [col for col in train.columns if col not in categorical_features + [target_column]]
test_numerical_features = [col for col in test.columns if col not in categorical_features + [id_column]]

# Save feature-engineered normalized datasets
normalized_datasets = {}

# Decimal Scaling
train_decimal_scaled, test_decimal_scaled = decimal_scaling(train, test, numerical_features)
normalized_datasets['decimal_scaled'] = (train_decimal_scaled, test_decimal_scaled)

# Min-Max Normalization
train_min_max_scaled, test_min_max_scaled = min_max_normalizer(train, test, numerical_features)
normalized_datasets['min_max_scaled'] = (train_min_max_scaled, test_min_max_scaled)

# Z-Score Normalization
train_z_score_scaled, test_z_score_scaled = z_score_normalizer(train, test, numerical_features)
normalized_datasets['z_score_scaled'] = (train_z_score_scaled, test_z_score_scaled)

# Save normalized datasets
output_path = "../Normalized_Datasets"
for norm_name, (train_set, test_set) in normalized_datasets.items():
    train_file_name = f"{output_path}/Train/train_{norm_name}.csv"
    test_file_name = f"{output_path}/Test/test_{norm_name}.csv"
    train_set.to_csv(train_file_name, index=False)
    test_set.to_csv(test_file_name, index=False)

# Display a summary of saved files
print("Normalization completed. Files saved:")
print("Raw Normalized Datasets:")
for norm_name in raw_normalized_datasets.keys():
    print(f"- raw_train_{norm_name}.csv")
    print(f"- raw_test_{norm_name}.csv")
print("\nFeature-Engineered Normalized Datasets:")
for norm_name in normalized_datasets.keys():
    print(f"- train_{norm_name}.csv")
    print(f"- test_{norm_name}.csv")
