# ntad
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# File path
file_path = r"C:\Users\chevu\uopi\synthetic_network_traffic.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values
for col in data.columns:
    if data[col].dtype == 'object':  # Handle categorical columns
        data[col] = data[col].fillna(data[col].mode()[0])  # Replace with mode
    else:  # Handle numerical columns
        data[col] = data[col].fillna(data[col].mean())  # Replace with mean

# Encode categorical features
categorical_cols = data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Scale numerical features
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Display the preprocessed data
print("\nPreprocessed Data Head:")
print(data.head())

# Save the preprocessed dataset for future use
output_path = r"C:\Users\chevu\uopi\preprocessed_network_traffic.csv"
data.to_csv(output_path, index=False)
print(f"\nPreprocessed data saved to: {output_path}")
# Plot the results
plt.figure(figsize=(10, 6))
sns.histplot(test_df['value'], bins=50, kde=True, hue=test_df['is_anomaly'], palette={1: 'red', 0: 'blue'})
plt.title('Distribution of Values with Anomalies Highlighted', fontsize=16)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(['Normal', 'Anomaly'], title='Legend')
plt.tight_layout()
plt.show()
