# %% [markdown]
# # Decision Tree Implementation for Telecom Customer Churn Prediction
# This script implements a Decision Tree classifier to predict customer churn.
# The process includes data loading, preprocessing, model training, evaluation, and visualization.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn libraries for preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Imblearn library for handling imbalanced datasets
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading

# %%
# Load the pre-processed dataset
# The dataset was saved fully processed (encoded) in the Dataset directory
data_path = '../Dataset/Processed_Telecom_Churn.csv'
try:
    df = pd.read_csv(data_path)
    print("Processed dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find the dataset at {data_path}. Please check the path.")

# %%
# Display the first few rows of the dataset
pd.set_option('display.max_columns', None)
print(df.head())

# %% [markdown]
# ## 2. Data Preprocessing and Splitting

# %%
# Define feature variables (X) and Target variable (y)
target_col = 'Churn'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Scale the numerical features (tenure, MonthlyCharges, TotalCharges)
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# %% [markdown]
# ## 3. Resolving Class Imbalance with SMOTE

# %%
# Apply SMOTE to the training data to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Original y_train class distribution:\n{y_train.value_counts()}")
print(f"Resampled y_train class distribution:\n{y_train_resampled.value_counts()}")

# %% [markdown]
# ## 4. Decision Tree Model Training

# %%
# Train a Decision Tree Classifier
# We set random_state for reproducibility and max_depth to prevent overfitting
dt_model = DecisionTreeClassifier(random_state=42, max_depth=8, criterion='gini')

# Fit the model on the SMOTE resampled training data
dt_model.fit(X_train_resampled, y_train_resampled)
print("Decision Tree model trained successfully.")

# Make predictions on the test set
y_pred = dt_model.predict(X_test_scaled)
y_pred_proba = dt_model.predict_proba(X_test_scaled)[:, 1]
