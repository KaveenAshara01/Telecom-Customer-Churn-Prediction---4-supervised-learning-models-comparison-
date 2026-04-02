# %% [markdown]
# # Random Forest Implementation for Telecom Customer Churn Prediction
# This notebook implements a Random Forest classifier to predict customer churn using the Telecom Customer Churn dataset.
# The process includes data loading, preprocessing, feature engineering, model training, evaluation, and visualization.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn libraries for preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Imblearn library for handling imbalanced datasets
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading

# %%
# Load the dataset

data_path = '../Dataset/Raw.csv'
try:
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find the dataset at {data_path}. Please check the path.")

# %%
# Display the first few rows of the dataset
pd.set_option('display.max_columns', None)
display(df.head())

# %%
# Check fundamental information about data types and missing values
df.info()

# %% [markdown]
# ## 2. Data Preprocessing and Feature Engineering

# %%
# Step 2.1: Drop unnecessary columns
# 'customerID' is basically a unique identifier and does not contribute to predicting churn.
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# %%
# Step 2.2: Handle missing or invalid values
# 'TotalCharges' is read as an object because it contains spaces " " for missing values.
# Let's convert it to numeric, coercing errors to NaN.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many missing values we currently have
missing_values = df.isnull().sum()
print("Missing values per column prior to handling:\n", missing_values[missing_values > 0])

# Since the number of missing values is very small (usually around 11 for TotalCharges), 
# dropping these rows is a safe approach without losing much information.
df = df.dropna()
print(f"Dataset shape after removing missing values: {df.shape}")

# %%
# Step 2.3: Encode categorical variables
# Let's separate features into categorical and numerical
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['number']).columns.tolist()

# The target variable is 'Churn'
target_col = 'Churn'
if target_col in cat_cols:
    cat_cols.remove(target_col)

print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)

# Define feature variables (X) and Target variable (y)
X = df.drop(target_col, axis=1)
y = df[target_col].map({'Yes': 1, 'No': 0}) # Encode target manually

# For categorical features, we will use Label Encoding for binary features, 
# and One-Hot Encoding for features with more than 2 categories.
binary_cols = [col for col in cat_cols if X[col].nunique() == 2]
multi_cat_cols = [col for col in cat_cols if X[col].nunique() > 2]

# Label encode binary categorical columns
le = LabelEncoder()
for col in binary_cols:
    X[col] = le.fit_transform(X[col])

# One-hot encode multi-categorical columns
X = pd.get_dummies(X, columns=multi_cat_cols, drop_first=True)

print(f"Shape of feature matrix X after encoding: {X.shape}")

# %% [markdown]
# ## 2.5 Save Processed Dataset
# To ensure consistency across the group assignment when comparing multiple models (like SVM or Logistic Regression), 
# we save the perfectly pre-processed (encoded and cleaned, but NOT scaled or SMOTEd) dataset.

# %%
import os

processed_df = X.copy()
processed_df['Churn'] = y

# Create Dataset directory if it doesn't exist just in case
if not os.path.exists('../Dataset'):
    os.makedirs('../Dataset')

save_path = '../Dataset/Processed_Telecom_Churn.csv'
processed_df.to_csv(save_path, index=False)
print(f"Successfully saved fully processed dataset to: {save_path}")
print(f"Processed dataset shape: {processed_df.shape}")

# %%
# Random Forest does not strictly require feature scaling, but it's good practice 
# and can help performance when comparing with other models.

# %% [markdown]
# ## 3. Train-Test Split

# %%
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Scale the numerical features
# (tenure, MonthlyCharges, TotalCharges)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# %% [markdown]
# ## 3.1 Resolving Class Imbalance with SMOTE
# Telecom churn datasets usually have fewer instances of "Churn = Yes" compared to "Churn = No". 
# To prevent the model from being biased towards the majority class, we apply SMOTE (Synthetic Minority Over-sampling Technique)
# to oversample the minority class specifically on the training set.

# %%
# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Original y_train class distribution:\n{y_train.value_counts()}")
print(f"Resampled y_train class distribution:\n{y_train_resampled.value_counts()}")

# %% [markdown]
# ## 4. Random Forest Model Training

# %%
# Train a Random Forest Classifier
# We set class_weight='balanced' to handle the imbalanced nature of churn dataset.
rf_model = RandomForestClassifier(n_estimators=100, 
                                  random_state=42, 
                                  max_depth=10, 
                                  n_jobs=-1)

# Fit the model on the SMOTE resampled training data
rf_model.fit(X_train_resampled, y_train_resampled)

# %% [markdown]
# ## 5. Model Evaluation and Metrics

# %%
# Make predictions on the test set
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate classification evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("--- Random Forest Model Performance ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churn', 'Churn'], 
            yticklabels=['Not Churn', 'Churn'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %%
# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## 6. Feature Importance

# %%
# Extract feature importance from the model
feature_importances = rf_model.feature_importances_
features = X_train_scaled.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15 most important features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
plt.title('Top 15 Most Important Features - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Conclusion & Discussion
# The Random Forest model provides a robust baseline for predicting telecom customer churn. 
# 
# **Data Preprocessing & Feature Engineering**: Missing values in `TotalCharges` were pruned since there were very few, and variables were systematically encoded (Label/One-Hot encoding based on class quantity). Numerical variables were scaled. 
# 
# **Handling Imbalance**: Instead of just using the model's built-in `class_weight` adjustments, we utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to algorithmically duplicate and synthesize new minority class data points, ensuring a perfectly balanced training dataset.
# 
# **Modeling**: A Random Forest Classifier was trained on the resampled training subset.
#
# **Metrics**: Precision, Recall, Accuracy, F1-Score, and ROC-AUC are calculated to ensure a holistic viewing. Specifically for churn, achieving a higher recall (finding all customers who quit) sometimes holds higher priority than overarching accuracy.
#
# **Hyperparameters**: `max_depth = 10` is used to restrict the growth of individual trees to mitigate overfitting. 
# %% [markdown]
# ## 7. Export Evaluation Metrics

# %%
# Create a dictionary of metrics
metrics_dict = {
    'Model': ['Random Forest'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1],
    'ROC-AUC': [roc_auc]
}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Save to CSV in the current directory ('Random forest')
metrics_path = 'random_forest_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"Evaluation metrics successfully exported to: {metrics_path}")
