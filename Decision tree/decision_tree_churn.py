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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

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

# %% [markdown]
# ## 5. Model Evaluation and Metrics

# %%
# Calculate classification evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("--- Decision Tree Model Performance ---")
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
plt.title('Confusion Matrix - Decision Tree')
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
feature_importances = dt_model.feature_importances_
features = X_train_scaled.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15 most important features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
plt.title('Top 15 Most Important Features - Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
