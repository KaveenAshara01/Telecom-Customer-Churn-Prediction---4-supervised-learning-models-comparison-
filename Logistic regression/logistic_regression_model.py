import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os

# Set paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'Dataset', 'Processed_Telecom_Churn.csv')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split dataset into training and testing sets (80% train, 20% test)
print(f"Splitting dataset into {len(X)} samples...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
print("Training Logistic Regression model...")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predict on the test set
print("Evaluating model...")
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Display evaluation metrics
print("\n" + "="*40)
print("       MODEL EVALUATION METRICS       ")
print("="*40)
print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("-" * 40)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("-" * 40)
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("                Predicted No  Predicted Yes")
print(f"Actual No Churn : {cm[0][0]:<12}  {cm[0][1]}")
print(f"Actual Churn    : {cm[1][0]:<12}  {cm[1][1]}")
print("="*40)
print("Model training and evaluation successfully completed!")

# Export predictions to CSV
print("Exporting predictions to CSV...")
predictions_df = X_test.copy()
predictions_df['Actual_Churn'] = y_test
predictions_df['Predicted_Churn'] = y_pred
predictions_df['Predicted_Probability'] = y_pred_proba.round(4)

# Create Results directory if it doesn't exist just in case
results_dir = os.path.join(base_dir, 'Results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

save_path = os.path.join(results_dir, 'Logistic_Regression_Predictions.csv')
predictions_df.to_csv(save_path, index=False)
print(f"Successfully saved predictions to: {save_path}")
print("Done!")
