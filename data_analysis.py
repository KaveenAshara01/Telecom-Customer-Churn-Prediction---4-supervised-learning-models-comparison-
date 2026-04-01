# %% [markdown]
# # Data Analysis for Telecom Customer Churn Prediction
# This script analyzes the `Dataset/Raw.csv` dataset, providing essential charts and analyses.
# It is formatted with `# %%` to allow easy execution cell-by-cell or export to a Jupyter Notebook.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set standard plotting style
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load Data
# Let's load the raw dataset and examine the first few rows and basic information.

# %%
# Define dataset path
dataset_path = 'Dataset/Raw.csv'

# Load the dataset
df = pd.read_csv(dataset_path)

# Display basic information
print("Dataset Information:")
print("-" * 30)
df.info()

print("\nFirst 5 rows of the dataset:")
print("-" * 30)
# 'display' is available in Jupyter/IPython, falling back to print for standard python
if 'get_ipython' in globals() and get_ipython() is not None:
    display(df.head())
else:
    print(df.head())

# %% [markdown]
# ## 2. Data Preprocessing
# We need to correctly parse the numerical features. `TotalCharges` is often loaded as an object type due to empty spaces, so we'll convert it.

# %%
# Convert TotalCharges to numeric, coercing invalid parsing (like empty spaces) to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Let's simply drop the few rows with missing TotalCharges as they make up a tiny fraction of the data.
df.dropna(subset=['TotalCharges'], inplace=True)

print(f"\nRemaining rows after dropping missing values: {len(df)}")

# %% [markdown]
# ## 3. Target Variable Distribution
# Visualizing the distribution of our target variable `Churn` to understand class imbalance.

# %%
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x='Churn', palette='Set2', hue='Churn', legend=False)
plt.title('Target Variable Distribution (Churn)')
plt.xlabel('Churn')
plt.ylabel('Count')

# Add value counts above the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.show()

# %% [markdown]
# ## 4. Distribution of Numerical Features
# Let's inspect the distribution of key numerical variables: `tenure`, `MonthlyCharges`, and `TotalCharges`.

# %%
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

plt.figure(figsize=(18, 5))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[feature], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Boxplot of Key Features by Target
# Boxplots help visualize how our numerical features differ across the `Churn` classes, outlining the medians and any outliers.

# %%
plt.figure(figsize=(18, 5))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df, x='Churn', y=feature, palette='Set2', hue='Churn', legend=False)
    plt.title(f'Boxplot of {feature} by Churn')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Correlation Heatmap
# To understand the relationships between different numeric variables and the target, we use a correlation heatmap. 
# First, we'll map `Churn` to binary values (1 for Yes, 0 for No).

# %%
# Create a copy for correlation analysis
df_corr = df.copy()

# Convert target to numeric
df_corr['Churn_Binary'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})

# Select numeric columns for correlation matrix
numeric_cols = df_corr[numerical_features + ['Churn_Binary']]
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# ## 7. Categorical Features Analysis
# Let's explore how some key categorical features relate to the Customer Churn rate.

# %%
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract', 'InternetService']

plt.figure(figsize=(15, 12))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=df, x=feature, hue='Churn', palette='Set2')
    plt.title(f'{feature} vs Churn')
    # Rotate x labels if there's text overlap (especially useful for features with many classes like PaymentMethod)
    plt.xticks(rotation=45 if df[feature].nunique() > 2 else 0)

plt.tight_layout()
plt.show()
