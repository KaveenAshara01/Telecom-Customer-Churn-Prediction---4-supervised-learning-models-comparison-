# %% [markdown]
# # Model Evaluation and Comparison
# This script reads the evaluation metrics from each model's directory (Decision Tree, Logistic Regression, Random Forest, SVM) 
# and compares them visually and analytically to suggest the best model for Telecom Customer Churn Prediction.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set standard plotting style
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load Evaluation Metrics
# We load the evaluation metrics saved in the CSV files from each model's respective directory.

# %%
# Define paths to the CSV files
file_paths = {
    'Decision Tree': 'Decision tree/decision_tree_metrics.csv',
    'Logistic Regression': 'Logistic regression/logistic_regression_metrics.csv',
    'Random Forest': 'Random forest/random_forest_metrics.csv',
    'SVM': 'SVM/svm_results.csv'
}

# Load and combine all metrics into a single DataFrame
dfs = []
for model_name, path in file_paths.items():
    try:
        # Read the csv file
        df = pd.read_csv(path)
        # We can enforce naming consistency just in case the models used varying internal labels
        df['Model'] = model_name 
        dfs.append(df)
    except FileNotFoundError:
        print(f"Warning: File {path} not found.")

# Concatenate into one dataframe
metrics_df = pd.concat(dfs, ignore_index=True)

# Format the numerical columns simply for display purposes
display_df = metrics_df.copy()
numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
display_df[numeric_cols] = display_df[numeric_cols].round(4)

# Display the combined metrics
print("Combined Evaluation Metrics:")
print("-" * 50)
if 'get_ipython' in globals() and get_ipython() is not None:
    display(display_df)
else:
    print(display_df)

# %% [markdown]
# ## 2. Comparative Visualization
# Now we will visualize the different metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) using grouped bar charts to compare model performance side-by-side.

# %%
# Melt the dataframe for easier plotting with seaborn's grouped barplot
metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Set up the matplotlib figure
plt.figure(figsize=(14, 8))

# Create a bar plot grouped by metric with model hues
ax = sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='viridis')

# Add score labels on top of the bars for exact readability
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)

plt.title('Comparison of Evaluation Metrics Across Models', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.xlabel('Metric', fontsize=12)
plt.ylim(0, 1.1)

# Format legend
plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11, title_fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Detailed Analysis by Metric
# 
# Let's break down the comparative performance based on the specific evaluation values:
# 
# *   **Accuracy**: **Logistic Regression (~78.7%)** achieved the highest accuracy, followed closely by Random Forest (~76.0%). Logistic Regression is the most universally "correct" model for both churn and non-churn predictions.
# *   **Precision**: **Logistic Regression (~62.1%)** leads significantly in precision. This means when Logistic Regression predicts that a customer will churn, it has the highest likelihood of being correct. The other models hover around 50-53%.
# *   **Recall**: **Decision Tree (~80.5%)** and **Random Forest (~75.9%)** absolutely dominate in terms of recall, whereas Logistic Regression lags significantly behind (~51.6%). This indicates that tree-based models are vastly superior at finding the *actual* churners, suffering fewer "false negatives."
# *   **F1-Score**: F1-Score is the harmonic mean of precision and recall. **Random Forest (~62.8%)** slightly edges out the Decision Tree (~61.6%) and SVM (~61.0%) to offer the best balanced metric.
# *   **ROC-AUC**: The ROC-Area Under Curve measures the general ability of the model to distinguish between classes. All models perform consistently well here (between 0.81 and 0.83), but **Logistic Regression (0.832)** and **Random Forest (0.830)** top the list, indicating very strong classification discriminative power.

# %% [markdown]
# ## 4. Best Model Suggestion
# 
# **Recommendation:** **Random Forest** 
# 
# **Business Context & Justification:**
# In telecom customer churn prediction, the cost of failing to identify a churner (a False Negative) is typically much more expensive to the business than incorrectly predicting a loyal customer will churn (a False Positive, which usually just results in sending a redundant promotional/retention email). 
# 
# Therefore, business logic dictates that we should strongly prioritize **Recall** without letting precision dip so low that our retention budget is wasted. The metric that best captures this priority balance is the **F1-Score**.
# 
# 1.  **Why not Decision Tree?** While it has the absolute best Recall (caught 80.5% of churners), its precision is the lowest (49.8%). It flags far too many false positives.
# 2.  **Why not Logistic Regression?** While it has the highest Accuracy and Precision, its Recall is terribly low (51.6%). It simply misses too many actual churners, allowing nearly half of them to leave without any intervention from the business.
# 3.  **Why Random Forest?** The Random Forest provides the "goldilocks" middle ground. It maintains a highly effective **Recall (75.9%)** to capture the vast majority of churners. At the same time, it maintains an acceptable **Precision (53.5%)**, yielding the overall highest **F1-Score (62.8%)**. Furthermore, its **ROC-AUC (83.0%)** proves its strong overall separation power.
# 
# **Final Verdict**: To maximize customer retention and deploy budget effectively, the **Random Forest** model is recommended.
