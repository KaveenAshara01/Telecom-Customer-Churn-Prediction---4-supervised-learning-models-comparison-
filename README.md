# Telecom Customer Churn Prediction: 4 Supervised Learning Models Comparison

##  Project Overview
This repository contains a comprehensive data science project aimed at predicting telecommunications customer churn. By analyzing fundamental demographic data, service subscriptions, and complex billing histories, we implemented and rigorously compared four distinct supervised machine learning models to identify "at-risk" customers proactively. 

The core objective of this study is to identify behavioral patterns that historically precede contract termination, empowering telecom retention departments to successfully transition from reactive damage control into proactive, highly targeted intervention campaigns.

##  Machine Learning Models Implemented
We carefully evaluated the following algorithmic pipelines against a strictly isolated 20% holdout testing set:
- **Logistic Regression**: A probabilistic baseline model internally utilizing standard L2 (Ridge) regularization.
- **Decision Tree**: A highly structured, non-parametric boundary model fully optimized via exhaustive `GridSearchCV`.
- **Random Forest**: A robust ensemble bagging classifier strictly manually optimized (`n_estimators=100`, `max_depth=10`) to provide maximum structural F1-Score balance.
- **Support Vector Machine (SVM)**: A geometric mapping algorithm inherently optimized with a non-linear Radial Basis Function (`rbf`) kernel and heavy `C=10` boundary regularization.

##  Dataset
The base dataset (`Dataset/Raw.csv`) contains approximately 7,000 anonymized historical records of active telecom subscribers.
Key features isolated for prediction include:
- **Demographics**: Gender, Senior Citizen status, Partner, Dependents.
- **Service Utilization**: Phone/Internet service tiers (DSL/Fiber Optic), Tech Support, Online Security, and active streaming subscriptions.
- **Financial Architecture**: Overarching Tenure (months), Contract operational type (e.g., Month-to-Month), Monthly Charges, and comprehensive Total Charges.

*Note: The dataset natively exhibited a severe predictive class imbalance (with significantly more retained customers than active churned cases). This structural algorithmic hurdle was systematically neutralized exclusively on the training subset using the **Synthetic Minority Over-sampling Technique (SMOTE)**.*

##  Repository Structure
```text
├── Dataset/                   # Contains raw and pre-processed CSV data files
├── Decision tree/             # Jupyter notebook, Python scripts, and evaluation metrics for Decision Tree
├── Logistic regression/       # Python scripts and evaluation metrics for Logistic Regression
├── Random forest/             # Jupyter notebook, Python scripts, and evaluation metrics for Random Forest
├── SVM/                       # Jupyter notebook and evaluation metrics for Support Vector Machine
├── data_analysis.py           # Exploratory Data Analysis (EDA) script generating visualizations
├── model_comparison.py        # Central script merging and comparing metrics across all four models
└── README.md                  # Project documentation
```

## Key Findings & Results
Raw evaluation metrics heavily mask the financial realities of telecommunications customer churn, where actively missing a departing customer (a False Negative) is substantially more expensive to the operational bottom line than inadvertently issuing a redundant retention alert (a False Positive). 

Our comprehensive evaluation matrix highlighted the **Random Forest** as the absolute premier predictive engine. While the tuned Decision Tree computationally captured the most absolute churners (bearing the highest Recall), its inherently poor Precision ultimately rendered it economically unviable. Conversely, the Logistic Regression model possessed statistically high Accuracy but fundamentally failed to catch outgoing departing subscribers (rendering low Recall). Ultimately, the Random Forest model successfully negotiated the highest structural harmony, resulting in peak predictive **F1-Score (0.6276)** and incredibly strong general discriminative power (**ROC-AUC 0.8303**).

##  How to Run the Project
1. **Prerequisites**: Ensure you have Python 3.8+ installed along with standard data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`).
2. **Exploratory Data Analysis**: Run the `data_analysis.py` file interactively (cell-by-cell) utilizing `# %%` markers in compatible IDEs (like Jupyter, VS Code, or Spyder) to visualize the core dataset distributions and correlation heatmaps.
3. **Model Processing and Training**: Navigate into any of the individual model sub-directories (e.g., `Random forest/`) and execute the respective scripts or notebooks to encode and preprocess the dataset, effectively apply SMOTE, and directly train that specific classification algorithm.
4. **Evaluate & Systematically Compare**: Execute `model_comparison.py` located in the root repository directory to instantly load the locally generated individual metric CSV files, seamlessly combine them into a single dataframe entity, and automatically generate the final comparative analytical bar charts.

##  Future Work
- **Dynamic Time-Series Ingestion**: Transition subsequent models from observing static historical snapshots to dynamic, recurrent (RNN/LSTM) tracking of rolling monthly network usage and billing degradation.
- **Expanded Sentiment Features**: Systematically integrate Natural Language Processing (NLP) analytics derived directly from customer support transcripts to successfully flag acute conversational frustration triggers.
- **Commercial A/B ROI Testing**: Realistically deploy the Random Forest model's live predictive outputs within a highly active telecom call center to definitively measure specific retention conversion rates across competing promotional campaigns.
