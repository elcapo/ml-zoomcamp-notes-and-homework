# Quick Start Guide

This guide will help you get started with the Poverty Risk Analysis project.

---

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (or pip)

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd projects/capstone
```

### 2. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Data Preparation

### 1. Verify Data Files

Ensure you have the following data files:
```
data/
├── ECV_Th_2024/CSV/ECV_Th_2024.tab    # Household data
└── ECV_Tp_2024/CSV/ECV_Tp_2024.tab    # Person data
```

### 2. Merge Household and Person Data

Run the merging script:
```bash
uv run merge.py
```

This creates `data/merged.csv` with:
- 29,781 households
- ~35 features (household-level + aggregated person-level)
- Target variable: `vhPobreza`

---

## Exploratory Data Analysis

### Load Merged Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged data
df = pd.read_csv('data/merged.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
```

### Explore Target Variable

```python
# Poverty risk distribution
print("\nTarget distribution:")
print(df['vhPobreza'].value_counts())
print(df['vhPobreza'].value_counts(normalize=True))

# Visualize
plt.figure(figsize=(8, 5))
df['vhPobreza'].value_counts().plot(kind='bar')
plt.title('Poverty Risk Distribution')
plt.xlabel('At Risk of Poverty')
plt.ylabel('Count')
plt.xticks([0, 1], ['No (0)', 'Yes (1)'], rotation=0)
plt.show()
```

### Explore Key Features

```python
# Income distribution by poverty risk
plt.figure(figsize=(10, 6))
df.boxplot(column='HY020', by='vhPobreza')
plt.title('Income Distribution by Poverty Risk')
plt.xlabel('At Risk of Poverty')
plt.ylabel('Total Household Income (€)')
plt.suptitle('')
plt.show()

# Correlation with target
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corrwith(df['vhPobreza']).sort_values(ascending=False)
print("\nTop 10 features correlated with poverty risk:")
print(correlations.head(10))
```

### Check Missing Values

```python
# Missing value analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
})
print("\nMissing values:")
print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
```

---

## Build Baseline Model

### Prepare Data

```python
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop(['vhPobreza'], axis=1)
y = df['vhPobreza']

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### Train Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Feature Importance

```python
# Get feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Visualize
plt.figure(figsize=(10, 8))
feature_importance.head(15).plot(x='Feature', y='Coefficient', kind='barh')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.show()
```

---

## Train Advanced Models

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Train
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 features by importance:")
print(rf_importance.head(10))
```

### XGBoost

```python
from xgboost import XGBClassifier

# Calculate class ratio for imbalanced data
class_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# Train
xgb_model = XGBClassifier(
    scale_pos_weight=class_ratio,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_xgb):.4f}")
```

---

## Model Comparison

```python
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=class_ratio, random_state=42)
}

results = {}
for name, model in models.items():
    # Use scaled data for Logistic Regression
    if name == 'Logistic Regression':
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

    results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Visualize
plt.figure(figsize=(10, 6))
plt.boxplot(results.values(), labels=results.keys())
plt.title('Model Comparison (5-Fold Cross-Validation F1-Score)')
plt.ylabel('F1-Score')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```

---

## Save Model

```python
import joblib

# Save best model
joblib.dump(xgb_model, 'model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model saved successfully!")
```

---

## Next Steps

1. **Feature Engineering:** Create additional features based on domain knowledge
   - See [machine-learning-strategy.md](machine-learning-strategy.md) for ideas

2. **Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV
   ```python
   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [3, 5, 7],
       'learning_rate': [0.01, 0.1, 0.3]
   }

   grid_search = GridSearchCV(
       XGBClassifier(scale_pos_weight=class_ratio),
       param_grid,
       cv=5,
       scoring='f1',
       n_jobs=-1
   )
   grid_search.fit(X_train, y_train)
   ```

3. **Model Interpretation:** Use SHAP values for feature importance
   ```python
   import shap

   explainer = shap.TreeExplainer(xgb_model)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test)
   ```

4. **Deployment:** Create prediction API or web interface

---

## Troubleshooting

### Import Errors
If you get import errors, ensure all packages are installed:
```bash
uv add pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Memory Issues
If the dataset is too large, try:
- Reducing the number of features
- Using a smaller sample for initial exploration
- Using `low_memory=False` in `pd.read_csv()`

### Poor Model Performance
- Check for data leakage
- Verify preprocessing steps
- Try different feature engineering approaches
- Consider addressing class imbalance with SMOTE

---

## Additional Resources

- [Data Structure](data-structure.md) - Detailed dataset documentation
- [Variables Reference](variables-reference.md) - Complete variable descriptions
- [ML Strategy](machine-learning-strategy.md) - Advanced modeling techniques
