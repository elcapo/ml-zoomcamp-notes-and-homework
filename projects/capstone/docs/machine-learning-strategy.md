# Machine Learning Strategy

## Problem Type

**Binary Classification**

Since `vhPobreza` is binary (0/1), we're predicting whether a household is at risk of poverty.

---

## Data Merging Strategy

### Hybrid Aggregation Approach

We use a **hybrid strategy** that combines:
1. **Household head characteristics** (individual profile)
2. **Aggregated household statistics** (family composition)
3. **Derived features** (employment rate, dependency ratio, etc.)

This provides the best balance between capturing individual profiles and household dynamics.

#### Part 1: Household Head Features
Extract characteristics of the first person in each household:
- Age, sex, marital status
- Education level
- Employment status and work hours
- Personal income
- Health status

#### Part 2: Household Composition
Aggregate statistics across all household members:
- Household size
- Average age and standard deviation
- Number of males/females
- Maximum education level
- Number employed
- Total employee income and pensions
- Average health

#### Part 3: Derived Features
Calculate meaningful metrics:
- **Employment rate:** employed members / household size
- **Age composition:** children (<18), elderly (≥65), working-age adults
- **Dependency ratio:** (children + elderly) / working-age adults
- **Income per capita:** total income / household size

---

## Recommended Feature Engineering

### Age-Related
- Calculate age from birth year: `2024 - PB110`
- Create age groups: 0-18, 19-35, 36-50, 51-65, 65+

### Income-Related
- **Income-to-members ratio:** `HY020 / HB070` (per capita income)
- **Diversified income:** Count number of active income sources
- **Income concentration:** Standard deviation / mean within household

### Household Composition
- **Dependency ratio:** `(children + elderly) / working-age adults`
- **Household employment:** `Number employed / total adults`
- **Maximum education level in household:** `max(PE021)` per household

### Material Deprivation
- **Deprivation index:** Sum of HS/HH binary indicators
- **Essential needs:** Combine heating, food, emergency funds

---

## Model Selection

### Classification Models

#### Baseline
- **Logistic Regression**
  - Interpretable coefficients
  - Fast training
  - Good for understanding feature importance

#### Advanced Models
- **Random Forest**
  - Handles non-linear relationships
  - Feature importance built-in
  - Robust to outliers

- **Gradient Boosting** (XGBoost, LightGBM, CatBoost)
  - Often best performance
  - Good with imbalanced data
  - Handles missing values well

- **SVM with RBF kernel**
  - Good for complex decision boundaries

- **Neural Networks**
  - If sufficient data (may be overkill)

### Alternative: Regression Approach

Predict continuous income (`HY020`) and derive risk threshold:
- **Linear Regression** (baseline)
- **Ridge/Lasso/ElasticNet** (with regularization)
- **Gradient Boosting Regression**

---

## Validation Strategy

### Cross-Validation
- **Method:** Stratified K-Fold (k=5 or 10)
- **Why stratified:** Maintains class distribution in each fold
- **Metric:** Track consistency across folds

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

## Evaluation Metrics

### Primary Metric
- **F1-Score** (balanced between precision and recall)
  - Important if dataset is imbalanced
  - Harmonic mean of precision and recall

### Secondary Metrics
- **Precision:** Of predicted poor households, how many are actually poor?
- **Recall:** Of actual poor households, how many did we identify?
- **AUC-ROC:** Overall model discrimination ability
- **Confusion Matrix:** Detailed breakdown of predictions

### Metric Selection Rationale
```python
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

---

## Important Considerations

### 1. Class Imbalance

Check target distribution:
```python
print(y.value_counts())
print(y.value_counts(normalize=True))
```

**If imbalanced, use:**
- SMOTE for oversampling minority class
- `class_weight='balanced'` in scikit-learn models
- Stratified sampling in cross-validation
- Adjust classification threshold based on business needs

### 2. Missing Values

Replace special codes with NaN:
```python
import numpy as np
df = df.replace([-1, -2, -3, -4, -5, -6], np.nan)
```

**Imputation strategies:**
- Median for numeric features
- Mode for categorical features
- Advanced: KNN imputation or iterative imputation
- Consider adding "missing" indicator features

### 3. Multicollinearity

**Income variables are highly correlated:**
- `HY020`, `HY022`, `HY023` (different income definitions)
- Consider using only `HY020` (total disposable income)
- Check VIF (Variance Inflation Factor)
- Use Ridge/Lasso regularization
- Consider PCA if many correlated variables

### 4. Feature Scaling

**Normalize/standardize continuous variables:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
```

**When to scale:**
- Always for: Linear models, SVM, Neural Networks
- Not necessary for: Tree-based models (RF, XGBoost)

### 5. Categorical Encoding

**For categorical variables:**
- **One-hot encoding** for nominal variables (sex, marital status)
- **Ordinal encoding** for ordered variables (education level)
- **Label encoding** for tree-based models

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)
```

---

## Pipeline Template

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Define feature types
numeric_features = ['HY020', 'HB070', 'head_age', 'employment_rate', ...]
categorical_features = ['head_PB150', 'head_PB190', ...]

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Suggested Research Questions

1. **What demographic factors are most predictive of poverty risk?**
   - Analyze feature importance for age, sex, marital status, household composition

2. **How does education influence poverty risk?**
   - Compare education levels (PE021) with vhPobreza
   - Does maximum household education matter more than average?

3. **What is the relative importance of employment vs. pensions/subsidies?**
   - Analyze income sources (PY010N vs PY090N, PY050N)
   - Does income source diversity reduce risk?

4. **Is there a clear income threshold for poverty risk?**
   - Distribution of HY020 by vhPobreza
   - ROC curve analysis for income-based prediction

5. **Is household size a significant risk factor?**
   - Relationship between HB070 and vhPobreza controlling for income
   - Interaction between household size and income

6. **Is material deprivation (vhMATDEP) an independent predictor?**
   - Correlation between vhMATDEP and vhPobreza
   - Does it add predictive power beyond income?

---

## Model Comparison Strategy

1. **Start simple:** Logistic Regression baseline
2. **Add complexity:** Random Forest, XGBoost
3. **Compare models:** Cross-validation scores
4. **Feature importance:** Identify key predictors
5. **Hyperparameter tuning:** Grid search on best model
6. **Final evaluation:** Test set performance

```python
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'XGBoost': XGBClassifier(scale_pos_weight=class_ratio)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='f1')
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Next Steps

1. Run `merge.py` to create merged dataset
2. Perform exploratory data analysis (EDA)
3. Build baseline Logistic Regression model
4. Iterate with feature engineering
5. Try advanced models (Random Forest, XGBoost)
6. Hyperparameter tuning
7. Final model evaluation and interpretation
