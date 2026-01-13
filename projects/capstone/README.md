# Poverty Risk Analysis in Spain

Machine learning project to predict poverty risk in Spanish households using data from the Living Conditions Survey (ECV) 2024.

## Project Overview

### Objective

Develop a classification model to predict whether a household is at risk of poverty using demographic and socioeconomic data.

### Dataset

Living Conditions Survey (Encuesta de Condiciones de Vida - ECV) 2024 from the Spanish National Statistics Institute (INE). The ECV is the Spanish version of the EU-SILC (European Union Statistics on Income and Living Conditions) survey.

#### Target Variable

`vhPobreza` - Binary indicator of poverty risk (0=No, 1=Yes).

#### Key Features

- Household income and composition
- Individual demographics (age, sex, education)
- Employment status and work patterns
- Health indicators
- Material deprivation measures
- Housing conditions

#### Dataset Summary

| File | Records | Description |
|------|---------|-------------|
| **ECV_Th_2024** | 29,781 households | Household-level data (includes target variable) |
| **ECV_Tp_2024** | 61,526 persons | Person-level data (demographics, employment, income) |

The project merges these files using a **hybrid aggregation strategy** that combines household head characteristics with aggregated household statistics.

## Getting Started

### Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Merge household and person data:**
   ```bash
   uv run merge.py
   ```
   This generates `data/merged.csv` with all features ready for modeling.

3. **Start exploring and modeling:**
   ```python
   import pandas as pd
   df = pd.read_csv('data/merged.csv')
   ```

## Documentation

### Core Documentation

- **[Quick Start Guide](docs/quick-start.md)**: Get up and running quickly with step-by-step instructions.
- **[Data Structure](docs/data-structure.md)**: Understand the dataset structure and key variables.
- **[Machine Learning Strategy](docs/machine-learning-strategy.md)**: Recommended approaches for modeling.
- **[Variables Reference](docs/variables-reference.md)**: Complete reference for the 46 most important variables.

### Key Concepts

#### Data Merging

We use a **hybrid aggregation strategy** that combines:

- **Household head features:** Individual characteristics (age, education, employment, income).
- **Household composition:** Aggregated statistics (size, employment rate, dependency ratio).
- **Derived features:** Income per capita, age composition, employment intensity.

#### Target Variable

- **`vhPobreza`** - At risk of poverty indicator
- Located in household file (ECV_Th_2024)
- Binary: 0 (Not at risk) / 1 (At risk)

## Project Structure

```
.
├── data/                              # Dataset files
│   ├── ECV_Th_2024/                   # Household data
│   ├── ECV_Tp_2024/                   # Person data
│   └── merged.csv                     # Generated merged dataset
├── docs/                              # Documentation
│   ├── quick-start.md                 # Getting started guide
│   ├── data-structure.md              # Dataset documentation
│   ├── machine-learning-strategy.md
│   └── variables-reference.md
├── merge.py                           # Data merging script
├── pyproject.toml                     # Project dependencies
└── README.md                          # This file
```

## Modeling Approach

### Problem Type

Binary classification (predict poverty risk: 0 or 1).

### Recommended Models

1. **Logistic Regression**: Interpretable baseline.
2. **Random Forest**: Handles non-linear relationships.
3. **XGBoost/LightGBM**: Often best performance.
4. **SVM**: Good for complex decision boundaries.

### Evaluation Metrics

- **Primary**: F1-Score (handles class imbalance).
- **Secondary**: Precision, Recall, AUC-ROC.
- **Validation**: Stratified K-Fold cross-validation.

### Key Considerations

- **Class imbalance**: Use `class_weight='balanced'` or SMOTE.
- **Missing values**: Replace codes -1 to -6 with NaN.
- **Feature scaling**: Required for linear models, SVM, neural networks.
- **Multicollinearity**: Income variables are highly correlated.

## Example Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('data/merged.csv')
X = df.drop('vhPobreza', axis=1)
y = df['vhPobreza']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Research Questions

Some interesting questions to explore:

1. What demographic factors are most predictive of poverty risk?
2. How does education level influence poverty risk?
3. What is the relative importance of employment vs. social transfers?
4. Is there a clear income threshold for poverty risk?
5. Does household size matter after controlling for income?
6. Is material deprivation an independent predictor beyond income?

## Technical Notes

- **Encoding:** Data files use `latin-1` encoding (Spanish characters)
- **Separator:** TAB (`\t`) in `.tab` files
- **Missing values:** Coded as -1 to -6 (replace with NaN)
- **Reference year:** 2024 data may refer to income from 2023

## Data Source

**INE (Instituto Nacional de Estadística)**
- **Survey**: Encuesta de Condiciones de Vida (ECV) 2024
- **Website**: https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176807

## Additional Resources

- **EU-SILC Methodology:** European documentation on poverty and social exclusion indicators
- **Variable Documentation:** See `data/disreg_ecv_24/*.xlsx` for complete variable descriptions
- **Metadata:** See `data/ECV_T*/md_ECV_T*_2024.txt` for detailed data dictionaries

---

**Created for:** ML Zoomcamp Capstone Project
**Date:** January 2026
