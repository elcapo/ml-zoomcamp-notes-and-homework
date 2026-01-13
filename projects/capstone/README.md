# Poverty Risk Analysis in Spain

## Living Conditions Survey (ECV) 2024 - INE

### Dataset Description

This dataset comes from the **Living Conditions Survey (Encuesta de Condiciones de Vida - ECV) 2024** from the Spanish National Statistics Institute (INE). The ECV is the Spanish version of the EU-SILC (European Union Statistics on Income and Living Conditions) survey and provides detailed information on income, living conditions, and poverty risk in Spanish households.

**Project objective:** Develop a regression model to predict the poverty risk of individuals/households in Spain using demographic and socioeconomic data.

---

## Data Structure

The dataset consists of **4 cross-sectional files** (Base 2013):

### 1. **ECV_Td_2024** - Basic Household Data

- **Records:** 29,781 households
- **Content:** Basic household-level variables
- **Format:** CSV/TAB in `data/ECV_Td_2024/CSV/`

### 2. **ECV_Tr_2024** - Basic Person Data

- **Content:** Basic individual-level variables
- **Format:** CSV/TAB in `data/ECV_Tr_2024/CSV/`

### 3. **ECV_Th_2024** - Detailed Household Data ⭐

- **Records:** 29,781 households
- **Content:** Detailed household variables including **poverty risk** (target variable)
- **Main file:** `data/ECV_Th_2024/CSV/ECV_Th_2024.tab`
- **Format:** TSV (tab-separated values)

### 4. **ECV_Tp_2024** - Detailed Adult Data ⭐

- **Records:** 61,526 adults
- **Content:** Demographic variables, education, employment, health, individual income
- **Main file:** `data/ECV_Tp_2024/CSV/ECV_Tp_2024.tab`
- **Format:** TSV (tab-separated values)

### Additional Documentation

- **Record designs:** `data/disreg_ecv_24/*.xlsx` (variable descriptions in Excel)
- **Metadata:** `data/ECV_T*/md_ECV_T*_2024.txt` (detailed data dictionaries)
- **Instructions:** `data/LeemeECV_2024.txt`

---

## Target Variable: Poverty Risk

### `vhPobreza` (Column 131 in ECV_Th_2024)

- **Location:** Detailed household file (`ECV_Th_2024`)
- **Type:** Binary (categorical)
- **Values:**
  - `0` = Not at risk of poverty
  - `1` = At risk of poverty
- **Description:** Indicates whether the household is at risk of poverty according to EU-SILC criteria

### Real Data Example:

```
Household 1: Size=5 persons, vhPobreza=1, Income=€4,600
Household 2: Size=8 persons, vhPobreza=0, Income=€42,420
Household 3: Size=10 persons, vhPobreza=0, Income=€32,569
```

### Complementary Variable: `vhMATDEP` (Column 132)

- **Description:** Material deprivation of the household
- **Values:** 0 (No) / 1 (Yes)

---

## Key Predictor Variables

### A. Income Variables (File: ECV_Th_2024)

| Variable | Column | Description |
|----------|--------|-------------|
| **HY020** | 16 | Total household disposable income (net) |
| **HY022** | 18 | Household disposable income before social transfers |
| **HY023** | 20 | Household disposable income before social transfers except pensions |
| **HY030N** | - | Imputed rent |
| **HY040N** | - | Income from property rental |
| **HY050N** | - | Family/children related allowances |
| **HY070N** | - | Income from interest, dividends, etc. |

### B. Household Demographic Variables (File: ECV_Th_2024)

| Variable | Column | Description |
|----------|--------|-------------|
| **HB070** | 8 | Number of household members (household size) |
| **HB100** | 12 | Number of persons aged 0-15 in the household |
| **HB060** | - | Household type (single person, couple with children, etc.) |

### C. Individual Demographic Variables (File: ECV_Tp_2024)

| Variable | Column | Description | Values |
|----------|--------|-------------|--------|
| **PB110** | 8 | Year of birth | - |
| **PB140** | 12 | Year of birth | - |
| **PB150** | 14 | Sex | 1=Male, 2=Female |
| **PB180** | 20 | Country of birth (spouse) | - |
| **PB190** | 22 | Marital status | 1=Single, 2=Married, 3=Separated, 4=Widowed, 5=Divorced |

### D. Education Variables (File: ECV_Tp_2024)

| Variable | Column | Description | Values |
|----------|--------|-------------|--------|
| **PE021** | 34 | Highest education level attained | 00=Less than primary<br>10=Primary<br>20=Lower secondary<br>30=Upper secondary<br>40=Post-secondary non-tertiary<br>50=Tertiary |
| **PE041** | - | Detailed education level (16-34 years) | - |

### E. Employment Variables (File: ECV_Tp_2024)

| Variable | Description |
|----------|-------------|
| **PL051A** | Current employment status |
| **PL051B** | Employment status previous year |
| **PL060** | Hours worked per week |
| **PL073-076** | Months worked full-time/part-time |
| **PL080** | Occupation (ISCO code) |

### F. Individual Income Variables (File: ECV_Tp_2024)

| Variable | Description |
|----------|-------------|
| **PY010N** | Gross employee income |
| **PY020N** | Gross self-employment income |
| **PY050N** | Unemployment benefits |
| **PY090N** | Old-age pension |
| **PY100N** | Survivor's benefit |
| **PY110N** | Sickness/disability benefits |
| **PY120N** | Widow's pension |
| **PY130N** | Family/children related allowances |
| **PY140N** | Other social benefits |

### G. Health Variables (File: ECV_Tp_2024)

| Variable | Description |
|----------|-------------|
| **PH010** | General health status (1=Very good to 5=Very bad) |
| **PH020** | Limitation in activities due to health problems |
| **PH030** | Chronic illness |

### H. Material Deprivation Variables (File: ECV_Th_2024)

| Variable | Description |
|----------|-------------|
| **HS010-HS190** | Ability to afford unexpected expenses, go on vacation, pay for housing, etc. |
| **HH010-HH090** | Housing problems (dampness, light, etc.) |

---

## Machine Learning Analysis Strategy

### 1. **Data Merging**

To create a complete dataset for the model, you'll need to merge the files:

```python
# Pseudocode
households = pd.read_csv('ECV_Th_2024.tab', sep='\t')  # Target variable here
persons = pd.read_csv('ECV_Tp_2024.tab', sep='\t')

# Merge by household ID (HB030 in households, PB030 in persons)
# Option 1: Aggregate person data by household
# Option 2: Use only head of household/main breadwinner
```

**Merge keys:**
- `HB030` (ECV_Th) ↔ `PB030` (ECV_Tp): Household ID
- `DB030` (ECV_Td) ↔ `HB030` (ECV_Th): Household ID

### 2. **Recommended Feature Engineering**

- **Age:** Calculate from year of birth (2024 - PB110)
- **Income-to-members ratio:** HY020 / HB070 (per capita income)
- **Dependency ratio:** (children + elderly) / working-age adults
- **Household employment:** Number employed / total adults
- **Maximum education level in household:** Max(PE021) per household
- **Diversified income:** Number of active income sources
- **Family composition:** Dummy variables for household types
- **Deprivation indicators:** Sum of HS/HH variables

### 3. **Possible Model Types**

#### Classification

Since `vhPobreza` is binary (0/1):
- **Logistic Regression** (interpretable baseline)
- **Random Forest / Gradient Boosting** (XGBoost, LightGBM)
- **SVM with RBF kernel**
- **Neural Networks** (if sufficient data)

#### Regression

Predict continuous income (HY020) and derive risk:
- **Linear Regression** (baseline)
- **Ridge/Lasso/ElasticNet** (with regularization)
- **Gradient Boosting Regression**

### 4. **Validation and Metrics**

For the classification problem:
- **Primary metric:** F1-Score (dataset likely imbalanced)
- **Secondary metrics:** Precision, Recall, AUC-ROC
- **Cross-validation:** Stratified (k-fold=5 or 10)
- **Analysis:** Confusion matrix, ROC curve

### 5. **Important Considerations**

#### Class Imbalance

```python
# Check vhPobreza distribution
print(households['vhPobreza'].value_counts())

# If imbalanced (likely), use:
# - SMOTE for oversampling
# - class_weight='balanced' in scikit-learn
# - StratifiedKFold for validation
```

#### Missing Values

Files use special codes:
- `-1`, `-2`, `-3`, `-4`, `-5`, `-6`: Different types of missing data
- Variables with `_F` suffix: Data quality flags

```python
# Cleaning example
df = df.replace([-1, -2, -3, -4, -5, -6], np.nan)
```

#### Multicollinearity

- Be careful with income variables (HY020, HY022, HY023) which are highly correlated
- Use VIF (Variance Inflation Factor) or eliminate redundant variables
- Consider PCA if many correlated variables

#### Feature Scaling

- Normalize/standardize continuous variables (income, age, hours worked)
- Leave categorical variables as dummy variables

---

## Quick Start Guide

### 1. Load Data

```python
import pandas as pd

households = pd.read_csv(
   'data/ECV_Th_2024/CSV/ECV_Th_2024.tab',
   sep='\t',
   encoding='latin-1'
)

persons = pd.read_csv(
   'data/ECV_Tp_2024/CSV/ECV_Tp_2024.tab',
   sep='\t',
   encoding='latin-1'
)

print(f"Households: {households.shape}")
print(f"Persons: {persons.shape}")
```

### 2. Explore Target Variable

```python
# Poverty risk distribution
print(households['vhPobreza'].value_counts())
print(households['vhPobreza'].value_counts(normalize=True))

# Relationship with income
households.groupby('vhPobreza')['HY020'].describe()
```

### 3. Select Key Variables

```python
# Household variables
vars_household = ['HB030', 'HB070', 'HB100', 'HY020', 'HY022', 'vhPobreza']
households_subset = households[vars_household]

# Person variables (head of household)
vars_person = ['PB030', 'PB150', 'PE021', 'PL051A', 'PY010N']
persons_subset = persons[vars_person]
```

### 4. Basic Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# 1. Preprocess and merge data
# 2. Handle missing values
# 3. Encode categorical variables
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Train model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Suggested Research Questions

1. **What demographic factors are most predictive of poverty risk?**
   - Age, sex, marital status, household composition

2. **How does education influence poverty risk?**
   - Compare education levels (PE021) with vhPobreza

3. **What is the relative importance of employment vs. pensions/subsidies?**
   - Analyze income sources (PY010N vs PY090N, PY050N)

4. **Is there a clear income threshold for poverty risk?**
   - Distribution of HY020 by vhPobreza

5. **Is household size a significant risk factor?**
   - Relationship between HB070 and vhPobreza controlling for income

6. **Is material deprivation (vhMATDEP) an independent predictor?**
   - Correlation between vhMATDEP and vhPobreza

---

## Additional Resources

- **INE Website:** https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176807&menu=resultados&idp=1254735976608
- **EU-SILC Methodology:** Consult European documentation on poverty and social exclusion indicators
- **Flag variables (_F):** Indicate data quality/validity (11=good, other codes=problems)

---

## Technical Notes

- **Encoding:** CSV files use `latin-1` or `ISO-8859-1` (Spanish characters with accents)
- **Separator:** TAB (`\t`) in `.tab` files
- **Sample weights:** Some variables include elevation factors for population representativeness
- **Reference year:** 2024 data may refer to income from previous year (2023)

---

## Directory Structure

```
data/
├── LeemeECV_2024.txt                    # General instructions
├── disreg_ecv_24/                       # Record designs (Excel)
│   ├── dr_ECV_CM_Td_2024.xlsx           # Td variable descriptions
│   ├── dr_ECV_CM_Th_2024.xlsx           # Th variable descriptions
│   ├── dr_ECV_CM_Tp_2024.xlsx           # Tp variable descriptions
│   └── dr_ECV_CM_Tr_2024.xlsx           # Tr variable descriptions
├── ECV_Td_2024/
│   ├── CSV/
│   │   ├── ECV_Td_2024.tab              # Basic household data (TSV)
│   │   └── esudb24d.csv                 # Alternative format (CSV)
│   └── md_ECV_Td_2024.txt               # Metadata (dictionary)
├── ECV_Th_2024/                         # Target variable
│   ├── CSV/
│   │   ├── ECV_Th_2024.tab              # Detailed household data (TSV)
│   │   └── esudb24h.csv                 # Alternative format (CSV)
│   └── md_ECV_Th_2024.txt               # Metadata (dictionary)
├── ECV_Tp_2024/                         # Demographic variables
│   ├── CSV/
│   │   ├── ECV_Tp_2024.tab              # Detailed person data (TSV)
│   │   └── esudb24p.csv                 # Alternative format (CSV)
│   └── md_ECV_Tp_2024.txt               # Metadata (dictionary)
└── ECV_Tr_2024/
    ├── CSV/
    │   ├── ECV_Tr_2024.tab              # Basic person data (TSV)
    │   └── esudb24r.csv                 # Alternative format (CSV)
    └── md_ECV_Tr_2024.txt               # Metadata (dictionary)
```

---

**Created for:** Machine Learning Project with Scikit-Learn
**Dataset:** Living Conditions Survey (ECV) 2024 - INE Spain
**Date:** January 2024
