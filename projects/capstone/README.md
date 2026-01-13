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
| **PE021** | 34 | Highest education level attained | 00=Less than primary, 10=Primary, 20=Lower secondary, 30=Upper secondary, 40=Post-secondary non-tertiary, 50=Tertiary |
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

To create a complete dataset for the model, we use a **hybrid aggregation strategy** that combines:
- Individual characteristics of the household head
- Aggregated statistics for the entire household

This approach provides a good balance between capturing individual profiles and household dynamics.

**Merge keys:**
- `HB030` (ECV_Th) = Household ID
- `PB030` (ECV_Tp) = Person ID (household_id = floor(PB030 / 100))
- Example: Persons 101, 102 → Household 1

#### Hybrid Aggregation Strategy

The merging process consists of three parts:

**Part 1: Household Head Features**
- Extract characteristics of the first person (household head) in each household
- Features: age, sex, marital status, education, employment status, work hours, income, health

**Part 2: Household Composition**
- Aggregate person-level data across all household members
- Statistics: household size, average age, number of males, maximum education, number employed
- Income totals: employee income, pensions

**Part 3: Derived Features**
- Employment rate: employed members / household size
- Age-based composition: children (<18), elderly (≥65), working-age adults
- Dependency ratio: (children + elderly) / working-age adults
- Income per capita: total income / household size

#### Implementation

Use the provided `merge_data.py` script:

```bash
python merge_data.py
```

This generates `merged_data.csv` with:
- Household-level features (income, size, etc.)
- Household head characteristics (with `head_` prefix)
- Aggregated household composition features
- Target variable: `vhPobreza`

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

## Complete Variable Reference

### Quick Reference: Most Important Variables

Below is a curated list of the 46 most important variables for poverty risk analysis:

#### Target Variables

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **vhPobreza** | Household | At risk of poverty | 0=No, 1=Yes |
| **vhMATDEP** | Household | Severe material deprivation | 0=No, 1=Yes |

#### Demographics

| Variable | File | Description | Type | Values |
|----------|------|-------------|------|--------|
| **PB140** | Person | Year of birth | Numeric | 1950-2008 |
| **PB150** | Person | Sex | Categorical | 1=Male, 2=Female |
| **PB190** | Person | Marital status | Categorical | 1=Single, 2=Married, 3=Separated, 4=Widowed, 5=Divorced |
| **HB120** | Household | Number of household members | Numeric | 1-10+ |

#### Education

| Variable | File | Description | Type | Values |
|----------|------|-------------|------|--------|
| **PE021** | Person | Education level | Ordinal | 0=Less than primary, 10=Primary, 20=Lower secondary, 30=Upper secondary, 40=Post-secondary, 50=Tertiary |

#### Employment

| Variable | File | Description | Type |
|----------|------|-------------|------|
| **PL051A** | Person | Current employment status | Categorical (ISCO08) |
| **PL060** | Person | Hours worked per week | Numeric (0-80) |
| **PL073** | Person | Months worked full-time (employee) | Numeric (0-12) |
| **PL074** | Person | Months worked part-time (employee) | Numeric (0-12) |
| **PL075** | Person | Months worked full-time (self-employed) | Numeric (0-12) |
| **PL076** | Person | Months worked part-time (self-employed) | Numeric (0-12) |
| **PL080** | Person | Months unemployed | Numeric (0-12) |

#### Individual Income (All refer to previous year, net amounts in €)

| Variable | File | Description |
|----------|------|-------------|
| **PY010N** | Person | Net employee income |
| **PY020N** | Person | Net non-monetary employee income |
| **PY050N** | Person | Net self-employment income |
| **PY090N** | Person | Unemployment benefits |
| **PY100N** | Person | Retirement pension |
| **PY110N** | Person | Survivor's benefits |
| **PY120N** | Person | Sickness benefits |
| **PY130N** | Person | Disability benefits |
| **PY140N** | Person | Education allowances |

#### Household Income (All refer to previous year, net amounts in €)

| Variable | File | Description |
|----------|------|-------------|
| **HY020** | Household | **Total disposable income** (most important predictor) |
| **HY022** | Household | Income before social transfers (except pensions) |
| **HY023** | Household | Income before all social transfers |
| **HY030N** | Household | Imputed rent |
| **HY040N** | Household | Rental income |
| **HY050N** | Household | Family/children allowances |
| **HY060N** | Household | Social assistance |
| **HY070N** | Household | Housing allowances |

#### Health

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **PH010** | Person | General health status | 1=Very good, 2=Good, 3=Fair, 4=Bad, 5=Very bad |
| **PH020** | Person | Chronic illness | 1=Yes, 2=No |
| **PH030** | Person | Activity limitation | 1=Severely limited, 2=Limited, 3=Not limited |

#### Material Deprivation

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **HS040** | Household | Can afford vacation | 1=Yes, 2=No |
| **HS050** | Household | Can afford protein meals | 1=Yes, 2=No |
| **HS060** | Household | Can face unexpected expenses | 1=Yes, 2=No |
| **HS090** | Household | Has computer | 1=Yes, 2=No |
| **HS110** | Household | Can afford car | 1=Yes, 2=No |

#### Housing

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **HH010** | Household | Type of dwelling | 1=Detached, 2=Semi-detached, 3=Apartment |
| **HH030** | Household | Number of rooms | Numeric (1-10+) |
| **HH040** | Household | Leaking roof/damp | 1=Yes, 2=No |
| **HH050** | Household | Can keep home warm | 1=Yes, 2=No |
| **HH070** | Household | Housing costs (rent/mortgage + utilities) | Numeric (€) |

#### Linking Variables

| Variable | File | Description |
|----------|------|-------------|
| **PB030** | Person | Person ID = household_id × 100 + person_number |
| **HB030** | Household | Household ID |

**Relationship:** `household_id = floor(PB030 / 100)`
- Example: Person 101 → Household 1, Person 202 → Household 2

### Variable Conventions

#### Missing Values
Replace negative codes with NaN before analysis:
```python
df = df.replace([-1, -2, -3, -4, -5, -6], np.nan)
```

Codes meaning:
- `-1` = Missing
- `-2` = Not applicable
- `-3` = Not selected respondent
- `-4` = Not able to establish
- `-5` = Not filled (gross series filled instead)
- `-6` = Mix of values and missing

#### Variable Suffixes
- **_F** = Flag variable (data quality indicator)
  - `1` or `11` = Good quality data
  - Negative values = Issues (see missing values above)
- **N** = Net amount (after taxes)
- **G** = Gross amount (before taxes)

#### Variable Prefixes
- **P** = Person level
- **H** = Household level
- **B** = Basic information
- **E** = Education
- **L** = Labor/Employment
- **Y** = Income (Yield)
- **H** (second position) = Health
- **S** = Material deprivation (Shortage)

### Calculating Age
```python
age = 2024 - PB140  # Current age
```

### Feature Engineering Examples

**Dependency Ratio:**
```python
dependency_ratio = (num_children + num_elderly) / num_working_age
```

**Income Per Capita:**
```python
income_per_capita = HY020 / HB120
```

**Employment Intensity:**
```python
work_months = PL073 + PL074 + PL075 + PL076
employment_intensity = work_months / 12
```

**Material Deprivation Score:**
```python
deprivation_score = (
    (HS040 == 2) +  # No vacation
    (HS050 == 2) +  # No protein meals
    (HS060 == 2) +  # No emergency funds
    (HS110 == 2) +  # No car
    (HH050 == 2)    # Cannot heat home
)
```

**Income Diversification:**
```python
income_sources = sum([
    PY010N > 0,  # Employee income
    PY050N > 0,  # Self-employment
    PY090N > 0,  # Unemployment
    PY100N > 0,  # Pension
    HY040N > 0,  # Rental income
])
```

### Full Variable Documentation

- **Person variables:** 208 total variables in `data/disreg_ecv_24/dr_ECV_CM_Tp_2024.xlsx`
- **Household variables:** 187 total variables in `data/disreg_ecv_24/dr_ECV_CM_Th_2024.xlsx`
- **Quick reference CSV:** See `quick_reference_variables.csv` for the 46 most important variables

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
