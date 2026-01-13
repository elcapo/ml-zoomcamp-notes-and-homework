# Complete Variable Reference

This document provides detailed information on the 46 most important variables for poverty risk analysis.

---

## Target Variables

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **vhPobreza** | Household | At risk of poverty | 0=No, 1=Yes |
| **vhMATDEP** | Household | Severe material deprivation | 0=No, 1=Yes |

---

## Demographics

| Variable | File | Description | Type | Values |
|----------|------|-------------|------|--------|
| **PB140** | Person | Year of birth | Numeric | 1950-2008 |
| **PB150** | Person | Sex | Categorical | 1=Male, 2=Female |
| **PB190** | Person | Marital status | Categorical | 1=Single, 2=Married, 3=Separated, 4=Widowed, 5=Divorced |
| **HB120** | Household | Number of household members | Numeric | 1-10+ |

---

## Education

| Variable | File | Description | Type | Values |
|----------|------|-------------|------|--------|
| **PE021** | Person | Education level | Ordinal | 0=Less than primary, 10=Primary, 20=Lower secondary, 30=Upper secondary, 40=Post-secondary, 50=Tertiary |

---

## Employment

| Variable | File | Description | Type |
|----------|------|-------------|------|
| **PL051A** | Person | Current employment status | Categorical (ISCO08) |
| **PL060** | Person | Hours worked per week | Numeric (0-80) |
| **PL073** | Person | Months worked full-time (employee) | Numeric (0-12) |
| **PL074** | Person | Months worked part-time (employee) | Numeric (0-12) |
| **PL075** | Person | Months worked full-time (self-employed) | Numeric (0-12) |
| **PL076** | Person | Months worked part-time (self-employed) | Numeric (0-12) |
| **PL080** | Person | Months unemployed | Numeric (0-12) |

---

## Individual Income

All refer to previous year, net amounts in €:

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

---

## Household Income

All refer to previous year, net amounts in €:

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

---

## Health

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **PH010** | Person | General health status | 1=Very good, 2=Good, 3=Fair, 4=Bad, 5=Very bad |
| **PH020** | Person | Chronic illness | 1=Yes, 2=No |
| **PH030** | Person | Activity limitation | 1=Severely limited, 2=Limited, 3=Not limited |

---

## Material Deprivation

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **HS040** | Household | Can afford vacation | 1=Yes, 2=No |
| **HS050** | Household | Can afford protein meals | 1=Yes, 2=No |
| **HS060** | Household | Can face unexpected expenses | 1=Yes, 2=No |
| **HS090** | Household | Has computer | 1=Yes, 2=No |
| **HS110** | Household | Can afford car | 1=Yes, 2=No |

---

## Housing

| Variable | File | Description | Values |
|----------|------|-------------|--------|
| **HH010** | Household | Type of dwelling | 1=Detached, 2=Semi-detached, 3=Apartment |
| **HH030** | Household | Number of rooms | Numeric (1-10+) |
| **HH040** | Household | Leaking roof/damp | 1=Yes, 2=No |
| **HH050** | Household | Can keep home warm | 1=Yes, 2=No |
| **HH070** | Household | Housing costs (rent/mortgage + utilities) | Numeric (€) |

---

## Linking Variables

| Variable | File | Description |
|----------|------|-------------|
| **PB030** | Person | Person ID = household_id × 100 + person_number |
| **HB030** | Household | Household ID |

**Relationship:** `household_id = floor(PB030 / 100)`
- Example: Person 101 → Household 1, Person 202 → Household 2

---

## Feature Engineering Examples

### Calculating Age
```python
age = 2024 - PB140  # Current age
```

### Dependency Ratio
```python
dependency_ratio = (num_children + num_elderly) / num_working_age
```

### Income Per Capita
```python
income_per_capita = HY020 / HB120
```

### Employment Intensity
```python
work_months = PL073 + PL074 + PL075 + PL076
employment_intensity = work_months / 12
```

### Material Deprivation Score
```python
deprivation_score = (
    (HS040 == 2) +  # No vacation
    (HS050 == 2) +  # No protein meals
    (HS060 == 2) +  # No emergency funds
    (HS110 == 2) +  # No car
    (HH050 == 2)    # Cannot heat home
)
```

### Income Diversification
```python
income_sources = sum([
    PY010N > 0,  # Employee income
    PY050N > 0,  # Self-employment
    PY090N > 0,  # Unemployment
    PY100N > 0,  # Pension
    HY040N > 0,  # Rental income
])
```

---

## Full Documentation

For complete variable documentation:
- **Person variables:** 208 total variables in `data/disreg_ecv_24/dr_ECV_CM_Tp_2024.xlsx`
- **Household variables:** 187 total variables in `data/disreg_ecv_24/dr_ECV_CM_Th_2024.xlsx`
