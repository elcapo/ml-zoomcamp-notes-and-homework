# Data Structure

## Overview

The Living Conditions Survey (ECV) 2024 dataset consists of **4 cross-sectional files** (Base 2013):

### Files

| File | Records | Description | Format |
|------|---------|-------------|--------|
| **ECV_Td_2024** | 29,781 | Basic household data | CSV/TAB |
| **ECV_Tr_2024** | - | Basic person data | CSV/TAB |
| **ECV_Th_2024** (*) | 29,781 | Detailed household data (includes target variable) | TSV |
| **ECV_Tp_2024** (*) | 61,526 | Detailed adult data (demographics, employment, income) | TSV |

(*) Primary files used for analysis

## File Relationships

### Household-Person Linking

- **Households:** `HB030` = Household ID (1, 2, 3, ...)
- **Persons:** `PB030` = Person ID (101, 102, 201, 202, ...)
- **Relationship:** `Household ID = floor(PB030 / 100)`

**Example:**
- Household 1 → Persons: 101, 102
- Household 2 → Persons: 201, 202, 203
- Household 3 → Persons: 301, 302

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

### Complementary Variable

- **`vhMATDEP`** (Column 132): Material deprivation of the household (0=No, 1=Yes)

## Key Predictor Variables

### A. Income Variables (File: ECV_Th_2024)

| Variable | Description |
|----------|-------------|
| **HY020** | Total household disposable income (net) - **Most important predictor** |
| **HY022** | Household disposable income before social transfers (except pensions) |
| **HY023** | Household disposable income before all social transfers |
| **HY030N** | Imputed rent |
| **HY040N** | Income from property rental |
| **HY050N** | Family/children related allowances |
| **HY060N** | Social assistance |
| **HY070N** | Housing allowances |

### B. Household Demographics (File: ECV_Th_2024)

| Variable | Description |
|----------|-------------|
| **HB070** | Number of household members (household size) |
| **HB100** | Number of persons aged 0-15 in the household |
| **HB060** | Household type (single person, couple with children, etc.) |

### C. Individual Demographics (File: ECV_Tp_2024)

| Variable | Description | Values |
|----------|-------------|--------|
| **PB140** | Year of birth | 1950-2008 |
| **PB150** | Sex | 1=Male, 2=Female |
| **PB190** | Marital status | 1=Single, 2=Married, 3=Separated, 4=Widowed, 5=Divorced |

### D. Education (File: ECV_Tp_2024)

| Variable | Description | Values |
|----------|-------------|--------|
| **PE021** | Highest education level attained | 0=Less than primary, 10=Primary, 20=Lower secondary, 30=Upper secondary, 40=Post-secondary, 50=Tertiary |

### E. Employment (File: ECV_Tp_2024)

| Variable | Description |
|----------|-------------|
| **PL051A** | Current employment status |
| **PL060** | Hours worked per week |
| **PL073-076** | Months worked full-time/part-time |
| **PL080** | Occupation (ISCO code) |

### F. Individual Income (File: ECV_Tp_2024)

All refer to previous year, net amounts in €:

| Variable | Description |
|----------|-------------|
| **PY010N** | Net employee income |
| **PY020N** | Net self-employment income |
| **PY050N** | Unemployment benefits |
| **PY090N** | Old-age pension |
| **PY100N** | Survivor's benefit |
| **PY110N** | Sickness/disability benefits |

### G. Health (File: ECV_Tp_2024)

| Variable | Description | Values |
|----------|-------------|--------|
| **PH010** | General health status | 1=Very good, 2=Good, 3=Fair, 4=Bad, 5=Very bad |
| **PH020** | Chronic illness | 1=Yes, 2=No |
| **PH030** | Activity limitation | 1=Severely limited, 2=Limited, 3=Not limited |

### H. Material Deprivation (File: ECV_Th_2024)

| Variable | Description | Values |
|----------|-------------|--------|
| **HS040** | Can afford vacation | 1=Yes, 2=No |
| **HS050** | Can afford protein meals | 1=Yes, 2=No |
| **HS060** | Can face unexpected expenses | 1=Yes, 2=No |
| **HS090** | Has computer | 1=Yes, 2=No |
| **HS110** | Can afford car | 1=Yes, 2=No |

### I. Housing (File: ECV_Th_2024)

| Variable | Description | Values |
|----------|-------------|--------|
| **HH010** | Type of dwelling | 1=Detached, 2=Semi-detached, 3=Apartment |
| **HH030** | Number of rooms | Numeric (1-10+) |
| **HH040** | Leaking roof/damp | 1=Yes, 2=No |
| **HH050** | Can keep home warm | 1=Yes, 2=No |
| **HH070** | Housing costs (rent/mortgage + utilities) | Numeric (€) |

## Missing Values

Files use special codes for missing data:
- `-1` = Missing
- `-2` = Not applicable
- `-3` = Not selected respondent
- `-4` = Not able to establish
- `-5` = Not filled (gross series filled instead)
- `-6` = Mix of values and missing

**Cleaning example:**
```python
df = df.replace([-1, -2, -3, -4, -5, -6], np.nan)
```

## Variable Conventions

### Suffixes
- **_F** = Flag variable (data quality indicator)
  - `1` or `11` = Good quality data
  - Negative values = Issues
- **N** = Net amount (after taxes)
- **G** = Gross amount (before taxes)

### Prefixes
- **P** = Person level
- **H** = Household level
- **B** = Basic information
- **E** = Education
- **L** = Labor/Employment
- **Y** = Income (Yield)
- **H** (second position) = Health
- **S** = Material deprivation (Shortage)

## Directory Structure

```
data/
├── LeemeECV_2024.txt                    # General instructions
├── disreg_ecv_24/                       # Record designs (Excel)
│   ├── dr_ECV_CM_Td_2024.xlsx           # Variable descriptions
│   ├── dr_ECV_CM_Th_2024.xlsx
│   ├── dr_ECV_CM_Tp_2024.xlsx
│   └── dr_ECV_CM_Tr_2024.xlsx
├── ECV_Td_2024/
│   ├── CSV/
│   │   └── ECV_Td_2024.tab              # Basic household data (TSV)
│   └── md_ECV_Td_2024.txt               # Metadata
├── ECV_Th_2024/                         # Contains target variable
│   ├── CSV/
│   │   └── ECV_Th_2024.tab              # Detailed household data (TSV)
│   └── md_ECV_Th_2024.txt
├── ECV_Tp_2024/                         # Contains demographic variables
│   ├── CSV/
│   │   └── ECV_Tp_2024.tab              # Detailed person data (TSV)
│   └── md_ECV_Tp_2024.txt
└── ECV_Tr_2024/
    ├── CSV/
    │   └── ECV_Tr_2024.tab              # Basic person data (TSV)
    └── md_ECV_Tr_2024.txt
```

## Technical Notes

- **Encoding:** CSV files use `latin-1` or `ISO-8859-1` (Spanish characters with accents)
- **Separator:** TAB (`\t`) in `.tab` files
- **Sample weights:** Some variables include elevation factors for population representativeness
- **Reference year:** 2024 data may refer to income from previous year (2023)

## Additional Documentation

- **Record designs:** `data/disreg_ecv_24/*.xlsx` (variable descriptions in Excel)
- **Metadata:** `data/ECV_T*/md_ECV_T*_2024.txt` (detailed data dictionaries)
- **Complete variable reference:** See [variables-reference.md](variables-reference.md) for detailed information on the 46 most important variables
