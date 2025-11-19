# Labour Force Survey (LFS) Analysis

## Introduction

This project analyzes the [Labour Force Survey](https://www.ine.es/dyngs/Prensa/EPA3T25.htm) (LFS) microdata (downloaded from [ine.es](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176918&menu=resultados&idp=1254735976595)) from the third quarter of 2025, conducted by the Spanish National Statistics Institute (INE). The primary goal is to identify and understand the sociodemographic and economic factors that most strongly predict employment status.

## Project Structure

- `preprocess.py`: Data preprocessing and cleaning functions
- `process.py`: Data processing and feature engineering
- `model_selection.py`: Model training, evaluation, and selection
- `notebook.py`: Marimo notebook with detailed analysis and visualization

## Key Features

- Machine learning models:
  - Random Forest
  - XGBoost
- Model interpretability techniques:
  - Feature importance analysis
  - SHAP (SHapley Additive exPlanations) analysis

## Data Overview

The dataset contains microdata from the Labour Force Survey, focusing on key features such as:
- Province
- Age groups
- Sex
- Marital status
- Educational level

## Main Findings

### Top Predictive Features for Employment

1. **Sex**: Being male is the most significant predictor of employment
2. **Education**: Higher education level strongly correlates with employment
3. **Marital Status**: Being married appears to be an influential factor

These questions can be better observed in our SHAP charts. Note that to read them, you must know that:

* the yellow color represents elements in the category
* the blue color represents elements that do not belong to the category
* negative values (left side) means correlation with not having a paid occupation
* positive values (right side) means correlation with having a paid occupation

#### Sex

![Man](assets/shap/sex_man.png)

#### Education

![Higher education](assets/shap/education_higher.png)
![Primary education](assets/shap/education_primary.png)

#### Marital Status

![Married](assets/shap/marital_status_married.png)
![Single](assets/shap/marital_status_single.png)

#### Provinces

![Province](assets/shap/province.png)

### Model Performance

The XGBoost model demonstrated superior performance in predicting employment status, providing deeper insights into the factors influencing employment.

## Dependencies

- Pandas
- Seaborn
- Matplotlib
- XGBoost
- SHAP
- Marimo

## Usage

To run the analysis, ensure you have the required dependencies installed and use the Marimo notebook:

```bash
marimo run notebook.py
```
