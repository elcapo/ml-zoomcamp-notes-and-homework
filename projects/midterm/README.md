# Labour Force Survey (LFS) Analysis

## Introduction

This project analyzes the [Labour Force Survey](https://www.ine.es/dyngs/Prensa/EPA3T25.htm) (LFS) microdata (downloaded from [ine.es](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176918&menu=resultados&idp=1254735976595)) from the third quarter of 2025, conducted by the Spanish National Statistics Institute (INE). The primary goal is to identify and understand the sociodemographic and economic factors that most strongly predict employment status.

## Project Structure

- `preprocess.py`: Data preprocessing and cleaning functions
- `process.py`: Data processing and feature engineering
- `model_selection.py`: Model training, evaluation, and selection
- `notebook.py`: Marimo notebook with detailed analysis and visualization

## Key Features

This project leverages advanced machine learning techniques to understand the complex dynamics of employment prediction. We employ two powerful ensemble learning models: Random Forest and XGBoost. These algorithms are particularly well suited for capturing intricate, non-linear relationships within sociodemographic data. 

To ensure transparency and interpretability, we go beyond traditional black box modeling approaches. Our analysis incorporates model interpretability techniques, including feature importance analysis and SHAP (SHapley Additive exPlanations) analysis.

These methods allow us to not just predict employment status, but to understand the precise contributions of each feature to the model's predictions, providing actionable insights into the factors driving employment.

## Data Overview

Our analysis draws from a comprehensive microdata collection of the Labour Force Survey, meticulously compiled by the Spanish National Statistics Institute. The dataset offers a multidimensional view of employment dynamics, capturing crucial sociodemographic and economic characteristics. 

We focus on a rich set of features that provide insights into an individual's employment status. These include geographical context through province identification, demographic information such as age groups and sex, personal circumstances like marital status, and a critical factor in economic opportunity: educational level.

Each of these features serves as a lens through which we can understand the complex ecosystem of employment, revealing how various personal and structural factors intersect to influence an individual's professional trajectory.

## Main Findings

### Top Predictive Features for Employment

Our analysis reveals a landscape of employment predictors, highlighting the multifaceted nature of professional opportunities. Sex emerges as the most significant predictor, with male participants showing a substantially higher probability of employment. This finding underscores persistent gender disparities in the labor market, reflecting deeper structural inequalities.

Education stands as the second most powerful predictor, with higher education levels demonstrating a strong positive correlation with employment. This reinforces the critical role of human capital in accessing professional opportunities, suggesting that investment in education can be a powerful lever for improving employment prospects.

Marital status also reveals intriguing patterns, with married individuals showing a notable advantage in employment probability. This could be interpreted through multiple lenses: as a potential proxy for age and stability, a reflection of social networks, or an indication of broader life circumstances that might influence professional opportunities.

### Charts

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
