import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Poverty Risk Analysis in Spain

    Machine learning project to predict poverty risk in Spanish households using data from the Living Conditions Survey (ECV) 2024.

    /// admonition | Context.

    This project is part of [Carlos Capote](https://github.com/elcapo/)'s capstone project for the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).
    ///

    ## Project Overview

    ### Objective

    Develop a classification model to predict whether a household would be considered to be at risk of poverty using demographic and socioeconomic data. In other words, we are reverse engineering the risk poverty calculation in order to create a model that can reproduce and explain the calculation.

    ### Dataset

    Living Conditions Survey (Encuesta de Condiciones de Vida - ECV) 2024 from the Spanish National Statistics Institute (INE). The ECV is the Spanish version of the EU-SILC (European Union Statistics on Income and Living Conditions) survey.

    #### Target Variable

    * **`vhPobreza`** - Binary indicator of poverty risk (0=No, 1=Yes).

    #### Key Features

    - Household income and composition.
    - Individual demographics (age, sex, education).
    - Employment status and work patterns.
    - Health indicators.
    - Material deprivation measures.
    - Housing conditions.

    #### Dataset Summary

    The project merges these files using a **hybrid aggregation strategy** that combines household head characteristics with aggregated household statistics.

    | File | Records | Description |
    |------|---------|-------------|
    | **ECV_Th_2024** | 29,781 households | Household-level data (includes target variable) |
    | **ECV_Tp_2024** | 61,526 persons | Person-level data (demographics, employment, income) |

    To execute the merge, there is a **merge.py** script that does the job of merging the housholds and persons files into a single **data/merge.csv** file, which we'll be using as our dataset in this notebook:

    ```bash
    uv run merge.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load merged data
    df = pd.read_csv('data/merged.csv')
    df.head()
    return df, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * By aggregating people rows to the household records, we obtained a dataset with the same number of rows as households but with additional columns.

    At the same time, during the merge:

    * We also replaced negative values (which represent codes for different missing values) with _NaN_.
    * We applied some data transformations in order to keep the notebook clean.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Column Renaming

    As the default column names are quite abstract, we'll renaming taking into consideration the information that we extracted into the **docs/variables-reference.md** file. Also, we'll remove some columns that we won't use.
    """)
    return


@app.cell
def _(df):
    df_renamed = df.rename(columns={
        "HY020": "total_disposable_income",
        "HY023": "income_before_all_social_transfers",
        "head_PB150": "head_sex",
        "head_PB190": "head_marital_status",
        "head_PE021": "head_education_level",
        "head_PL051A": "head_employment_status",
        "head_PL060": "head_hours_worked_per_week",
        "head_PY010N": "head_net_employee_income",
        "head_PH010": "head_general_health_status",
        "vhPobreza": "poverty_risk",
    }).drop(columns=["HY022", "HB070", "HB100"])
    return (df_renamed,)


@app.cell
def _(df_renamed):
    df_renamed.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Target Variable

    The target variable, `vhPobreza`, which indicates whether the people in the given household lives under risk of poverty (0 = no, 1 = yes) is an imbalanced variable with about a 17% of the households in the dataset being considered to be in risk of poverty.
    """)
    return


@app.cell(hide_code=True)
def _(df_renamed, plt):
    plt.figure(figsize=(12, 6))
    df_renamed['poverty_risk'].value_counts().plot(kind='bar', alpha=0.5, color='steelblue', edgecolor='gray')
    plt.title('Poverty Risk Distribution')
    plt.xlabel('At Risk of Poverty')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No (0)', 'Yes (1)'], rotation=0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Features

    In this section, we'll take a look at the most important feature variables.

    ### Household head
    """)
    return


@app.cell
def _(plt):
    def plot_histogram(data, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(12, 6))
        data.plot.hist(bins=10, alpha=0.5, color='steelblue', edgecolor='gray', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig

    def plot_bars(data, title, xlabel, ylabel, xticklabels):
        fig, ax = plt.subplots(figsize=(12, 6))
        data.value_counts().plot(kind='bar', alpha=0.5, color='steelblue', edgecolor='gray', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(xticklabels, rotation=0)
        return fig
    return plot_bars, plot_histogram


@app.cell(hide_code=True)
def _(df_renamed, mo, plot_bars, plot_histogram):
    def plot_head_total_disposable_outcome():
        return plot_histogram(
            data=df_renamed['total_disposable_income'],
            title='Total disposable income',
            xlabel='Income (€)',
            ylabel='Count',
        )

    def plot_head_age():
        return plot_histogram(
            data=df_renamed['head_age'],
            title='Age of the household head',
            xlabel='Age (years)',
            ylabel='Count',
        )

    def plot_head_sex():
        return plot_bars(
            data=df_renamed['head_sex'],
            title='Sex',
            xlabel='Sex',
            ylabel='Count',
            xticklabels=['Man', 'Woman'],
        )

    def plot_head_marital_status():
        return plot_bars(
            data=df_renamed['head_marital_status'],
            title='Marital status',
            xlabel='Marital status',
            ylabel='Count',
            xticklabels=['Single', 'Married', 'Separated', 'Widowed', 'Divorced'],
        )

    def plot_head_hours_worked_per_week():
        return plot_histogram(
            data=df_renamed['head_hours_worked_per_week'],
            title='Hours worked per week',
            xlabel='Hours per week',
            ylabel='Count',
        )

    def plot_head_general_health_status():
        return plot_bars(
            data=df_renamed['head_general_health_status'],
            title='Health status',
            xlabel='Health status',
            ylabel='Count',
            xticklabels=['Very good', 'Good', 'Fair', 'Bad', 'Very bad'],
        )

    mo.ui.tabs({
        "Total disposable outcome": plot_head_total_disposable_outcome(),
        "Age": plot_head_age(),
        "Sex": plot_head_sex(),
        "Marital status": plot_head_marital_status(),
        "Hours Worked per Week": plot_head_hours_worked_per_week(),
        "General Health Status": plot_head_general_health_status(),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Household members
    """)
    return


@app.cell(hide_code=True)
def _(df_renamed, mo, plot_histogram):
    def plot_household_size():
        return plot_histogram(
            data=df_renamed['household_size'],
            title='Household size',
            xlabel='People',
            ylabel='Count'
        )

    def plot_mean_age():
        return plot_histogram(
            data=df_renamed['mean_age'],
            title='Mean age',
            xlabel='Age',
            ylabel='Count'
        )

    def plot_dependency_ratio():
        return plot_histogram(
            data=df_renamed['dependency_ratio'],
            title='Dependency ratio',
            xlabel='Ratio',
            ylabel='Count'
        )

    def plot_income_per_capita():
        return plot_histogram(
            data=df_renamed['income_per_capita'],
            title='Income per capita',
            xlabel='Income (€)',
            ylabel='Count'
        )

    mo.ui.tabs({
        "Household size": plot_household_size(),
        "Mean age": plot_mean_age(),
        "Dependency ratio": plot_dependency_ratio(),
        "Income per capita": plot_income_per_capita(),
    })
    return


if __name__ == "__main__":
    app.run()
