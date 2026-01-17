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

    Take into account that, during the merge process:

    * We also replaced negative values (which represent codes for different missing values) with _NaN_.
    * We applied some data transformations in order to keep the notebook clean.
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

    def load_dataset():
        return pd.read_csv('data/merged.csv')

    load_dataset().head()
    return load_dataset, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By aggregating people rows to the household records, we obtained a dataset with the same number of rows as households but with additional columns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalization

    As the default column names are quite abstract, we'll renaming taking into consideration the information that we extracted into the **docs/variables-reference.md** file. Also, we'll remove some columns that we won't use. We'll also use an imputer to handle missing values by filling them with their corresponding medians.
    """)
    return


@app.cell
def _(load_dataset, pd):
    from sklearn.impute import SimpleImputer

    def rename_columns(dataset):
        return dataset.rename(columns={
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

    def handle_missing_values(dataset):
        imputer = SimpleImputer(strategy='median')
        return pd.DataFrame(
            imputer.fit_transform(dataset),
            columns=dataset.columns
        )

    def normalize_dataset(dataset):
        renamed_dataset = rename_columns(dataset)
        return handle_missing_values(renamed_dataset)

    df = normalize_dataset(load_dataset())
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Target Variable

    The target variable, `vhPobreza`, which indicates whether the people in the given household lives under risk of poverty (0 = no, 1 = yes) is an imbalanced variable with about a 17% of the households in the dataset being considered to be in risk of poverty.
    """)
    return


@app.cell(hide_code=True)
def _(df, plt):
    plt.figure(figsize=(12, 6))
    df['poverty_risk'].value_counts().plot(kind='bar', alpha=0.5, color='steelblue', edgecolor='gray')
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
def _(df, mo, plot_bars, plot_histogram):
    def plot_head_total_disposable_outcome():
        return plot_histogram(
            data=df['total_disposable_income'],
            title='Total disposable income',
            xlabel='Income (€)',
            ylabel='Count',
        )

    def plot_head_age():
        return plot_histogram(
            data=df['head_age'],
            title='Age of the household head',
            xlabel='Age (years)',
            ylabel='Count',
        )

    def plot_head_sex():
        return plot_bars(
            data=df['head_sex'],
            title='Sex',
            xlabel='Sex',
            ylabel='Count',
            xticklabels=['Man', 'Woman'],
        )

    def plot_head_marital_status():
        return plot_bars(
            data=df['head_marital_status'],
            title='Marital status',
            xlabel='Marital status',
            ylabel='Count',
            xticklabels=['Single', 'Married', 'Separated', 'Widowed', 'Divorced'],
        )

    def plot_head_hours_worked_per_week():
        return plot_histogram(
            data=df['head_hours_worked_per_week'],
            title='Hours worked per week',
            xlabel='Hours per week',
            ylabel='Count',
        )

    def plot_head_general_health_status():
        return plot_bars(
            data=df['head_general_health_status'],
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
def _(df, mo, plot_histogram):
    def plot_household_size():
        return plot_histogram(
            data=df['household_size'],
            title='Household size',
            xlabel='People',
            ylabel='Count'
        )

    def plot_mean_age():
        return plot_histogram(
            data=df['mean_age'],
            title='Mean age',
            xlabel='Age',
            ylabel='Count'
        )

    def plot_dependency_ratio():
        return plot_histogram(
            data=df['dependency_ratio'],
            title='Dependency ratio',
            xlabel='Ratio',
            ylabel='Count'
        )

    def plot_income_per_capita():
        return plot_histogram(
            data=df['income_per_capita'],
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Splits

    Now that we have the dataset ready and we also have some knowledge about what it contains, we'll create our dataset splits:

    * **train**: used to train our models before comparing them
    * **val**: used to validate hyperparameters while still choosing the best model
    * **test**: used to evaluate the final model
    * **full**: used to train the production model (combines the train and validation splots)
    """)
    return


@app.cell
def _(df):
    from sklearn.model_selection import train_test_split

    def split_dataset(dataset, random_state=1):
        full, test = train_test_split(dataset, test_size=0.2, random_state=random_state)
        train, val = train_test_split(full, test_size=0.25, random_state=random_state)

        return train, full, val, test

    df_train, df_full, df_val, df_test = split_dataset(df)

    {
        "len(df_train)": len(df_train),
        "len(df_full)": len(df_full),
        "len(df_val)": len(df_val),
        "len(df_test)": len(df_test),
    }
    return df_full, df_test, df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Separating Features and Target

    We'll now separate our features from the target variable.
    """)
    return


@app.cell
def _(df_full, df_test, df_train, df_val):
    def separate_features_and_target(dataset):
        target = dataset.poverty_risk.copy()
        features = dataset.copy()
        del features["poverty_risk"]

        return features, target

    X_train, y_train = separate_features_and_target(df_train)
    X_full, y_full = separate_features_and_target(df_full)
    X_val, y_val = separate_features_and_target(df_val)
    X_test, y_test = separate_features_and_target(df_test)
    return X_full, X_test, X_train, X_val, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scale Features

    As some models operate better with scaled features, we'll train a standard scaler.
    """)
    return


@app.cell
def _(X_full, X_test, X_train, X_val):
    from sklearn.preprocessing import StandardScaler

    def scale_features(X_train, X_full, X_val, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_full_scaled = scaler.transform(X_full)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_full_scaled, X_val_scaled, X_test_scaled

    X_train_scaled, X_full_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_full, X_val, X_test)
    return X_test_scaled, X_train_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling

    In this section, we'll use different machine learning classification models to predict the poverty risk. We'll finish the section by evaluating them and comparing them.

    ### Logistic Regression

    #### Training
    """)
    return


@app.cell
def _(X_train_scaled, y_train):
    from sklearn.linear_model import LogisticRegression

    def train_logistic_regressor(X, y):
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=500,
            random_state=1,
        )

        model.fit(X, y)

        return model

    logistic_regression_model = train_logistic_regressor(X_train_scaled, y_train)
    return (logistic_regression_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Evaluate
    """)
    return


@app.cell
def _(X_test_scaled, logistic_regression_model, y_test):
    from sklearn.metrics import classification_report, f1_score, roc_auc_score

    def evaluate_classifier(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        report["f1 score"] = f1_score(y_test, y_pred)
        report["auc-roc score"] = roc_auc_score(y_test, y_proba)

        return report

    logistic_regression_evaluation = evaluate_classifier(logistic_regression_model, X_test_scaled, y_test)

    logistic_regression_evaluation
    return (evaluate_classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Precision**

    From the previous output we see that this model gets right most if its attempts to predict that a household is not in poverty risk, having a 99.23% of success at predicting them. But it has a lot less precission when it predicts households are in poverty risk, having a 75.73% success rate in those cases.

    **Recall**

    If rather than focusing at what it predicts, we focus in what's real, we see that the model correctly identified 93.35% of the households that were not in poverty risk and a 96.76% of the households that were in poverty risk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Feature Importances
    """)
    return


@app.cell
def _(X_train, logistic_regression_model, pd, plt):
    def plot_logistic_regression_feature_importance():
        logistic_regression_feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': logistic_regression_model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=True)
    
        fig, ax = plt.subplots(figsize=(12, 6))
        logistic_regression_feature_importance.tail(15).plot(x='Feature', y='Coefficient', kind='barh', ax=ax)
        ax.set_title('Logistic regression - Feature importances')
        ax.set_xlabel('Coefficient')
        plt.show()

    plot_logistic_regression_feature_importance()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forest Classifier

    #### Train
    """)
    return


@app.cell
def _(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier

    def train_random_forest_classifier(X, y):
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=1,
            n_jobs=-1,
        )

        model.fit(X, y)

        return model

    random_forest_model = train_random_forest_classifier(X_train, y_train)
    return (random_forest_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Evaluate
    """)
    return


@app.cell
def _(X_test, evaluate_classifier, random_forest_model, y_test):
    random_forest_evaluation = evaluate_classifier(random_forest_model, X_test, y_test)

    random_forest_evaluation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Precision**

    From the previous output we see that this model also gets right most if its attempts to predict that a household is not in poverty risk, having a 98.20% of success at predicting them. But this time, the model also has a very  high precission when it predicts households are in poverty risk, having a 97.37% success rate in those cases.

    **Recall**

    If rather than focusing at what it predicts, we focus in what's real, we see that the model correctly identified 99.47% of the households that were not in poverty risk and a 91.53% of the households that were in poverty risk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Feature Importances
    """)
    return


@app.cell
def _(X_train, pd, plt, random_forest_model):
    def plot_random_forest_feature_importance():
        random_forest_feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': random_forest_model.feature_importances_
        }).sort_values('Importance', ascending=True)
    
        fig, ax = plt.subplots(figsize=(12, 6))
        random_forest_feature_importance.tail(15).plot(x='Feature', y='Importance', kind='barh', ax=ax)
        ax.set_title('Random forest - Feature importances')
        ax.set_xlabel('Importance')
        plt.show()

    plot_random_forest_feature_importance()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### XGBoost

    #### Train
    """)
    return


@app.cell
def _(X_train, y_train):
    from xgboost import XGBClassifier

    def train_xgboost_classifier():
        class_ratio = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            scale_pos_weight=class_ratio,
            random_state=1,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        return model

    xgboost_classifier_model = train_xgboost_classifier()
    return (xgboost_classifier_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Evaluate
    """)
    return


@app.cell
def _(X_test, evaluate_classifier, xgboost_classifier_model, y_test):
    xgboost_classifier_evaluation = evaluate_classifier(xgboost_classifier_model, X_test, y_test)

    xgboost_classifier_evaluation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Precision**

    From the previous output we see that this model also gets right most if its attempts to predict that a household is not in poverty risk, having a 98.83% of success at predicting them. But this time, the model also has a very  high precission when it predicts households are in poverty risk, having a 93.42% success rate in those cases.

    **Recall**

    If rather than focusing at what it predicts, we focus in what's real, we see that the model correctly identified 98.57% of the households that were not in poverty risk and a 94.57% of the households that were in poverty risk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparisson

    | Metric | Logistic regression (sensible) | Random forest (conservative) | XGBoost (balanced) |
    | --- | --- | --- | --- |
    | Precision (class 1) | 0.76 | 0.97 | 0.93 |
    | Recall (class 1) | 0.97 | 0.92 | 0.95 |
    | F1 (class 1) | 0.85 | 0.94 | 0.94 |
    | False negatives | Very few | More | Few |
    | False positives | More | Very few | Some |
    | Accuracy global | 0.94 | 0.98 | 0.98 |
    | AUC-ROC | 0.991 | 0.994 | 0.996 |
    | Profile | Inclusive | Restrictive | Balanced |
    """)
    return


if __name__ == "__main__":
    app.run()
