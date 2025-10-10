import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, pd, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Machine Learning Zoomcamp

    ## Module 3: **Classification**
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"

    chapters = pd.DataFrame([

        {
            "title": "Churn Prediction Project",
            "youtube_id": "0Zw04wdeTQo",
            "contents": repository_root+"03-classification/01-churn-project.md"
        },
        {
            "title": "Data Preparation",
            "youtube_id": "VSGGU9gYvdg",
            "contents": repository_root+"03-classification/02-data-preparation.md"
        },
        {
            "title": "Setting up the Validation Framework",
            "youtube_id": "_lwz34sOnSE",
            "contents": repository_root+"03-classification/03-validation.md"
        },
        {
            "title": "Exporatory Data Analysis",
            "youtube_id": "BNF1wjBwTQA",
            "contents": repository_root+"03-classification/04-eda.md"
        },
        {
            "title": "Feature Importance: Churn Rate and Risk Ratio",
            "youtube_id": "fzdzPLlvs40",
            "contents": repository_root+"03-classification/05-risk.md"
        },
        {
            "title": "Feature Importance: Mutual Information",
            "youtube_id": "_u2YaGT6RN0",
            "contents": repository_root+"03-classification/06-mutual-info.md"
        },
        {
            "title": "Feature importance: Correlation",
            "youtube_id": "mz1707QVxiY",
            "contents": repository_root+"03-classification/07-correlation.md"
        },
        {
            "title": "One-hot Encoding",
            "youtube_id": "L-mjQFN5aR0",
            "contents": repository_root+"03-classification/08-ohe.md"
        },
        {
            "title": "Logistic Regression",
            "youtube_id": "7KFE2ltnBAg",
            "contents": repository_root+"03-classification/09-logistic-regression.md"
        },
        {
            "title": "Training Logistic Regression with Scikit-Learn",
            "youtube_id": "hae_jXe2fN0",
            "contents": repository_root+"03-classification/10-training-log-reg.md"
        },
        {
            "title": "Model Interpretation",
            "youtube_id": "OUrlxnUAAEA",
            "contents": repository_root+"03-classification/11-log-reg-interpretation.md"
        },
        {
            "title": "Using the Model",
            "youtube_id": "Y-NGmnFpNuM",
            "contents": repository_root+"03-classification/12-using-log-reg.md"
        },
        {
            "title": "Summary",
            "youtube_id": "Zz6oRGsJkW4",
            "contents": repository_root+"03-classification/13-summary.md"
        },
        {
            "title": "Explore More",
            "contents": repository_root+"03-classification/14-explore-more.md"
        },
    ])

    chapters.insert(loc=0, column="snapshot", value="https://img.youtube.com/vi/"+chapters.youtube_id.astype(str)+"/hqdefault.jpg")
    chapters.insert(loc=2, column="youtube", value="https://youtube.com/watch?v="+chapters.youtube_id.astype(str))

    videos = chapters[chapters["youtube_id"].notnull()]
    videos[["snapshot", "title", "youtube"]]
    return (chapters,)


@app.cell(hide_code=True)
def _(chapters):
    contents = chapters[chapters["contents"].notnull()]
    contents[["title", "contents"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Churn Prediction Project

    We'll be working with the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, with the goal of implementing a model that's able to predict if a customer is likely to "churn" (stop using the service).

    ### The Dataset

    The raw data contains 7043 rows (customers) and 21 columns (features).
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    pd.DataFrame([
        {
            "column": "customerID",
            "description": "Customer ID"
        },
        {
            "column": "gender",
            "description": "Whether the customer is a male or a female"
        },
        {
            "column": "SeniorCitizen",
            "description": "Whether the customer is a senior citizen or not (1, 0)"
        },
        {
            "column": "Partner",
            "description": "Whether the customer has a partner or not (Yes, No)"
        },
        {
            "column": "Dependents",
            "description": "Whether the customer has dependents or not (Yes, No)"
        },
        {
            "column": "tenure",
            "description": "Number of months the customer has stayed with the company"
        },
        {
            "column": "PhoneService",
            "description": "Whether the customer has a phone service or not (Yes, No)"
        },
        {
            "column": "MultipleLines",
            "description": "Whether the customer has multiple lines or not (Yes, No, No phone service)"
        },
        {
            "column": "InternetService",
            "description": "Customer’s internet service provider (DSL, Fiber optic, No)"
        },
        {
            "column": "OnlineSecurity",
            "description": "Whether the customer has online security or not (Yes, No, No internet service)"
        },
        {
            "column": "OnlineBackup",
            "description": "Whether the customer has online backup or not (Yes, No, No internet service)"
        },
        {
            "column": "DeviceProtection",
            "description": "Whether the customer has device protection or not (Yes, No, No internet service)"
        },
        {
            "column": "TechSupport",
            "description": "Whether the customer has tech support or not (Yes, No, No internet service)"
        },
        {
            "column": "StreamingTV",
            "description": "Whether the customer has streaming TV or not (Yes, No, No internet service) "
        },
        {
            "column": "StreamingMovies",
            "description": "Whether the customer has streaming movies or not (Yes, No, No internet service)"
        },
        {
            "column": "Contract",
            "description": "The contract term of the customer (Month-to-month, One year, Two year)"
        },
        {
            "column": "PaperlessBilling",
            "description": "Whether the customer has paperless billing or not (Yes, No)"
        },
        {
            "column": "PaymentMethod",
            "description": "The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card"
        },
        {
            "column": "MonthlyCharges",
            "description": "The amount charged to the customer monthly"
        },
        {
            "column": "TotalCharges",
            "description": "The total amount charged to the customer"
        },
        {
            "column": "Churn",
            "description": "Whether the customer churned or not (Yes or No)"
        },
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Local Dataset Path

    For convenience, the dataset has already been downloaded into [module-3/data/customer-churn.csv](module-3/data/customer-churn.csv).
    ///
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("module-3/data/customer-churn.csv")

    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The Goal

    What we'll try to do is to predict the **Churn** column, which will be our target variable and represents the customers who actually left the service.

    ### The Method

    We'll train a binary classification model that will attempt to classify customers in two groups, the ones that are right about to churn and the ones who are not.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Preparation

    ### Histograms of Numeric Columns
    """
    )
    return


@app.cell
def _(df, sns):
    sns.histplot(df.SeniorCitizen, bins=15)
    return


@app.cell
def _(df, sns):
    sns.histplot(df.tenure, bins=15)
    return


@app.cell
def _(df, sns):
    sns.histplot(df.MonthlyCharges, bins=15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Standardize Column Names""")
    return


@app.cell
def _(df, pd):
    def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
        standardized = dataframe.copy()
        standardized.columns = standardized.columns.str.lower().str.replace(' ', '_')

        return standardized

    standardize_column_names(df)
    return (standardize_column_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Standardize Column Values""")
    return


@app.cell
def _(df, pd):
    def get_cateogorical_columns(dataframe: pd.DataFrame) -> list[str]:
        return list(list(dataframe.dtypes[dataframe.dtypes == 'object'].index))

    def standardize_categorical_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        standardized = dataframe.copy()

        for column in get_cateogorical_columns(standardized):
            standardized[column] = standardized[column].str.lower().str.replace(' ', '_')

        return standardized

    standardize_categorical_values(df)
    return (standardize_categorical_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prepare a Standardized Dataset""")
    return


@app.cell
def _(df, standardize_categorical_values, standardize_column_names):
    df_standardized = standardize_column_names(df)
    df_standardized = standardize_categorical_values(df_standardized)

    df_standardized
    return (df_standardized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Fix Column Data Types

    #### Total Charges

    First we'll convert the `totalcharges` column to numeric coercing (setting to null) the values that cannot be converted to numeric.
    """
    )
    return


@app.cell
def _(df_standardized, pd):
    totalcharges = pd.to_numeric(df_standardized.totalcharges, errors='coerce')

    totalcharges
    return (totalcharges,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, we'll check what values were lost during the transformation.""")
    return


@app.cell
def _(df_standardized, totalcharges):
    df_standardized[totalcharges.isnull()][["customerid", "totalcharges"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, we'll fill the empty values with zeroes.""")
    return


@app.cell(hide_code=True)
def _(df_standardized, sns, totalcharges):
    df_standardized.totalcharges = totalcharges.fillna(0)

    sns.histplot(df_standardized.totalcharges, bins=25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Churn


    Rather than a boolean column, the `churn` column contains strings with the values **yes** or **no**.
    """
    )
    return


@app.cell
def _(df_standardized):
    df_standardized.churn.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's fix it!""")
    return


@app.cell
def _(df_standardized):
    df_standardized.churn = (df_standardized.churn == 'yes').astype(int)
    return


if __name__ == "__main__":
    app.run()
