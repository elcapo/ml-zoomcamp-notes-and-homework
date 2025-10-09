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
    return mo, pd


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


if __name__ == "__main__":
    app.run()
