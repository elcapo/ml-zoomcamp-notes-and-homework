import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Machine Learning Zoomcamp

    ## Module 2: **Regression**
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"

    chapters = pd.DataFrame([
        {
            "title": "Car price prediction project",
            "youtube_id": "vM3SqPNlStE",
            "contents": repository_root+"02-regression/01-car-price-intro.md"
        },
        {
            "title": "Data preparation",
            "youtube_id": "Kd74oR4QWGM",
            "contents": repository_root+"02-regression/02-data-preparation.md"
        },
        {
            "title": "Exploratory data analysis",
            "youtube_id": "k6k8sQ0GhPM",
            "contents": repository_root+"02-regression/03-eda.md"
        },
        {
            "title": "Setting up the validation framework",
            "youtube_id": "ck0IfiPaQi0",
            "contents": repository_root+"02-regression/04-validation-framework.md"
        },
        {
            "title": "Linear regression",
            "youtube_id": "Dn1eTQLsOdA",
            "contents": repository_root+"02-regression/05-linear-regression-simple.md"
        },
        {
            "title": "Linear regression: vector form",
            "youtube_id": "YkyevnYyAww",
            "contents": repository_root+"02-regression/06-linear-regression-vector.md"
        },
        {
            "title": "Training linear regression: Normal equation",
            "youtube_id": "hx6nak-Y11g",
            "contents": repository_root+"02-regression/07-linear-regression-training.md"
        },
        {
            "title": "Baseline model for car price prediction project",
            "youtube_id": "SvPpMMYtYbU",
            "contents": repository_root+"02-regression/08-baseline-model.md"
        },
        {
            "title": "Root mean squared error",
            "youtube_id": "0LWoFtbzNUM",
            "contents": repository_root+"02-regression/09-rmse.md"
        },
        {
            "title": "Using RMSE on validation data",
            "youtube_id": "rawGPXg2ofE",
            "contents": repository_root+"02-regression/10-car-price-validation.md"
        },
        {
            "title": "Feature engineering",
            "youtube_id": "-aEShw4ftB0",
            "contents": repository_root+"02-regression/11-feature-engineering.md"
        },
        {
            "title": "Categorical variables",
            "youtube_id": "sGLAToAAMa4",
            "contents": repository_root+"02-regression/12-categorical-variables.md"
        },
        {
            "title": "Regularization",
            "youtube_id": "91ve3EJlHBc",
            "contents": repository_root+"02-regression/13-regularization.md"
        },
        {
            "title": "Tuning the model",
            "youtube_id": "lW-YVxPgzQw",
            "contents": repository_root+"02-regression/14-tuning-model.md"
        },
        {
            "title": "Using the model",
            "youtube_id": "KT--uIJozes",
            "contents": repository_root+"02-regression/15-using-model.md"
        },
        {
            "title": "Car price prediction project summary",
            "youtube_id": "_qI01YXbyro",
            "contents": repository_root+"02-regression/16-summary.md"
        },
        {
            "title": "Explore more",
            "youtube_id": "",
            "contents": repository_root+"02-regression/17-explore-more.md"
        },
        {
            "title": "Homework",
            "youtube_id": "",
            "contents": repository_root+"02-regression/homework.md"
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
    ## Car Price Prediction Project

    In this chapter we'll be working with the [Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset) Kaggle dataset.

    1. Prepare the data and do some Exploratory Data Analysis (EDA)
    2. Use linear regression for predicting price
    3. Understand the internals of linear regression
    4. Evaluate the model using Root Mean Square Error (RMSE)
    5. Do some feature engineering
    6. Regualization
    7. Use the model
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Preparation

    To facilitate the analysis, the CSV file has been already downloaded in **data/car-prices.csv**.
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    car_prices = pd.read_csv('./module-2/data/car-prices.csv')
    car_prices.head()
    return (car_prices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Standardize Column Names

    Some of the columns are named using spaces as separator and other columns were named using underscores as a separator. We'll now standardize their names.
    """
    )
    return


@app.cell(hide_code=True)
def _(car_prices):
    car_prices.columns = car_prices.columns.str.lower().str.replace(' ', '_')
    car_prices.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Standardize Values

    First we'll get the list of all string columns.
    """
    )
    return


@app.cell(hide_code=True)
def _(car_prices):
    string_columns = list(car_prices.dtypes[car_prices.dtypes == 'object'].index)
    string_columns
    return (string_columns,)


@app.cell(hide_code=True)
def _(car_prices, string_columns):
    for column in string_columns:
        car_prices[column] = car_prices[column].str.lower().str.replace(' ', '_')
    car_prices.head()
    return


if __name__ == "__main__":
    app.run()
