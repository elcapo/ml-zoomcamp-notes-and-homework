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
    return mo, np, pd, sns


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
    for string_column in string_columns:
        car_prices[string_column] = car_prices[string_column].str.lower().str.replace(' ', '_')
    car_prices.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exploratory Data Analysis

    ### Check Unique Values per Column
    """
    )
    return


@app.cell(hide_code=True)
def _(car_prices):
    unique_values = {}

    for described_column in car_prices.columns:
        unique_values[described_column] = {
            "sample": car_prices[described_column].unique()[:5],
            "count": car_prices[described_column].nunique()
        }

    unique_values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Check the Price Column

    A tiny number of cars with a very high price is ruining our plot.
    """
    )
    return


@app.cell(hide_code=True)
def _(car_prices, sns):
    sns.histplot(car_prices.msrp, bins=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We could filter them to get a better view but that implies making a choice on where to cut. Which could confuse our model.""")
    return


@app.cell(hide_code=True)
def _(car_prices, sns):
    sns.histplot(car_prices.msrp[car_prices.msrp < 100000], bins=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Instead of making a manual cut, we can take the logarithm of the price. By doing so, outlier (few cases with extreme values) are handled in a more elegant way.""")
    return


@app.cell(hide_code=True)
def _(car_prices, np, sns):
    price_logs = np.log1p(car_prices.msrp)

    sns.histplot(price_logs, bins=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Check for Empty Values""")
    return


@app.cell(hide_code=True)
def _(car_prices):
    car_prices.isna().sum()[car_prices.isna().any()]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Setting Up the Validation Framework

    We want to define our feature matrix $X$ and a target variable $y$ and then separate it into our train $X_T$ and $y_T$, validation $X_V$ and $y_V$ and test $X_{test}$ and $t_{test}$ splits.
    """
    )
    return


@app.cell(hide_code=True)
def _(car_prices):
    n = len(car_prices)
    n_val = n_test = int(len(car_prices) * 0.2)
    n_train = len(car_prices) - n_val - n_test

    {"n": n, "n_train": n_train, "n_val": n_val, "n_test": n_test}
    return n, n_train, n_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To avoid spliting the records in the same order they have in the dataset (so that we don't get oddly distributed splits), we'll make a randomized selection.

    First, we create a list with the identifiers of each sample in a random order.
    """
    )
    return


@app.cell(hide_code=True)
def _(n, np):
    np.random.seed(2)
    shuffled_indexes = np.arange(n)
    np.random.shuffle(shuffled_indexes)

    shuffled_indexes
    return (shuffled_indexes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Then we use that as our reference for creating our splits.""")
    return


@app.cell
def _(car_prices, n_train, n_val, shuffled_indexes):
    car_prices_train = car_prices.iloc[shuffled_indexes[:n_train]]
    car_prices_val = car_prices.iloc[shuffled_indexes[n_train:n_train+n_val]]
    car_prices_test = car_prices.iloc[shuffled_indexes[n_train+n_val:]]
    return car_prices_test, car_prices_train, car_prices_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As our index column has now a random order, we'll reset it in each of the splits.""")
    return


@app.cell
def _(car_prices_test, car_prices_train, car_prices_val):
    car_prices_train.reset_index(drop=True, inplace=True)
    car_prices_val.reset_index(drop=True, inplace=True)
    car_prices_test.reset_index(drop=True, inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, we'll define our target variables adjusting the price column using the logarithm function described above.""")
    return


@app.cell
def _(car_prices_test, car_prices_train, car_prices_val, np):
    y_train = np.log1p(car_prices_train.msrp.values)
    y_val = np.log1p(car_prices_val.msrp.values)
    y_test = np.log1p(car_prices_test.msrp.values)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And, finally, we'll define our feature matrices.""")
    return


@app.cell
def _(car_prices_test, car_prices_train, car_prices_val):
    X_train = car_prices_train
    X_val = car_prices_val
    X_test = car_prices_test

    del X_train["msrp"]
    del X_val["msrp"]
    del X_test["msrp"]
    return


if __name__ == "__main__":
    app.run()
