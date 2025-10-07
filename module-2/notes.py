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
    return mo, np, pd, plt, sns


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
            "youtube_id": None,
            "contents": repository_root+"02-regression/17-explore-more.md"
        },
        {
            "title": "Homework",
            "youtube_id": None,
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


@app.cell
def _(car_prices):
    def get_string_columns(df):
        return list(df.dtypes[df.dtypes == 'object'].index)

    get_string_columns(car_prices)
    return (get_string_columns,)


@app.cell(hide_code=True)
def _(car_prices, get_string_columns):
    for string_column in get_string_columns(car_prices):
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


@app.cell
def _(car_prices):
    def show_unique_values(df):
        unique_values = {}

        for described_column in df.columns:
            unique_values[described_column] = {
                "sample": df[described_column].unique()[:5],
                "count": df[described_column].nunique()
            }

        return unique_values

    show_unique_values(car_prices)
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
    def plot_price_logs(df):
        price_logs = np.log1p(df.msrp)
        sns.histplot(price_logs, bins=100)

    plot_price_logs(car_prices)
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
    return y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And, finally, we'll define our feature matrices.""")
    return


@app.cell
def _(car_prices_test, car_prices_train, car_prices_val):
    X_train = car_prices_train.fillna(0)
    X_val = car_prices_val.fillna(0)
    X_test = car_prices_test.fillna(0)

    del X_train["msrp"]
    del X_val["msrp"]
    del X_test["msrp"]
    return X_train, X_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, let's wrap all this into a funciton.""")
    return


@app.cell
def _(np):
    def prepare_split(df):
        df = df.copy()
        df.reset_index(drop=True, inplace=True)

        y = np.log1p(df.msrp.values)

        del df["msrp"]

        return df, y

    def split_dataset(df):
        n = len(df)
        n_val = n_test = int(len(df) * 0.2)
        n_train = len(df) - n_val - n_test

        np.random.seed(2)
        shuffled_indexes = np.arange(n)
        np.random.shuffle(shuffled_indexes)

        train = prepare_split(df.iloc[shuffled_indexes[:n_train]])
        val = prepare_split(df.iloc[shuffled_indexes[n_train:n_train+n_val]])
        test = prepare_split(df.iloc[shuffled_indexes[n_train+n_val:]])

        return train, val, test
    return (split_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Linear Regression

    We are trying to compute a $g$ estimator function so that given the features of a car $i$:

    \[
        x_i = (x_{i1}, x_{i2}, ..., x_{in})
    \]

    Our function evaluated with those features approaches the target price:

    \[
        g_i(x_{i1}, x_{i2}, ..., x_{in}) \approx y_i
    \]
    """
    )
    return


@app.cell
def _(np):
    def raw_linear_regression(x, w):
        assert len(x) + 1 == len(w), "The size of the features vector does not match the size of the weights"

        y_log = w[0] + np.sum([x[i] * w[i + 1] for i in range(len(x))])

        return np.expm1(y_log).astype(int)
    return (raw_linear_regression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In order to show how linear regression works, we'll manually compute it for a selection of features: `engine_hp`, `city_mpg` and `popularity` of a randomly selected car.""")
    return


@app.cell
def _(X_train):
    def select_car(df, id):
        return df.iloc[id:id + 1]

    random_car_id = 10
    select_car(X_train, random_car_id)
    return random_car_id, select_car


@app.cell
def _(X_train):
    def select_features(df):
        return df[["engine_hp", "city_mpg", "popularity"]].values

    select_features(X_train)
    return (select_features,)


@app.cell
def _(X_train, random_car_id, select_car, select_features):
    def get_random_car():
        return select_features(select_car(X_train, random_car_id))

    get_random_car()
    return (get_random_car,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Given the "randomly" chosen car, we'll extract the selected features and its corresponding price:""")
    return


@app.cell
def _(get_random_car, np, random_car_id, y_train):
    {
        "x": get_random_car()[0],
        "y": np.expm1(y_train[random_car_id:random_car_id + 1]).astype(int)
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we need to define some weights. Typically, we'd be running a process that finds the optimal values but for the moment we'll manually assign them some numbers:""")
    return


@app.cell
def _(get_random_car, raw_linear_regression):
    raw_linear_regression(get_random_car()[0], [7.5, 0.01, 0.07, 0.003])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Linear Regression in Vector Form

    We want now to generalize our previous linear regression formula so that it works with any number of features:

    \[
        g_i(x_i) = w_o + \sum_{j=1}^n x_{ij} w_j
    \]

    As the right term of the equation corresponds with a dot product, we can rewrite it in a simplified form:

    \[
        g_i(x_i) = w_0 + \bold{w}^T \bold{x}_i
    \]
    """
    )
    return


@app.cell
def _(np):
    def vectorial_linear_regression(x, w):
        assert len(x) + 1 == len(w), "The size of the features vector does not match the size of the weights"

        y_log = w[0] + np.dot(x, w[1:])

        return np.expm1(y_log).astype(int)
    return (vectorial_linear_regression,)


@app.cell
def _(get_random_car, vectorial_linear_regression):
    vectorial_linear_regression(get_random_car()[0], [7.5, 0.01, 0.07, 0.003])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    If we want, we can simplify our expression even further by defining an extra feature that always equals $1$. By doing so, we'd get $\bold{w}$ and $\bold{x}$ vectors with $n+1$ elements:

    \[
        w = [w_0, w_1, ..., w_n]
    \]

    \[
        x_i = [1, x_{i1}, x_{i2}, ..., x_{in}]
    \]

    Therefore, our estimator could be written as:

    \[
        g_i(x_i) = \bold{w}^T \bold{x}_i
    \]
    """
    )
    return


@app.cell
def _(np):
    def final_linear_regression(x, w):
        assert len(x) + 1 == len(w), "The size of the features vector does not match the size of the weights"

        x = np.concat((np.array([1]), np.array(x)))
        y_log = np.dot(x, w)

        return np.expm1(y_log).astype(int)
    return (final_linear_regression,)


@app.cell
def _(final_linear_regression, get_random_car):
    final_linear_regression(get_random_car()[0], [7.5, 0.01, 0.07, 0.003])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can write our $g$ function for all samples as a matrix multiplication.

    \[
    \bold{y_p} = 
    \begin{bmatrix}
    1 & x_{11} & x_{12} & \cdots & x_{1n} \\
    1 & x_{21} & x_{22} & \cdots & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & x_{m1} & x_{m2} & \cdots & x_{mn}
    \end{bmatrix}
    \begin{bmatrix}
    w_0 \\
    w_1 \\
    w_2 \\
    \vdots \\
    w_n
    \end{bmatrix}
    =
    \begin{bmatrix}
    \bold{x_1}^T · \bold{w} \\
    \bold{x_2}^T · \bold{w} \\
    \vdots \\
    \bold{x_m}^T · \bold{w} \\
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell
def _(X_train, np, select_features):
    X_matrix = select_features(X_train.iloc[10:15])
    X_matrix = np.c_[np.ones(X_matrix.shape[0]), X_matrix]

    X_matrix
    return (X_matrix,)


@app.cell
def _(np, y_train):
    y_matrix = y_train[10:15]

    np.expm1(y_matrix).astype(int)
    return (y_matrix,)


@app.cell
def _():
    w = [7.5, 0.01, 0.07, 0.003]

    w
    return (w,)


@app.cell
def _(X_matrix, np, w):
    np.expm1(X_matrix.dot(w)).astype(int)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training a Linear Regression Model

    We already saw that our $g$ estimator looks like this:

    \[
        \bold{X} \bold{w} \approx \bold{y}
    \]

    So now we have to solve $\bold{w}$ so that we can find the weights.

    \[
        \bold{X}^{-1} \bold{X} \bold{w} \approx \bold{X}^{-1} \bold{y}
    \]

    What gives us:

    \[
        \bold{w} \approx \bold{X}^{-1} \bold{y}
    \]

    But there is an issue: we are working with a rectangular matrix, not with a squared one. So we won't be able to find its inverse.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Gram Matrix

    The first thing we can do to try to solve this is to take into account that $X^T X$ is actually an squared matrix, it may have an inverse.

    Therefore, we could start with:

    \[
        \bold{X}^{T} \bold{X} \bold{w} \approx \bold{X}^{T} \bold{y}
    \]

    ... and try to solve it with:

    \[
        (\bold{X}^{T} \bold{X})^{-1} \bold{X}^{T} \bold{X} \bold{w} \approx (\bold{X}^{T} \bold{X})^{-1} \bold{X}^{T} \bold{y}
    \]

    ... which simplifies to:

    \[
        \bold{w} \approx (\bold{X}^{T} \bold{X})^{-1} \bold{X}^{T} \bold{y}
    \]
    """
    )
    return


@app.cell
def _(X_matrix, np, y_matrix):
    def solve_linear_regression(X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    solve_linear_regression(X_matrix, y_matrix)
    return (solve_linear_regression,)


@app.cell
def _(X_matrix, np, solve_linear_regression, y_matrix):
    def predict_price(X, y):
        w = solve_linear_regression(X, y)
        y = X.dot(w)
        return np.expm1(y).astype(int)

    predict_price(X_matrix, y_matrix)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Baseline Model for Car Price Prediction Project

    We'll start by selecting all the numerical features from our train dataset.
    """
    )
    return


@app.cell
def _(X_train):
    def naive_prepare_X(X):
        return X.select_dtypes(include="number").values

    naive_prepare_X(X_train)
    return (naive_prepare_X,)


@app.cell
def _(y_train):
    y_train
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, following the same formula we used in the previous section, we define our training function, which will take the features and target variable and return the corresponding weights.""")
    return


@app.cell
def _(X_train, naive_prepare_X, solve_linear_regression, y_train):
    solve_linear_regression(naive_prepare_X(X_train), y_train)
    return


@app.cell
def _(X_train, naive_prepare_X, solve_linear_regression, y_train):
    def naive_predict(X, y):
        w = solve_linear_regression(naive_prepare_X(X), y)
        return naive_prepare_X(X).dot(w)

    naive_predict(X_train, y_train)
    return (naive_predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By plotting at the same time the target variable we used during training and our predictions, we can have a first idea about how good our model is.""")
    return


@app.cell
def _(X_train, naive_predict, sns, y_train):
    sns.histplot(y_train, color="blue")
    sns.histplot(naive_predict(X_train, y_train), color="red", alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Root Mean Squared Error

    In order to measure how good our model performs, we'll use `RMSE`, which is a metric that will tell us how good (or bad) our model predicted by checking the average of the squared difference between our predictions`g(x_i)` and the target values `y_i`:

    \[
        RMSE = \frac{1}{n} \sum_{i=0}^n (g(x_i) - y_i)^2
    \]
    """
    )
    return


@app.function
def rmse(y_predicted, y_real):
    n = len(y_real)
    return (((y_predicted - y_real)**2) / n).sum()


@app.cell
def _(X_train, naive_predict, y_train):
    rmse(
        naive_predict(X_train, y_train),
        y_train
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Using RMSE on Validation Data

    In the previous chapter we used `RMSE` on the same dataset we used to train our model, which isn't what we want. Instead, we want to use the validate split we created already.
    """
    )
    return


@app.cell
def _(X_val, naive_predict, y_val):
    naive_predict(X_val, y_val)
    return


@app.cell
def _(X_val, naive_predict, y_val):
    rmse(
        naive_predict(X_val, y_val),
        y_val
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Feature Engineering

    Let's add a feature that computes the age of the cars.
    """
    )
    return


@app.cell
def _():
    from datetime import datetime

    def feature_prepare_X(X):
        X = X.copy()
        X["age"] = datetime.today().year - X.year
        return X.select_dtypes(include="number").values
    return datetime, feature_prepare_X


@app.cell
def _(X_train, feature_prepare_X):
    feature_prepare_X(X_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Retraining our model we can see how it performs better with the additional feature.""")
    return


@app.cell
def _(feature_prepare_X, solve_linear_regression):
    def feature_train_and_predict(X_train, y_train, X_val, y_val):
        w = solve_linear_regression(feature_prepare_X(X_train), y_train)
        return feature_prepare_X(X_val).dot(w)
    return (feature_train_and_predict,)


@app.cell
def _(X_train, X_val, feature_train_and_predict, y_train, y_val):
    rmse(
        feature_train_and_predict(X_train, y_train, X_val, y_val),
        y_val
    )
    return


@app.cell
def _(X_train, X_val, feature_train_and_predict, sns, y_train, y_val):
    sns.histplot(y_val, color="blue")
    sns.histplot(feature_train_and_predict(X_train, y_train, X_val, y_val), color="red", alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Categorical Variables""")
    return


@app.cell
def _(X_train):
    def categorize_column(df, column_name, transform=None, drop_first=True):
        df = df.copy()

        if not transform:
            unique_values = sorted(df[column_name].unique().astype(str))
        else:
            unique_values = sorted(df[column_name].unique())

        # Skip the first value if drop_first is True
        start_idx = 1 if drop_first else 0

        for value in unique_values[start_idx:]:
            transformed_value = transform(value) if callable(transform) else value
            df[column_name + "_" + str(transformed_value)] = (df[column_name] == value).astype(int)

        return df

    categorize_column(X_train, "number_of_doors", int)
    return (categorize_column,)


@app.cell
def _(car_prices, categorize_column, datetime):
    def categorize_prepare(
        X,
        categorical_columns = ["make", "engine_fuel_type", "transmission_type", "driven_wheels", "vehicle_size", "vehicle_style"]
    ):
        X = X.copy()
        X = X.fillna(0)

        X["age"] = datetime.today().year - X.year

        # We'll ignore model and market category as they are too fragmented
        if "model" in X:
            del X["model"]

        if "market_category" in X:
            del X["market_category"]

        X = categorize_column(X, "number_of_doors", int)

        for categorical_column in categorical_columns:
            X = categorize_column(X, categorical_column)

        del X["number_of_doors"]
        del X["make"]
        del X["engine_fuel_type"]
        del X["transmission_type"]
        del X["driven_wheels"]
        del X["vehicle_size"]
        del X["vehicle_style"]

        return X

    categorize_prepare(car_prices).head()
    return (categorize_prepare,)


@app.cell
def _(solve_linear_regression):
    def categorized_train_and_predict(X_train, y_train, X_val, y_val):
        w = solve_linear_regression(X_train, y_train)
        return X_val.dot(w)
    return (categorized_train_and_predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""At this point we'd expect to solve our system and obtain a better performance. But we'll very likely get a horrible result. Maybe even a _Singular matrix_ error!""")
    return


@app.cell
def _(
    car_prices,
    categorize_prepare,
    categorized_train_and_predict,
    split_dataset,
):
    def categorized_eval():
        df_categorized = categorize_prepare(car_prices, ["make", "vehicle_size"])
        df_train, df_val, df_test = split_dataset(df_categorized)

        X_train, y_train = df_train
        X_val, y_val = df_val

        return rmse(
            categorized_train_and_predict(X_train, y_train, X_val, y_val),
            y_val
        )

    categorized_eval()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Regularization

    When we introduced categorical variables in our model, we also increased the chance of having features that are dependent of each other. For instance, if we had 3 possible values for **number of doors** and we created 3 boolean columns, one for each of the values, then each of the columns could be deduced from the others. In other words, they would be linearly dependent. That means that the inverse of $X^T X$ won't even be defined.

    In order to solve this, one approach is to add a diagonal multiplied by a small factor so that we get a $X^T X + \delta \mathbb{I}$ that has an inverse.
    """
    )
    return


@app.cell
def _():
    delta = 0.01
    return (delta,)


@app.cell
def _(np):
    def regularize_and_solve_linear_regression(X, y, delta):
        return np.linalg.inv(X.T.dot(X) + delta * np.eye(X.shape[1])).dot(X.T).dot(y)
    return (regularize_and_solve_linear_regression,)


@app.cell
def _(regularize_and_solve_linear_regression):
    def regularized_train_and_predict(X_train, y_train, X_val, y_val, delta):
        w = regularize_and_solve_linear_regression(X_train, y_train, delta)
        return X_val.dot(w)
    return (regularized_train_and_predict,)


@app.cell
def _(
    car_prices,
    categorize_prepare,
    delta,
    regularized_train_and_predict,
    split_dataset,
):
    def regularize_categorized_eval(delta):
        df_categorized = categorize_prepare(car_prices)
        df_train, df_val, df_test = split_dataset(df_categorized)

        X_train, y_train = df_train
        X_val, y_val = df_val

        return rmse(
            regularized_train_and_predict(X_train, y_train, X_val, y_val, delta),
            y_val
        )

    regularize_categorized_eval(delta)
    return (regularize_categorized_eval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tuning the Model

    To obtain the best $delta$ for our model, we can start by just trying some values:
    """
    )
    return


@app.cell
def _(np, regularize_categorized_eval):
    def find_delta():
        results = {}

        for n in np.linspace(9.5, 10, 20):
            delta = 1/(10**n)
            results[delta] = regularize_categorized_eval(delta)

        return results

    regularization_deltas = find_delta()

    regularization_deltas
    return (regularization_deltas,)


@app.cell
def _(plt, regularization_deltas):
    def plot_deltas(regularization_deltas):
        x = regularization_deltas.keys()
        y = regularization_deltas.values()
        plt.plot(x, y)
        plt.xlabel('δ')
        plt.ylabel('RMSE')
        plt.show()

    plot_deltas(regularization_deltas)
    return


@app.cell
def _(regularization_deltas):
    def get_min_regularization(regularization_deltas):
        min_delta = None
        min_rmse = None

        for delta, rmse in regularization_deltas.items():
            if min_rmse is None or rmse < min_rmse:
                min_rmse = rmse
                min_delta = delta

        return (min_delta, min_rmse)

    (regularization_delta, _) = get_min_regularization(regularization_deltas)
    return (regularization_delta,)


@app.cell
def _(regularization_delta, regularize_categorized_eval):
    regularize_categorized_eval(regularization_delta)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Using the Model

    At this point, we'll train our `g` model on a split that combines the train and validation splits.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prepare a Merged Dataset with Train and Validation""")
    return


@app.cell
def _(car_prices, categorize_prepare, np, split_dataset):
    def get_full_train():
        df_categorized = categorize_prepare(car_prices)
        df_train, df_val, df_test = split_dataset(df_categorized)

        X_train, y_train = df_train
        X_val, y_val = df_val
        X_test, y_test = df_test

        X = np.concat([X_train, X_val])
        y = np.concat([y_train, y_val])

        return X, y, X_test.values, y_test

    get_full_train()
    return (get_full_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Train the Model

    Here we are training the model in our merged dataset and displaying its weights.
    """
    )
    return


@app.cell
def _(
    get_full_train,
    regularization_delta,
    regularize_and_solve_linear_regression,
):
    def solve_full_train(delta):
        X, y, X_test, y_test = get_full_train()

        return regularize_and_solve_linear_regression(X, y, delta)

    solve_full_train(regularization_delta)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Check that we Can Predict Prices""")
    return


@app.cell
def _(
    get_full_train,
    regularization_delta,
    regularize_and_solve_linear_regression,
):
    def full_train_and_predict(delta):
        X, y, X_test, y_test = get_full_train()
        w = regularize_and_solve_linear_regression(X, y, delta)

        return X.dot(w)

    full_train_and_predict(regularization_delta)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Evaluate the Model on the Test Split""")
    return


@app.cell
def _(
    get_full_train,
    regularization_delta,
    regularize_and_solve_linear_regression,
):
    def full_train_eval(delta):
        X, y, X_test, y_test = get_full_train()
        w = regularize_and_solve_linear_regression(X, y, delta)

        return rmse(
            X_test.dot(w),
            y_test
        )

    full_train_eval(regularization_delta)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Use the Model to Predict a Car's Price

    Now we can use our model to predict the price of a given car.
    """
    )
    return


@app.cell
def _(
    get_full_train,
    np,
    regularization_delta,
    regularize_and_solve_linear_regression,
):
    def get_test_car_price(car_id, delta):
        X, y, X_test, y_test = get_full_train()
        w = regularize_and_solve_linear_regression(X, y, delta)

        return {
            "y": np.expm1(y_test[car_id]),
            "y_predicted": np.expm1(np.array([X_test[car_id]]).dot(w)[0])
        }

    car_id = 20
    get_test_car_price(car_id, regularization_delta)
    return


if __name__ == "__main__":
    app.run()
