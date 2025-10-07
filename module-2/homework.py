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
    # Module 2 - [Regression](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/02-regression)

    ## Homework

    ### Dataset

    For this homework, we'll use the Car Fuel Efficiency dataset. Download it from here. You can do it with wget:

    wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv

    The goal of this homework is to create a regression model for predicting the car fuel efficiency (column `fuel_efficiency_mpg`).
    """
    )
    return


@app.cell
def _(pd):
    def get_dataframe():
        return pd.read_csv("./module-2/data/car_fuel_efficiency.csv")

    get_dataframe()
    return (get_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Preparing the dataset 

    Use only the following columns:

    * `'engine_displacement'`,
    * `'horsepower'`,
    * `'vehicle_weight'`,
    * `'model_year'`,
    * `'fuel_efficiency_mpg'`
    """
    )
    return


@app.cell
def _(get_dataframe):
    def get_simplified_dataframe():
        df = get_dataframe()

        return df[["engine_displacement", "horsepower", "vehicle_weight", "model_year", "fuel_efficiency_mpg"]]

    dataframe = get_simplified_dataframe()

    dataframe
    return (dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### EDA

    * Look at the `fuel_efficiency_mpg` variable. Does it have a long tail?

    Looking at its histogram, it does not look like it has a long tail.
    """
    )
    return


@app.cell
def _(dataframe, sns):
    sns.histplot(dataframe.fuel_efficiency_mpg, bins=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 1

    There's one column with missing values. What is it?

    * `'engine_displacement'`
    * `'horsepower'`
    * `'vehicle_weight'`
    * `'model_year'`
    """
    )
    return


@app.cell
def _(dataframe):
    dataframe.isna().any()[dataframe.isna().any().values]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 2

    What's the median (50% percentile) for variable `'horsepower'`?

    - 49
    - 99
    - 149
    - 199
    """
    )
    return


@app.cell
def _(dataframe):
    dataframe.horsepower.median()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Prepare and split the dataset

    * Shuffle the dataset (the filtered one you created above), use seed `42`.
    * Split your data in train/val/test sets, with 60%/20%/20% distribution.

    Use the same code as in the lectures
    """
    )
    return


@app.cell
def _(dataframe, np):
    def split_dataset(df, seed = 42):
        n = len(df)
        n_val = n_test = int(len(df) * 0.2)
        n_train = len(df) - n_val - n_test

        np.random.seed(seed)
        shuffled_indexes = np.arange(n)
        np.random.shuffle(shuffled_indexes)

        train = df.iloc[shuffled_indexes[:n_train]]
        val = df.iloc[shuffled_indexes[n_train:n_train+n_val]]
        test = df.iloc[shuffled_indexes[n_train+n_val:]]

        return train, val, test

    split_dataset(dataframe)
    return (split_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 3

    * We need to deal with missing values for the column from Q1.
    * We have two options: fill it with 0 or with the mean of this variable.
    * Try both options. For each, train a linear regression model without regularization using the code from the lessons.
    * For computing the mean, use the training only!
    * Use the validation dataset to evaluate the models and compare the RMSE of each option.
    * Round the RMSE scores to 2 decimal digits using `round(score, 2)`
    * Which option gives better RMSE?

    Options:

    - With 0
    - With mean
    - Both are equally good
    """
    )
    return


@app.cell
def _(dataframe):
    def get_zero_filled_horsepower():
        df_zero_filled = dataframe.copy()
        df_zero_filled.horsepower = dataframe.horsepower.fillna(0)

        return df_zero_filled

    get_zero_filled_horsepower()
    return (get_zero_filled_horsepower,)


@app.cell
def _(dataframe, split_dataset):
    def get_mean_filled_horsepower():
        df_mean_filled = dataframe.copy()
        train, val, test = split_dataset(dataframe)
        df_mean_filled.horsepower = dataframe.horsepower.fillna(train.horsepower.mean())

        return df_mean_filled

    get_mean_filled_horsepower()
    return (get_mean_filled_horsepower,)


@app.cell
def _(dataframe):
    def separate_target(dataframe):
        X = dataframe.copy()
        del X["fuel_efficiency_mpg"]
        X.insert(0, "ones", 1)
        y = dataframe.fuel_efficiency_mpg.copy()

        return X, y

    separate_target(dataframe)
    return (separate_target,)


@app.cell
def _(np):
    def solve_linear_regression(X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return (solve_linear_regression,)


@app.cell
def _(np):
    def rmse(y_predicted, y_real):
        n = len(y_real)
        return np.sqrt(((y_predicted - y_real)**2).sum() / n)
    return (rmse,)


@app.cell
def _(
    get_zero_filled_horsepower,
    rmse,
    separate_target,
    solve_linear_regression,
    split_dataset,
):
    def evaluate_zero_filled_horsepower():
        df = get_zero_filled_horsepower()
        train, val, test = split_dataset(df)

        X_train, y_train = separate_target(train)
        w = solve_linear_regression(X_train, y_train)

        X_val, y_val = separate_target(val)
        y_predicted = X_val.dot(w)

        return rmse(y_predicted, y_val)

    evaluate_zero_filled_horsepower()
    return (evaluate_zero_filled_horsepower,)


@app.cell
def _(
    get_mean_filled_horsepower,
    rmse,
    separate_target,
    solve_linear_regression,
    split_dataset,
):
    def evaluate_mean_filled_horsepower():
        df = get_mean_filled_horsepower()
        train, val, test = split_dataset(df)

        X_train, y_train = separate_target(train)
        w = solve_linear_regression(X_train, y_train)

        X_val, y_val = separate_target(val)
        y_predicted = X_val.dot(w)

        return rmse(y_predicted, y_val)

    evaluate_mean_filled_horsepower()
    return (evaluate_mean_filled_horsepower,)


@app.cell
def _(evaluate_mean_filled_horsepower, evaluate_zero_filled_horsepower):
    {
        "rmse_zero_filled": evaluate_zero_filled_horsepower().round(2),
        "rmse_mean_filled": evaluate_mean_filled_horsepower().round(2),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 4

    * Now let's train a regularized linear regression.
    * For this question, fill the NAs with 0. 
    * Try different values of `r` from this list: `[0, 0.01, 0.1, 1, 5, 10, 100]`.
    * Use RMSE to evaluate the model on the validation dataset.
    * Round the RMSE scores to 2 decimal digits.
    * Which `r` gives the best RMSE?

    If multiple options give the same best RMSE, select the smallest `r`.

    Options:

    - 0
    - 0.01
    - 1
    - 10
    - 100
    """
    )
    return


@app.cell
def _(np):
    def solve_regularized_linear_regression(X, y, r = 0.1):
        XTX = X.T.dot(X)
        XTX = XTX + r * np.eye(XTX.shape[0])
        return np.linalg.inv(XTX).dot(X.T).dot(y)
    return (solve_regularized_linear_regression,)


@app.cell
def _(
    get_zero_filled_horsepower,
    rmse,
    separate_target,
    solve_regularized_linear_regression,
    split_dataset,
):
    def evaluate_with_regularization(r = 0.1):
        df = get_zero_filled_horsepower()
        train, val, test = split_dataset(df)

        X_train, y_train = separate_target(train)
        w = solve_regularized_linear_regression(X_train, y_train, r)

        X_val, y_val = separate_target(val)
        y_predicted = X_val.dot(w)

        return rmse(y_predicted, y_val).round(2)

    regularization_scores = {
        "0": evaluate_with_regularization(r = 0),
        "0.01": evaluate_with_regularization(r = 0.01),
        "1": evaluate_with_regularization(r = 1),
        "10": evaluate_with_regularization(r = 10),
        "100": evaluate_with_regularization(r = 100),
    }

    regularization_scores
    return (regularization_scores,)


@app.cell
def _(np, regularization_scores):
    np.min(list(regularization_scores.values()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 5 

    * We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
    * Try different seed values: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
    * For each seed, do the train/validation/test split with 60%/20%/20% distribution.
    * Fill the missing values with 0 and train a model without regularization.
    * For each seed, evaluate the model on the validation dataset and collect the RMSE scores. 
    * What's the standard deviation of all the scores? To compute the standard deviation, use `np.std`.
    * Round the result to 3 decimal digits (`round(std, 3)`)

    What's the value of std?

    - 0.001
    - 0.006
    - 0.060
    - 0.600

    > Note: Standard deviation shows how different the values are.
    > If it's low, then all values are approximately the same.
    > If it's high, the values are different. 
    > If standard deviation of scores is low, then our model is *stable*.
    """
    )
    return


@app.cell
def _(
    get_zero_filled_horsepower,
    rmse,
    separate_target,
    solve_linear_regression,
    split_dataset,
):
    def evaluate_seed_influence(seed = 0):
        df = get_zero_filled_horsepower()
        train, val, test = split_dataset(df, seed=seed)

        X_train, y_train = separate_target(train)
        w = solve_linear_regression(X_train, y_train)

        X_val, y_val = separate_target(val)
        y_predicted = X_val.dot(w)

        return rmse(y_predicted, y_val)

    seed_scores = {}
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        seed_scores[seed] = evaluate_seed_influence(seed)

    seed_scores
    return (seed_scores,)


@app.cell
def _(np, seed_scores):
    np.std(list(seed_scores.values())).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 6

    * Split the dataset like previously, use seed 9.
    * Combine train and validation datasets.
    * Fill the missing values with 0 and train a model with `r=0.001`. 
    * What's the RMSE on the test dataset?

    Options:

    - 0.15
    - 0.515
    - 5.15
    - 51.5
    """
    )
    return


@app.cell
def _(
    get_zero_filled_horsepower,
    np,
    rmse,
    separate_target,
    solve_regularized_linear_regression,
    split_dataset,
):
    def evaluate_final():
        df = get_zero_filled_horsepower()
        train, val, test = split_dataset(df, seed=9)

        X_train, y_train = separate_target(train)
        X_val, y_val = separate_target(val)
        X_test, y_test = separate_target(test)

        X = np.concat([X_train, X_val])
        y = np.concat([y_train, y_val])
        w = solve_regularized_linear_regression(X, y, r=0.001)
    
        y_predicted = X_test.dot(w)

        return rmse(y_predicted, y_test)

    evaluate_final()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Submit the results

    * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw02
    * If your answer doesn't match options exactly, select the closest one
    """
    )
    return


if __name__ == "__main__":
    app.run()
