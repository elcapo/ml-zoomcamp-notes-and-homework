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
    # Module 6: [Decision Trees](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/06-trees)

    > Note: sometimes your answer doesn't match one of 
    > the options exactly. That's fine. 
    > Select the option that's closest to your solution.
    > If it's exactly in between two options, select the higher value.


    ## Dataset

    In this homework, we continue using the fuel efficiency dataset.
    Download it from <a href='https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'>here</a>.

    You can do it with wget:

    ```bash
    wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
    ```

    The goal of this homework is to create a regression model for predicting the car fuel efficiency (column `'fuel_efficiency_mpg'`).
    """
    )
    return


@app.cell
def _(pd):
    def get_dataframe():
        return pd.read_csv("./module-2/data/car_fuel_efficiency.csv")

    df_raw = get_dataframe()
    df_raw.head()
    return df_raw, get_dataframe


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Preparing the dataset 

    Preparation:

    * Fill missing values with zeros.
    * Do train/validation/test split with 60%/20%/20% distribution. 
    * Use the `train_test_split` function and set the `random_state` parameter to 1.
    * Use `DictVectorizer(sparse=True)` to turn the dataframes into matrices.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Fill missing values""")
    return


@app.cell
def _(df_raw, get_dataframe, pd):
    def fill_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = get_dataframe()

        copy = df.copy()
        copy = copy.fillna(0)

        return copy

    df_filled = fill_dataframe(df_raw)
    df_filled.head()
    return (df_filled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Create Data Splits""")
    return


@app.cell
def _(df_filled, pd):
    from sklearn.model_selection import train_test_split

    def get_splits(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        df_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
        df_train, df_val = train_test_split(df_full, test_size=0.25, random_state=1)

        return df_train, df_val, df_test

    df_train, df_val, df_test = get_splits(df_filled)

    {
        "len(df_train)": len(df_train),
        "len(df_val)": len(df_val),
        "len(df_test)": len(df_test)
    }
    return (df_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Train a Dictionary Vectorizer""")
    return


@app.cell
def _(df_train, pd):
    from sklearn.feature_extraction import DictVectorizer

    def separate_target(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        y = df.copy().fuel_efficiency_mpg
        X = df.copy()
        del X["fuel_efficiency_mpg"]

        return X, y

    def train_dict_vectorizer(df: pd.DataFrame) -> DictVectorizer:
        X, y = separate_target(df)
        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(X.to_dict(orient="records"))

        return dict_vectorizer

    dict_vectorizer = train_dict_vectorizer(df_train)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 1

    Let's train a decision tree regressor to predict the `fuel_efficiency_mpg` variable. 

    * Train a model with `max_depth=1`.


    Which feature is used for splitting the data?


    * `'vehicle_weight'`
    * `'model_year'`
    * `'origin'`
    * `'fuel_type'`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 2

    Train a random forest regressor with these parameters:

    * `n_estimators=10`
    * `random_state=1`
    * `n_jobs=-1` (optional - to make training faster)


    What's the RMSE of this model on the validation data?

    * 0.045
    * 0.45
    * 4.5
    * 45.0
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 3

    Now let's experiment with the `n_estimators` parameter

    * Try different values of this parameter from 10 to 200 with step 10.
    * Set `random_state` to `1`.
    * Evaluate the model on the validation dataset.


    After which value of `n_estimators` does RMSE stop improving?
    Consider 3 decimal places for calculating the answer.

    - 10
    - 25
    - 80
    - 200

    If it doesn't stop improving, use the latest iteration number in
    your answer.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 4

    Let's select the best `max_depth`:

    * Try different values of `max_depth`: `[10, 15, 20, 25]`
    * For each of these values,
      * try different values of `n_estimators` from 10 till 200 (with step 10)
      * calculate the mean RMSE 
    * Fix the random seed: `random_state=1`


    What's the best `max_depth`, using the mean RMSE?

    * 10
    * 15
    * 20
    * 25
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 5

    We can extract feature importance information from tree-based models. 

    At each step of the decision tree learning algorithm, it finds the best split. 
    When doing it, we can calculate "gain" - the reduction in impurity before and after the split. 
    This gain is quite useful in understanding what are the important features for tree-based models.

    In Scikit-Learn, tree-based models contain this information in the
    [`feature_importances_`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.feature_importances_)
    field. 

    For this homework question, we'll find the most important feature:

    * Train the model with these parameters:
      * `n_estimators=10`,
      * `max_depth=20`,
      * `random_state=1`,
      * `n_jobs=-1` (optional)
    * Get the feature importance information from this model


    What's the most important feature (among these 4)? 

    * `vehicle_weight`
    *	`horsepower`
    * `acceleration`
    * `engine_displacement`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 6

    Now let's train an XGBoost model! For this question, we'll tune the `eta` parameter:

    * Install XGBoost
    * Create DMatrix for train and validation
    * Create a watchlist
    * Train a model with these parameters for 100 rounds:

    ```
    xgb_params = {
        'eta': 0.3, 
        'max_depth': 6,
        'min_child_weight': 1,
    
        'objective': 'reg:squarederror',
        'nthread': 8,
    
        'seed': 1,
        'verbosity': 1,
    }
    ```

    Now change `eta` from `0.3` to `0.1`.

    Which eta leads to the best RMSE score on the validation dataset?

    * 0.3
    * 0.1
    * Both give equal value
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Submit the results

    * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw06
    * If your answer doesn't match options exactly, select the closest one. If the answer is exactly in between two options, select the higher value.
    """
    )
    return


if __name__ == "__main__":
    app.run()
