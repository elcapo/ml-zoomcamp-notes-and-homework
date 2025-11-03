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
    return mo, pd, plt, sns


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
    return df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Train a Dictionary Vectorizer""")
    return


@app.cell
def _(df_train, pd):
    from sklearn.feature_extraction import DictVectorizer

    def separate_target(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        target = df.copy().fuel_efficiency_mpg
        features = df.copy()
        del features["fuel_efficiency_mpg"]

        return features, target

    def train_dict_vectorizer(df: pd.DataFrame) -> DictVectorizer:
        features, _ = separate_target(df)
        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(features.to_dict(orient="records"))

        return dict_vectorizer

    dict_vectorizer = train_dict_vectorizer(df_train)
    return DictVectorizer, dict_vectorizer, separate_target


@app.cell(hide_code=True)
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
def _(DictVectorizer, df_train, dict_vectorizer, pd, separate_target):
    from sklearn.tree import DecisionTreeRegressor

    def train_smallest_decision_tree(df: pd.DataFrame, dict_vectorizer: DictVectorizer) -> DecisionTreeRegressor:
        features, y = separate_target(df)
        X = dict_vectorizer.transform(features.to_dict(orient="records"))
        decision_tree = DecisionTreeRegressor(max_depth=1)
        decision_tree.fit(X, y)

        return decision_tree

    smallest_decision_tree = train_smallest_decision_tree(df_train, dict_vectorizer)
    return (smallest_decision_tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As we can see below, **vehicle_weight** is the principal feature used to split the data.""")
    return


@app.cell
def _(dict_vectorizer, smallest_decision_tree):
    from sklearn.tree import plot_tree

    plot_tree(smallest_decision_tree, feature_names=dict_vectorizer.get_feature_names_out().tolist())
    return


@app.cell(hide_code=True)
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
def _(DictVectorizer, df_train, dict_vectorizer, pd, separate_target):
    from typing import Optional
    from sklearn.ensemble import RandomForestRegressor

    def train_random_forest(
        df: pd.DataFrame,
        dict_vectorizer: DictVectorizer,
        n_estimators: int = 10,
        max_depth: Optional[int] = None
    ) -> RandomForestRegressor:
        features, y = separate_target(df)
        X = dict_vectorizer.transform(features.to_dict(orient="records"))
        random_forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=1, n_jobs=-1)
        random_forest.fit(X, y)

        return random_forest

    random_forest = train_random_forest(df_train, dict_vectorizer)
    return RandomForestRegressor, random_forest, train_random_forest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The RMSE of the random forest model on the validation data is close to 0.45.""")
    return


@app.cell
def _(
    DictVectorizer,
    RandomForestRegressor,
    df_val,
    dict_vectorizer,
    pd,
    random_forest,
    separate_target,
):
    from sklearn.metrics import root_mean_squared_error

    def predict_random_forest(
        df: pd.DataFrame,
        dict_vectorizer: DictVectorizer,
        random_forest: RandomForestRegressor
    ) -> (pd.DataFrame, pd.DataFrame):
        features, y = separate_target(df)
        X = dict_vectorizer.transform(features.to_dict(orient="records"))

        return random_forest.predict(X), y

    def eval_random_forest(
        df: pd.DataFrame,
        dict_vectorizer: DictVectorizer,
        random_forest: RandomForestRegressor
    ) -> float:
        y_pred, y = predict_random_forest(df_val, dict_vectorizer, random_forest)

        return root_mean_squared_error(y_true=y, y_pred=y_pred)

    eval_random_forest(df_val, dict_vectorizer, random_forest)
    return (eval_random_forest,)


@app.cell(hide_code=True)
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
def _(
    DictVectorizer,
    df_train,
    df_val,
    dict_vectorizer,
    eval_random_forest,
    pd,
    train_random_forest,
):
    def evaluate_n_estimators(df_train: pd.DataFrame, df_val: pd.DataFrame, dict_vectorizer: DictVectorizer) -> dict:
        evals = []

        for n in range(10, 201, 10):
            random_forest = train_random_forest(df_train, dict_vectorizer, n_estimators=n)
            eval = eval_random_forest(df_val, dict_vectorizer, random_forest)
            evals.append((n, eval))

        return pd.DataFrame(evals, columns=["n_estimators", "rmse"])

    n_estimators_evals = evaluate_n_estimators(df_train, df_val, dict_vectorizer)

    n_estimators_evals.sort_values("rmse", ascending=True)[:5]
    return (n_estimators_evals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The improvement stops at about 200 estimators.""")
    return


@app.cell
def _(n_estimators_evals, plt, sns):
    best_n_estimators = n_estimators_evals.sort_values("rmse", ascending=True)[0:1].n_estimators.min()

    plt.figure(figsize=(12, 7))
    plt.title("Improvement limit")
    ax = sns.lineplot(x=n_estimators_evals.n_estimators, y=n_estimators_evals.rmse, color="limegreen")
    ax.axvline(x = best_n_estimators, ymin = 0, ymax = 1)
    return


@app.cell(hide_code=True)
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

    def evaluate_n_estimators(df_train: pd.DataFrame, df_val: pd.DataFrame, dict_vectorizer: DictVectorizer) -> dict:
        evals = []

        for n in range(10, 201, 10):
            random_forest = train_random_forest(df_train, dict_vectorizer, n_estimators=n)
            eval = eval_random_forest(df_val, dict_vectorizer, random_forest)
            evals.append((n, eval))

        return pd.DataFrame(evals, columns=["n_estimators", "rmse"])

    n_estimators_evals = evaluate_n_estimators(df_train, df_val, dict_vectorizer)

    n_estimators_evals.sort_values("rmse", ascending=True)[:5]
    What's the best `max_depth`, using the mean RMSE?

    * 10
    * 15
    * 20
    * 25
    """
    )
    return


@app.cell
def _(
    DictVectorizer,
    df_train,
    df_val,
    dict_vectorizer,
    eval_random_forest,
    pd,
    train_random_forest,
):
    def evaluate_n_estimators_and_max_depth(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        dict_vectorizer: DictVectorizer
    ) -> dict:
        evals = []

        for d in [10, 15, 20, 25]:
            for n in range(10, 201, 10):
                random_forest = train_random_forest(df_train, dict_vectorizer, n_estimators=n, max_depth=d)
                eval = eval_random_forest(df_val, dict_vectorizer, random_forest)
                evals.append((d, n, eval))

        return pd.DataFrame(evals, columns=["max_depth", "n_estimators", "rmse"])

    n_estimators_and_max_depth_evals = evaluate_n_estimators_and_max_depth(df_train, df_val, dict_vectorizer)

    n_estimators_and_max_depth_evals.sort_values("rmse", ascending=True)[:5]
    return (n_estimators_and_max_depth_evals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The best **max_depth** according to the RMSE metric is 10.""")
    return


@app.cell
def _(n_estimators_and_max_depth_evals, plt, sns):
    plt.figure(figsize=(12, 7))
    plt.title("Best parameters")

    n_estimators_and_max_depth_evals.max_depth = "max_depth=" + n_estimators_and_max_depth_evals.max_depth.astype(str)
    n_estimators_and_max_depth_evals.pivot(index="n_estimators", columns="max_depth", values="rmse")

    sns.heatmap(
        n_estimators_and_max_depth_evals.pivot(index="n_estimators", columns=["max_depth"]),
        annot=True,
        fmt=".3f"
    )
    return


@app.cell(hide_code=True)
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


@app.cell
def _(
    DictVectorizer,
    df_train,
    df_val,
    dict_vectorizer,
    pd,
    train_random_forest,
):
    def find_importances(df_train: pd.DataFrame, df_val: pd.DataFrame, dict_vectorizer: DictVectorizer):
        random_forest = train_random_forest(df_train, dict_vectorizer, n_estimators=10, max_depth=20)

        importances = {}
        for name, importance in zip(dict_vectorizer.get_feature_names_out(), random_forest.feature_importances_):
            importances[name] = importance

        return importances

    importances = find_importances(df_train, df_val, dict_vectorizer)
    importances
    return (importances,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The most important feature is **vehicle_weight**.""")
    return


@app.cell
def _(importances):
    importances["vehicle_weight"], importances["horsepower"], importances["acceleration"], importances["engine_displacement"]
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
def _(DictVectorizer, pd, separate_target):
    import xgboost as xgb
    from xgboost.core import Booster

    def train_xgboost(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        dict_vectorizer: DictVectorizer,
        eta: float = 0.3
    ) -> (Booster, dict):
        features_train, y_train = separate_target(df_train)
        X_train = dict_vectorizer.transform(features_train.to_dict(orient="records"))

        dmatrix_train = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=dict_vectorizer.get_feature_names_out().tolist(),
        )

        features_val, y_val = separate_target(df_val)
        X_val = dict_vectorizer.transform(features_val.to_dict(orient="records"))

        dmatrix_val = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=dict_vectorizer.get_feature_names_out().tolist(),
        )

        watchlist = [(dmatrix_train, "train"), (dmatrix_val, "val")]
        evals = {}

        xgb_params = {
            "eta": eta,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 8,
            "seed": 1,
            "verbosity": 1,
            "eval_metric": "rmse",
        }

        booster = xgb.train(
            xgb_params,
            dmatrix_train,
            num_boost_round=100,
            evals=watchlist,
            evals_result=evals,
            verbose_eval=False
        )

        return booster, evals
    return (train_xgboost,)


@app.cell
def _(df_train, df_val, dict_vectorizer, train_xgboost):
    first_booster, first_evals = train_xgboost(df_train, df_val, dict_vectorizer, eta=0.3)
    second_booster, second_evals = train_xgboost(df_train, df_val, dict_vectorizer, eta=0.1)
    return first_evals, second_evals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The best RMSE is found for ETA = 0.1 when the number of rounds approaches 100.""")
    return


@app.cell
def _(first_evals, second_evals):
    {
        "min_rmse(eta=0.3)": min(first_evals["val"]["rmse"]),
        "min_rmse(eta=0.1)": min(second_evals["val"]["rmse"]),
    }
    return


@app.cell
def _(first_evals, second_evals, sns):
    sns.lineplot(first_evals["val"]["rmse"], label="ETA = 0.3")
    sns.lineplot(second_evals["val"]["rmse"], label="ETA = 0.1")
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
