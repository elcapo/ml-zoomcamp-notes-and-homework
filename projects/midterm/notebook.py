import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pylab as plt

    import preprocess
    import process
    return mo, pd, plt, preprocess, process


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Labour Force Survey (LFS)

    ## Exploratory Data Analysis

    This notebook presents an exploratory data analysis of the microdata from the [third quarter of 2025 of the Labour Force Survey](https://www.ine.es/dyngs/Prensa/EPA3T25.htm) (LFS) conducted by the [Spanish National Statistics Institute](https://www.ine.es) (INE). The main goal of this phase is to become familiar with the data structure, understand the key variables, and detect potential patterns, inconsistencies, or outliers that may influence the analysis.

    ## Goals

    Throughout the EDA, the following aspects will be addressed:

    * Review of the structure and coding of variables (individuals, households, and dwellings).
    * Distribution of the population by demographic and labour characteristics.
    * Identification of basic relationships between activity status, occupation, sector, and educational level.
    * Assessment of data quality: missing values, duplicates, and internal consistency between variables.

    This analysis does not aim to draw final statistical conclusions, but rather to establish a solid foundation for future modelling, ensuring that the data are properly understood and prepared for use.
    """)
    return


@app.cell(hide_code=True)
def _(preprocess):
    df = preprocess.read_dataset()
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Inventory

    Although the full dataset contains 91 columns, we'll focus our analysis on a selection of them, which we will describe and document in this section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Field `prov`

    The field `prov` refers to the province of Spain of the interviewed person.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    preprocess.map_prov(df).value_counts()[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Field `edad1`

    The field `edad1` classifies the interviewed person's age on different age groups.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    preprocess.map_edad1(df).value_counts()[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Field `sexo1`

    The field `sexo1` identifies the interviewed person as either man or woman.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    preprocess.map_sexo1(df).value_counts()[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Field `eciv1`

    The field `eciv1` corresponds with the marital status.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    preprocess.map_eciv1(df).value_counts()[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Field `nforma`

    The field `nforma` contains the educational level.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    preprocess.map_nforma(df).value_counts()[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Target variable `trarem`

    Finally, the field `trarem` is our target variable and answers to the question whether the person did any paid job during the week before the interview.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    preprocess.map_trarem(df).value_counts()[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reduced Dataset

    Taking into account only the fields that we added to our inventory, documented above, we'll prepare now a "reduced" dataset which will also be categorized so that it's almost ready to be processed by a dictionary vectorizer.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    df_reduced = preprocess.reduced_dataset(df)
    df_reduced.head()
    return (df_reduced,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Empty Fields

    Here we analyze null values and take decisions on how to handle them. To get started, we quickly check the number of records that contain null values for each column.
    """)
    return


@app.cell(hide_code=True)
def _(df_reduced):
    df_reduced.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the previous results, it seems likely that the records missing `eciv1` are the same records missing `nforma` and `trarem`. Let's first check that.
    """)
    return


@app.cell(hide_code=True)
def _(df_reduced):
    (df_reduced[df_reduced.eciv1.isnull()].index == df_reduced[df_reduced.nforma.isnull()].index).all() \
      and \
    (df_reduced[df_reduced.eciv1.isnull()].index == df_reduced[df_reduced.trarem.isnull()].index).all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As the records are the same, checking which ones are null in one of the columns would suffice:
    """)
    return


@app.cell(hide_code=True)
def _(df_reduced):
    df_reduced[df_reduced.eciv1.isnull()].edad1.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the last results we can see that the only records without the variables `eciv1`, `nforma` and the target `traren` are the records that correspond to people who is less than 16 years old. We can then safely exclude those records from our study.
    """)
    return


@app.cell(hide_code=True)
def _(df, preprocess):
    df_filtered = preprocess.filtered_dataset(df)
    df_filtered.head()
    return (df_filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up the Validation Framework

    We now need to split our dataset into 3 splits: train, validation and test.
    """)
    return


@app.cell(hide_code=True)
def _(df_filtered, process):
    df_train, df_val, df_test = process.split_dataset(df_filtered)

    {
        "len(df_train)": len(df_train),
        "len(df_val)": len(df_val),
        "len(df_test)": len(df_test),
    }
    return df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dictionary Vectorizer

    As all our features are categoric columns, we'll nead to prepare the data using a dictionary vectorizer.
    """)
    return


@app.cell
def _(df_train, process):
    train_features, train_target = process.separate_features_and_target(df_train)
    dict_vectorizer = process.train_dict_vectorizer(train_features)
    return dict_vectorizer, train_features, train_target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Features and Target
    """)
    return


@app.cell
def _(df_val, dict_vectorizer, process, train_features, train_target):
    X_train = dict_vectorizer.transform(train_features.to_dict(orient="records"))
    y_train = train_target == "Yes"

    val_features, val_target = process.separate_features_and_target(df_val)

    X_val = dict_vectorizer.transform(val_features.to_dict(orient="records"))
    y_val = val_target == "Yes"
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest

    Our goal with this dataset is not be able to predict if someone will have an occupation or not depending on certain parameters but to understand what parameters correlate better with the fact that someone has an occupation. So creating a random forest models and analyzing their feature importances seems like the way to go.
    """)
    return


@app.cell
def _(X_train, pd, y_train):
    from time import time, sleep
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier

    param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 1, 2, 3, 5, 10, 25, 50],
        "min_samples_leaf": [1, 25, 50, 100, 500, 1000],
    }

    def search_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, param_distributions: dict) -> RandomizedSearchCV:
        random_forest = RandomForestClassifier(random_state=1)
        randomized_search = RandomizedSearchCV(random_forest, param_distributions, scoring="roc_auc")
        randomized_search.fit(X_train, y_train)

        return randomized_search

    start_time = time()
    randomized_search = search_random_forest(X_train, y_train, param_distributions)
    end_time = time()

    print("Execution time: %s s" % int(end_time - start_time))
    return RandomizedSearchCV, end_time, randomized_search, start_time, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Best Params

    We will now examine the results of each experiment to obtain the parameters of the best model.

    First, we'll check each of the experiments looking at its mean score and fit time.
    """)
    return


@app.cell(hide_code=True)
def _(RandomizedSearchCV, plt, randomized_search):
    def plot_experiments(search: RandomizedSearchCV):
        fit_times = search.cv_results_["mean_fit_time"]
        scores = search.cv_results_["mean_test_score"]
    
        fig, ax_scores = plt.subplots(figsize=(12, 6))
        ax_fit_times = ax_scores.twinx()

        colors = ["royalblue" for _ in range(len(search.cv_results_))]
        colors[search.best_index_] = "forestgreen"
    
        ax_scores.bar(x=range(len(scores)), height=scores, color=colors)
        ax_scores.set_xlabel("Experiment")
        ax_scores.set_ylabel("Mean test score")
    
        ax_fit_times.plot(range(len(scores)), fit_times, color="red")
        ax_scores.set_xticks(range(len(scores)))
        ax_fit_times.set_ylabel("Mean fit time")
    
        plt.title("Experiment analysis")
        plt.show()

    plot_experiments(randomized_search)
    return (plot_experiments,)


@app.cell(hide_code=True)
def _(RandomizedSearchCV, plt, randomized_search):
    def plot_random_forest_parameters(search: RandomizedSearchCV):
        fig, axis = plt.subplots(1, 3, figsize=(16, 3))

        colors = ["royalblue" for _ in range(len(search.cv_results_))]
        colors[search.best_index_] = "forestgreen"
    
        n_estimators = [param["n_estimators"] if param["n_estimators"] != None else 0 for param in search.cv_results_["params"]]
        ax_estimators = axis[0]
        ax_estimators.bar(x=range(len(n_estimators)), height=n_estimators, color=colors)
        ax_estimators.set_xticks(range(len(n_estimators)))
        ax_estimators.set_xlabel("Experiment")
        ax_estimators.set_title("Number of estimators")

        min_samples_leaf = [param["min_samples_leaf"] for param in search.cv_results_["params"]]
        ax_min_samples_leaf = axis[1]
        ax_min_samples_leaf.bar(x=range(len(min_samples_leaf)), height=min_samples_leaf, color=colors)
        ax_min_samples_leaf.set_xticks(range(len(min_samples_leaf)))
        ax_min_samples_leaf.set_xlabel("Experiment")
        ax_min_samples_leaf.set_title("Mimimum samples per leaf")

        max_depth = [param["max_depth"] if param["max_depth"] != None else 0 for param in search.cv_results_["params"]]
        ax_max_depth = axis[2]
        ax_max_depth.bar(x=range(len(max_depth)), height=max_depth, color=colors)
        ax_max_depth.set_xticks(range(len(max_depth)))
        ax_max_depth.set_xlabel("Experiment")
        ax_max_depth.set_title("Maximum depth")

        plt.show()

    plot_random_forest_parameters(randomized_search)
    return


@app.cell
def _(randomized_search):
    randomized_search.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation

    As we can see below, with the params that achieved the best fit, the model reached a 75% accuracy.
    """)
    return


@app.cell
def _(X_val, randomized_search, y_val):
    print("Randomized search score: %.2f %%" % (randomized_search.score(X_val, y_val) * 100))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost

    At this point, we want to check if with XGBoost we would achieve better results than the ones we obtained with random forest.
    """)
    return


@app.cell
def _(
    RandomizedSearchCV,
    X_train,
    X_val,
    end_time,
    pd,
    start_time,
    time,
    y_train,
    y_val,
):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    booster_param_distributions = {
        "eta": [0.3, 0.5, 0.7],
        "max_depth": [0, 3, 6],
        "min_child_weight": [0, 5, 10],
    }

    def search_xgboost(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        param_distributions: dict
    ) -> RandomizedSearchCV:
        booster = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=1, eval_metric=roc_auc_score, random_state=1)
        booster_search = RandomizedSearchCV(booster, param_distributions)
        booster_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        return booster_search

    booster_start_time = time()
    booster_search = search_xgboost(X_train, y_train, X_val, y_val, booster_param_distributions)
    booster_end_time = time()

    print("Execution time: %s s" % int(end_time - start_time))
    return (booster_search,)


@app.cell
def _(booster_search, plot_experiments):
    plot_experiments(booster_search)
    return


@app.cell(hide_code=True)
def _(RandomizedSearchCV, booster_search, plt):
    def plot_xgboost_parameters(search: RandomizedSearchCV):
        fig, axis = plt.subplots(1, 3, figsize=(16, 3))

        colors = ["royalblue" for _ in range(len(search.cv_results_))]
        colors[search.best_index_] = "forestgreen"
    
        eta = [param["eta"] for param in search.cv_results_["params"]]
        ax_eta = axis[0]
        ax_eta.bar(x=range(len(eta)), height=eta, color=colors)
        ax_eta.set_xticks(range(len(eta)))
        ax_eta.set_xlabel("Experiment")
        ax_eta.set_title("Eta")

        max_depth = [param["max_depth"] if param["max_depth"] != None else 0 for param in search.cv_results_["params"]]
        ax_max_depth = axis[1]
        ax_max_depth.bar(x=range(len(max_depth)), height=max_depth, color=colors)
        ax_max_depth.set_xticks(range(len(max_depth)))
        ax_max_depth.set_xlabel("Experiment")
        ax_max_depth.set_title("Maximum depth")

        min_child_weight = [param["min_child_weight"] for param in search.cv_results_["params"]]
        ax_min_child_weight = axis[2]
        ax_min_child_weight.bar(x=range(len(min_child_weight)), height=min_child_weight, color=colors)
        ax_min_child_weight.set_xticks(range(len(min_child_weight)))
        ax_min_child_weight.set_xlabel("Experiment")
        ax_min_child_weight.set_title("Mimimum child weight")

        plt.show()

    plot_xgboost_parameters(booster_search)
    return


@app.cell
def _(X_val, booster_search, y_val):
    print("XGBoost search score: %.2f %%" % (booster_search.score(X_val, y_val) * 100))
    return


if __name__ == "__main__":
    app.run()
