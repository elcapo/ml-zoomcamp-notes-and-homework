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
    import model_selection
    return mo, model_selection, plt, preprocess, process


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
    df_train, df_full, df_val, df_test = process.split_dataset(df_filtered)

    {
        "len(df_train)": len(df_train),
        "len(df_full)": len(df_full),
        "len(df_val)": len(df_val),
        "len(df_test)": len(df_test),
    }
    return df_full, df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dictionary Vectorizer

    As all our features are categoric columns, we'll need to prepare the data using a dictionary vectorizer.
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
def _(df_full, df_val, dict_vectorizer, process, train_features, train_target):
    X_train = dict_vectorizer.transform(train_features.to_dict(orient="records"))
    y_train = train_target == "Yes"

    full_features, full_target = process.separate_features_and_target(df_full)
    X_full = dict_vectorizer.transform(full_features.to_dict(orient="records"))
    y_full = full_target == "Yes"

    val_features, val_target = process.separate_features_and_target(df_val)
    X_val = dict_vectorizer.transform(val_features.to_dict(orient="records"))
    y_val = val_target == "Yes"
    return X_full, X_train, X_val, y_full, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest

    Our goal with this dataset is not be able to predict if someone will have an occupation or not depending on certain parameters, but to understand what parameters correlate better with the fact that someone has an occupation instead. So we'll focus on interpretable (decision-tree based) models. Let's start by training a random forest.
    """)
    return


@app.cell(hide_code=True)
def _(X_train, model_selection, y_train):
    from time import time, sleep

    random_forest_param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 1, 2, 3, 5, 10, 25, 50],
        "min_samples_leaf": [1, 25, 50, 100, 500, 1000],
    }

    start_time = time()
    random_forest_search = model_selection.search_random_forest(X_train, y_train, random_forest_param_distributions)
    end_time = time()

    print("Execution time: %s s" % int(end_time - start_time))
    return end_time, random_forest_search, start_time, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Best Params

    Now, let's examine the results of each experiment to obtain the parameters of the best random forest model. First, we'll check each of the experiments looking at its mean score and fit time highlighting in green the winner experiment.
    """)
    return


@app.cell(hide_code=True)
def _(model_selection, random_forest_search):
    model_selection.plot_experiments(random_forest_search)
    return


@app.cell(hide_code=True)
def _(model_selection, random_forest_search):
    model_selection.plot_random_forest_parameters(random_forest_search)
    return


@app.cell
def _(random_forest_search):
    random_forest_search.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation

    With this params we'll now train an evaluate a Random Forest model using the full (train + validation) dataset.
    """)
    return


@app.cell(hide_code=True)
def _(X_full, X_val, model_selection, random_forest_search, y_full, y_val):
    optimized_random_forest = model_selection.train_random_forest(X_full, y_full, param_distributions=random_forest_search.best_params_)

    print("Randomized search score: %.2f %%" % (model_selection.eval_model(X_val, y_val, optimized_random_forest) * 100))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost

    At this point, we want to check if with XGBoost we would achieve better results than the ones we obtained with random forest.
    """)
    return


@app.cell(hide_code=True)
def _(
    X_train,
    X_val,
    end_time,
    model_selection,
    start_time,
    time,
    y_train,
    y_val,
):
    import xgboost as xgb

    booster_param_distributions = {
        "eta": [0.1, 0.2, 0.3, 1.0],
        "max_depth": [5, 25, 50],
        "min_child_weight": [1, 3, 5, 7],
    }

    booster_start_time = time()
    booster_search = model_selection.search_xgboost(X_train, y_train, X_val, y_val, booster_param_distributions)
    booster_end_time = time()

    print("Execution time: %s s" % int(end_time - start_time))
    return (booster_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Best Params

    As we did before, we'll now examine the results of each experiment to obtain the best parameters for the XGBoost model. First, we'll check each of the experiments looking at its mean score and fit time highlighting in green the winner experiment.
    """)
    return


@app.cell
def _(booster_search, model_selection):
    model_selection.plot_experiments(booster_search)
    return


@app.cell(hide_code=True)
def _(booster_search, model_selection):
    model_selection.plot_xgboost_parameters(booster_search)
    return


@app.cell
def _(booster_search):
    booster_search.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation

    With this params we'll now train an evaluate a XGDBooster model using the full (train + validation) dataset.
    """)
    return


@app.cell
def _(
    X_full,
    X_val,
    booster_search,
    dict_vectorizer,
    model_selection,
    y_full,
    y_val,
):
    optimized_booster = model_selection.train_booster(X_full, y_full, param_distributions=booster_search.best_params_)
    optimized_booster.get_booster().feature_names = dict_vectorizer.get_feature_names_out().tolist()

    print("XGBoost search score: %.2f %%" % (model_selection.eval_model(X_val, y_val, optimized_booster) * 100))
    return (optimized_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspection

    The most influential features in the model are:

    * **sexo1=Man** (Score: 4560.0): This is the single most important feature. Being a man is, by far, the most effective feature for making splits that accurately predict the target variable. Which, as a recall, identifies whether the interviewed person had a paid occupation the week before the interview happened. This suggests a strong correlation between being male and having an occupation.

    * **nforma=Higher education** (Score: 2665.0): This is the second most important feature. Having a higher education level is extremely predictive, likely indicating a strong positive association with being employed.

    * **eciv1=Married** (Score: 2414.0): Being married is also a very influential feature, possibly acting as a proxy for factors like age or stable employment history.
    """)
    return


@app.cell(hide_code=True)
def _(optimized_booster, plt):
    from xgboost import plot_importance

    plt.figure(figsize=(12, 6))

    plot_importance(
        optimized_booster.get_booster(),
        max_num_features=10,
        height=.7,
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, optimized_booster):
    from xgboost import to_graphviz
    from pathlib import Path

    booster_graph = to_graphviz(
        optimized_booster.get_booster(),
        tree_idx=1,
        rankdir="LR",
        layout="dot",
        size="100,60",
        ratio="fill",
        dpi="300",
        condition_node_params={
            "fontsize": "128",
            "shape": "rect",
        },
        leaf_node_params={
            "fontsize": "128",
        }
    )

    graph_pdf = booster_graph.render("booster_tree", format="pdf", view=False)
    mo.pdf(src=Path(graph_pdf))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save the Model

    Finally, let's save the model for later use.
    """)
    return


@app.cell(hide_code=True)
def _(optimized_booster):
    optimized_booster.save_model("booster_model.json")
    return


if __name__ == "__main__":
    app.run()
