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

    ## Module 6: **Decision Trees**
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = (
        "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"
    )

    chapters = pd.DataFrame(
        [
            {
                "title": "Credit Risk",
                "youtube_id": "GJGmlfZoCoU",
                "contents": repository_root + "06-trees/01-credit-risk.md",
            },
            {
                "title": "Data Preparation",
                "youtube_id": "tfuQdI3YO2c",
                "contents": repository_root + "06-trees/02-data-prep.md",
            },
            {
                "title": "Decision Trees",
                "youtube_id": "YGiQvFbSIg8",
                "contents": repository_root + "06-trees/03-decision-trees.md",
            },
            {
                "title": "Decision Tree Learning",
                "youtube_id": "XODz6LwKY7g",
                "contents": repository_root + "06-trees/04-decision-tree-learning.md",
            },
            {
                "title": "Decision Tree Tuning",
                "youtube_id": "XJaxwH50Qok",
                "contents": repository_root + "06-trees/05-decision-tree-tuning.md",
            },
            {
                "title": "Random Forest",
                "youtube_id": "FZhcmOfNNZE",
                "contents": repository_root + "06-trees/06-random-forest.md",
            },
            {
                "title": "Gradient boosting and XGBoost",
                "youtube_id": "xFarGClszEM",
                "contents": repository_root + "06-trees/07-boosting.md",
            },
            {
                "title": "XGBoost Parameter Tuning",
                "youtube_id": "VX6ftRzYROM",
                "contents": repository_root + "06-trees/08-xgb-tuning.md",
            },
            {
                "title": "Final Model",
                "youtube_id": "lqdnyIVQq-M",
                "contents": repository_root + "06-trees/09-final-model.md",
            },
            {
                "title": "Summary",
                "youtube_id": "JZ6sRZ_5j_c",
                "contents": repository_root + "06-trees/10-summary.md",
            },
            {
                "title": "Explore More",
                "contents": repository_root + "06-trees/11-explore-more.md",
            },
        ]
    )

    chapters.insert(
        loc=0,
        column="snapshot",
        value="https://img.youtube.com/vi/"
        + chapters.youtube_id.astype(str)
        + "/hqdefault.jpg",
    )
    chapters.insert(
        loc=2,
        column="youtube",
        value="https://youtube.com/watch?v=" + chapters.youtube_id.astype(str),
    )

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
    ## Credit Risk

    In this module we'll be creating a credit risk scoring model. To do so, we'll be using a specific [CreditScoring](https://github.com/gastonstat/CreditScoring) dataset that has been copied into the data folder for easier access and loading.

    The model will decide whether a client is likely to return a credit:

    - if the model returns 0, the client is very likely to payback and the loan is approved
    - if the model returns 1, the client is not likely to payback and the loan is rejected
    """
    )
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("module-6/data/CreditScoring.csv")

    df_raw.head()
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Cleaning and Preparation

    ### Quick Look at the Data Types
    """
    )
    return


@app.cell
def _(df_raw):
    df_raw.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Normalize Column Names""")
    return


@app.cell
def _(df_raw, pd):
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        copy = df.copy()
        copy.columns = copy.columns.str.lower()

        return copy

    def normalize_column_names_preview(df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = normalize_column_names(df_raw)

        return df_normalized.head()

    normalize_column_names_preview(df_raw)
    return (normalize_column_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Set Category Names""")
    return


@app.cell
def _(df_raw, normalize_column_names, pd):
    def set_category_names(df: pd.DataFrame) -> pd.DataFrame:
        copy = df.copy()
        copy.status = copy.status.map({1: 'ok', 2: 'default', 0: 'unknown'})
        copy.home = copy.home.map({1: 'rent', 2: 'owner', 3: 'private', 4: 'ignore', 5: 'parents', 6: 'other', 0: 'unknown',})
        copy.marital = copy.marital.map({1: 'single', 2: 'married', 3: 'widow', 4: 'separated', 5: 'divorced', 0: 'unknown'})
        copy.records = copy.records.map({1: 'no', 2: 'yes', 0: 'unknown'})
        copy.job = copy.job.map({1: 'fixed', 2: 'partime', 3: 'freelance', 4: 'others', 0: 'unknown'})

        return copy

    def set_category_names_preview(df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = normalize_column_names(df_raw)
        df_categorized = set_category_names(df_normalized)

        return df_categorized.head()

    set_category_names_preview(df_raw)
    return (set_category_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Handle Missing Values""")
    return


@app.cell
def _(df_raw):
    df_raw.describe().round()
    return


@app.cell
def _(df_raw, normalize_column_names, np, pd, set_category_names):
    def tag_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        copy = df.copy()
        copy.income = copy.income.replace(to_replace=99999999, value=np.nan)
        copy.assets = copy.assets.replace(to_replace=99999999, value=np.nan)
        copy.debt = copy.debt.replace(to_replace=99999999, value=np.nan)

        return copy

    def tag_missing_values_preview(df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = normalize_column_names(df_raw)
        df_categorized = set_category_names(df_normalized)
        df_missing = tag_missing_values(df_categorized)

        return df_missing.describe()

    tag_missing_values_preview(df_raw)
    return (tag_missing_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Filter Target Variable

    We can only make use of the part of the dataset that has clearly defined outcomes, but the target variable contains records where it's unknown. We'll now filter those records.
    """
    )
    return


@app.cell
def _(
    df_raw,
    normalize_column_names,
    pd,
    set_category_names,
    tag_missing_values,
):
    def filter_target(df: pd.DataFrame) -> pd.DataFrame:
        copy = df.copy()
        copy = copy[copy.status != 'unknown'].reset_index(drop=True)

        return copy

    def filter_target_preview(df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = normalize_column_names(df_raw)
        df_categorized = set_category_names(df_normalized)
        df_missing = tag_missing_values(df_categorized)
        df_filtered = filter_target(df_missing)

        return df_filtered.head()

    filter_target_preview(df_raw)
    return (filter_target,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Set up the Validation Framework""")
    return


@app.cell
def _(
    df_raw,
    filter_target,
    normalize_column_names,
    pd,
    set_category_names,
    tag_missing_values,
):
    from sklearn.model_selection import train_test_split

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = normalize_column_names(df_raw)
        df_categorized = set_category_names(df_normalized)
        df_missing = tag_missing_values(df_categorized)

        return filter_target(df_missing)

    def split(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        df_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
        df_train, df_val = train_test_split(df_full, test_size=0.25, random_state=11)

        df_full = df_full.reset_index(drop=True)
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        return df_full, df_train, df_val, df_test

    df_full, df_train, df_val, df_test = split(preprocess(df_raw))
    return df_full, df_test, df_train, df_val


@app.cell
def _(df_full, pd):
    def separate_target(df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        target = (df.status == 'default').astype(int).values
        del features["status"]

        return features, target

    separate_target(df_full)
    return (separate_target,)


@app.cell
def _(df_full, pd, separate_target):
    from typing import Optional
    from sklearn.feature_extraction import DictVectorizer

    def train_dictionary_vectorizer(df: pd.DataFrame) -> (DictVectorizer, dict):
        dictionary = df.to_dict(orient='records')
        dict_vectorizer = DictVectorizer(sparse=False)
        X = dict_vectorizer.fit_transform(dictionary)

        return dict_vectorizer, X

    def get_features_and_target(df: pd.DataFrame, dict_vectorizer: Optional[DictVectorizer] = None) \
        -> (pd.DataFrame, pd.DataFrame, DictVectorizer
    ):
        features, y = separate_target(df)

        if not dict_vectorizer:
            dict_vectorizer, X = train_dictionary_vectorizer(features)
        else:
            X = dict_vectorizer.transform(features.to_dict(orient='records'))

        return X, y, dict_vectorizer

    get_features_and_target(df_full)
    return DictVectorizer, Optional, get_features_and_target


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Decision Tree

    Decision trees are a data structure that encodes information about a dataset in the form of conditions (if statements). Each of the conditions typically relates with a field from the dataset, a comparison symbol (<, <=, >=, >) and a value. From each node, two branches are maintained to the records that match the condition and the records that don't match it.
    """
    )
    return


@app.cell
def _(DictVectorizer, df_train, get_features_and_target, pd):
    from sklearn.tree import DecisionTreeClassifier

    def train_decision_tree(df: pd.DataFrame, max_depth: int = None, min_samples_leaf: int = 1) \
        -> (DecisionTreeClassifier, DictVectorizer
    ):
        decision_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        X, y, dict_vectorizer = get_features_and_target(df)
        decision_tree.fit(X, y)

        return decision_tree, dict_vectorizer

    overfitted_decision_tree, dict_vectorizer = train_decision_tree(df_train)
    return dict_vectorizer, overfitted_decision_tree, train_decision_tree


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Evaluate the model

    At the moment, we have 100% of accuracy on the train set but an accuracy of around 66% on our validation set. We are **overfitting** our model.
    """
    )
    return


@app.cell
def _(
    DictVectorizer,
    df_train,
    df_val,
    dict_vectorizer,
    get_features_and_target,
    overfitted_decision_tree,
    pd,
):
    from sklearn.metrics import roc_auc_score
    from sklearn.base import ClassifierMixin

    def get_roc_auc_score(df: pd.DataFrame, dict_vectorizer: DictVectorizer, model: ClassifierMixin) -> float:
        X, y, _ = get_features_and_target(df, dict_vectorizer)
        y_pred = model.predict_proba(X)[:,1]

        return roc_auc_score(y, y_pred)

    {
        "roc_auc_val": get_roc_auc_score(df_val, dict_vectorizer, overfitted_decision_tree),
        "roc_auc_train": get_roc_auc_score(df_train, dict_vectorizer, overfitted_decision_tree)
    }
    return get_roc_auc_score, roc_auc_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By default, decision trees can grow as much as they want, what makes them prone to overfitting. To address this, we can set a maximum depth when we create them.""")
    return


@app.cell
def _(
    df_train,
    df_val,
    dict_vectorizer,
    get_roc_auc_score,
    train_decision_tree,
):
    decision_tree, _ = train_decision_tree(df_train, max_depth=3)

    {
        "roc_auc_val": get_roc_auc_score(df_val, dict_vectorizer, decision_tree),
        "roc_auc_train": get_roc_auc_score(df_train, dict_vectorizer, decision_tree)
    }
    return (decision_tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Explore the tree""")
    return


@app.cell
def _(decision_tree, dict_vectorizer, plt):
    from sklearn.tree import plot_tree

    plt.figure(figsize=(16, 9))
    plot_tree(decision_tree, feature_names=dict_vectorizer.get_feature_names_out(), fontsize=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Decision Tree Learning

    A decision tree is a model that makes predictions by following a series of yes/no questions based on the features of the data. Each internal node represents a question (for example, **income <= 74.5**), each branch represents an answer (*yes* or *no*) and each leaf node gives the final decision or prediction.

    The tree learns these questions automatically from training data by finding, at each step, the feature and threshold that best separate the examples into groups that are as homogeneous as possible with respect to the target (for instance: all *yes* or all *no*).

    This process continues recursively until the data are well classified or other stopping conditions are met, producing a model that can later be used to classify new, unseen examples by following the same sequence of decisions.

    ### Simplified algorithm

    This pseudocode shows how we could implement an algorithm that finds the best split.

    ```python
    decision_tree = {}

    for feature in features:
        thresholds = find_all_thresholds(feature)
        impurities = {}

        for threshold in thresholds:
            condition = define_condition(feature, '<=', threshold)
            splitted_dataset = split(condition)
            impurity = compute_impurity(splitted_dataset)
            impurities[condition] = impurity

        decision_tree[feature] = select_condition_with_lowest_impurity(impurities)
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Decision Tree Tuning""")
    return


@app.cell
def _(
    df_train,
    df_val,
    dict_vectorizer,
    get_roc_auc_score,
    pd,
    train_decision_tree,
):
    def search_best_decision_tree():
        scores = []

        for depth in [None, 1, 2, 4, 8, 16, 32]:
            for min_samples in [1, 5, 10, 50, 100, 250, 500]:
                decision_tree, _ = train_decision_tree(df_train, max_depth=depth, min_samples_leaf=min_samples)
                score = get_roc_auc_score(df_val, dict_vectorizer, decision_tree)
                scores.append((depth, min_samples, score))

        columns = ["max_depth", "min_samples_leaf", "roc_auc"]
        df_scores = pd.DataFrame(scores, columns=columns)

        return df_scores

    decision_tree_scores = search_best_decision_tree()
    decision_tree_scores.sort_values("roc_auc", ascending=False).head()
    return (decision_tree_scores,)


@app.cell
def _(decision_tree_scores, sns):
    sns.heatmap(decision_tree_scores.pivot(index="min_samples_leaf", columns=["max_depth"]), annot=True, fmt='.2f')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Random Forest

    A random forest is an ensemble model that improves on decision trees by combining many of them to make more reliable predictions. Each tree in the forest is trained on a slightly different random subset of the data and considers only a random selection of features when choosing splits, which helps reduce overfitting and increases generalization.

    When making a prediction, all the trees “vote”: for classification, the class chosen by most trees is the final output; for regression, their average prediction is used. In essence, a random forest uses the wisdom of many diverse trees to make decisions that are more stable and accurate than any single tree alone.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Train""")
    return


@app.cell
def _(DictVectorizer, Optional, df_train, get_features_and_target, pd):
    from sklearn.ensemble import RandomForestClassifier

    def train_random_forest(df: pd.DataFrame, n_estimators: int = 10, max_depth: Optional[int] = None, min_samples_leaf: int = 1) \
        -> (RandomForestClassifier, DictVectorizer
    ):
        random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        X, y, dict_vectorizer = get_features_and_target(df)
        random_forest.fit(X, y)

        return random_forest, dict_vectorizer

    overfitted_random_forest, _ = train_random_forest(df_train)
    return overfitted_random_forest, train_random_forest


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Evaluate

    With a quick evaluation we can see that, as it happened before, we are overfitting.
    """
    )
    return


@app.cell
def _(
    df_train,
    df_val,
    dict_vectorizer,
    get_roc_auc_score,
    overfitted_random_forest,
):
    {
        "roc_auc_val": get_roc_auc_score(df_val, dict_vectorizer, overfitted_random_forest),
        "roc_auc_train": get_roc_auc_score(df_train, dict_vectorizer, overfitted_random_forest)
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Choose the Number of Estimators""")
    return


@app.cell
def _(
    df_train,
    df_val,
    dict_vectorizer,
    get_roc_auc_score,
    pd,
    train_random_forest,
):
    def choose_n_estimators():
        scores = []

        for estimators in [1, 5, 10, 50, 100, 200, 500]:
            random_forest, _ = train_random_forest(df_train, n_estimators=estimators)
            score = get_roc_auc_score(df_val, dict_vectorizer, random_forest)
            scores.append((estimators, score))

        columns = ["estimators", "roc_auc"]
        df_scores = pd.DataFrame(scores, columns=columns)

        return df_scores

    n_estimators_scores = choose_n_estimators()
    n_estimators_scores.sort_values("roc_auc", ascending=False).head()
    return (n_estimators_scores,)


@app.cell
def _(n_estimators_scores, sns):
    sns.lineplot(x=n_estimators_scores.estimators, y=n_estimators_scores.roc_auc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Tuning""")
    return


@app.cell
def _(
    df_train,
    df_val,
    dict_vectorizer,
    get_roc_auc_score,
    pd,
    train_random_forest,
):
    def search_best_random_forest():
        scores = []

        for depth in [None, 1, 2, 4, 8, 16, 32]:
            for min_samples in [1, 5, 10, 50, 100, 250, 500]:
                random_forest, _ = train_random_forest(df_train, max_depth=depth, n_estimators=50)
                score = get_roc_auc_score(df_val, dict_vectorizer, random_forest)
                scores.append((depth, min_samples, score))

        columns = ["max_depth", "min_samples_leaf", "roc_auc"]
        df_scores = pd.DataFrame(scores, columns=columns)

        return df_scores

    random_forest_scores = search_best_random_forest()
    random_forest_scores.sort_values("roc_auc", ascending=False).head()
    return (random_forest_scores,)


@app.cell
def _(random_forest_scores, sns):
    sns.heatmap(random_forest_scores.pivot(index="min_samples_leaf", columns=["max_depth"]), annot=True, fmt='.2f')
    return


@app.cell
def _(
    df_train,
    df_val,
    dict_vectorizer,
    get_roc_auc_score,
    train_random_forest,
):
    random_forest, _ = train_random_forest(df_train, min_samples_leaf=100, max_depth=8, n_estimators=50)

    {
        "roc_auc_val": get_roc_auc_score(df_val, dict_vectorizer, random_forest),
        "roc_auc_train": get_roc_auc_score(df_train, dict_vectorizer, random_forest)
    }
    return (random_forest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Gradient boosting and XGBoost

    XGBoost (Extreme Gradient Boosting) is an advanced machine learning algorithm that builds a powerful model by combining many weak decision trees in sequence, where each new tree focuses on correcting the mistakes made by the previous ones. Instead of training all trees independently (as in random forests), XGBoost adds them one at a time, optimizing the overall model through a process called gradient boosting, which minimizes errors using ideas from calculus.

    It also includes clever techniques like regularization to prevent overfitting, handling of missing data, and efficient use of memory and computation. The result is a fast, scalable, and highly accurate model widely used in data science competitions and real-world applications.
    """
    )
    return


@app.cell
def _(DictVectorizer, df_train, df_val, get_features_and_target, pd):
    import xgboost as xgb
    from xgboost.core import Booster

    def train_booster(
        df_train: pd.DataFrame, df_val: pd.DataFrame, xgb_params: dict = {}
    ) -> (Booster, DictVectorizer):
        X_train, y_train, dict_vectorizer = get_features_and_target(df_train)
        dmatrix_train = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=dict_vectorizer.get_feature_names_out().tolist(),
        )

        X_val, y_val, _ = get_features_and_target(df_val, dict_vectorizer=dict_vectorizer)
        dmatrix_val = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=dict_vectorizer.get_feature_names_out().tolist(),
        )

        watchlist = [(dmatrix_train, "train"), (dmatrix_val, "val")]
        evals_result = {}

        booster = xgb.train(
            xgb_params,
            dmatrix_train,
            num_boost_round=100,
            evals=watchlist,
            evals_result=evals_result,
            verbose_eval=False
        )

        return booster, dict_vectorizer, evals_result


    xgb_params = {
        "eta": 0.3,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "binary:logistic",
        "nthread": 8,
        "seed": 1,
        "verbosity": 0,
        "eval_metric": "auc"
    }

    non_tuned_booster, _, evals_result = train_booster(df_train, df_val, xgb_params)
    return (
        Booster,
        evals_result,
        non_tuned_booster,
        train_booster,
        xgb,
        xgb_params,
    )


@app.cell
def _(evals_result, sns):
    sns.lineplot(evals_result["train"]["auc"], legend="brief", label="Train")
    sns.lineplot(evals_result["val"]["auc"], legend="brief", label="Validation")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Predict""")
    return


@app.cell
def _(
    Booster,
    DictVectorizer,
    df_val,
    dict_vectorizer,
    get_features_and_target,
    non_tuned_booster,
    pd,
    xgb,
):
    def booster_predict(df: pd.DataFrame, booster: Booster, dict_vectorizer: DictVectorizer):
        X, y, _ = get_features_and_target(df, dict_vectorizer=dict_vectorizer)
        dmatrix = xgb.DMatrix(X, label=y, feature_names=dict_vectorizer.get_feature_names_out().tolist())

        return booster.predict(dmatrix)

    booster_predict(df_val[10:20], non_tuned_booster, dict_vectorizer)
    return (booster_predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Evaluate""")
    return


@app.cell
def _(
    Booster,
    booster_predict,
    df_val,
    dict_vectorizer,
    get_features_and_target,
    non_tuned_booster,
    pd,
    roc_auc_score,
):
    def booster_evaluate(df: pd.DataFrame, booster: Booster):
        X, y, _ = get_features_and_target(df, dict_vectorizer=dict_vectorizer)
        y_pred = booster_predict(df, booster, dict_vectorizer)

        return roc_auc_score(y, y_pred)

    booster_evaluate(df_val, non_tuned_booster)
    return (booster_evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## XGBoost Parameter Tuning

    In this section we'll be tunning these 3 parameters:

    * `eta`
    * `max_depth`
    * `min_child_weight`
    """
    )
    return


@app.cell
def _(df_train, df_val, train_booster):
    def evaluate_parameters(eta: int = 0.3, max_depth: int = 6, min_child_weight: int = 1):
        xgb_params = {
            "eta": eta,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "objective": "binary:logistic",
            "nthread": 8,
            "seed": 1,
            "verbosity": 1,
            "eval_metric": "auc"
        }

        _, _, evals_result = train_booster(df_train, df_val, xgb_params)

        return evals_result
    return (evaluate_parameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Parameter `eta`""")
    return


@app.cell
def _(evaluate_parameters, plt, sns):
    def evaluate_eta():
        eta_values = [0.1, 0.5, 1.0]

        for eta_value in eta_values:
            eta_eval = evaluate_parameters(eta=eta_value)
            sns.lineplot(eta_eval["val"]["auc"], legend="brief", label="Eta = %s" % eta_value)

        plt.show()

    evaluate_eta()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Parameter `max_depth`""")
    return


@app.cell
def _(evaluate_parameters, plt, sns):
    def evaluate_max_depth():
        max_depth_values = [5, 10, 20]

        for max_depth_value in max_depth_values:
            max_depth_eval = evaluate_parameters(max_depth=max_depth_value)
            sns.lineplot(max_depth_eval["val"]["auc"], legend="brief", label="Maximum depth = %s" % max_depth_value)

        plt.show()

    evaluate_max_depth()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Parameter `min_child_weight`""")
    return


@app.cell
def _(evaluate_parameters, plt, sns):
    def evaluate_min_child_weight():
        min_child_weight_values = [1, 5, 10]

        for min_child_weight_value in min_child_weight_values:
            min_child_weight_eval = evaluate_parameters(min_child_weight=min_child_weight_value)
            sns.lineplot(min_child_weight_eval["val"]["auc"], legend="brief", label="Minimun child weight = %s" % min_child_weight_value)

        plt.show()

    evaluate_min_child_weight()
    return


@app.cell
def _(df_train, df_val, train_booster, xgb_params):
    final_xgb_params = {
        "eta": 1,
        "max_depth": 5,
        "min_child_weight": 10,
        "objective": "binary:logistic",
        "nthread": 8,
        "seed": 1,
        "verbosity": 0,
        "eval_metric": "auc"
    }

    tuned_booster, _, tuned_evals = train_booster(df_train, df_val, xgb_params)
    return final_xgb_params, tuned_booster


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Final Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Compare the Current Models""")
    return


@app.cell(hide_code=True)
def _(
    booster_evaluate,
    decision_tree,
    df_val,
    dict_vectorizer,
    get_roc_auc_score,
    random_forest,
    tuned_booster,
):
    {
        "decision_tree_auc": get_roc_auc_score(df_val, dict_vectorizer, decision_tree),
        "random_forest_auc": get_roc_auc_score(df_val, dict_vectorizer, random_forest),
        "xg_booster": booster_evaluate(df_val, tuned_booster),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Train the Final Model""")
    return


@app.cell
def _(
    booster_evaluate,
    df_full,
    df_test,
    df_val,
    final_xgb_params,
    train_booster,
):
    final_booster, _, final_evals = train_booster(df_full, df_val, final_xgb_params)
    booster_evaluate(df_test, final_booster)
    return


if __name__ == "__main__":
    app.run()
