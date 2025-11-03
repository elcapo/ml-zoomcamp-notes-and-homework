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
    return mo, np, pd, plt


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
    return df_full, df_train, df_val


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
    return DictVectorizer, get_features_and_target


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

    def train_decision_tree(df: pd.DataFrame, max_depth: int = None) -> (DecisionTreeClassifier, DictVectorizer):
        decision_tree = DecisionTreeClassifier(max_depth=max_depth)
        X, y, dict_vectorizer = get_features_and_target(df)
        decision_tree.fit(X, y)

        return decision_tree, dict_vectorizer

    overfitted_decision_tree, dict_vectorizer = train_decision_tree(df_train)
    return (
        DecisionTreeClassifier,
        dict_vectorizer,
        overfitted_decision_tree,
        train_decision_tree,
    )


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
    DecisionTreeClassifier,
    DictVectorizer,
    df_train,
    df_val,
    dict_vectorizer,
    get_features_and_target,
    overfitted_decision_tree,
    pd,
):
    from sklearn.metrics import roc_auc_score

    def get_roc_auc_score(df: pd.DataFrame, dict_vectorizer: DictVectorizer, decision_tree: DecisionTreeClassifier) -> float:
        X, y, _ = get_features_and_target(df, dict_vectorizer)
        y_pred = decision_tree.predict_proba(X)[:,1]

        return roc_auc_score(y, y_pred)

    {
        "roc_auc_val": get_roc_auc_score(df_val, dict_vectorizer, overfitted_decision_tree),
        "roc_auc_train": get_roc_auc_score(df_train, dict_vectorizer, overfitted_decision_tree)
    }
    return (get_roc_auc_score,)


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


if __name__ == "__main__":
    app.run()
