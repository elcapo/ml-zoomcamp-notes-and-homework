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
    return mo, np, pd


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
    return (df_full,)


@app.cell
def _(df_full, pd):
    def separate_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        y = (df.status == 'default').astype(int).values
        del X["status"]

        return X, y

    X_full, y_full = separate_features_and_target(df_full)
    return


if __name__ == "__main__":
    app.run()
