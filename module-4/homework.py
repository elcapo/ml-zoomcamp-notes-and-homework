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
    # Module 4: [Evaluation](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/04-evaluation)

    ## Homework

    ### Dataset

    For this homework, we'll use the lead scoring Bank Marketing dataset. Download it from here. You can do it with wget:

    wget [https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv)

    In this dataset our desired target for classification task will be **converted** variable - has the client signed up to the platform or not.

    """
    )
    return


@app.cell
def _(pd):
    def read_dataframe():
        return pd.read_csv("./module-4/data/course_lead_scoring.csv")

    raw_dataframe = read_dataframe()
    raw_dataframe.head()
    return (raw_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data preparation

    ### Check if the missing values are presented in the features
    """
    )
    return


@app.cell
def _(raw_dataframe):
    raw_dataframe.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### If there are missing values:

    #### For categorical features, replace them with 'NA'
    """
    )
    return


@app.cell
def _(raw_dataframe):
    categorical_columns = ["lead_source", "industry", "employment_status", "location"]

    def fill_categorical_values(dataframe):
        dataframe[categorical_columns] = dataframe[categorical_columns].fillna('NA')

        return dataframe

    fill_categorical_values(raw_dataframe).head()
    return categorical_columns, fill_categorical_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### For numerical features, replace with with 0.0""")
    return


@app.cell
def _(raw_dataframe):
    numeric_columns = ["number_of_courses_viewed", "annual_income", "interaction_count", "lead_score"]

    def fill_numerical_values(dataframe):
        dataframe[numeric_columns] = dataframe[numeric_columns].fillna(0.0)

        return dataframe

    fill_numerical_values(raw_dataframe).head()
    return fill_numerical_values, numeric_columns


@app.cell
def _(fill_categorical_values, fill_numerical_values, raw_dataframe):
    filled_dataframe = fill_numerical_values(fill_categorical_values(raw_dataframe.copy()))
    filled_dataframe.head()
    return (filled_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Split the data into 3 parts

    Split the data into 3 parts: train/validation/test with 60% / 20% / 20% distribution. Use train_test_split function for that with random_state=1.
    """
    )
    return


@app.cell
def _(filled_dataframe):
    from sklearn.model_selection import train_test_split

    full_dataframe, test_dataframe = train_test_split(filled_dataframe, test_size=0.2, random_state=1)
    train_dataframe, val_dataframe = train_test_split(full_dataframe, test_size=0.25, random_state=1)

    {
        "len(train_dataframe)": len(train_dataframe),
        "len(val_dataframe)": len(val_dataframe),
        "len(test_dataframe)": len(test_dataframe),
    }
    return train_dataframe, val_dataframe


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 1: ROC AUC feature importance

    ROC AUC could also be used to evaluate feature importance of numerical variables. 

    Let's do that

    * For each numerical variable, use it as score (aka prediction) and compute the AUC with the `y` variable as ground truth.
    * Use the training dataset for that


    If your AUC is < 0.5, invert this variable by putting "-" in front

    (e.g. `-df_train['balance']`)

    AUC can go below 0.5 if the variable is negatively correlated with the target variable. You can change the direction of the correlation by negating this variable - then negative correlation becomes positive.
    """
    )
    return


@app.cell
def _(numeric_columns, pd, train_dataframe):
    from sklearn.metrics import roc_auc_score

    def compute_auc_for_numeric_features(dataframe: pd.DataFrame) -> dict:
        auc_scores = {}

        for feature in numeric_columns:
            auc_scores[feature] = roc_auc_score(dataframe.converted, dataframe[feature])

        return auc_scores

    compute_auc_for_numeric_features(train_dataframe)
    return (roc_auc_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Which numerical variable (among the following 4) has the highest AUC?

    - `lead_score`
    - `number_of_courses_viewed`
    - `interaction_count`
    - `annual_income`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `number_of_courses_viewed` feature has the highest AUC with a value of $0.7564$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 2: Training the model

    Apply one-hot-encoding using `DictVectorizer` and train the logistic regression with these parameters:

    ```python
    LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    ```
    """
    )
    return


@app.cell
def _(
    categorical_columns,
    numeric_columns,
    pd,
    roc_auc_score,
    train_dataframe,
    val_dataframe,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction import DictVectorizer

    def separate_target(dataframe: pd.DataFrame):
        y = dataframe.converted
        X = dataframe.drop(columns=['converted'])
        return X[numeric_columns + categorical_columns], y

    def train(dataframe: pd.DataFrame):
        X, y = separate_target(dataframe)

        dict_vectorizer = DictVectorizer(sparse=False)
        X_dict = dict_vectorizer.fit_transform(X.to_dict(orient="records"))

        model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
        model.fit(X_dict, y)

        return dict_vectorizer, model

    def evaluate(
        model: LogisticRegression,
        dict_vectorizer: DictVectorizer,
        dataframe: pd.DataFrame
    ) -> float:
        X, y = separate_target(dataframe)
        X_dict = dict_vectorizer.transform(X.to_dict(orient="records"))

        y_pred = model.predict_proba(X_dict)[:, 1] >= 0.55

        return round(roc_auc_score(y, y_pred), 3)

    dict_vectorizer, model = train(train_dataframe)
    evaluate(model, dict_vectorizer, val_dataframe)
    return (
        DictVectorizer,
        LogisticRegression,
        dict_vectorizer,
        model,
        separate_target,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    What's the AUC of this model on the validation dataset? (round to 3 digits)

    - 0.32
    - 0.52
    - 0.72
    - 0.92
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The AUC of this model is 0.72.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 3: Precision and Recall

    Now let's compute precision and recall for our model.

    * Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
    * For each threshold, compute precision and recall
    * Plot them
    """
    )
    return


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    dict_vectorizer,
    model,
    np,
    pd,
    plt,
    separate_target,
    val_dataframe,
):
    from sklearn.metrics import recall_score, precision_score

    def precision_and_recall_for_threshold(
        model: LogisticRegression,
        dict_vectorizer: DictVectorizer,
        dataframe: pd.DataFrame,
        threshold: float
    ) -> tuple:
        X, y = separate_target(dataframe)
        X_dict = dict_vectorizer.transform(X.to_dict(orient="records"))
        y_pred = model.predict_proba(X_dict)[:, 1] >= threshold

        return precision_score(y, y_pred, zero_division=1.0), recall_score(y, y_pred)

    def evaluate_precision_and_recall(
        model: LogisticRegression,
        dict_vectorizer: DictVectorizer,
        dataframe: pd.DataFrame,
    ) -> list:
        evaluation = []

        for threshold in np.linspace(0, 1, 100):
            precision, recall = precision_and_recall_for_threshold(model, dict_vectorizer, dataframe, threshold)
            evaluation.append((threshold, precision, recall,))

        return pd.DataFrame(evaluation, columns=["threshold", "precision", "recall"])

    def plot_precision_and_recall(
        model: LogisticRegression,
        dict_vectorizer: DictVectorizer,
        dataframe: pd.DataFrame,
    ):
        evaluation = evaluate_precision_and_recall(model, dict_vectorizer, dataframe)

        plt.plot(evaluation.threshold, evaluation.precision, label="Precision", color="g")
        plt.plot(evaluation.threshold, evaluation.recall, label="Recall", color="r")
        plt.xlabel("Threshold")
        plt.show()

    plot_precision_and_recall(model, dict_vectorizer, val_dataframe)
    return (evaluate_precision_and_recall,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    At which threshold precision and recall curves intersect?

    * 0.145
    * 0.345
    * 0.545
    * 0.745
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Precision and recall intersect at 0.745.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 4: F1 score

    Precision and recall are conflicting - when one grows, the other goes down. That's why they are often combined into the F1 score - a metrics that takes into account both

    This is the formula for computing F1:

    $$F_1 = 2 \cdot \cfrac{P \cdot R}{P + R}$$

    Where $P$ is precision and $R$ is recall.

    Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01
    """
    )
    return


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    dict_vectorizer,
    evaluate_precision_and_recall,
    model,
    pd,
    plt,
    val_dataframe,
):
    def evaluate_with_f1(
        model: LogisticRegression,
        dict_vectorizer: DictVectorizer,
        dataframe: pd.DataFrame,
    ):
        evaluation = evaluate_precision_and_recall(model, dict_vectorizer, dataframe)
        evaluation["f1"] = 2 * evaluation.precision * evaluation.recall / (evaluation.precision + evaluation.recall)

        return evaluation

    def plot_f1(
        model: LogisticRegression,
        dict_vectorizer: DictVectorizer,
        dataframe: pd.DataFrame,
    ):
        evaluation = evaluate_with_f1(model, dict_vectorizer, val_dataframe)

        plt.plot(evaluation.threshold, evaluation.f1, label="F1", color="b")
        plt.xlabel("Threshold")
        plt.show()

    plot_f1(model, dict_vectorizer, val_dataframe)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    At which threshold F1 is maximal?

    - 0.14
    - 0.34
    - 0.54
    - 0.74
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""F1 is maximal at 0.74.""")
    return


if __name__ == "__main__":
    app.run()
