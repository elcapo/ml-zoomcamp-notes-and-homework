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

    ## Module 4: **Evaluation**
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"

    chapters = pd.DataFrame([
        {
            "title": "Evaluation Metrics: Session Overview",
            "youtube_id": "gmg5jw1bM8A",
            "contents": repository_root+"04-classification/01-overview.md"
        },
        {
            "title": "Accuracy and Dummy Model",
            "youtube_id": "FW_l7lB0HUI",
            "contents": repository_root+"04-classification/02-accuracy.md"
        },
        {
            "title": "Confusion Table",
            "youtube_id": "Jt2dDLSlBng",
            "contents": repository_root+"04-classification/03-confusion-table.md"
        },
        {
            "title": "Precision and Recall",
            "youtube_id": "gRLP_mlglMM",
            "contents": repository_root+"04-classification/04-precision-recall.md"
        },
        {
            "title": "ROC Curves",
            "youtube_id": "dnBZLk53sQI",
            "contents": repository_root+"04-classification/05-roc.md"
        },
        {
            "title": "ROC AUC",
            "youtube_id": "hvIQPAwkVZo",
            "contents": repository_root+"04-classification/06-auc.md"
        },
        {
            "title": "Cross Validation",
            "youtube_id": "BIIZaVtUbf4",
            "contents": repository_root+"04-classification/07-cross-validation.md"
        },
        {
            "title": "Summary",
            "youtube_id": "-v8XEQ2AHvQ",
            "contents": repository_root+"04-classification/08-summary"
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
    ## Evaluation Metrics: Session Overview

    In the last module we created a model that predicted if a customer was about to churn or not. During the last part of the module we found that our model had an accuracy of around 81%. In this module we'll try to understand how to decide wether that's a good accuracy, or not.

    ### Churn Predictor

    To get started, let's quickly set up our model again.

    #### Read and Standardize the Dataset
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
        standardized = dataframe.copy()
        standardized.columns = standardized.columns.str.lower().str.replace(' ', '_')

        return standardized

    def get_cateogorical_columns(dataframe: pd.DataFrame) -> list[str]:
        return list(list(dataframe.dtypes[dataframe.dtypes == 'object'].index))

    def standardize_categorical_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        standardized = dataframe.copy()

        for column in get_cateogorical_columns(standardized):
            standardized[column] = standardized[column].str.lower().str.replace(' ', '_')

        return standardized

    def standardize_non_categorical_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        totalcharges = pd.to_numeric(df_standardized.totalcharges, errors='coerce')
        df_standardized.totalcharges = totalcharges.fillna(0)

        return df_standardized

    df_raw = pd.read_csv("module-3/data/customer-churn.csv")
    df_standardized = standardize_column_names(df_raw)
    df_standardized = standardize_categorical_values(df_standardized)
    df_standardized = standardize_non_categorical_values(df_standardized)

    df_standardized.head()
    return (df_standardized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Setup the Validation Framework""")
    return


@app.cell(hide_code=True)
def _(df_standardized, pd):
    from sklearn.model_selection import train_test_split

    def split_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        full_train, test = train_test_split(dataframe, test_size=0.2, random_state=1)
        train, val = train_test_split(full_train, test_size=0.25, random_state=1)

        return train, val, test, full_train

    df_train, df_val, df_test, df_full = split_dataframe(df_standardized)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_full = df_full.reset_index(drop=True)

    {
        "df_train": len(df_train),
        "df_val": len(df_val),
        "df_test": len(df_test),
        "df_full": len(df_full),
    }
    return df_full, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Prepare the Features and Target Tensors""")
    return


@app.cell(hide_code=True)
def _(df_full, df_val, pd):
    from sklearn.feature_extraction import DictVectorizer

    numerical_columns = ["tenure", "monthlycharges", "totalcharges"]

    categorical_columns = [
        'gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
        'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
        'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod'
    ]

    def get_full_trained_vectorizer(dataframe: pd.DataFrame) -> list[dict]:
        copy = dataframe.copy()
        del copy["churn"]
        dictionary = copy.to_dict(orient="records")

        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(dictionary)

        return dict_vectorizer, dictionary

    def get_features_and_target(dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer, dictionary):
        X = dict_vectorizer.transform(dictionary)
        y = dataframe.churn == "yes"

        return X, y

    dict_vectorizer_full, dictionary_full = get_full_trained_vectorizer(df_full)
    X_full, y_full = get_features_and_target(df_full, dict_vectorizer_full, dictionary_full)

    dictionary_val = df_val[numerical_columns + categorical_columns].to_dict(orient="records")
    X_val, y_val = get_features_and_target(df_val, dict_vectorizer_full, dictionary_val)

    {
        "X_full": len(X_full),
        "y_full": len(y_full),
        "X_val": len(X_val),
        "y_val": len(y_val),
    }
    return X_full, X_val, y_full, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Train the Model""")
    return


@app.cell
def _(X_full, X_val, y_full, y_val):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=5000)
    model.fit(X_full, y_full)
    model.score(X_val, y_val)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Accuracy and Dummy Model

    As part of the implementation of our predictor, we chose a $0.5$ threshold, which we interpreted as a probability.

    Now we'll start by created a predictor where we can manually set that threshold.
    """
    )
    return


@app.cell(hide_code=True)
def _(X_val, model, np, pd, y_val):
    def predict(X: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
        predictions = pd.DataFrame()
        predictions["probability"] = model.predict_proba(X)[:, 1]
        predictions["prediction"] = predictions["probability"] > threshold

        return predictions

    def evaluate(X: np.ndarray, y: np.ndarray, threshold: float = 0.5):
        y_pred = predict(X, threshold)

        return (y == y_pred.prediction).mean()

    threshold_tries = np.linspace(0, 1, 21)

    accuracies_per_threshold = {}
    for threshold in threshold_tries:
        accuracies_per_threshold[round(threshold, 2)] = round(evaluate(X_val, y_val, threshold), 3)

    accuracies_per_threshold
    return accuracies_per_threshold, predict, threshold_tries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""According to this, the best threshold is $0.55$ with an accuracy of around $81%$. We can easily check it by plotting those accuracies.""")
    return


@app.cell(hide_code=True)
def _(accuracies_per_threshold, plt, threshold_tries):
    plt.scatter(y = accuracies_per_threshold.values(), x = threshold_tries)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The first and last thresholds can be interpreted as follows:

    * a model with a threshold of $0$ predict that almost every customer will churn, as it will consider if the score computed for each customer satisfies $score > 0$
    * on the other hand, a model with a threshold of $1$ predict that no customer will churn, as it will consider if the score computed for each customer satisfies $score >= 1$

    Let's check how many records do we have with those scores.
    """
    )
    return


@app.cell
def _(X_val, predict):
    {
        "score > 0": (predict(X_val, threshold = 0).prediction == True).sum(),
        "score > 1": (predict(X_val, threshold = 1).prediction == True).sum(),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here we have to stop and reflect about one thing:

    * the **trained model** has an accuracy of around **81.3%**
    * the **dummy model** that always predicts that our customers won't churn has an acuracy of **72.6%**

    Why should we bother, after all, training a model?

    The answer, as we'll see below, is that accuracy is not telling us the full story.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Confusion Matrix

    If we want a complete version of the story, we'll have to check the confusion matrix. It's the matrix that divides our dataset into at least four categories:

    |                            | Predicted as: Churn | Predicted as: No Churn |
    | ---                        | ---                 | ---                    |
    | **Actual state: Churn**    | True positive       | False negative         |
    | **Actual state: No Churn** | False positive      | True negative          |

    In our case there are 4 categories because we are dealing with a binary problem but in other classification problems we'll see a bigger confusion matrix.

    ### Actual values

    Let's obtain the actual counts of those four categories for our validation dataset with our $0.55$ chosen threshold.
    """
    )
    return


@app.cell(hide_code=True)
def _(X_val, df_val, predict):
    chosen_threshold = 0.55

    TP = (df_val[predict(X_val, chosen_threshold).prediction == True].churn == "yes").sum()
    FN = (df_val[predict(X_val, chosen_threshold).prediction == False].churn == "yes").sum()
    FP = (df_val[predict(X_val, chosen_threshold).prediction == True].churn != "yes").sum()
    TN = (df_val[predict(X_val, chosen_threshold).prediction == False].churn != "yes").sum()

    {
        "True positive": TP,
        "False negative": FN,
        "False positive": FP,
        "True negative": TN,
        "All cases": TP + FN + FP + TN,
    }
    return FN, FP, TN, TP


@app.cell(hide_code=True)
def _(FN, FP, TN, TP, np):
    confusion_matrix = np.array([
        [TP, FN],
        [FP, TN],
    ])

    confusion_matrix
    return (confusion_matrix,)


@app.cell
def _(confusion_matrix):
    (confusion_matrix / confusion_matrix.sum()).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Precision and Recall""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Precision and Recall

    In this section, we'll see how we can extract some interesting and useful metrics from the confusion matrix.

    ### Accuracy

    Actually, we've already been working with accuracy, which we could express in terms of elements of the confusion matrix as follows:

    \[
        Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(FN, FP, TN, TP):
    (TP + TN) / (TP + TN + FP + FN)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Precision

    Precision tells us how many positive predictions turn out to be correct.

    \[
        Precision = \frac{TP}{TP + FP}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(FP, TP):
    TP / (TP + FP)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Recall

    Recall tells us the fraction of correctly identified positive examples.

    \[
        Recall = \frac{TP}{TP + FN}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(FN, TP):
    TP / (TP + FN)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we can see that our model, which had an accuracy of 81% has a precision of 72% and a recall of 51%. This is telling us a richer story than the accuracy by itself. If we only looked at the accuracy we may have thought that our model was good enough but now we can see why the model is actually not that good.""")
    return


if __name__ == "__main__":
    app.run()
