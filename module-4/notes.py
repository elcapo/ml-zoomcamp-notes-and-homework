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
    return FN, FP, TN, TP, chosen_threshold


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ROC Curves

    We'll introduce here two new metrics that we can read from the confusion matrix.

    ### False Positive Rate (FPR)

    The False Positive Rate metric tells us how many of the negative cases we predicted as positive.

    \[
        FPR = \frac{FP}{TN + FP}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(FP, TN):
    FP / (TN + FP)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### True Positive Rate (TPR)

    The True Positive Rate metric tells us how many of the positive cases were predicted as positive.

    \[
        TPR = \frac{TP}{TP + FN}
    \]

    This is actually the same as recall.
    """
    )
    return


@app.cell(hide_code=True)
def _(FN, TP):
    TP / (TP + FN)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We want to plot FPR and TPR for each possible threshold.""")
    return


@app.cell
def _(X_val, df_val, np, plt, predict):
    def get_confusion_matrix(threshold):
        TP = (df_val[predict(X_val, threshold).prediction == True].churn == "yes").sum()
        FN = (df_val[predict(X_val, threshold).prediction == False].churn == "yes").sum()
        FP = (df_val[predict(X_val, threshold).prediction == True].churn != "yes").sum()
        TN = (df_val[predict(X_val, threshold).prediction == False].churn != "yes").sum()

        return np.array([[TP, FN], [FP, TN]])

    def track_tprs_and_fprs():
        thresholds = np.linspace(0, 1, 50)
        tprs = []
        fprs = []

        for threshold in thresholds:
            confusion_matrix = get_confusion_matrix(threshold)

            TP = confusion_matrix[0][0]
            FN = confusion_matrix[0][1]
            FP = confusion_matrix[1][0]
            TN = confusion_matrix[1][1]

            tprs.append(TP / (TP + FN))
            fprs.append(FP / (TN + FP))

        return thresholds, tprs, fprs

    def plot_tprs_and_fprs(axis):
        thresholds, tprs, fprs = track_tprs_and_fprs()

        axis.plot(thresholds, tprs, label="Our model TPR", color='b')
        axis.plot(thresholds, fprs, label="Our model FPR", color='b', linestyle='dashed')
        axis.legend()
        axis.set_xlabel('Threshold')

    def plot_tprs_vs_fprs(axis):
        thresholds, tprs, fprs = track_tprs_and_fprs()

        axis.plot(fprs, tprs, label="Our model's TPR vs. FPR", color='b')
        axis.legend()
        axis.set_xlabel('FPR')
        axis.set_ylabel('TPR')

    def plot_model():
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))

        plot_tprs_and_fprs(ax[0])
        plot_tprs_vs_fprs(ax[1])

    plot_model()
    plt.show()
    return plot_tprs_and_fprs, plot_tprs_vs_fprs, track_tprs_and_fprs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Random Model

    The issue with the TPR vs FPR plot is that we have nothing to compare it with. To fix that, we'll now create a model that decides randomly wether a client will churn, or not.
    """
    )
    return


@app.cell
def _(X_val, np, pd):
    def get_random_predictions(X: pd.DataFrame, threshold = 0.5) -> np.array:
        return np.random.uniform(0, 1, size=len(X)) > threshold

    get_random_predictions(X_val)
    return (get_random_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's quickly check the accuracy of this model:""")
    return


@app.cell
def _(X_val, get_random_predictions, y_val):
    (get_random_predictions(X_val) == y_val).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, let's plot its corresponding TPR and FPR curves:""")
    return


@app.cell
def _(X_val, df_val, get_random_predictions, np, plt):
    def get_random_confusion_matrix(threshold):
        random_predictions = get_random_predictions(X_val, threshold)

        TP = (df_val[random_predictions == True].churn == "yes").sum()
        FN = (df_val[random_predictions == False].churn == "yes").sum()
        FP = (df_val[random_predictions == True].churn != "yes").sum()
        TN = (df_val[random_predictions == False].churn != "yes").sum()

        return np.array([[TP, FN], [FP, TN]])

    def track_random_tprs_and_fprs():
        thresholds = np.linspace(0, 1, 50)
        tprs = []
        fprs = []

        for threshold in thresholds:
            confusion_matrix = get_random_confusion_matrix(threshold)

            TP = confusion_matrix[0][0]
            FN = confusion_matrix[0][1]
            FP = confusion_matrix[1][0]
            TN = confusion_matrix[1][1]

            tprs.append(TP / (TP + FN))
            fprs.append(FP / (TN + FP))

        return thresholds, tprs, fprs

    def plot_random_tprs_and_fprs(axis):
        thresholds, tprs, fprs = track_random_tprs_and_fprs()

        axis.plot(thresholds, tprs, label="Random TPR", color='r')
        axis.plot(thresholds, fprs, label="Random FPR", color='r', linestyle='dashed')
        axis.legend()
        axis.set_xlabel('Threshold')

    def plot_random_tprs_vs_fprs(axis):
        thresholds, tprs, fprs = track_random_tprs_and_fprs()

        axis.plot(fprs, tprs, label="Random TPR vs. FPR", color='r')
        axis.legend()
        axis.set_xlabel('FPR')
        axis.set_ylabel('TPR')

    def plot_random_model():
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))

        plot_random_tprs_and_fprs(ax[0])
        plot_random_tprs_vs_fprs(ax[1])

    plot_random_model()
    plt.show()
    return plot_random_tprs_and_fprs, plot_random_tprs_vs_fprs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ideal Model

    Now we have an idea about how the TPR vs FPR plot would look for the worse possible model. But we haven't seen how it should look like for the best model. Let's set it up.
    """
    )
    return


@app.cell
def _(X_val, np, pd, y_val):
    positive_val_count = (y_val == True).sum()
    negative_val_count = (y_val == False).sum()

    y_ideal = np.repeat([False, True], [negative_val_count, positive_val_count])

    def get_ideal_predictions(X: pd.DataFrame, threshold = 0.5) -> np.array:
        predictions = np.linspace(0, 1, len(X_val))

        return predictions >= threshold

    get_ideal_predictions(X_val)
    return (
        get_ideal_predictions,
        negative_val_count,
        positive_val_count,
        y_ideal,
    )


@app.cell
def _(mo):
    mo.md(r"""Evaluated at the ideal threshold, the accuracy of this model should be 100%.""")
    return


@app.cell
def _(
    X_val,
    get_ideal_predictions,
    negative_val_count,
    positive_val_count,
    y_ideal,
):
    ideal_threshold = negative_val_count / (negative_val_count + positive_val_count)

    (get_ideal_predictions(X_val, ideal_threshold) == y_ideal).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's now plot its corresponding TPR and FPR curves:""")
    return


@app.cell
def _(X_val, get_ideal_predictions, np, plt, y_ideal):
    def get_ideal_confusion_matrix(threshold):
        ideal_predictions = get_ideal_predictions(X_val, threshold)

        TP = (y_ideal[ideal_predictions == True] == True).sum()
        FN = (y_ideal[ideal_predictions == False] == True).sum()
        FP = (y_ideal[ideal_predictions == True] == False).sum()
        TN = (y_ideal[ideal_predictions == False] == False).sum()

        return np.array([[TP, FN], [FP, TN]])

    def track_ideal_tprs_and_fprs():
        thresholds = np.linspace(0, 1, 50)
        tprs = []
        fprs = []

        for threshold in thresholds:
            confusion_matrix = get_ideal_confusion_matrix(threshold)

            TP = confusion_matrix[0][0]
            FN = confusion_matrix[0][1]
            FP = confusion_matrix[1][0]
            TN = confusion_matrix[1][1]

            tprs.append(TP / (TP + FN))
            fprs.append(FP / (TN + FP))

        return thresholds, tprs, fprs

    def plot_ideal_tprs_and_fprs(axis):
        thresholds, tprs, fprs = track_ideal_tprs_and_fprs()

        axis.plot(thresholds, tprs, label="Ideal TPR", color='g')
        axis.plot(thresholds, fprs, label="Ideal FPR", color='g', linestyle='dashed')
        axis.legend()
        axis.set_xlabel('Threshold')

    def plot_ideal_tprs_vs_fprs(axis):
        thresholds, tprs, fprs = track_ideal_tprs_and_fprs()

        axis.plot(fprs, tprs, label="Ideal TPR vs. FPR", color='g')
        axis.legend()
        axis.set_xlabel('FPR')
        axis.set_ylabel('TPR')

    def plot_ideal_model():
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))

        plot_ideal_tprs_and_fprs(ax[0])
        plot_ideal_tprs_vs_fprs(ax[1])

    plot_ideal_model()
    plt.show()
    return (
        plot_ideal_tprs_and_fprs,
        plot_ideal_tprs_vs_fprs,
        track_ideal_tprs_and_fprs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Put it All Together

    Now, let's merge these three plots.
    """
    )
    return


@app.cell
def _(
    plot_ideal_tprs_and_fprs,
    plot_ideal_tprs_vs_fprs,
    plot_random_tprs_and_fprs,
    plot_random_tprs_vs_fprs,
    plot_tprs_and_fprs,
    plot_tprs_vs_fprs,
    plt,
):
    def plot_all_models():
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))

        plot_tprs_and_fprs(ax[0])
        plot_random_tprs_and_fprs(ax[0])
        plot_ideal_tprs_and_fprs(ax[0])

        plot_tprs_vs_fprs(ax[1])
        plot_random_tprs_vs_fprs(ax[1])
        plot_ideal_tprs_vs_fprs(ax[1])

    plot_all_models()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plot ROC Curves with Scikit Learn""")
    return


@app.cell
def _(X_val, chosen_threshold, plt, predict, y_val):
    from sklearn.metrics import roc_curve

    def track_scikit_tprs_and_fprs():
        fprs, tprs, thresholds = roc_curve(y_val, predict(X_val, chosen_threshold).probability)

        return thresholds, tprs, fprs

    def plot_scikit_tprs_and_fprs(axis):
        thresholds, tprs, fprs = track_scikit_tprs_and_fprs()

        axis.plot(thresholds, tprs, label="Our model TPR", color='b')
        axis.plot(thresholds, fprs, label="Our model FPR", color='b', linestyle='dashed')
        axis.legend()
        axis.set_xlabel('Threshold')

    def plot_scikit_tprs_vs_fprs(axis):
        thresholds, tprs, fprs = track_scikit_tprs_and_fprs()

        axis.plot(fprs, tprs, label="Our model TPR vs. FPR", color='b')
        axis.legend()
        axis.set_xlabel('FPR')
        axis.set_ylabel('TPR')

    def plot_scikit_model():
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))

        plot_scikit_tprs_and_fprs(ax[0])
        plot_scikit_tprs_vs_fprs(ax[1])

    plot_scikit_model()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ROC AUC

    The ROC AUC refers to the "area under the curve" and when applied to the "TPR vs. FPR" curve is a good metric for our classification model. To get an intuition of what it measures, we can quickly check that:

    * if we applied it to the **ideal** model, we'd get an area of around 1.0, its maximum value
    * if we applied it to the **random** model, we'd get an area of around 0.5, its minimum value
    """
    )
    return


@app.cell
def _(plot_ideal_tprs_vs_fprs, plot_random_tprs_vs_fprs, plt):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    plot_random_tprs_vs_fprs(ax)
    plot_ideal_tprs_vs_fprs(ax)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We'll use Scikit's learn specific function to obtain it.""")
    return


@app.cell
def _(track_ideal_tprs_and_fprs, track_tprs_and_fprs):
    from sklearn.metrics import auc

    def get_all_auc():
        _, tprs, fprs = track_tprs_and_fprs()
        _, ideal_tprs, ideal_fprs = track_ideal_tprs_and_fprs()

        return {
            "our_model_auc": auc(fprs, tprs),
            "ideal_auc": auc(ideal_fprs, ideal_tprs),
        }

    get_all_auc()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Interpreting ROC AUC

    The AUC metric tells us how good our model is at sorting customers according to how likely they are to churn. A way to picture it is to imagine that it takes pairs of customers where one of them churned and the other way didn't, and checks if our model actually predicted a greater posibility of churn for the positive one.
    """
    )
    return


@app.cell
def _(X_val, chosen_threshold, predict, y_val):
    from random import randint

    def manually_estimate_auc():
        y_pred = predict(X_val, chosen_threshold)

        negatives = y_pred[y_val == False].probability.to_list()
        positives = y_pred[y_val == True].probability.to_list()

        n = 100000
        success = 0

        for i in range(n):
            positive_index = randint(0, len(positives) - 1)
            negative_index = randint(0, len(negatives) - 1)

            if positives[positive_index] > negatives[negative_index]:
                success += 1

        return success / n

    manually_estimate_auc()
    return


if __name__ == "__main__":
    app.run()
