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
    return full_dataframe, train_dataframe, val_dataframe


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
    mo.md(r"""The closest suggested answer to the AUC of this model is 0.72.""")
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
        plt.vlines(0.64, 0, 1)
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
    mo.md(r"""The closest suggested is 0.545.""")
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

    def plot_f1(evaluation: pd.DataFrame):
        plt.plot(evaluation.threshold, evaluation.f1, label="F1", color="b")
        plt.xlabel("Threshold")
        plt.show()

    def find_max(evaluation: pd.DataFrame):
        f1_max = evaluation[evaluation.f1 == evaluation.f1.max()].iloc[0]
        print("F1 max is {:.3f} and it's reach at {:.3f}".format(f1_max.f1, f1_max.threshold))

    f1_evaluation = evaluate_with_f1(model, dict_vectorizer, val_dataframe)
    plot_f1(f1_evaluation)
    find_max(f1_evaluation)
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
    mo.md(r"""The closest suggested option to the threshold where F1 is maximal is 0.54.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 5: 5-Fold CV


    Use the `KFold` class from Scikit-Learn to evaluate our model on 5 different folds:

    ```
    KFold(n_splits=5, shuffle=True, random_state=1)
    ```

    * Iterate over different folds of `df_full_train`
    * Split the data into train and validation
    * Train the model on train with these parameters: `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`
    * Use AUC to evaluate the model on validation
    """
    )
    return


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    categorical_columns,
    full_dataframe,
    np,
    numeric_columns,
    pd,
    roc_auc_score,
):
    from sklearn.model_selection import KFold

    def get_trained_vectorizer(dataframe: pd.DataFrame) -> list[dict]:
        copy = dataframe.copy()
        del copy["converted"]
        dictionary = copy.to_dict(orient="records")

        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(dictionary)

        return dict_vectorizer, dictionary

    def get_features_and_target(dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer, dictionary):
        X = dict_vectorizer.transform(dictionary)
        y = dataframe.converted == 1

        return X, y

    def train_folds(df_full: pd.DataFrame):
        kfolds = KFold(n_splits=5, shuffle=True, random_state=1)

        auc_scores = []
        for train_idx, val_idx in kfolds.split(df_full):
            df_train = df_full.iloc[train_idx]
            df_val = df_full.iloc[val_idx]

            dict_vectorizer, dictionary = get_trained_vectorizer(df_train)
            X_train, y_train = get_features_and_target(df_train, dict_vectorizer, dictionary)

            model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
            model.fit(X_train, y_train)

            dictionary_val = df_val[numeric_columns + categorical_columns].to_dict(orient="records")
            X_val, y_val = get_features_and_target(df_val, dict_vectorizer, dictionary_val)
            y_pred = model.predict_proba(X_val)[:,1]

            auc_scores.append(roc_auc_score(y_val, y_pred))

        print("{:.3f} +- {:.3f}".format(np.mean(auc_scores), np.std(auc_scores)))

        return auc_scores

    train_folds(full_dataframe)
    return KFold, get_features_and_target, get_trained_vectorizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    How large is standard deviation of the scores across different folds?

    - 0.0001
    - 0.006
    - 0.06
    - 0.36
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The closest value to the standard deviation across folds is $0.06$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 6: Hyperparameter Tuning

    Now let's use 5-Fold cross-validation to find the best parameter `C`

    * Iterate over the following `C` values: `[0.000001, 0.001, 1]`
    * Initialize `KFold` with the same parameters as previously
    * Use these parameters for the model: `LogisticRegression(solver='liblinear', C=C, max_iter=1000)`
    * Compute the mean score as well as the std (round the mean and std to 3 decimal digits)
    """
    )
    return


@app.cell
def _(
    KFold,
    LogisticRegression,
    categorical_columns,
    full_dataframe,
    get_features_and_target,
    get_trained_vectorizer,
    np,
    numeric_columns,
    pd,
    roc_auc_score,
):
    def hypertune_folds(df_full: pd.DataFrame, C: float):
        kfolds = KFold(n_splits=5, shuffle=True, random_state=1)

        auc_scores = []
        for train_idx, val_idx in kfolds.split(df_full):
            df_train = df_full.iloc[train_idx]
            df_val = df_full.iloc[val_idx]

            dict_vectorizer, dictionary = get_trained_vectorizer(df_train)
            X_train, y_train = get_features_and_target(df_train, dict_vectorizer, dictionary)

            model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
            model.fit(X_train, y_train)

            dictionary_val = df_val[numeric_columns + categorical_columns].to_dict(orient="records")
            X_val, y_val = get_features_and_target(df_val, dict_vectorizer, dictionary_val)
            y_pred = model.predict_proba(X_val)[:,1]

            auc_scores.append(roc_auc_score(y_val, y_pred))

        print("C={:.3f}: {:.3f} +- {:.3f}".format(C, np.mean(auc_scores), np.std(auc_scores)))

        return auc_scores

    def test_c_values(df_full: pd.DataFrame, C_values: list[float]):
        auc_scores = []
        for C_value in C_values:
            auc_scores.append(hypertune_folds(full_dataframe, C_value))

        return auc_scores

    test_c_values(full_dataframe, [0.000001, 0.001, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Which `C` leads to the best mean score?

    - 0.000001
    - 0.001
    - 1
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The best score corresponds to $C = 0.001$.""")
    return


if __name__ == "__main__":
    app.run()
