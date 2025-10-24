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
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Machine Learning Zoomcamp

    ## Module 5: **Deployment**
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
                "title": "Session Overview",
                "youtube_id": "agIFak9A3m8",
                "contents": repository_root + "05-deployment/01-intro.md",
            },
            {
                "title": "Saving and Loading the Model",
                "youtube_id": "EJpqZ7OlwFU",
                "contents": repository_root + "05-deployment/02-pickle.md",
            },
            {
                "title": "Web Services: Introduction to Flask",
                "youtube_id": "W7ubna1Rfv8",
                "contents": repository_root + "05-deployment/03-flask-intro.md",
            },
            {
                "title": "Serving the Churn Model with Flask",
                "youtube_id": "Q7ZWPgPnRz8",
                "contents": repository_root + "05-deployment/04-flask-deployment.md",
            },
            {
                "title": "Python Virtual Environment: Pipenv",
                "youtube_id": "BMXh8JGROHM",
                "contents": repository_root + "05-deployment/05-pipenv.md",
            },
            {
                "title": "Environment Management: Docker",
                "youtube_id": "wAtyYZ6zvAs",
                "contents": repository_root + "05-deployment/06-docker.md",
            },
            {
                "title": "Deployment to the Cloud: AWS Elastic Beanstalk (optional)",
                "youtube_id": "HGPJ4ekhcLg",
                "contents": repository_root + "05-deployment/07-aws-eb.md",
            },
            {
                "title": "Summary",
                "youtube_id": "sSAqYSk7Br4",
                "contents": repository_root + "05-deployment/08-summary.md",
            },
            {
                "title": "Explore More",
                "contents": repository_root + "05-deployment/09-explore-more.md",
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
    ## Previous Sessions Refresh

    In the previous modules we created a model that predicted if a customer was about to churn or not. This time we'll learn how to save the module and serve it in a production environment. But first we have to reproduce the steps to build the model.

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
    return X_full, X_val, dict_vectorizer_full, y_full, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Saving and Loading the Model""")
    return


@app.cell
def _(X_full, X_val, dict_vectorizer_full, y_full, y_val):
    import pickle
    import os
    from sklearn.linear_model import LogisticRegression

    model_file = "module-5/data/model-weights.bin"

    def load_model():
        with open(model_file, "rb") as f_in:
            (model_loaded, dict_vectorizer_full_loaded) = pickle.load(f_in)
        return model_loaded, dict_vectorizer_full_loaded

    def train_and_save_model():
        model = LogisticRegression(max_iter=5000)
        model.fit(X_full, y_full)
        with open(model_file, "wb") as f_out:
            pickle.dump((model, dict_vectorizer_full), f_out)

        return model, dict_vectorizer_full

    if not os.path.isfile("module-5/data/model-weights.bin"):
        train_and_save_model()

    model_loaded, dict_vectorizer_full_loaded = load_model()

    {
        model_loaded.score(X_val, y_val)
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To facilitate the deployment, a the code necessary to train, save and load the model has been extracted to these files:

    * [model.py](https://github.com/elcapo/ml-zoomcamp-notes-and-homework/blob/main/module-5/model_package/model.py)
    * [train.py](https://github.com/elcapo/ml-zoomcamp-notes-and-homework/blob/main/module-5/model_package/train.py)
    * [predict.py](https://github.com/elcapo/ml-zoomcamp-notes-and-homework/blob/main/module-5/model_package/predict.py)

    Before running any of these programs, first make sure that you have loaded the virtual environment for the course:

    ```bash
    source .venv/bin/activate
    cd module-5/
    ```

    ### Train

    To run the training:

    ```bash
    python -m model_package.train
    ```

    ### Predict

    To run a prediction:

    ```bash
    python -m model_package.predict
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Web Services: Introduction to Flask

    The REST API server has been implemented in [api.py](https://github.com/elcapo/ml-zoomcamp-notes-and-homework/blob/main/module-5/model_package/api.py) a single method that responds **PONG** when the `/ping` method is accessed via GET.

    To run it:

    ```bash
    python -m model_package.api
    ```

    When you run it, you'll see a message telling you where the service is listening. Something like:

    > Running on http://127.0.0.1:5000

    You can use that address to test the service:

    ```bash
    curl http://127.0.0.1:5000/ping
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Serving the Churn Model with Flask

    In this chapter we added a `/predict` method to our API that responds to POST requests with JSONs similar to the dictionaries that we were obtaining from the dataset:

    ```json
    {
        "customerid": "7590-vhveg",
        "gender": "female",
        "seniorcitizen": 0,
        "partner": "yes",
        "dependents": "no",
        "tenure": 1,
        "phoneservice": "no",
        "multiplelines": "no_phone_service",
        "internetservice": "dsl",
        "onlinesecurity": "no",
        "onlinebackup": "yes",
        "deviceprotection": "no",
        "techsupport": "no",
        "streamingtv": "no",
        "streamingmovies": "no",
        "contract": "month-to-month",
        "paperlessbilling": "yes",
        "paymentmethod": "electronic_check",
        "monthlycharges": 29.85,
        "totalcharges": 29.85,
        "churn": "no"
    }
    ```

    ... and returns the corresponding prediction.
    """
    )
    return


if __name__ == "__main__":
    app.run()
