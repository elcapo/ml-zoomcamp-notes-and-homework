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
    return mo, pd, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Module 3: [Classification](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/03-classification)

    ## Homework

    ### Dataset

    For this homework, we'll use the Bank Marketing dataset. Download it from here. You can do it with wget:

    wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv

    In this dataset our desired target for classification task will be **converted** variable - has the client signed up to the platform or not.
    """
    )
    return


@app.cell
def _(pd):
    def read_dataframe():
        return pd.read_csv("./module-3/data/course_lead_scoring.csv")

    raw_dataframe = read_dataframe()
    raw_dataframe.head()
    return (raw_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### List Numeric and Categorical Columns""")
    return


@app.cell
def _(raw_dataframe):
    raw_dataframe.dtypes
    return


@app.cell
def _():
    numeric_columns = ["number_of_courses_viewed", "annual_income", "interaction_count", "lead_score"]
    categorical_columns = ["lead_source", "industry", "employment_status", "location"]
    return categorical_columns, numeric_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Preparation

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
    mo.md(r"""### For categorical features, replace them with 'NA'""")
    return


@app.cell
def _(categorical_columns, raw_dataframe):
    def fill_categorical_values(dataframe):
        dataframe[categorical_columns] = dataframe[categorical_columns].fillna('NA')

        return dataframe

    fill_categorical_values(raw_dataframe).head()
    return (fill_categorical_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### For numerical features, replace with with 0.0""")
    return


@app.cell
def _(numeric_columns, raw_dataframe):
    def fill_numerical_values(dataframe):
        dataframe[numeric_columns] = dataframe[numeric_columns].fillna(0)

        return dataframe

    fill_numerical_values(raw_dataframe).head()
    return (fill_numerical_values,)


@app.cell
def _(fill_categorical_values, fill_numerical_values, raw_dataframe):
    filled_dataframe = fill_categorical_values(raw_dataframe)
    filled_dataframe = fill_numerical_values(filled_dataframe)
    filled_dataframe.head()
    return (filled_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 1

    What is the most frequent observation (mode) for the column `industry`?

    - `NA`
    - `technology`
    - `healthcare`
    - `retail`
    """
    )
    return


@app.cell
def _(filled_dataframe):
    filled_dataframe.industry.mode()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 2

    Create the [correlation matrix](https://www.google.com/search?q=correlation+matrix) for the numerical features of your dataset. 
    In a correlation matrix, you compute the correlation coefficient between every pair of features.

    What are the two features that have the biggest correlation?

    - `interaction_count` and `lead_score`
    - `number_of_courses_viewed` and `lead_score`
    - `number_of_courses_viewed` and `interaction_count`
    - `annual_income` and `interaction_count`

    Only consider the pairs above when answering this question.
    """
    )
    return


@app.cell
def _(filled_dataframe, numeric_columns):
    filled_dataframe[numeric_columns].corr()
    return


@app.cell
def _(filled_dataframe, numeric_columns, sns):
    sns.heatmap(filled_dataframe[numeric_columns].corr(), annot=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The pair of features with greatest correlation is `annual_income` and `interaction_count`.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Split the data

    * Split your data in train/val/test sets with 60%/20%/20% distribution.
    * Use Scikit-Learn for that (the `train_test_split` function) and set the seed to `42`.
    * Make sure that the target value `y` is not in your dataframe.
    """
    )
    return


@app.cell
def _(filled_dataframe, raw_dataframe):
    from sklearn.model_selection import train_test_split

    full_dataframe, test_dataframe = train_test_split(filled_dataframe, test_size=0.2)
    train_dataframe, val_dataframe = train_test_split(full_dataframe, test_size=0.25)

    {
        "raw_size": len(raw_dataframe),
        "full_size": len(full_dataframe),
        "train_size": len(train_dataframe),
        "val_size": len(val_dataframe),
        "test_size": len(test_dataframe),
    }
    return train_dataframe, val_dataframe


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 3

    * Calculate the mutual information score between `y` and other categorical variables in the dataset. Use the training set only.
    * Round the scores to 2 decimals using `round(score, 2)`.

    Which of these variables has the biggest mutual information score?
  
    - `industry`
    - `location`
    - `lead_source`
    - `employment_status`
    """
    )
    return


@app.cell
def _(train_dataframe):
    from sklearn.metrics import mutual_info_score

    {
        "industry": round(mutual_info_score(train_dataframe.industry, train_dataframe.converted), 2),
        "location": round(mutual_info_score(train_dataframe.location, train_dataframe.converted), 2),
        "lead_source": round(mutual_info_score(train_dataframe.lead_source, train_dataframe.converted), 2),
        "employment_status": round(mutual_info_score(train_dataframe.employment_status, train_dataframe.converted), 2),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The variable with biggest mutual information score (compared with our **y** variable) is **lead_source**.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 4

    * Now let's train a logistic regression.
    * Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
    * Fit the model on the training dataset.
        - To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
        - `model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`
    * Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

    What accuracy did you get?

    - 0.64
    - 0.74
    - 0.84
    - 0.94
    """
    )
    return


@app.cell
def _(categorical_columns, numeric_columns, train_dataframe):
    from sklearn.feature_extraction import DictVectorizer

    def get_dict_vectorizer(dataframe):
        dict_vectorizer = DictVectorizer(sparse=False)
        dictionary = dataframe[numeric_columns + categorical_columns].to_dict(orient="records")
        dict_vectorizer.fit(dictionary)

        return dict_vectorizer

    train_dict_vectorizer = get_dict_vectorizer(train_dataframe)
    return DictVectorizer, train_dict_vectorizer


@app.cell
def _(categorical_columns, numeric_columns):
    def prepare_dataframe(dataframe, dict_vectorizer):
        y = dataframe.converted

        copy = dataframe.copy()
        del copy["converted"]

        dictionary = dataframe[numeric_columns + categorical_columns].to_dict(orient="records")
        X = dict_vectorizer.transform(dictionary)

        return X, y
    return (prepare_dataframe,)


@app.cell
def _(
    prepare_dataframe,
    train_dataframe,
    train_dict_vectorizer,
    val_dataframe,
):
    from sklearn.linear_model import LogisticRegression

    def eval_model():
        X_train, y_train = prepare_dataframe(train_dataframe, train_dict_vectorizer)
    
        model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
    
        X_val, y_val = prepare_dataframe(val_dataframe, train_dict_vectorizer)
    
        return round(model.score(X_val, y_val), 2)

    eval_model()
    return (LogisticRegression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 5 

    * Let's find the least useful feature using the *feature elimination* technique.
    * Train a model using the same features and parameters as in Q4 (without rounding).
    * Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
    * For each feature, calculate the difference between the original accuracy and the accuracy without the feature. 

    Which of following feature has the smallest difference?

    - `'industry'`
    - `'employment_status'`
    - `'lead_score'`

    > **Note**: The difference doesn't have to be positive.
    """
    )
    return


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    categorical_columns,
    numeric_columns,
    train_dataframe,
    val_dataframe,
):
    def get_reduced_dict_vectorizer(dataframe, ignored_column):
        dict_vectorizer = DictVectorizer(sparse=False)

        reduced_columns = numeric_columns + categorical_columns
        reduced_columns.remove(ignored_column)

        dictionary = dataframe[reduced_columns].to_dict(orient="records")
        dict_vectorizer.fit(dictionary)

        return dict_vectorizer

    def prepare_reduced_dataframe(dataframe, dict_vectorizer, ignored_column):
        y = dataframe.converted

        copy = dataframe.copy()
        del copy["converted"]

        reduced_columns = numeric_columns + categorical_columns
        reduced_columns.remove(ignored_column)

        dictionary = dataframe[reduced_columns].to_dict(orient="records")
        X = dict_vectorizer.transform(dictionary)

        return X, y

    def evaluate_without_feature(train_dataframe, val_dataframe, ignored_column):
        train_reduced_dataframe = train_dataframe.copy()
        del train_reduced_dataframe[ignored_column]

        train_reduced_dict_vectorizer = get_reduced_dict_vectorizer(train_reduced_dataframe, ignored_column)

        X_train_reduced, y_train_reduced = prepare_reduced_dataframe(
            train_reduced_dataframe,
            train_reduced_dict_vectorizer,
            ignored_column
        )
    
        model_reduced = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        model_reduced.fit(X_train_reduced, y_train_reduced)

        val_reduced_dataframe = val_dataframe.copy()
        del val_reduced_dataframe[ignored_column]

        X_val_reduced, y_val_reduced = prepare_reduced_dataframe(
            val_reduced_dataframe,
            train_reduced_dict_vectorizer,
            ignored_column
        )
    
        return round(model_reduced.score(X_val_reduced, y_val_reduced), 2)

    {
        "industry": evaluate_without_feature(train_dataframe, val_dataframe, "industry"),
        "employment_status": evaluate_without_feature(train_dataframe, val_dataframe, "employment_status"),
        "lead_score": evaluate_without_feature(train_dataframe, val_dataframe, "lead_score"),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The feature that caused the smallest change was **lead_score**.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 6

    * Now let's train a regularized logistic regression.
    * Let's try the following values of the parameter `C`: `[0.01, 0.1, 1, 10, 100]`.
    * Train models using all the features as in Q4.
    * Calculate the accuracy on the validation dataset and round it to 3 decimal digits.

    Which of these `C` leads to the best accuracy on the validation set?

    - 0.01
    - 0.1
    - 1
    - 10
    - 100

    > **Note**: If there are multiple options, select the smallest `C`.
    """
    )
    return


@app.cell
def _(
    LogisticRegression,
    prepare_dataframe,
    train_dataframe,
    train_dict_vectorizer,
    val_dataframe,
):
    def eval_model_parameter(C):
        X_train, y_train = prepare_dataframe(train_dataframe, train_dict_vectorizer)
    
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
    
        X_val, y_val = prepare_dataframe(val_dataframe, train_dict_vectorizer)
    
        return round(model.score(X_val, y_val), 2)

    {
        "0.01": eval_model_parameter(0.01),
        "0.1": eval_model_parameter(0.1),
        "1": eval_model_parameter(1),
        "10": eval_model_parameter(10),
        "100": eval_model_parameter(100),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The value C = **0.01** lead for the greatest accuracy.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Submit the results

    * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw03
    * If your answer doesn't match options exactly, select the closest one
    """
    )
    return


if __name__ == "__main__":
    app.run()
