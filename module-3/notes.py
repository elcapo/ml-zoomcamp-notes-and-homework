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

    ## Module 3: **Classification**
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"

    chapters = pd.DataFrame([

        {
            "title": "Churn Prediction Project",
            "youtube_id": "0Zw04wdeTQo",
            "contents": repository_root+"03-classification/01-churn-project.md"
        },
        {
            "title": "Data Preparation",
            "youtube_id": "VSGGU9gYvdg",
            "contents": repository_root+"03-classification/02-data-preparation.md"
        },
        {
            "title": "Setting up the Validation Framework",
            "youtube_id": "_lwz34sOnSE",
            "contents": repository_root+"03-classification/03-validation.md"
        },
        {
            "title": "Exporatory Data Analysis",
            "youtube_id": "BNF1wjBwTQA",
            "contents": repository_root+"03-classification/04-eda.md"
        },
        {
            "title": "Feature Importance: Churn Rate and Risk Ratio",
            "youtube_id": "fzdzPLlvs40",
            "contents": repository_root+"03-classification/05-risk.md"
        },
        {
            "title": "Feature Importance: Mutual Information",
            "youtube_id": "_u2YaGT6RN0",
            "contents": repository_root+"03-classification/06-mutual-info.md"
        },
        {
            "title": "Feature importance: Correlation",
            "youtube_id": "mz1707QVxiY",
            "contents": repository_root+"03-classification/07-correlation.md"
        },
        {
            "title": "One-hot Encoding",
            "youtube_id": "L-mjQFN5aR0",
            "contents": repository_root+"03-classification/08-ohe.md"
        },
        {
            "title": "Logistic Regression",
            "youtube_id": "7KFE2ltnBAg",
            "contents": repository_root+"03-classification/09-logistic-regression.md"
        },
        {
            "title": "Training Logistic Regression with Scikit-Learn",
            "youtube_id": "hae_jXe2fN0",
            "contents": repository_root+"03-classification/10-training-log-reg.md"
        },
        {
            "title": "Model Interpretation",
            "youtube_id": "OUrlxnUAAEA",
            "contents": repository_root+"03-classification/11-log-reg-interpretation.md"
        },
        {
            "title": "Using the Model",
            "youtube_id": "Y-NGmnFpNuM",
            "contents": repository_root+"03-classification/12-using-log-reg.md"
        },
        {
            "title": "Summary",
            "youtube_id": "Zz6oRGsJkW4",
            "contents": repository_root+"03-classification/13-summary.md"
        },
        {
            "title": "Explore More",
            "contents": repository_root+"03-classification/14-explore-more.md"
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
    ## Churn Prediction Project

    We'll be working with the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, with the goal of implementing a model that's able to predict if a customer is likely to "churn" (stop using the service).

    ### The Dataset

    The raw data contains 7043 rows (customers) and 21 columns (features).
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    pd.DataFrame([
        {
            "column": "customerID",
            "description": "Customer ID"
        },
        {
            "column": "gender",
            "description": "Whether the customer is a male or a female"
        },
        {
            "column": "SeniorCitizen",
            "description": "Whether the customer is a senior citizen or not (1, 0)"
        },
        {
            "column": "Partner",
            "description": "Whether the customer has a partner or not (Yes, No)"
        },
        {
            "column": "Dependents",
            "description": "Whether the customer has dependents or not (Yes, No)"
        },
        {
            "column": "tenure",
            "description": "Number of months the customer has stayed with the company"
        },
        {
            "column": "PhoneService",
            "description": "Whether the customer has a phone service or not (Yes, No)"
        },
        {
            "column": "MultipleLines",
            "description": "Whether the customer has multiple lines or not (Yes, No, No phone service)"
        },
        {
            "column": "InternetService",
            "description": "Customer’s internet service provider (DSL, Fiber optic, No)"
        },
        {
            "column": "OnlineSecurity",
            "description": "Whether the customer has online security or not (Yes, No, No internet service)"
        },
        {
            "column": "OnlineBackup",
            "description": "Whether the customer has online backup or not (Yes, No, No internet service)"
        },
        {
            "column": "DeviceProtection",
            "description": "Whether the customer has device protection or not (Yes, No, No internet service)"
        },
        {
            "column": "TechSupport",
            "description": "Whether the customer has tech support or not (Yes, No, No internet service)"
        },
        {
            "column": "StreamingTV",
            "description": "Whether the customer has streaming TV or not (Yes, No, No internet service) "
        },
        {
            "column": "StreamingMovies",
            "description": "Whether the customer has streaming movies or not (Yes, No, No internet service)"
        },
        {
            "column": "Contract",
            "description": "The contract term of the customer (Month-to-month, One year, Two year)"
        },
        {
            "column": "PaperlessBilling",
            "description": "Whether the customer has paperless billing or not (Yes, No)"
        },
        {
            "column": "PaymentMethod",
            "description": "The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card"
        },
        {
            "column": "MonthlyCharges",
            "description": "The amount charged to the customer monthly"
        },
        {
            "column": "TotalCharges",
            "description": "The total amount charged to the customer"
        },
        {
            "column": "Churn",
            "description": "Whether the customer churned or not (Yes or No)"
        },
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Local Dataset Path

    For convenience, the dataset has already been downloaded into [module-3/data/customer-churn.csv](module-3/data/customer-churn.csv).
    ///
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("module-3/data/customer-churn.csv")

    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The Goal

    What we'll try to do is to predict the **Churn** column, which will be our target variable and represents the customers who actually left the service.

    ### The Method

    We'll train a binary classification model that will attempt to classify customers in two groups, the ones that are right about to churn and the ones who are not.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Preparation

    ### Histograms of Numeric Columns
    """
    )
    return


@app.cell
def _(df, plt, sns):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(df.SeniorCitizen, bins=15, ax=axes[0])
    axes[0].set_title('Senior Citizen')

    sns.histplot(df.tenure, bins=15, ax=axes[1])
    axes[1].set_title('Tenure')

    sns.histplot(df.MonthlyCharges, bins=15, ax=axes[2])
    axes[2].set_title('Monthly Charges')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Standardize Column Names""")
    return


@app.cell
def _(df, pd):
    def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
        standardized = dataframe.copy()
        standardized.columns = standardized.columns.str.lower().str.replace(' ', '_')

        return standardized

    standardize_column_names(df)
    return (standardize_column_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Standardize Column Values""")
    return


@app.cell
def _(df, pd):
    def get_cateogorical_columns(dataframe: pd.DataFrame) -> list[str]:
        return list(list(dataframe.dtypes[dataframe.dtypes == 'object'].index))

    def standardize_categorical_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        standardized = dataframe.copy()

        for column in get_cateogorical_columns(standardized):
            standardized[column] = standardized[column].str.lower().str.replace(' ', '_')

        return standardized

    standardize_categorical_values(df)
    return (standardize_categorical_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prepare a Standardized Dataset""")
    return


@app.cell
def _(df, standardize_categorical_values, standardize_column_names):
    df_standardized = standardize_column_names(df)
    df_standardized = standardize_categorical_values(df_standardized)

    df_standardized
    return (df_standardized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Fix Column Data Types

    #### Total Charges

    First we'll convert the `totalcharges` column to numeric coercing (setting to null) the values that cannot be converted to numeric.
    """
    )
    return


@app.cell
def _(df_standardized, pd):
    totalcharges = pd.to_numeric(df_standardized.totalcharges, errors='coerce')

    totalcharges
    return (totalcharges,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, we'll check what values were lost during the transformation.""")
    return


@app.cell
def _(df_standardized, totalcharges):
    df_standardized[totalcharges.isnull()][["customerid", "totalcharges"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, we'll fill the empty values with zeroes.""")
    return


@app.cell(hide_code=True)
def _(df_standardized, sns, totalcharges):
    df_standardized.totalcharges = totalcharges.fillna(0)

    sns.histplot(df_standardized.totalcharges, bins=25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Churn


    Rather than a boolean column, the `churn` column contains strings with the values **yes** or **no**.
    """
    )
    return


@app.cell
def _(df_standardized):
    df_standardized.churn.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's fix it!""")
    return


@app.cell
def _(df_standardized):
    df_standardized.churn = (df_standardized.churn == 'yes').astype(int)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setting up the Validation Framework""")
    return


@app.cell
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
    return df_full, df_test, df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Check the Size of the Splits""")
    return


@app.cell
def _(df_full, df_test, df_train, df_val):
    {
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
        "full": len(df_full),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exploratory Data Analysis

    ### Check for Missing Values

    After a quick check we see that there are no missing values.
    """
    )
    return


@app.cell
def _(df_full):
    df_full.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Look at the Target Variable""")
    return


@app.cell
def _(df_full):
    df_full.churn.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Get the global average churn.""")
    return


@app.cell
def _(df_full):
    df_full.churn.mean().round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Display the Numerical Features""")
    return


@app.cell
def _(df_full):
    numerical_columns = ["tenure", "monthlycharges", "totalcharges"]

    df_full[["customerid"] + numerical_columns]
    return (numerical_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Look at the Categorical Variables""")
    return


@app.cell
def _(df_full):
    df_full.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Count the Different Values for Each Categorical Value""")
    return


@app.cell
def _(df_full):
    categorical_columns = [
        'gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
        'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
        'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod'
    ]

    df_full[categorical_columns].nunique()
    return (categorical_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Feature Importance: Churn Rate and Risk Ratio

    ### Churn Rate
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Check the Churn Rate for Each Gender

    As the mean of the churn for each gender is almost equal to the general mean, we can start guessing that gender is not going to be a very important feature when trying to predict the churn.
    """
    )
    return


@app.cell
def _(df_full):
    {
        "churn_female": df_full[df_full.gender == "female"].churn.mean().round(2),
        "churn_male": df_full[df_full.gender == "male"].churn.mean().round(2),
        "churn_general": df_full.churn.mean().round(2),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Check the Churn Rate for Each Partner

    As the mean of the churn changes significantly depending on whether the client has a partner or not, we can guess that this column will have a bigger impact when trying to predict the churn.
    """
    )
    return


@app.cell
def _(df_full):
    {
        "churn_partner": df_full[df_full.partner == "yes"].churn.mean().round(2),
        "churn_no_partner": df_full[df_full.partner == "no"].churn.mean().round(2),
        "churn_general": df_full.churn.mean().round(2),
    }
    return


@app.cell
def _(df_full, pd):
    def check_importance(column_name: str, dataframe: pd.DataFrame) -> pd.DataFrame:
        group = dataframe.groupby(column_name).churn.agg(["mean", "count"]).round(2)
        group["diff"] = (group["mean"] - dataframe.churn.mean()).round(2)
        group["risk"] = (group["mean"] / dataframe.churn.mean()).round(2)

        return group

    check_importance("gender", df_full)
    return (check_importance,)


@app.cell
def _(categorical_columns, check_importance, df_full, pd):
    def check_importances(dataframe: pd.DataFrame) -> list[pd.DataFrame]:
        importances = []

        for categorical_column in categorical_columns:
            importances.append(check_importance(categorical_column, dataframe))

        return importances

    check_importances(df_full)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Feature Importance: Mutual Information

    The mutual information metric tells us how much we know a variable through another variable.
    """
    )
    return


@app.cell
def _(df_full):
    from sklearn.metrics import mutual_info_score

    mutual_info_score(df_full.churn, df_full.contract)
    return (mutual_info_score,)


@app.cell
def _(categorical_columns, df_full, mutual_info_score, pd):
    def check_churn_importance(series: pd.Series) -> list[pd.Series]:
        return round(mutual_info_score(df_full.churn, series), 3)

    df_full[categorical_columns].apply(check_churn_importance).sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Feature Importance: Correlation

    We'll now consider correlation metrics between our variables. In particular, we'll use the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
    """
    )
    return


@app.cell
def _(df_full, numerical_columns):
    df_full[numerical_columns].corrwith(df_full.churn).abs()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, we'll check that as tenure increases, churn decreases.""")
    return


@app.cell
def _(df_full):
    churn_by_tenure = {
        "to_2": df_full[df_full.tenure < 2].churn.mean(),
        "from_2_to_12": df_full[(df_full.tenure >= 2) & (df_full.tenure < 12)].churn.mean(),
        "from_12": df_full[df_full.tenure >= 12].churn.mean(),
    }

    churn_by_tenure
    return (churn_by_tenure,)


@app.cell
def _(churn_by_tenure, sns):
    sns.barplot(x=["x < 2", "2 <= x < 12", "x >= 12"], y=churn_by_tenure.values())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, we'll check that as monthly charges increase, churn also increases.""")
    return


@app.cell
def _(df_full):
    churn_by_monthly_charges = {
        "to_30": df_full[df_full.monthlycharges < 30].churn.mean(),
        "from_30_to_80": df_full[(df_full.monthlycharges >= 30) & (df_full.monthlycharges < 80)].churn.mean(),
        "from_80": df_full[df_full.monthlycharges >= 80].churn.mean(),
    }

    churn_by_monthly_charges
    return (churn_by_monthly_charges,)


@app.cell
def _(churn_by_monthly_charges, sns):
    sns.barplot(x=["x < 30", "30 <= x < 80", "x >= 80"], y=churn_by_monthly_charges.values())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## One-hot Encoding""")
    return


@app.cell
def _():
    from sklearn.feature_extraction import DictVectorizer
    return (DictVectorizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""First we define a dictionary vectorizer and fit it with our full train dataframe.""")
    return


@app.cell
def _(DictVectorizer, categorical_columns, df_full, numerical_columns, pd):
    def get_trained_vectorizer(dataframe: pd.DataFrame) -> list[dict]:
        dictionary = dataframe[numerical_columns + categorical_columns].to_dict(orient="records")

        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(dictionary)

        return dict_vectorizer, dictionary

    dict_vectorizer, dictionary = get_trained_vectorizer(df_full)
    return dict_vectorizer, get_trained_vectorizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, let's test it with the first rows.""")
    return


@app.cell
def _(categorical_columns, df_full, dict_vectorizer):
    dict_vectorizer.transform(df_full[categorical_columns].iloc[:3].to_dict(orient="records"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, let's check the list of columns for each of the generated features.""")
    return


@app.cell
def _(dict_vectorizer):
    list(dict_vectorizer.get_feature_names_out())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prepare our Feature Matrices and Target Vectors""")
    return


@app.cell
def _(
    DictVectorizer,
    categorical_columns,
    df_train,
    df_val,
    get_trained_vectorizer,
    numerical_columns,
    pd,
):
    def get_features_and_target(dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer, dictionary):
        X = dict_vectorizer_train.transform(dictionary)
        y = dataframe.churn

        return X, y

    dict_vectorizer_train, dictionary_train = get_trained_vectorizer(df_train)
    X_train, y_train = get_features_and_target(df_train, dict_vectorizer_train, dictionary_train)

    dictionary_val = df_val[numerical_columns + categorical_columns].to_dict(orient="records")
    X_val, y_val = get_features_and_target(df_val, dict_vectorizer_train, dictionary_val)
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Logistic Regression

    In logistic regresion, our target variable can only have two values:

    \[
      \bold y_i = \{0, 1\}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Similarity with Linear Regression

    As a recall, in linear regression we used this formula for our estimator function:

    \[
        g(\bold x_i) = w_0 + \bold w^T \bold x_i
    \]

    Which was a function that had an image that could take any value between $-\infty$ and $\infty$.

    In other words, $g(\bold x_i) \in \mathbb{R}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Sigmoid

    The difference with linear regresion is that in the case of the logistic regression, we'll use a sigmoid function that will keep the values constrained between $0$ and $1$:

    \[
        g(\bold x_i) = sigmoid(w_0 + \bold w^T \bold x_i)
    \]
    """
    )
    return


@app.cell
def _(np, sns):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def plot_sigmoid():
        z = np.linspace(-15, 15, 500)
        y = sigmoid(z)

        return sns.lineplot(x=z, y=y)

    plot_sigmoid()
    return (sigmoid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As a quick recall of previous chapters, this is how linear regression looks like:""")
    return


@app.function
def linear_regression(xi, w0, wi):
    result = w0

    for x, w in zip(xi, wi):
        result += x * w

    return result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""With that in mind, it's quite easy to implement the logistic regression.""")
    return


@app.cell
def _(sigmoid):
    def logistic_regression(xi, w0, wi):
        z = linear_regression(xi, w0, wi)

        return sigmoid(z)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Training Logistic Regression with Scikit-Learn""")
    return


@app.cell
def _(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    {
        "coefficients": model.coef_[0].round(3),
        "bias": model.intercept_[0].round(3),
    }
    return LogisticRegression, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Use the Model

    First, we'll use the model to create some **hard predictions** for us. In other words, we'll force the model to generate booleans for us. We'll also take a look at the corresponding estimated **probabilities** and we'll compare all that with the real values.
    """
    )
    return


@app.cell
def _(X_val, model, y_val):
    {
        "hard_predictions": model.predict(X_val[0:5]).astype(bool),
        "probabilities": model.predict_proba(X_val[0:5])[:,1].round(3),
        "references": y_val[0:5].values.astype(bool),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By setting a manual **cut** point we can obtain a different set of predictions.""")
    return


@app.cell
def _(X_val, model):
    cut = 0.5

    {
        "soft_predictions": model.predict_proba(X_val[0:5])[:,1] > cut,
    }
    return (cut,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, let's do a quick evaluation of our model.""")
    return


@app.cell
def _(X_val, cut, model, y_val):
    predicted_churn = model.predict_proba(X_val)[:,1] > cut

    (predicted_churn == y_val).mean().round(3)
    return (predicted_churn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Inspect the Correct and Wrong Predictions""")
    return


@app.cell
def _(X_val, model, pd, predicted_churn, y_val):
    def inspect_predictions():
        dataframe = pd.DataFrame()
        dataframe["probability"] = model.predict_proba(X_val)[:,1].round(2)
        dataframe["prediction"] = predicted_churn.astype(int)
        dataframe["reference"] = y_val.astype(int)
        dataframe["correct"] = dataframe["prediction"] == dataframe["reference"]

        return dataframe

    inspect_predictions()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model Interpretation

    Our model has too many coefficients to study them one by one.
    """
    )
    return


@app.cell
def _(dict_vectorizer, model):
    list(zip(dict_vectorizer.get_feature_names_out(), model.coef_[0].round(3)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""So we'll create a small version of it.""")
    return


@app.cell
def _(DictVectorizer, df_train, pd):
    small_feature_set = ["contract", "monthlycharges", "tenure"]

    def get_small_trained_vectorizer(dataframe: pd.DataFrame) -> list[dict]:
        dictionary = dataframe[small_feature_set].to_dict(orient="records")

        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(dictionary)

        return dict_vectorizer, dictionary

    def get_small_dataset():
        return df_train[small_feature_set + ["churn"]]

    def get_small_features_and_target(dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer, dictionary):
        X = dict_vectorizer.transform(dictionary)
        y = dataframe.churn

        return X, y

    df_small = get_small_dataset()
    dict_vectorizer_small, dictionary_small = get_small_trained_vectorizer(df_small)
    X_small, y_small = get_small_features_and_target(df_small, dict_vectorizer_small, dictionary_small)
    return X_small, dict_vectorizer_small, y_small


@app.cell
def _(LogisticRegression, X_small, dict_vectorizer_small, y_small):
    model_small = LogisticRegression(max_iter=5000)
    model_small.fit(X_small, y_small)

    coefficients = dict(zip(dict_vectorizer_small.get_feature_names_out(), model_small.coef_[0].round(3)))
    coefficients
    return coefficients, model_small


@app.cell
def _(model_small):
    bias_small = model_small.intercept_[0].round(3)

    {
        "bias": bias_small,
    }
    return (bias_small,)


@app.cell
def _(X_small, bias_small, coefficients, sigmoid):
    def evaluate_row(row_number: int):
        x = X_small[row_number]

        return sigmoid(
            bias_small +
            coefficients["contract=month-to-month"] * x[0] +
            coefficients["contract=one_year"] * x[1] +
            coefficients["contract=two_year"] * x[2] +
            coefficients["monthlycharges"] * x[3] +
            coefficients["tenure"] * x[4]
        ).round(2)

    [evaluate_row(row_number) for row_number in range(10)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Using the Model

    ### Train a Model on the Full Dataset
    """
    )
    return


@app.cell
def _(DictVectorizer, df_full, pd):
    def get_full_trained_vectorizer(dataframe: pd.DataFrame) -> list[dict]:
        copy = dataframe.copy()
        del copy["churn"]
        dictionary = copy.to_dict(orient="records")

        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(dictionary)

        return dict_vectorizer, dictionary

    def get_full_features_and_target(dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer, dictionary):
        X = dict_vectorizer.transform(dictionary)
        y = dataframe.churn

        return X, y

    dict_vectorizer_full, dictionary_full = get_full_trained_vectorizer(df_full)
    X_full, y_full = get_full_features_and_target(df_full, dict_vectorizer_full, dictionary_full)
    return X_full, dict_vectorizer_full, get_full_features_and_target, y_full


@app.cell
def _(LogisticRegression, X_full, y_full):
    model_full = LogisticRegression(max_iter=5000)
    model_full.fit(X_full, y_full)
    return (model_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Evaluate the Model on the Test Dataset""")
    return


@app.cell
def _(
    categorical_columns,
    df_test,
    dict_vectorizer_full,
    get_full_features_and_target,
    numerical_columns,
):
    dictionary_test = df_test[numerical_columns + categorical_columns].to_dict(orient="records")
    X_test, y_test = get_full_features_and_target(df_test, dict_vectorizer_full, dictionary_test)

    y_test[y_test.values == 1].sum() / y_test.count()
    return X_test, dictionary_test, y_test


@app.cell
def _(X_test, model_full, pd):
    def predict_full(X):
        predictions = pd.DataFrame()
        predictions["probability"] = model_full.predict_proba(X)[:, 1]
        predictions["prediction"] = predictions["probability"] > 0.5

        return predictions

    y_pred = predict_full(X_test)
    y_pred
    return predict_full, y_pred


@app.cell
def _(y_pred):
    (y_pred.prediction == True).sum() / len(y_pred)
    return


@app.cell
def _(X_test, predict_full, y_test):
    def evaluate_full(X, y):
        y_pred = predict_full(X)

        return (y == y_pred.prediction).mean()

    evaluate_full(X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Use the Model with Data from some Random Customer Records""")
    return


@app.cell
def _(dictionary_test):
    from random import randrange

    def get_random_customer_ids(n):
        test_records = len(dictionary_test)

        return [randrange(1, test_records) for _ in range(n)]

    get_random_customer_ids(10)
    return (get_random_customer_ids,)


@app.cell
def _(dict_vectorizer_full, dictionary_test, get_random_customer_ids, np):
    def get_random_customers(n):
        random_customer_ids = get_random_customer_ids(n)
        dictionary_items = np.array(dictionary_test)[random_customer_ids]
        return dict_vectorizer_full.transform(dictionary_items)

    X_random_test = get_random_customers(10)
    X_random_test
    return (X_random_test,)


@app.cell
def _(X_random_test, predict_full):
    predict_full(X_random_test)
    return


if __name__ == "__main__":
    app.run()
