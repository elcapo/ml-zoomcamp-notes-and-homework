import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Machine Learning Zoomcamp

    ## Module 1: **Introduction to Machine Learning**
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"

    chapters = pd.DataFrame([
        {
            "title": "Introduction to Machine Learning",
            "youtube_id": "Crm_5n4mvmg",
            "contents": repository_root+"01-intro/01-what-is-ml.md"
        },
        {
            "title": "Machine Learning vs. Rule-Based Systems",
            "youtube_id": "CeukwyUdaz8",
            "contents": repository_root+"01-intro/02-ml-vs-rules.md"
        },
        {
            "title": "Supervised Machine Learning",
            "youtube_id": "j9kcEuGcC2Y",
            "contents": repository_root+"01-intro/03-supervised-ml.md"
        },
        {
            "title": "CRoss-Industry Standard Process for Data Mining",
            "youtube_id": "dCa3JvmJbr0",
            "contents": repository_root+"01-intro/04-crisp-dm.md"
        },
        {
            "title": "Model Selection Process",
            "youtube_id": "OH_R0Sl9neM",
            "contents": repository_root+"01-intro/05-model-selection.md"
        },
        {
            "title": "GitHub Codespaces",
            "youtube_id": "pqQFlV3f9Bo",
            "contents": repository_root+"01-intro/06-environment.md"
        },
        {
            "title": "Introduction to NumPy",
            "youtube_id": "Qa0-jYtRdbY",
            "contents": repository_root+"01-intro/07-numpy.md"
        },
        {
            "title": "Linear Algebra Refresher",
            "youtube_id": "zZyKUeOR4Gg",
            "contents": repository_root+"01-intro/08-linear-algebra.md"
        },
        {
            "title": "Introduction to Pandas",
            "youtube_id": "0j3XK5PsnxA",
            "contents": repository_root+"01-intro/09-pandas.md"
        },
        {
            "title": "Final Summary",
            "youtube_id": "VRrEEVeJ440",
            "contents": repository_root+"01-intro/10-summary.md"
        },
        {
            "title": "Homework",
            "contents": repository_root+"01-intro/homework.md"
        }
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
    ## Introduction to Machine Learning

    /// details | Car Prices Dataset
        type: info

    To make this notes more realistic, we'll use Kaggle's [sidharth178/car-prices-dataset](https://www.kaggle.com/datasets/sidharth178/car-prices-dataset) dataset.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    car_prices = pd.read_csv("./module-1/data/car-prices/train.csv")
    car_prices.head()
    return (car_prices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To get a first idea of what is Machine Learning, we can imagine that we own a website where people can sell their used cars. A first problem in which we could use Machine Learning is to assist users when setting a price for their car.

    In this problem we'd start with the car's **features**. For instance:
    """
    )
    return


@app.cell(hide_code=True)
def _(car_prices):
    [column for column in car_prices.columns if not column in ("ID", "Price")]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ... and we'll try to guess its price. Typically, we'll call **target** to the variable that we are trying to guess.

    Having a list of cars that contains a certain set of features and their corresponding prices, we'll be able to:

    * First, we train a model so that it learns to relate prices with their **features**.
    * Then, we use the model so that given a set of features, it guesses its **target**; in our case, the car's price.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Machine Learning vs. Rule-Based Systems

    In this chapter we compare the "classic" way of creating programs with the Machine Learning approach. As an example, we imagine how could we create a program that works as an antispam detector, receiving an email and classifying it as spam or not spam.

    ```python
    class Email:
        from_email_address: str
        to_email_addresses: list[str]
        cc_email_addresses: list[str]
        subject: str
        message: str
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Rule Based

    To create a rule based antispam we would create a program that checks for different things, which we would usually add one by one after examining the previous emails that we've been able to manually classify as either legit or spam:

    ```python
    def has_suspicious_from(email: Email) -> bool:
        suspicious_literals = ["spam", "mailinator"]
        return any([
            email.from_email_address.find(literal) > -1
            for literal in suspicious_literals
        ])

    def is_spam(email: Email) -> bool:
        return has_suspicious_from(email)
    ```

    This looks great, it can work and detect some spam messages but we'd very likely going to have to adapt the code as we receive more emails and we find that our code needs a more complex logic, for instance, checking if the email has many targets or copies to too many people.

    ```python
    def has_suspicious_targets(email: Email) -> bool:
        return len(email.to_email_addresses) + len(email.cc_email_addresses) > 5

    def is_spam(email: Email) -> bool:
        return has_suspicious_from(email) or has_suspicious_targets(email)
    ```

    The more criteria we want to take into account, the more code we'll have and the more complex it will become.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Machine Learning

    To create a machine learning model that solves the same issue, we follow a simpler process:

    1. Get the data
    2. Define its features and create a dataset linking them to a target variable (the spam flag)
    3. Train a classifier model on the new dataset
    4. Use the trained model to check new emails

    Thanks to this approach we can create a **spam** button that users will click when they see a message that's spam and use it as our source of information to iteratively train our model on new spam and legitimate emails.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### From Rule Based Systems to Machine Learning

    When migrating from rule based systems to machine learning approaches we don't have to throw everything to the trash can. Instead, we can start using many of the initial rules as the base for the features of our model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Supervised Machine Learning

    * On one hand, as the **features** are in most cases many numeric variables per item, we'll use an upper case $X$ to represent its matrix nature.

    * On the other hand, as the **target** variables can be represented by a single number per item, we can use a vector to represent the answers that correspond to a given input matrix.

    Training a model that can predict our target variable given a list of features for a batch of items can be seen as creating a function $g$ such that:

    \[
    g(X) = y
    \]

    Before training a model, we'd transform all features into numerical representations.

    This is our $X$ matrix:
    """
    )
    return


@app.cell(hide_code=True)
def _(car_prices, pd):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import GammaRegressor
    from sklearn.preprocessing import StandardScaler

    def first_pass(original):
        passed = original.copy()
        passed["Levy"] = pd.to_numeric(passed["Levy"].replace("-", 0), errors="coerce")
        passed["Mileage"] = passed["Mileage"].str.replace(" km", "", regex=False).astype(float)
        passed["Engine volume"] = pd.to_numeric(passed["Engine volume"], errors="coerce")
        passed["Engine volume"] = passed["Engine volume"].fillna(passed["Engine volume"].median())
        return passed

    car_train = first_pass(car_prices)

    X_train = car_train.drop(columns=["ID", "Price"])
    y_train = car_train["Price"]

    numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)

    feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = list(numerical_features) + list(feature_names)

    X_train_preprocessed = pd.DataFrame(
        X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed,
        columns=all_feature_names
    )

    X_train_preprocessed.head()
    return (
        GammaRegressor,
        Pipeline,
        StandardScaler,
        X_train,
        first_pass,
        preprocessor,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, we can get a dataset for which we have the features but not the price and use our model to tell our trained model to compute a **Predicted Price**.""")
    return


@app.cell(hide_code=True)
def _(
    GammaRegressor,
    Pipeline,
    StandardScaler,
    X_train,
    first_pass,
    pd,
    preprocessor,
    y_train,
):
    car_test = pd.read_csv("./module-1/data/car-prices/test.csv")
    car_test = first_pass(car_test)

    X_test = car_test.drop(columns=["ID", "Price"])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("regressor", GammaRegressor(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    X_test.insert(1, "Predicted Price", value=y_predicted.astype(int))
    X_test
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## CRoss-Industry Standard Process for Data Mining

    ### The Problem

    Going back to the spam detection example, what we did was:

    * we defined our goal (to detect whether a message is spam, or not)
    * we extracted some features
    * we trained a model
    * we used the model with test data to evaluate it

    These steps were a basic representation of whet the "CRISP-DM" methodology tries to solve.
    """
    )
    return


@app.cell
def _(mo):
    mo.image("./module-1/assets/crisp-dm.jpeg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Business Understanding

    In a real world case, many departments of a big organization have participate in a data mining problem. The first step consists in uderstanding the problem that we are trying to solve but not from a technical point of view but from a business point of view instead. Actually, we shouldn't decide whether to start a Machine Learning problem until we really understand the business problem we are facing.

    At this step, the most important thing is to stablish measurable goals.

    /// details | **Example Goal for the Spam Detection Problem**
        type: info

    We want to reduce the number of spam messages to a 50%.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Data Understanding

    Once we understand the problem that we are trying to solve, we have to gather the data that we have available and make our best to understand it. At this step, we have to ask ourselves a few questions:

    * Where do the data come from?
    * Is it reliable?
    * Is the dataset big enough?
    * Can we collect more?

    /// details | **Example Questions for the Spam Detection Problem**
        type: info

    Can we ask our users to mark incoming messages as spam (or not spam)?
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Data Preparation

    To prepare the data so that it can be put into a Machine Learning algorithm involves several steps:

    * Extract some features from raw data
    * Remove duplicated records
    * Transform the data into numeric values
    * Creating different splits for training and validation

    /// details | **Example Preparations for the Spam Detection Problem**
        type: info

    Does the subject contain more than 25 characters?
    Does the sender contain "mailinator"?
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Modeling

    At this step we choose different models, train them with our training dataset split and choose the best one according to some metrics.

    /// details | **Example Models for the Spam Detection Problem**
        type: info

    In the spam detection case we could choose between logistic regression, decision trees, neural networks, etc.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Evaluation

    Here we check if we managed to reach the goals that we stablished during the first step.

    * Was the goal achievable?
    * Did we reach it?
    * What can we do to get closer in the next iteration?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Deployment

    Finally, we deploy our new models so that they are accessible by our end users.

    * Deployment usually is tied to evaluation because there is no better evaluation than the one of end users.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model Selection Process

    In this chapter, we are focusing on the **modeling** part that we described above.

    ### Train and Validation Splits

    The first and probably most important technique used to create models is to split our dataset into two different parts:

    * Around an 80% of the dataset will become our **train** dataset
    * We'll keep the remaining part hidden from the model for **validation** purposes

    That will let us use the model to create predictions of cases that it has not seen during its training. As we have the correct answers for those cases, we'll be able to measure the differences between the correct answers and the answers generated by the model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Multiple Comparison Problem

    When testing many models or hyperparameter settings, the chance of finding a good result just by luck increases with the number of models and different settings that we consider. So we may find models that perform well on one dataset but fail to generalize.

    A technique that helps with this is to split our dataset in three (not two) parts:

    * **Training** set: Used to fit the model parameters.
    * **Validation** set: Used to tune hyperparameters and compare models (avoids overfitting directly to the training set).
    * **Test** set: Held out until the very end to measure the true generalization performance.
    """
    )
    return


if __name__ == "__main__":
    app.run()
