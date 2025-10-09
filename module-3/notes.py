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


if __name__ == "__main__":
    app.run()
