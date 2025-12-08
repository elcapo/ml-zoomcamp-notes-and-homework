import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torchvision import models, transforms
    from PIL import Image
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning Zoomcamp

    ## Module 9: **Deploy with AWS Lambda**
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = (
        "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"
    )

    chapters = pd.DataFrame(
        [
            {
                "title": "Introduction to serverless",
                "youtube_id": "JLIVwIsU6RA",
                "contents": repository_root + "09-serverless/01-intro.md",
            },
            {
                "title": "AWS Lambda",
                "youtube_id": "_UX8-2WhHZo",
                "contents": repository_root + "09-serverless/02-aws-lambda.md",
            },
            {
                "title": "Serverless Deployment",
                "youtube_id": "sHQaeVm5hT8",
                "contents": repository_root + "09-serverless/workshop/README.md",
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
    mo.md(r"""
    ## Scikit Learn

    We'll start by deploying the Scikit Learn model that we developed during the 5th module. The model has been implemented under **module-9/scikit-model**.
    """)
    return


if __name__ == "__main__":
    app.run()
