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
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Module 5: [Deployment](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment)

    > Note: sometimes your answer doesn't match one of the options exactly. 
    > That's fine. 
    > Select the option that's closest to your solution.
    > If it's exactly in between two options, select the higher value.

    We recommend using python 3.12 or 3.13 in this homework.

    In this homework, we're going to continue working with the lead scoring dataset. You don't need the dataset: we will provide the model for you.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 1

    * Install `uv0.7.12`
    * What's the version of uv you installed?
    * Use `--version` to find out


    ## Initialize an empty uv project

    You should create an empty folder for homework
    and do it there.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Answer

    First, `uv` was installed following the official installation instructions:

    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    The, with `uv --version` we found that we are using **0.7.12**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 2

    * Use uv to install Scikit-Learn version 1.6.1
    * What's the first hash for Scikit-Learn you get in the lock file?
    * Include the entire string starting with sha256:, don't include quotes
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Answer

    The `scikit-learn` module was installed with:

    ```bash
    uv init
    uv add scikit-learn==1.6.1
    ```

    The first hash found for **scikit-learn** was:

    ```
    [[package]]
    name = "scikit-learn"
    version = "1.6.1"
    source = { registry = "https://pypi.org/simple" }
    dependencies = [
        { name = "joblib" },
        { name = "numpy" },
        { name = "scipy" },
        { name = "threadpoolctl" },
    ]
    sdist = { url = "https://files.pythonhosted.org/packages/9e/a5/4ae3b3a0755f7b35a280ac90b28817d1f380318973cff14075ab41ef50d9/scikit_learn-1.6.1.tar.gz", hash = "sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e", size = 7068312, upload-time = "2025-01-10T08:07:55.348Z" }
    ```

    Which refers to this hash:

    ```
    sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Models

    We have prepared a pipeline with a dictionary vectorizer and a model.

    It was trained (roughly) using this code:

    ```python
    categorical = ['lead_source']
    numeric = ['number_of_courses_viewed', 'annual_income']

    df[categorical] = df[categorical].fillna('NA')
    df[numeric] = df[numeric].fillna(0)

    train_dict = df[categorical + numeric].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )

    pipeline.fit(train_dict, y_train)
    ```

    > **Note**: You don't need to train the model. This code is just for your reference.

    And then saved with Pickle. Download it [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/cohorts/2025/05-deployment/pipeline_v1.bin).

    With `wget`:

    ```bash
    wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 3

    Let's use the model!

    * Write a script for loading the pipeline with pickle
    * Score this record:

    ```json
    {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }
    ```

    If you're getting errors when unpickling the files, check their checksum:

    ```bash
    $ md5sum pipeline_v1.bin
    7d17d2e4dfbaf1e408e1a62e6e880d49 *pipeline_v1.bin
    ```

    What's the probability that this lead will convert? 

    * 0.333
    * 0.533
    * 0.733
    * 0.933
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Answer

    With this code:

    ```python
    import pickle

    with open("pipeline_v1.bin", "rb") as f:
        dict_vectorizer, model = pickle.load(f)

    record = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    x = dict_vectorizer.transform(record)

    model.predict_proba(x)[:, 1]
    ```

    We obtain a prediction of a probability of **0.533**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 4

    Now let's serve this model as a web service

    * Install FastAPI
    * Write FastAPI code for serving the model
    * Now score this client using `requests`:

    ```python
    url = "YOUR_URL"
    client = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0
    }
    requests.post(url, json=client).json()
    ```

    What's the probability that this client will get a subscription?

    * 0.334
    * 0.534
    * 0.734
    * 0.934
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Answer

    A project with these dependencies was created:

    ```toml
    dependencies = [
        "fastapi[standard]>=0.120.0",
        "scikit-learn==1.6.1",
    ]
    ```

    Then, to set up our FastAPI server, this code was used:

    ```python
    import pickle
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Client(BaseModel):
        lead_source: str
        number_of_courses_viewed: int
        annual_income: float

    with open("pipeline_v1.bin", "rb") as f:
        dict_vectorizer, model = pickle.load(f)

    app = FastAPI()

    @app.post("/predict")
    def predict(client: Client):
        x = dict_vectorizer.transform(client.model_dump())

        return model.predict_proba(x)[:, 1]
    ```

    The server was started by running:

    ```bash
    uv run fastapi dev
    ```

    With the code provided above, we obtained **0.534**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Docker

    Install [Docker](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/06-docker.md). 
    We will use it for the next two questions.

    For these questions, we prepared a base image: `agrigorev/zoomcamp-model:2025`. 
    You'll need to use it (see Question 5 for an example).

    This image is based on `3.13.5-slim-bookworm` and has
    a pipeline with logistic regression (a different one)
    as well a dictionary vectorizer inside. 

    This is how the Dockerfile for this image looks like:

    ```docker 
    FROM python:3.13.5-slim-bookworm
    WORKDIR /code
    COPY pipeline_v2.bin .
    ```

    We already built it and then pushed it to [`agrigorev/zoomcamp-model:2025`](https://hub.docker.com/r/agrigorev/zoomcamp-model).

    > **Note**: You don't need to build this docker image, it's just for your reference.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 5

    Download the base image `agrigorev/zoomcamp-model:2025`. You can easily make it by using [docker pull](https://docs.docker.com/engine/reference/commandline/pull/) command.

    So what's the size of this base image?

    * 45 MB
    * 121 MB
    * 245 MB
    * 330 MB

    You can get this information when running `docker images` - it'll be in the "SIZE" column.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Answer

    These commands were used:

    ```bash
    docker pull agrigorev/zoomcamp-model:2025
    docker images | grep "agrigorev/zoomcamp-model"
    ```

    ... to check that the size of the image is **121MB**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dockerfile

    Now create your own `Dockerfile` based on the image we prepared.

    It should start like that:

    ```docker
    FROM agrigorev/zoomcamp-model:2025
    # add your stuff here
    ```

    Now complete it:

    * Install all the dependencies from pyproject.toml
    * Copy your FastAPI script
    * Run it with uvicorn 

    After that, you can build your docker image.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Answer

    We build this image:

    ```bash
    FROM agrigorev/zoomcamp-model:2025

    WORKDIR /code

    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    COPY ["main.py", "pipeline_v1.bin", "pyproject.toml", "uv.lock", "./"]

    RUN uv sync --frozen --no-cache

    EXPOSE 8001
    CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
    ```

    and then we built it and run it with:

    ```bash
    docker build -t ml-homework .
    docker run -p 8001:8001 -it --rm ml-homework
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Question 6

    Let's run your docker container!

    After running it, score this client once again:

    ```python
    url = "YOUR_URL"
    client = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0
    }
    requests.post(url, json=client).json()
    ```

    What's the probability that this lead will convert?

    * 0.39
    * 0.59
    * 0.79
    * 0.99
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Answer

    Similar to what we obtained in question 4 (as we are working with the same model and we used the same input) we obtained a churn estimated probability of: **0.534**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Submit the results

    * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw05
    * If your answer doesn't match options exactly, select the closest one. If the answer is exactly in between two options, select the higher value.



    ## Publishing to Docker hub

    This is just for reference, this is how we published an image to Docker hub.

    `Dockerfile_base`: 

    ```dockerfile
    FROM python:3.13.5-slim-bookworm
    WORKDIR /code
    COPY pipeline_v2.bin .
    ```

    Publishing:

    ```bash
    docker build -t mlzoomcamp2025_hw5 -f Dockerfile_base .
    docker tag mlzoomcamp2025_hw5:latest agrigorev/zoomcamp-model:2025
    docker push agrigorev/zoomcamp-model:2025
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
