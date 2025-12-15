import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning Zoomcamp

    ## Module 10: **Deploying with Kubernetes**
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
                "title": "Overview",
                "youtube_id": "mvPER7YfTkw",
                "contents": repository_root + "10-kubernetes/01-overview.md",
            },
            {
                "title": "TensorFlow Serving",
                "youtube_id": "deXR2fThYDw",
                "contents": repository_root + "10-kubernetes/02-tensorflow-serving.md",
            },
            {
                "title": "Creating a pre-processing service",
                "youtube_id": "OIlrS14Zi0o",
                "contents": repository_root + "10-kubernetes/03-preprocessing.md",
            },
            {
                "title": "Running everything locally with Docker-compose",
                "youtube_id": "ZhQQfpWfkKY",
                "contents": repository_root + "10-kubernetes/04-docker-compose.md",
            },
            {
                "title": "Introduction to Kubernetes",
                "youtube_id": "UjVkpszDzgk",
                "contents": repository_root + "10-kubernetes/05-kubernetes-intro.md",
            },
            {
                "title": "Deploying a simple service to Kubernetes",
                "youtube_id": "PPUCVRIV9t8",
                "contents": repository_root + "10-kubernetes/06-kubernetes-simple-service.md",
            },
            {
                "title": "Deploying TensorFlow models to Kubernetes",
                "youtube_id": "6vHLMdnjO2w",
                "contents": repository_root + "10-kubernetes/07-kubernetes-tf-serving.md",
            },
            {
                "title": "Deploying to EKS",
                "youtube_id": "89jxeddZtC0",
                "contents": repository_root + "10-kubernetes/08-eks.md",
            },
            {
                "title": "Summary",
                "youtube_id": "J5LMRTIu4jY",
                "contents": repository_root + "10-kubernetes/09-summary.md",
            },
            {
                "title": "Workshop",
                "youtube_id": "c_CzCsCnWoU",
                "contents": repository_root + "10-kubernetes/workshop/README.md",
            },
            {
                "title": "Homework",
                "contents": repository_root + "10-kubernetes/homework.md",
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
    ## Overview

    We'll be deploying models that we built on previous modules using Docker and Kubernetes.

    > Kubernetes automates operational tasks of container management and includes built-in commands for deploying applications, rolling out changes to your applications, scaling your applications up and down to fit changing needs, monitoring your applications, and more—making it easier to manage applications.

    Source: [https://cloud.google.com/learn/what-is-kubernetes?hl=es-419](https://cloud.google.com/learn/what-is-kubernetes?hl=en)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tools

    Assuming that we already have Docker installed, we still need to install **kubectl** and **kind**.

    > The Kubernetes command-line tool, **kubectl**, allows you to run commands against Kubernetes clusters. You can use kubectl to deploy applications, inspect and manage cluster resources, and view logs. For more information including a complete list of kubectl operations, see the kubectl reference documentation.
    >
    > **kind** lets you run Kubernetes on your local computer. This tool requires that you have either Docker or Podman installed.

    Source: https://kubernetes.io/docs/tasks/tools/
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tool Checks

    These tests help you to check that you have all the tools installed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `docker`

    By running this command:

    ```bash
    docker run --rm hello-world
    ```

    ... you should obtain a response that includes this output:

    ```
    Hello from Docker!
    This message shows that your installation appears to be working correctly.
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `kubectl`

    By running this command:

    ```bash
    kubectl --help
    ```

    ... you should obtain a response that includes this output:

    ```
    kubectl controls the Kubernetes cluster manager.

     Find more information at: https://kubernetes.io/docs/reference/kubectl/
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `kind`

    By running this command:

    ```bash
    kind --help
    ```

    ... you should obtain a response that includes this output:

    ```
    kind creates and manages local Kubernetes clusters using Docker container 'nodes'
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Serve a Model via API

    An API endpoint was implemented in the **module-10/workshop/** folder. To start it, go to the folder and run:

    ```bash
    uv run main.py
    ```

    That should open an endpoint that you can target:

    ```bash
    curl -X POST "http://localhost:9696/predict" \
      -H "Content-Type: application/json" \
      -d '{"url":"https://example.com/image.jpg"}'
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
