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

    ## Course Launch
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    chapters = pd.DataFrame([
        {
            "title": "2025 Course Launch Stream",
            "youtube_id": "z064DoidiKg"
        },
    ])

    chapters.insert(loc=0, column="snapshot", value="https://img.youtube.com/vi/"+chapters.youtube_id.astype(str)+"/hqdefault.jpg")
    chapters.insert(loc=2, column="youtube", value="https://youtube.com/watch?v="+chapters.youtube_id.astype(str))

    videos = chapters[chapters["youtube_id"].notnull()]
    videos[["snapshot", "title", "youtube"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Quick start checklist

    * Check [the course repository](https://github.com/DataTalksClub/machine-learning-zoomcamp) on GitHub and star it. All the materials are stored in this repo.
    * Subscribe to [DataTalks.Club's YouTube channel](https://www.youtube.com/c/DataTalksClub) and check [the course playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR).
    * Check the [FAQ](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit?tab=t.0) for the list of frequently asked questions.
    * Join the [Slack](https://datatalks-club.slack.com/ssb/redirect) course channel for discussions and the [Telegram channel](https://t.me/mlzoomcamp) for announcements.
    * Check the [course platform](https://courses.datatalks.club/) for deadlines and open submissions.

    """
    )
    return


if __name__ == "__main__":
    app.run()
