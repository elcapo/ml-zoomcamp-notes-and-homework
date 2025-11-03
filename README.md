# Machine Learning Zoomcamp

This repository contains my personal **notes and homework** of the Machine Learning Zoomcamp.

### Content

| Module | Content | Editable Version | Readable Version |
| --- | --- | --- | --- |
| Welcome | **Launch Stream** | [Marimo Notebook](./notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/notes.html) |
| Introduction to Machine Learning | **Notes**  | [Marimo Notebook](./module-1/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-1/notes.html) |
| Introduction to Machine Learning | **Homework** | [Marimo Notebook](./module-1/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-1/homework.html) |
| Linear Regression | **Notes** | [Marimo Notebook](./module-2/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-2/notes.html)
| Linear Regression | **Homework** | [Marimo Notebook](./module-2/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-2/homework.html) |
| Classification | **Notes** | [Marimo Notebook](./module-3/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-3/notes.html)
| Classification | **Homework** | [Marimo Notebook](./module-3/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-3/homework.html) |
| Evaluation | **Notes** | [Marimo Notebook](./module-4/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-4/notes.html)
| Evaluation | **Homework** | [Marimo Notebook](./module-4/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-4/homework.html) |
| Deployment | **Notes** | [Marimo Notebook](./module-5/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-5/notes.html)
| Deployment | **Homework** | [Marimo Notebook](./module-5/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-5/homework.html) |
| Deployment | **Notes** | [Marimo Notebook](./module-6/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-6/notes.html)

## Installation

### Clone the Repository

```bash
git clone https://github.com/elcapo/ml-zoomcamp-notes-and-homework
cd ml-zoomcamp-notes-and-homework
```

### Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Edit the Notebooks

```bash
marimo edit
```

### Export the Results (as HTML)

```bash
./export.sh
```

## Run the Model

To run the REST API implemented in module 5 that serves the model:

```bash
source .venv/bin/activate
cd module-5/
python -m model_package.api
```

This will run the model "naked". If we want it in the context of a Docker container, we should do:

```bash
cd module-5/

docker build -t ml-zoomcamp .
docker run -d -p 5000:5000 -it ml-zoomcamp
```