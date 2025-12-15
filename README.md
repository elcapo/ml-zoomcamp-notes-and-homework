# Machine Learning Zoomcamp

This repository contains my personal **notes and homework** of the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/).

### Notes

| Module | Editable Version | Readable Version |
| --- | --- | --- |
| Launch Stream | [Marimo Notebook](./notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/notes.html) |
| Introduction to Machine Learning | [Marimo Notebook](./module-1/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-1/notes.html) |
| Linear Regression | [Marimo Notebook](./module-2/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-2/notes.html)
| Classification | [Marimo Notebook](./module-3/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-3/notes.html)
| Evaluation | [Marimo Notebook](./module-4/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-4/notes.html)
| Deployment | [Marimo Notebook](./module-5/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-5/notes.html)
| Decision Trees | [Marimo Notebook](./module-6/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-6/notes.html)
| Deep Learning | [Marimo Notebook](./module-8/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-8/notes.html)
| Deploy with AWS Lambda | [Marimo Notebook](./module-9/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-9/notes.html) |
| Deploy with Kubernetes | [Marimo Notebook](./module-10/notes.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-10/notes.html) |

### Homework

| Module | Editable Version | Readable Version |
| --- | --- | --- |
| Introduction to Machine Learning | [Marimo Notebook](./module-1/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-1/homework.html) |
| Linear Regression | [Marimo Notebook](./module-2/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-2/homework.html) |
| Classification | [Marimo Notebook](./module-3/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-3/homework.html) |
| Evaluation | [Marimo Notebook](./module-4/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-4/homework.html) |
| Deployment | [Marimo Notebook](./module-5/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-5/homework.html) |
| Decision Trees | [Marimo Notebook](./module-6/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-6/homework.html) |
| Deep Learning | [Marimo Notebook](./module-8/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-8/homework.html) |
| Deploy with AWS Lambda | [Marimo Notebook](./module-9/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-9/homework.html) |
| Deploy with Kubernetes | [Marimo Notebook](./module-10/homework.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/module-10/homework.html) |

### Projects

| Module | Editable Version | Readable Version |
| --- | --- | --- |
| Midterm Project | [Marimo Notebook](./projects/midterm/notebook.py) | [Rendered HTML](https://raw.githack.com/elcapo/ml-zoomcamp-notes-and-homework/main/results/projects/midterm/notebook.html) |

## Installation

To download and edit a local copy of this project, follow this steps:

```bash
# Clone the repository
git clone https://github.com/elcapo/ml-zoomcamp-notes-and-homework
cd ml-zoomcamp-notes-and-homework

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

This will prepare an environment with all the dependencies needed to run all the notebooks, both for notes and homework.

## Edit the Notebooks

Additionally, you may want to know these commands, which are useful for editing the notebooks and publishing changes

```bash
# Edit the notebooks
marimo edit

# Publish the changes
./export.sh
```

Note that this way of publishing the changes (saving the results as standalone HTML files) is slow, as all the code from each notebook has to be executed.

As a more efficient alternative, you can use the "Download HTML" manual option whenever you finish editing a notebook.

## Special Environments

For certain projects, additional virtual environments are recommended.

### REST API of Module 5

To run the REST API implemented in module 5 that serves the model, these steps are recommended:

```bash
# Change to the project folder
cd module-5/

# Deactivate any active virtual environment
deactivate

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the model
python -m model_package.api
```

This will run the model "naked". If you want to run it in the context of a Docker container, we should do:

```bash
docker build -t ml-zoomcamp .
docker run -d -p 5000:5000 -it ml-zoomcamp
```
