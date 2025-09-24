import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Module 1 - [Introduction to Machine Learning](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/01-intro)

    ## Homework

    ### Set up the environment

    You need to install Python, NumPy, Pandas, Matplotlib and Seaborn. For that, you can use the instructions from
    [06-environment.md](../../../01-intro/06-environment.md).

    ### Q1. Pandas version

    What's the version of Pandas that you installed?
    """
    )
    return


@app.cell
def _(pd):
    pd.__version__
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Getting the data 

    For this homework, we'll use the Car Fuel Efficiency dataset. Download it from <a href='https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'>here</a>.

    You can do it with wget:
    ```bash
    wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
    ```

    Or just open it with your browser and click "Save as...".

    Now read it with Pandas.
    """
    )
    return


@app.cell
def _(pd):
    cars = pd.read_csv("./data/car_fuel_efficiency.csv")
    return (cars,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q2. Records count

    How many records are in the dataset?

    - 4704
    - 8704
    - 9704
    - 17704
    """
    )
    return


@app.cell
def _(cars):
    len(cars)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q3. Fuel types

    How many fuel types are presented in the dataset?

    - 1
    - 2
    - 3
    - 4
    """
    )
    return


@app.cell
def _(cars):
    cars.fuel_type.nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q4. Missing values

    How many columns in the dataset have missing values?

    - 0
    - 1
    - 2
    - 3
    - 4
    """
    )
    return


@app.cell
def _(cars):
    cars.isna().any().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q5. Max fuel efficiency

    What's the maximum fuel efficiency of cars from Asia?

    - 13.75
    - 23.75
    - 33.75
    - 43.75
    """
    )
    return


@app.cell
def _(cars):
    cars[cars.origin == "Asia"].fuel_efficiency_mpg.max()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q6. Median value of horsepower

    1. Find the median value of `horsepower` column in the dataset.
    2. Next, calculate the most frequent value of the same `horsepower` column.
    3. Use `fillna` method to fill the missing values in `horsepower` column with the most frequent value from the previous step.
    4. Now, calculate the median value of `horsepower` once again.

    Has it changed?

    - Yes, it increased
    - Yes, it decreased
    - No
    """
    )
    return


@app.cell
def _(cars):
    # 1. Find the median value of `horsepower` column in the dataset

    cars.horsepower.median()
    return


@app.cell
def _(cars):
    # 2. Next, calculate the most frequent value of the same horsepower column.

    horsepower_mode = int(cars.horsepower.mode().item())
    horsepower_mode
    return (horsepower_mode,)


@app.cell
def _(cars, horsepower_mode):
    # 3. Use fillna method to fill the missing values in horsepower column with the most frequent value from the previous step.

    filled_horsepower = cars.copy()
    filled_horsepower.horsepower = filled_horsepower.horsepower.fillna(horsepower_mode)
    filled_horsepower.horsepower
    return (filled_horsepower,)


@app.cell
def _(filled_horsepower):
    # 4. Now, calculate the median value of horsepower once again.

    filled_horsepower.horsepower.median()
    return


@app.cell
def _(cars, filled_horsepower):
    # Has it changed?

    {
        "Yes, it increased": cars.horsepower.median() <= filled_horsepower.horsepower.median(),
        "Yes, it decreased": cars.horsepower.median() >= filled_horsepower.horsepower.median(),
        "No": cars.horsepower.median() == filled_horsepower.horsepower.median()
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q7. Sum of weights

    1. Select all the cars from Asia
    2. Select only columns `vehicle_weight` and `model_year`
    3. Select the first 7 values
    4. Get the underlying NumPy array. Let's call it `X`.
    5. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.
    6. Invert `XTX`.
    7. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100, 1200]`.
    8. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
    9. What's the sum of all the elements of the result?

    > **Note**: You just implemented linear regression. We'll talk about it in the next lesson.

    - 0.051
    - 0.51
    - 5.1
    - 51
    """
    )
    return


@app.cell
def _(cars):
    # 1. Select all the cars from Asia

    cars_from_asia = cars[cars.origin == "Asia"]
    cars_from_asia
    return (cars_from_asia,)


@app.cell
def _(cars_from_asia):
    # 2. Select only columns vehicle_weight and model_year

    cars_from_asia[["vehicle_weight", "model_year"]]
    return


@app.cell
def _(cars_from_asia):
    # 3. Select the first 7 values

    cars_from_asia[["vehicle_weight", "model_year"]].head(7)
    return


@app.cell
def _(cars_from_asia):
    # 4. Get the underlying NumPy array. Let's call it X

    X = cars_from_asia[["vehicle_weight", "model_year"]].head(7).to_numpy()
    X
    return (X,)


@app.cell
def _(X):
    # 5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX

    XTX = X.T.dot(X)
    XTX
    return (XTX,)


@app.cell
def _(XTX, np):
    # 6. Invert XTX

    XTX_inv = np.linalg.inv(XTX)
    XTX_inv
    return (XTX_inv,)


@app.cell
def _(np):
    # 7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200]

    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    y
    return (y,)


@app.cell
def _(X, XTX_inv, y):
    # 8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w

    w = XTX_inv.dot(X.T).dot(y)
    w
    return (w,)


@app.cell
def _(w):
    # 9. What's the sum of all the elements of the result?

    w.sum()
    return


if __name__ == "__main__":
    app.run()
