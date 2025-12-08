import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import onnxruntime as ort
    import numpy as np
    from torchvision import transforms
    from keras_image_helper import create_preprocessor
    return create_preprocessor, mo, np, ort


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning Zoomcamp

    ## Module 9: **Deploy with AWS Lambda**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Homework

    In this homework, we'll deploy the Straight vs Curly Hair Type model we trained in the
    [previous homework](../08-deep-learning/homework.md).

    Download the model files from here:

    * https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx.data
    * https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx

    With wget:

    ```bash
    PREFIX="https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
    DATA_URL="${PREFIX}/hair_classifier_v1.onnx.data"
    MODEL_URL="${PREFIX}/hair_classifier_v1.onnx"
    wget ${DATA_URL}
    wget ${MODEL_URL}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 1

    To be able to use this model, we need to know the name of the input and output nodes.

    What's the name of the output:

    * `output`
    * `sigmoid`
    * `softmax`
    * `prediction`
    """)
    return


@app.cell
def _(ort):
    session = ort.InferenceSession("module-9/homework/hair_classifier_v1.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    (input_name, output_name)
    return input_name, output_name, session


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The name of the output is **output**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing the image

    You'll need some code for downloading and resizing images. You can use
    this code:
    """)
    return


@app.cell
def _():
    from io import BytesIO
    from urllib import request
    from PIL import Image

    def download_image(url):
        with request.urlopen(url) as resp:
            buffer = resp.read()
        stream = BytesIO(buffer)
        img = Image.open(stream)
        return img


    def prepare_image(img, target_size):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.NEAREST)
        return img
    return download_image, prepare_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2: Target size

    Let's download and resize this image:

    https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg

    Based on the previous homework, what should be the target size for the image?

    * 64x64
    * 128x128
    * 200x200
    * 256x256
    """)
    return


@app.cell
def _(download_image, prepare_image):
    image = download_image("https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
    prepared_image = prepare_image(image, (200, 200))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The target size for the image is **200x200**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 3

    Now we need to turn the image into numpy array and pre-process it.

    > Tip: Check the previous homework. What was the pre-processing
    > we did there?

    After the pre-processing, what's the value in the first pixel, the R channel?

    * -10.73
    * -1.073
    * 1.073
    * 10.73
    """)
    return


@app.cell
def _(create_preprocessor, np):
    def preprocess_pytorch(X):
        # X: shape (1, 299, 299, 3), dtype=float32, values in [0, 255]
        X = X / 255.0

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        # Convert NHWC → NCHW
        # from (batch, height, width, channels) → (batch, channels, height, width)
        X = X.transpose(0, 3, 1, 2)

        # Normalize
        X = (X - mean) / std

        return X.astype(np.float32)


    preprocessor = create_preprocessor(preprocess_pytorch, target_size=(200, 200))

    X = preprocessor.from_url("https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
    X[0][0][0][0]
    return (preprocessor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The value of the R channel of the firs pixel is **-1.073**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 4

    Now let's apply this model to this image. What's the output of the model?

    * 0.09
    * 0.49
    * 0.69
    * 0.89
    """)
    return


@app.cell
def _(input_name, output_name, preprocessor, session):
    classes = [
        'dress',
        'hat',
        'longsleeve',
        'outwear',
        'pants',
        'shirt',
        'shoes',
        'shorts',
        'skirt',
        't-shirt'
    ]

    def predict(url):
        X = preprocessor.from_url(url)
        result = session.run([output_name], {input_name: X})
        float_predictions = result[0][0].tolist()
        return dict(zip(classes, float_predictions))

    predict("https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare the lambda code

    Now you need to copy all the code into a separate python file. You will
    need to use this file for the next two questions.

    Tip: you can test this file locally with `ipython` or Jupyter Notebook
    by importing the file and invoking the function from this file.


    ## Docker

    For the next two questions, we'll use a Docker image that we already
    prepared. This is the Dockerfile that we used for creating the image:

    ```docker
    FROM public.ecr.aws/lambda/python:3.13

    COPY hair_classifier_empty.onnx.data .
    COPY hair_classifier_empty.onnx .
    ```

    Note that it uses Python 3.13.

    The docker image is published to [`agrigorev/model-2025-hairstyle:v1`](https://hub.docker.com/r/agrigorev/model-2025-hairstyle).

    A few notes:

    * The image already contains a model and it's not the same model
      as the one we used for questions 1-4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 5

    Download the base image `agrigorev/model-2025-hairstyle:v1`. You can do it with [`docker pull`](https://docs.docker.com/engine/reference/commandline/pull/).

    So what's the size of this base image?

    * 88 Mb
    * 208 Mb
    * 608 Mb
    * 1208 Mb

    You can get this information when running `docker images` - it'll be in the "SIZE" column.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```bash
    docker pull agrigorev/model-2025-hairstyle:v1
    docker image ls | grep agrigorev/model-2025-hairstyle
    ```

    > agrigorev/model-2025-hairstyle - v1 - 4528ad1525d5 - 6 days ago - 608MB
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The size of the downloaded image is **608 Mb**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 6

    Now let's extend this docker image, install all the required libraries
    and add the code for lambda.

    You don't need to include the model in the image. It's already included.
    The name of the file with the model is `hair_classifier_empty.onnx` and it's
    in the current workdir in the image (see the Dockerfile above for the
    reference).
    The provided model requires the same preprocessing for images regarding target size and rescaling the value range than used in homework 8.

    Now run the container locally.

    Score this image: https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg

    What's the output from the model?

    * -1.0
    * -0.10
    * 0.10
    * 1.0
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result is 0.09 which is almost **0.10**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Publishing it to AWS

    Now you can deploy your model to AWS!

    * Publish your image to ECR
    * Create a lambda function in AWS, use the ECR image
    * Give it more RAM and increase the timeout
    * Test it
    * Expose the lambda function using API Gateway

    This is optional and not graded.


    ## Submit the results

    * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw09
    * If your answer doesn't match options exactly, select the closest one. If the answer is exactly in between two options, select the higher value.

    ## Publishing to Docker hub

    Just for the reference, this is how we published our image to Docker hub:

    ```bash
    docker build -t model-2025-hairstyle -f homework.dockerfile .
    docker tag model-2025-hairstyle:latest agrigorev/model-2025-hairstyle:v1
    docker push agrigorev/model-2025-hairstyle:v1
    ```

    (You don't need to execute this code)
    """)
    return


if __name__ == "__main__":
    app.run()
