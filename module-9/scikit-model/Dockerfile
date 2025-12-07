FROM public.ecr.aws/lambda/python:3.13

COPY ["requirements.txt", "./"]
RUN pip install -r requirements.txt

COPY ["./data", "./data"]
COPY ["./model_package", "./model_package"]
COPY ["lambda_function.py", "./"]

CMD ["lambda_function.lambda_handler"]