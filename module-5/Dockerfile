FROM python:3.13.3-slim

WORKDIR /app

COPY ["requirements.txt", "./"]
COPY ["./data", "./data"]
COPY ["./model_package", "./model_package"]

RUN pip install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "model_package.api:app"]

EXPOSE 5000
