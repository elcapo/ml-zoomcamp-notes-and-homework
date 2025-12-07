import os
from pathlib import Path
from flask import Flask, request, json
from model_package.model import load_model, generate_random_sample

model_file = "/var/task/data/model-weights.bin"
assert os.path.isfile(model_file), "The model file was not found"
model, dict_vectorizer = load_model(model_file)

def lambda_handler(event, context):
    global model, dict_vectorizer

    customer = event
    X = dict_vectorizer.transform([customer])

    churn = model.predict(X)[0]

    return {
        "statusCode": 200,
        "predicted_churn": bool(churn),
        "customer": customer,
    }
