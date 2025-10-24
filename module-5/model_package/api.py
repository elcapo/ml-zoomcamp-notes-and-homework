import os
from pathlib import Path
from flask import Flask, request, json
from model_package.model import load_model

current_path = Path(__file__).parent.parent
model_file = f"{current_path}/data/model-weights.bin"
assert os.path.isfile(model_file), "The model file was not found"
model, dict_vectorizer = load_model(model_file)

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

@app.route('/predict', methods=['POST'])
def predict():
    assert request.is_json, "The request must be in JSON format"

    global model, dict_vectorizer

    customer = json.loads(request.data)
    X = dict_vectorizer.transform([customer])

    churn = model.predict(X)[0]

    return {
        "predicted_churn": bool(churn),
        "customer": customer,
    }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")