import os
from pathlib import Path
from random import choice
from flask import Flask, request, json
import model as model_loader
import preprocess

current_path = Path(__file__).parent
model_file = f"{current_path}/weights/booster_model.json"
print(model_file)
assert os.path.isfile(model_file), "The model file was not found"
model = model_loader.load_booster(model_file)

dict_vectorizer_file = f"{current_path}/weights/dict_vectorizer.bin"
assert os.path.isfile(dict_vectorizer_file), "The dictionary vectorizer file was not found"
dict_vectorizer = model_loader.load_dict_vectorizer(dict_vectorizer_file)

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

@app.route('/predict', methods=['POST'])
def predict():
    assert request.is_json, "The request must be in JSON format"

    global model, dict_vectorizer

    record = json.loads(request.data)
    X = dict_vectorizer.transform([record])

    occupation = model.predict(X)[0]

    return {
        "predicted_occupation": bool(occupation),
        "data": record,
    }

@app.route('/generate', methods=['GET'])
def generate():
    return {
        "prov": choice(list(preprocess.get_prov_values().items()))[1],
        "edad1": choice(list(preprocess.get_edad1_values().items()))[1],
        "sexo1": choice(list(preprocess.get_sexo1_values().items()))[1],
        "eciv1": choice(list(preprocess.get_eciv1_values().items()))[1],
        "nforma": choice(list(preprocess.get_nforma_values().items()))[1],
    }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")