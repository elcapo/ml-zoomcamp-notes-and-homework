import os
import pickle


def load_model():
    models_path = os.path.join(os.path.dirname(__file__), '../model')

    with open(f"{models_path}/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(f"{models_path}/imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    with open(f"{models_path}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, imputer, scaler
