import pickle
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

def load_booster(filename: str, params: dict = {}) -> xgb.XGBClassifier:
    booster = xgb.XGBClassifier(random_state=1, **params)
    booster.load_model(filename)

    return booster

def save_dict_vectorizer(dict_vectorizer: DictVectorizer, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(dict_vectorizer, f)

def load_dict_vectorizer(filename: str) -> DictVectorizer:
    with open(filename, "rb") as f:
        return pickle.load(f)
