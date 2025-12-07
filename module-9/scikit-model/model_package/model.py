from random import choice, randint, random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# Data preparation

def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    standardized = dataframe.copy()
    standardized.columns = standardized.columns.str.lower().str.replace(' ', '_')

    return standardized

def get_cateogorical_columns(dataframe: pd.DataFrame) -> list[str]:
    return list(list(dataframe.dtypes[dataframe.dtypes == 'object'].index))

def standardize_categorical_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    standardized = dataframe.copy()

    for column in get_cateogorical_columns(standardized):
        standardized[column] = standardized[column].str.lower().str.replace(' ', '_')

    return standardized

def standardize_non_categorical_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    standardized = dataframe.copy()

    totalcharges = pd.to_numeric(standardized.totalcharges, errors='coerce')
    standardized.totalcharges = totalcharges.fillna(0)

    return standardized

def get_standardized_dataframe(dataframe_path: str) -> pd.DataFrame:
    raw = pd.read_csv(dataframe_path)
    standardized = standardize_column_names(raw)
    standardized = standardize_categorical_values(standardized)
    standardized = standardize_non_categorical_values(standardized)

    return standardized

# Validation framework

def split_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    full, test = train_test_split(dataframe, test_size=0.2, random_state=1)
    train, val = train_test_split(full, test_size=0.25, random_state=1)

    full = full.reset_index(drop=True)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return full, train, val, test

# Extract the features

numerical_columns = ["tenure", "monthlycharges", "totalcharges"]

categorical_columns = [
    'gender', 'seniorcitizen', 'partner', 'dependents',
    'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
    'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
    'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod'
]

def get_full_trained_vectorizer(dataframe: pd.DataFrame) -> list[dict]:
    copy = dataframe.copy()
    del copy["churn"]
    dictionary = copy.to_dict(orient="records")

    dict_vectorizer = DictVectorizer(sparse=False)
    dict_vectorizer.fit(dictionary)

    return dict_vectorizer, dictionary

def get_features_and_target(dataframe: pd.DataFrame, dict_vectorizer: DictVectorizer, dictionary):
    X = dict_vectorizer.transform(dictionary)
    y = dataframe.churn == "yes"

    return X, y

# Train and and save the model

def train_and_save_model(model_file: str, X: np.ndarray, y: np.ndarray, dict_vectorizer: DictVectorizer):
    model = LogisticRegression(max_iter=5000)
    model.fit(X, y)
    with open(model_file, "wb") as f:
        pickle.dump((model, dict_vectorizer), f)
    
    return model, dict_vectorizer

# Load the model

def load_model(model_file: str):
    with open(model_file, "rb") as f:
        (model, dict_vectorizer) = pickle.load(f)

    return model, dict_vectorizer

# Generate random samples

def options_for_categorical_column(column: str, features: list[str]) -> list[str]:
    options = []
    for feature in features:
        if feature.startswith(column):
            parts = feature.split("=")
            if len(parts) == 2:
                options.append(feature.split("=")[1])
    return options

def generate_random_sample(dict_vectorizer: DictVectorizer) -> dict:
    features = [feature for feature in dict_vectorizer.get_feature_names_out() if not feature.startswith("customerid")]

    random_sample = {}
    for categorical_column in categorical_columns:
        options = options_for_categorical_column(categorical_column, features)
        if not len(options):
            random_sample[categorical_column] = None
            continue
        random_sample[categorical_column] = choice(options)

    random_sample["customerid"] = randint(0, 9999)
    random_sample["tenure"] = randint(0, 75)
    random_sample["monthlycharges"] = round(random() * 125, 2)
    random_sample["totalcharges"] = round(random() * 8750, 2)
    random_sample["seniorcitizen"] = 1 if random() > 0.5 else 0

    return random_sample
