import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

def split_dataset(df: pd.DataFrame, random_state: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full, test = train_test_split(df, test_size=0.2, random_state=random_state)
    train, val = train_test_split(full, test_size=0.25, random_state=random_state)

    return train, full, val, test

def separate_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = df.trarem.copy()
    features = df.copy()
    del features["trarem"]

    return features, target

def train_dict_vectorizer(features: pd.DataFrame) -> DictVectorizer:
    dict_vectorizer = DictVectorizer(sparse=False)
    dict_vectorizer.fit(features.to_dict(orient="records"))

    return dict_vectorizer

def vectorize_features(features: pd.DataFrame, dict_vectorizer: DictVectorizer):
    return dict_vectorizer.transform(features.to_dict(orient="records"))
