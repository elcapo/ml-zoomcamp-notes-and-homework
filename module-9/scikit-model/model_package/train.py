from pathlib import Path
from model_package.model import numerical_columns, categorical_columns, get_standardized_dataframe, split_dataframe, get_full_trained_vectorizer, get_features_and_target, train_and_save_model

if __name__ == "__main__":
    current_path = Path(__file__).parent.parent

    df_standardized = get_standardized_dataframe(f"{current_path}/data/customer-churn.csv")
    df_full, df_train, df_val, df_test = split_dataframe(df_standardized)

    dict_vectorizer_full, dictionary_full = get_full_trained_vectorizer(df_full)
    X_full, y_full = get_features_and_target(df_full, dict_vectorizer_full, dictionary_full)

    dictionary_val = df_val[numerical_columns + categorical_columns].to_dict(orient="records")
    X_val, y_val = get_features_and_target(df_val, dict_vectorizer_full, dictionary_val)

    model_file = f"{current_path}/data/model-weights.bin"
    model, dict_vectorizer = train_and_save_model(model_file, X_full, y_full, dict_vectorizer_full)

    accuracy = model.score(X_val, y_val)
    print(f"The model was loaded successfully and has an accuracy of {accuracy:.3f}")