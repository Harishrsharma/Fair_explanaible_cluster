# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_bank_dataset(path, sample_size=None):

    df = pd.read_csv(path)

    # Sensitive attribute example
    df["sensitive"] = df["marital"].apply(
        lambda x: 1 if x == "married" else 0
    )

    sensitive = df["sensitive"].values

    drop_cols = ["marital", "sensitive", "y"]
    X_raw = df.drop(columns=drop_cols)

    if sample_size and sample_size < len(X_raw):
        idx = np.random.choice(len(X_raw), sample_size, replace=False)
        X_raw = X_raw.iloc[idx].reset_index(drop=True)
        sensitive = sensitive[idx]

    categorical_cols = X_raw.select_dtypes(include=["object"]).columns
    numeric_cols = X_raw.select_dtypes(include=["number"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X_raw)
    feature_names = preprocessor.get_feature_names_out()

    return X_processed, sensitive, feature_names
