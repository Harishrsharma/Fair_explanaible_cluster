# data_processing.py
# Unified data loader for all 4 thesis datasets.
# Sensitive attributes aligned with thesis proposal (Nov 2025):
#   Bank Marketing : age_group (age > 35)
#   Adult          : gender    (Male=1, Female=0)
#   COMPAS         : race      (African-American=1, others=0)
#   German Credit  : sex       (male=1, female=0)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.config import SENSITIVE_CONFIGS, SAMPLE_SIZES


def _build_sensitive(df, cfg):
    """Convert a raw column into a binary sensitive array using config rules."""
    col = cfg["column"]
    kind = cfg["type"]

    if kind == "threshold":
        return (df[col] > cfg["threshold"]).astype(int).values

    if kind == "binary_map":
        mapping = cfg["map"]
        default = cfg.get("default", 0)
        return df[col].map(lambda v: mapping.get(v, default)).values

    raise ValueError(f"Unknown sensitive type: {kind}")


def _preprocess(df, drop_cols):
    """Standard numeric scaling + one-hot encoding, returning (X, feature_names)."""
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    X = preprocessor.fit_transform(X_raw)
    feature_names = preprocessor.get_feature_names_out()
    return X, feature_names


def _sample(X, sensitive, sample_size, random_state=42):
    if sample_size and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), sample_size, replace=False)
        return X[idx], sensitive[idx]
    return X, sensitive


# ─────────────────────────────────────────────────────────────────────────────
# Individual loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_bank_dataset(path, sample_size=None, random_state=42):
    """
    Bank Marketing dataset.
    Sensitive: age_group (age > 35 = 1, else 0)
    Target column 'y' is dropped. Marital status kept as feature.
    """
    df = pd.read_csv(path)
    cfg = SENSITIVE_CONFIGS["bank"]
    sensitive = _build_sensitive(df, cfg)

    drop_cols = ["y", cfg["column"]]   # drop age after encoding sensitive
    X, feature_names = _preprocess(df, drop_cols)
    X, sensitive = _sample(X, sensitive, sample_size, random_state)
    return X, sensitive, feature_names


def load_adult_dataset(path, sample_size=None, random_state=42):
    """
    Adult Census Income dataset.
    Sensitive: gender (Male=1, Female=0)
    Target column 'Class-label' is dropped.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    cfg = SENSITIVE_CONFIGS["adult"]
    sensitive = _build_sensitive(df, cfg)

    drop_cols = ["Class-label", cfg["column"]]
    X, feature_names = _preprocess(df, drop_cols)
    X, sensitive = _sample(X, sensitive, sample_size, random_state)
    return X, sensitive, feature_names


def load_compas_dataset(path, sample_size=None, random_state=42):
    """
    COMPAS Recidivism dataset.
    Sensitive: race (African-American=1, others=0)
    Only keeps numerically / categorically meaningful features; drops
    identifiers, dates, and raw score columns to avoid data leakage.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    cfg = SENSITIVE_CONFIGS["compas"]
    sensitive = _build_sensitive(df, cfg)

    # Keep predictive features; remove IDs, dates, names, target
    keep_cols = [
        "age", "juv_fel_count", "juv_misd_count", "juv_other_count",
        "priors_count", "c_charge_degree", "sex", "age_cat",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df_sub = df[available].copy()

    cat_cols = df_sub.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_sub.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    X = preprocessor.fit_transform(df_sub)
    feature_names = preprocessor.get_feature_names_out()
    X, sensitive = _sample(X, sensitive, sample_size, random_state)
    return X, sensitive, feature_names


def load_german_dataset(path, sample_size=None, random_state=42):
    """
    German Credit dataset.
    Sensitive: sex (male=1, female=0)
    Target column 'class-label' is dropped.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    cfg = SENSITIVE_CONFIGS["german"]
    sensitive = _build_sensitive(df, cfg)

    drop_cols = ["class-label", cfg["column"]]
    X, feature_names = _preprocess(df, drop_cols)
    X, sensitive = _sample(X, sensitive, sample_size, random_state)
    return X, sensitive, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = {
    "bank":   load_bank_dataset,
    "adult":  load_adult_dataset,
    "compas": load_compas_dataset,
    "german": load_german_dataset,
}


def load_dataset(name, path, random_state=42):
    """
    Load any of the 4 thesis datasets by name.
    Returns (X, sensitive, feature_names).
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(LOADERS)}")

    sample_size = SAMPLE_SIZES.get(name)
    return LOADERS[name](path, sample_size=sample_size, random_state=random_state)
