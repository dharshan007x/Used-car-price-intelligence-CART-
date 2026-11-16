"""
Preprocessing Pipeline Module for Used Car Price Intelligence.

Builds scikit-learn ColumnTransformer pipelines combining numerical imputation
and categorical one-hot encoding with unknown category handling.
Prevents train-test leakage and preserves feature names for tree explainability.
"""

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ── Core feature sets (baseline — no brand/model) ──
CORE_NUMERICAL_FEATURES = [
    "vehicle_age",
    "km_driven",
    "engine_cc",
]

# ── Extended features used in baseline model ──
EXTENDED_NUMERICAL_FEATURES = [
    "vehicle_age",
    "km_driven",
    "engine_cc",
    "mileage_val",
    "max_power_bhp",
]

CORE_CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "transmission",
]

# ── Market-grade full feature sets (used after upgrade) ──
MARKET_NUMERICAL_FEATURES = [
    "vehicle_age",
    "km_driven",
    "km_per_year",    # annual usage intensity (derived, leak-free)
    "km_ratio",       # km_driven / typical_km (age-normalised, >1 = high usage)
    "engine_cc",
    "mileage_val",
    "max_power_bhp",
    "owner_tier",   # ordinal int: 1=First … 5=TestDrive
    "seats",
    "torque_nm",
    "high_mileage_flag",
]

MARKET_CATEGORICAL_FEATURES = [
    "brand",
    "model_name",
    "fuel",
    "transmission",
    "seller_type",
    "trim_tier",
    "body_segment",
]

TARGET_COLUMN = "selling_price"


def build_preprocessor(
    num_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None,
    high_cardinality_features: Optional[List[str]] = None,
    min_frequency: int = 10,
) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer.

    - Numerical features: Median imputation (no scaling — CART is invariant).
    - Categorical features: Most-frequent imputation + OneHotEncoder
      with handle_unknown='ignore'.
    - High-cardinality categoricals (brand, model_name) get min_frequency
      to group rare values into 'infrequent_sklearn'.
    """
    if num_features is None:
        num_features = MARKET_NUMERICAL_FEATURES
    if cat_features is None:
        cat_features = MARKET_CATEGORICAL_FEATURES
    if high_cardinality_features is None:
        high_cardinality_features = ["brand", "model_name"]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Separate high-cardinality from standard categorical features
    hc_cols = [c for c in high_cardinality_features if c in cat_features]
    std_cat_cols = [c for c in cat_features if c not in hc_cols]

    transformers = [
        ("num", num_pipeline, num_features),
    ]

    if hc_cols:
        hc_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                min_frequency=min_frequency
            ))
        ])
        transformers.append(("cat_hc", hc_pipeline, hc_cols))

    if std_cat_cols:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat_std", cat_pipeline, std_cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """
    Safely extracts human-readable feature names after OneHotEncoding fitting.
    Essential for tree visualization and decision path explainability.
    """
    feature_names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" or trans == "drop":
            continue

        if hasattr(trans, "named_steps"):
            # It's a Pipeline
            last_step = trans.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                names = list(last_step.get_feature_names_out(cols))
                feature_names.extend(names)
            else:
                feature_names.extend(cols)
        elif hasattr(trans, "get_feature_names_out"):
            names = list(trans.get_feature_names_out(cols))
            feature_names.extend(names)
        else:
            feature_names.extend(cols)

    return feature_names


def save_pipeline(pipeline: Any, path: Path) -> None:
    """Persists fitted preprocessing pipeline or model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path) -> Any:
    """Loads fitted pipeline or model."""
    if not path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at {path}")
    return joblib.load(path)
