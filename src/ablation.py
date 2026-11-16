"""
Ablation Analysis Module for Used Car Price Intelligence.

Trains four progressively richer CART feature sets on the same train/val/test
splits and reports how each group of features shifts MAE, RMSE, and R².
This surfaces honest model-dominance information for academic review.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ── Feature-set definitions ────────────────────────────────────────────
# All sets use the SAME depth (12) and random_state=42 for fair comparison.

ABLATION_SETS: Dict[str, Dict[str, List[str]]] = {
    "A_core_only": {
        "description": "Core physics features only (age, km, engine, mileage, power)",
        "numerical": ["vehicle_age", "km_driven", "engine_cc", "mileage_val", "max_power_bhp"],
        "categorical": ["fuel", "transmission"],
    },
    "B_plus_brand_model": {
        "description": "Core + Brand + Model name",
        "numerical": ["vehicle_age", "km_driven", "engine_cc", "mileage_val", "max_power_bhp"],
        "categorical": ["brand", "model_name", "fuel", "transmission"],
    },
    "C_plus_market": {
        "description": "B + ownership tier, seats, torque, seller type, trim, body segment",
        "numerical": ["vehicle_age", "km_driven", "engine_cc", "mileage_val", "max_power_bhp",
                      "owner_tier", "seats", "torque_nm"],
        "categorical": ["brand", "model_name", "fuel", "transmission",
                        "seller_type", "trim_tier", "body_segment"],
    },
    "D_full": {
        "description": "Full market-grade feature set (C + high-mileage flag + km_per_year + km_ratio)",
        "numerical": ["vehicle_age", "km_driven", "km_per_year", "km_ratio", "engine_cc", "mileage_val",
                      "max_power_bhp", "owner_tier", "seats", "torque_nm", "high_mileage_flag"],
        "categorical": ["brand", "model_name", "fuel", "transmission",
                        "seller_type", "trim_tier", "body_segment"],
    },
}

ABLATION_DEPTH = 12
ABLATION_RANDOM_STATE = 42


def _filter_available_features(
    feature_list: List[str],
    df: pd.DataFrame
) -> List[str]:
    """Return only features that are present in the dataframe."""
    return [f for f in feature_list if f in df.columns]


def _build_simple_preprocessor(num_cols: List[str], cat_cols: List[str]):
    """Builds a minimal preprocessor inline to avoid circular imports."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    hc = ["brand", "model_name"]
    hc_cols = [c for c in hc if c in cat_cols]
    std_cols = [c for c in cat_cols if c not in hc_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if hc_cols:
        transformers.append(("cat_hc", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10))
        ]), hc_cols))
    if std_cols:
        transformers.append(("cat_std", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), std_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_ablation_study(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    depth: int = ABLATION_DEPTH,
    random_state: int = ABLATION_RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains four feature sets on training data, evaluates on val and test.

    Returns:
        ablation_df  — DataFrame with one row per feature set
        result_dict  — Raw results keyed by set name
    """
    rows = []
    result_dict: Dict[str, Any] = {}

    for set_name, cfg in ABLATION_SETS.items():
        num_cols = _filter_available_features(cfg["numerical"], X_train)
        cat_cols = _filter_available_features(cfg["categorical"], X_train)
        all_cols = num_cols + cat_cols

        if not all_cols:
            logger.warning(f"[Ablation] Set {set_name}: no usable columns found, skipping.")
            continue

        logger.info(f"[Ablation] Training set {set_name}: {len(num_cols)} num + {len(cat_cols)} cat features")

        try:
            # Fit preprocessor on training subset only
            prep = _build_simple_preprocessor(num_cols, cat_cols)
            Xtr = prep.fit_transform(X_train[all_cols])
            Xvl = prep.transform(X_val[all_cols])
            Xts = prep.transform(X_test[all_cols])

            # Train CART
            cart = DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=depth,
                random_state=random_state
            )
            cart.fit(Xtr, y_train)

            # Evaluate
            yv_pred = cart.predict(Xvl)
            yt_pred = cart.predict(Xts)

            val_mae = mean_absolute_error(y_val, yv_pred)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, yv_pred)))
            val_r2 = r2_score(y_val, yv_pred)
            test_mae = mean_absolute_error(y_test, yt_pred)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, yt_pred)))
            test_r2 = r2_score(y_test, yt_pred)

            row = {
                "feature_set": set_name,
                "description": cfg["description"],
                "n_input_features": len(all_cols),
                "n_transformed_cols": Xtr.shape[1],
                "val_mae": round(val_mae, 0),
                "val_rmse": round(val_rmse, 0),
                "val_r2": round(val_r2, 4),
                "test_mae": round(test_mae, 0),
                "test_rmse": round(test_rmse, 0),
                "test_r2": round(test_r2, 4),
            }
            rows.append(row)
            result_dict[set_name] = {**row, "model": cart, "preprocessor": prep}

            logger.info(f"  → Val MAE=₹{val_mae:,.0f}  Test R²={test_r2:.4f}")

        except Exception as exc:
            logger.error(f"[Ablation] Set {set_name} failed: {exc}")

    ablation_df = pd.DataFrame(rows)
    return ablation_df, result_dict
