"""
CART Regression Tree Module for Used Car Price Intelligence.

Implements CART (Classification and Regression Trees) for continuous price estimation
using scikit-learn's DecisionTreeRegressor with squared_error reduction criterion.

Monotonic constraints (sklearn >= 1.4):
  km_driven, km_per_year, km_ratio → -1  (higher usage → lower price)
  vehicle_age                       → -1  (older → lower price)
  owner_tier                        → -1  (more owners → lower price)
  All other features                →  0  (unconstrained)
"""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Features that must be monotonically decreasing with respect to price
_DECREASING_FEATURES = {"km_driven", "km_per_year", "km_ratio", "vehicle_age", "owner_tier"}


def build_monotonic_cst(feature_names: List[str]) -> List[int]:
    """
    Builds a monotonic_cst list aligned to the given (post-transform) feature names.

    Returns a list of length len(feature_names) where:
      -1 = decreasing (higher feature value → lower price)
       0 = no constraint

    sklearn DecisionTreeRegressor accepts a list/array for monotonic_cst
    (each element corresponds to the matching column index in X).

    Only pass-through numerical features are constrained; OHE-expanded
    categorical columns all receive 0.
    """
    cst: List[int] = []
    for name in feature_names:
        # Strip ColumnTransformer prefixes (num__, cat_hc__, cat_std__)
        raw = name
        for pfx in ("num__", "cat_hc__", "cat_std__"):
            if name.startswith(pfx):
                raw = name[len(pfx):]
                break
        cst.append(-1 if raw in _DECREASING_FEATURES else 0)
    return cst


def create_cart_regressor(
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    ccp_alpha: float = 0.0,
    random_state: int = 42,
    criterion: str = "squared_error",
    monotonic_cst: Optional[List[int]] = None,
) -> DecisionTreeRegressor:
    """
    Instantiates a CART DecisionTreeRegressor.

    Splitting Criterion:
    Squared error minimization (MSE reduction at each node split):
        Q(X_m) = (1 / N_m) * sum((y_i - y_hat_m)^2)
    Split s* chosen to maximize reduction in weighted impurity:
        Delta Q = Q(X_m) - (N_left/N_m)*Q(X_left) - (N_right/N_m)*Q(X_right)
    Leaf prediction:
        y_hat_leaf = mean(y_i in leaf)

    Args:
        monotonic_cst: Dict mapping feature names → {-1, 0, 1} constraints.
            Requires sklearn >= 1.4. Features with -1 must be monotonically
            decreasing (e.g., km_driven → lower price as km increases).
            Pass None to disable constraints (backward-compatible).
    """
    kwargs: Dict[str, Any] = dict(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )
    if monotonic_cst is not None:
        kwargs["monotonic_cst"] = monotonic_cst
    return DecisionTreeRegressor(**kwargs)


def extract_tree_properties(tree_model: DecisionTreeRegressor) -> Dict[str, Any]:
    """Extracts structural characteristics from a fitted DecisionTreeRegressor."""
    tree = tree_model.tree_
    return {
        "max_depth": int(tree_model.get_depth()),
        "total_nodes": int(tree.node_count),
        "leaf_nodes": int(tree_model.get_n_leaves()),
        "ccp_alpha": float(tree_model.ccp_alpha),
        "criterion": str(tree_model.criterion)
    }


def evaluate_cart_performance(
    tree_model: DecisionTreeRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """
    Computes rigorous training and validation metrics for a fitted CART model.
    """
    # Predictions
    y_train_pred = tree_model.predict(X_train)
    y_val_pred = tree_model.predict(X_val)

    # Training metrics
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))
    train_r2 = float(r2_score(y_train, y_train_pred))

    # Validation metrics
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    val_mse = float(mean_squared_error(y_val, y_val_pred))
    val_rmse = float(np.sqrt(val_mse))
    val_r2 = float(r2_score(y_val, y_val_pred))

    props = extract_tree_properties(tree_model)

    return {
        **props,
        "train_mae": train_mae,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "generalization_gap_rmse": float(val_rmse - train_rmse),
        "generalization_gap_r2": float(train_r2 - val_r2)
    }
