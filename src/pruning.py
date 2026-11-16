"""
Cost-Complexity Pruning (CCP) Module for Used Car Price Intelligence.

Implements CART Minimal Cost-Complexity Pruning:
    R_alpha(T) = R(T) + alpha * |T|
Extracts pruning paths, evaluates effective alphas on training and validation splits,
and selects the optimal complexity penalty to prevent overfitting.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from src.cart_model import create_cart_regressor, evaluate_cart_performance


def compute_pruning_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, DecisionTreeRegressor]:
    """
    Fits an initial large tree and extracts the cost-complexity pruning path.
    Returns:
        ccp_alphas: Array of effective alpha values.
        impurities: Array of corresponding total impurities.
        initial_tree: The fitted unpruned base tree.
    """
    base_tree = create_cart_regressor(max_depth=max_depth, random_state=random_state)
    base_tree.fit(X_train, y_train)

    path = base_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Ensure alphas are non-negative and finite
    ccp_alphas = np.clip(ccp_alphas, a_min=0.0, a_max=None)
    return ccp_alphas, impurities, base_tree


def select_candidate_alphas(ccp_alphas: np.ndarray, max_candidates: int = 25) -> np.ndarray:
    """
    Subsamples alphas to a computationally clean, representative sequence
    spanning from 0 to max alpha, removing duplicates.
    """
    unique_alphas = np.unique(ccp_alphas)
    if len(unique_alphas) <= max_candidates:
        return unique_alphas

    # Always include 0.0, the max alpha, and log-spaced / quantile intermediate values
    non_zero = unique_alphas[unique_alphas > 0]
    if len(non_zero) == 0:
        return np.array([0.0])

    indices = np.linspace(0, len(non_zero) - 1, max_candidates - 1, dtype=int)
    chosen = [0.0] + list(non_zero[indices])
    return np.unique(chosen)


def run_pruning_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_depth: Optional[int] = None,
    candidate_alphas: Optional[np.ndarray] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any], DecisionTreeRegressor]:
    """
    Executes cost-complexity pruning experiment:
    1. Computes CCP path.
    2. Trains a decision tree for each alpha.
    3. Evaluates error and tree size on train and val sets.
    4. Selects optimal alpha based on validation RMSE.
    """
    all_alphas, impurities, base_tree = compute_pruning_path(
        X_train, y_train, max_depth=max_depth, random_state=random_state
    )

    if candidate_alphas is None:
        candidate_alphas = select_candidate_alphas(all_alphas, max_candidates=25)

    records = []
    models = []

    for alpha in candidate_alphas:
        model = create_cart_regressor(
            max_depth=max_depth,
            ccp_alpha=float(alpha),
            random_state=random_state
        )
        model.fit(X_train, y_train)
        metrics = evaluate_cart_performance(model, X_train, y_train, X_val, y_val)
        metrics["alpha"] = float(alpha)
        records.append(metrics)
        models.append(model)

    results_df = pd.DataFrame(records)

    # Select best alpha: lowest val_rmse
    best_idx = results_df["val_rmse"].idxmin()
    best_row = results_df.loc[best_idx]
    best_alpha = float(best_row["alpha"])
    best_model = models[best_idx]

    unpruned_row = results_df.iloc[0] # alpha = 0.0

    interpretation = {
        "best_alpha": best_alpha,
        "best_tree_depth": int(best_row["max_depth"]),
        "best_leaf_nodes": int(best_row["leaf_nodes"]),
        "best_total_nodes": int(best_row["total_nodes"]),
        "best_val_rmse": float(best_row["val_rmse"]),
        "best_val_r2": float(best_row["val_r2"]),
        "unpruned_val_rmse": float(unpruned_row["val_rmse"]),
        "unpruned_val_r2": float(unpruned_row["val_r2"]),
        "unpruned_leaves": int(unpruned_row["leaf_nodes"]),
        "node_reduction_pct": round(
            (1.0 - best_row["total_nodes"] / unpruned_row["total_nodes"]) * 100, 2
        ) if unpruned_row["total_nodes"] > 0 else 0.0,
        "summary": (
            f"Cost-complexity pruning with optimal alpha={best_alpha:.2e} "
            f"reduced tree complexity from {int(unpruned_row['total_nodes'])} to {int(best_row['total_nodes'])} nodes "
            f"({round((1.0 - best_row['total_nodes'] / unpruned_row['total_nodes']) * 100, 1)}% pruning). "
            f"Validation RMSE changed from ₹{unpruned_row['val_rmse']:,.0f} to ₹{best_row['val_rmse']:,.0f}, "
            f"demonstrating {'generalization gain' if best_row['val_rmse'] < unpruned_row['val_rmse'] else 'comparable accuracy with superior parsimony'}."
        )
    }

    return results_df, interpretation, best_model
