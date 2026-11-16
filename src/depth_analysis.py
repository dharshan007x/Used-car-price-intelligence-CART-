"""
Tree Depth Experimentation and Analysis Module for Used Car Price Intelligence.

Evaluates CART regression trees across various maximum depth constraints
to investigate underfitting, overfitting, and optimal tree complexity.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from src.cart_model import create_cart_regressor, evaluate_cart_performance


DEFAULT_DEPTH_RANGE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]


def run_depth_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    depth_list: Optional[List[Optional[int]]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains CART models across multiple max_depth values on training partition,
    evaluating performance on both training and held-out validation sets.
    """
    if depth_list is None:
        depth_list = DEFAULT_DEPTH_RANGE

    records = []
    trained_models = {}

    for d in depth_list:
        label = f"Depth {d}" if d is not None else "Unrestricted (None)"
        model = create_cart_regressor(max_depth=d, random_state=random_state)
        model.fit(X_train, y_train)
        
        metrics = evaluate_cart_performance(model, X_train, y_train, X_val, y_val)
        metrics["depth_param"] = d if d is not None else -1
        metrics["depth_label"] = label
        records.append(metrics)
        trained_models[label] = model

    results_df = pd.DataFrame(records)

    # Determine best depth based purely on lowest validation RMSE
    best_row = results_df.loc[results_df["val_rmse"].idxmin()]
    best_depth_param = None if best_row["depth_param"] == -1 else int(best_row["depth_param"])

    # Underfitting check (shallow trees)
    shallow_row = results_df.iloc[0] # depth 1
    # Overfitting check (unrestricted tree)
    unrestricted_row = results_df.iloc[-1] # None

    interpretation = {
        "best_depth": best_depth_param,
        "best_depth_label": best_row["depth_label"],
        "best_val_rmse": float(best_row["val_rmse"]),
        "best_val_r2": float(best_row["val_r2"]),
        "underfitting_diagnosis": (
            f"At shallow depth ({shallow_row['depth_label']}), the model underfits severely: "
            f"Train R² is only {shallow_row['train_r2']:.4f} and Val R² is {shallow_row['val_r2']:.4f}. "
            f"The tree is overly constrained and cannot capture non-linear relationships."
        ),
        "overfitting_diagnosis": (
            f"At {unrestricted_row['depth_label']} depth, the tree memorizes training observations: "
            f"Train R² climbs to {unrestricted_row['train_r2']:.4f} (Train RMSE ₹{unrestricted_row['train_rmse']:,.0f}), "
            f"while Val R² is {unrestricted_row['val_r2']:.4f} (Val RMSE ₹{unrestricted_row['val_rmse']:,.0f}). "
            f"The generalization gap is ₹{unrestricted_row['generalization_gap_rmse']:,.0f}, demonstrating clear overfitting."
        ),
        "optimal_tradeoff_summary": (
            f"Optimal validation performance is achieved at {best_row['depth_label']} "
            f"with {int(best_row['leaf_nodes'])} leaves and {int(best_row['total_nodes'])} total nodes, "
            f"yielding Val R² of {best_row['val_r2']:.4f} and Val RMSE of ₹{best_row['val_rmse']:,.0f}."
        )
    }

    return results_df, interpretation
