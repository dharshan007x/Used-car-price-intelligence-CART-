"""
Model Evaluation and Diagnostics Module for Used Car Price Intelligence.

Computes comprehensive regression metrics (MAE, MSE, RMSE, R2, MAPE),
residual error distributions, and segment-wise accuracy breakdowns.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1000.0) -> float:
    """Computes Mean Absolute Percentage Error with protection against zero or near-zero denominators."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_t), epsilon)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Test"
) -> Dict[str, Any]:
    """
    Computes all standard regression evaluation metrics with rigorous definitions.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    
    residuals = y_true - y_pred
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))
    median_abs_error = float(np.median(np.abs(residuals)))

    return {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape,
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "median_abs_error": median_abs_error
    }


def compute_segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyzes prediction performance partitioned across price tiers.
    """
    if bins is None:
        bins = [0, 300000, 700000, 1500000, float("inf")]
    if labels is None:
        labels = ["Budget (< ₹3L)", "Mid-Range (₹3L-₹7L)", "Premium (₹7L-₹15L)", "Luxury (> ₹15L)"]

    df_eval = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    df_eval["segment"] = pd.cut(df_eval["actual"], bins=bins, labels=labels, right=False)

    rows = []
    for label in labels:
        sub = df_eval[df_eval["segment"] == label]
        if len(sub) > 0:
            act = sub["actual"].values
            prd = sub["predicted"].values
            rows.append({
                "Price Segment": label,
                "Sample Count": len(sub),
                "MAE (₹)": mean_absolute_error(act, prd),
                "RMSE (₹)": np.sqrt(mean_squared_error(act, prd)),
                "R² Score": r2_score(act, prd) if len(sub) > 1 else np.nan,
                "MAPE (%)": safe_mape(act, prd)
            })

    return pd.DataFrame(rows)


def build_model_comparison_table(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Constructs the master model comparison table evaluating:
    1. Unrestricted CART Tree
    2. Depth-Restricted CART Tree
    3. Cost-Complexity Pruned CART Tree
    4. Optional Linear Baseline (if provided)
    """
    records = []

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        
        # Structural properties if decision tree
        if hasattr(model, "tree_"):
            depth = int(model.get_depth())
            nodes = int(model.tree_.node_count)
            leaves = int(model.get_n_leaves())
        else:
            depth = "N/A"
            nodes = "N/A"
            leaves = "N/A"

        metrics = compute_comprehensive_metrics(y_test, y_pred, dataset_name=name)
        records.append({
            "Model": name,
            "Tree Depth": depth,
            "Nodes": nodes,
            "Leaves": leaves,
            "MAE (₹)": metrics["mae"],
            "RMSE (₹)": metrics["rmse"],
            "R² Score": metrics["r2"],
            "MAPE (%)": metrics["mape_pct"]
        })

    return pd.DataFrame(records)
