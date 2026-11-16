"""
Explainability Module for Used Car Price Intelligence.

Extracts exact CART decision paths using sklearn's decision_path() and apply().
Generates transparent, human-readable step-by-step audit trails and leaf statistics.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def explain_prediction_path(
    tree_model: DecisionTreeRegressor,
    feature_names: List[str],
    sample_vector: np.ndarray,
    raw_input_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Traces and explains the decision path taken by a single vehicle input through the CART tree.
    
    Args:
        tree_model: Fitted DecisionTreeRegressor
        feature_names: List of all transformed column names (from OneHotEncoder + numericals)
        sample_vector: 1D or 2D preprocessed feature array for this car
        raw_input_dict: Original user-entered values (for intuitive context in descriptions)
        
    Returns:
        Structured explanation dict containing node path steps, thresholds, direction,
        leaf sample count, leaf value, and natural language summary.
    """
    if sample_vector.ndim == 1:
        sample_vector = sample_vector.reshape(1, -1)

    tree = tree_model.tree_
    node_indicator = tree_model.decision_path(sample_vector)
    leaf_id = int(tree_model.apply(sample_vector)[0])

    node_indices = node_indicator.indices[
        node_indicator.indptr[0]: node_indicator.indptr[1]
    ]

    steps = []
    text_narrative = []

    for step_num, node_id in enumerate(node_indices, start=1):
        # Check if internal decision node or leaf
        if node_id == leaf_id:
            leaf_samples = int(tree.n_node_samples[node_id])
            predicted_val = float(tree.value[node_id][0][0])
            steps.append({
                "step": step_num,
                "node_id": int(node_id),
                "is_leaf": True,
                "feature": "Terminal Leaf",
                "condition": "N/A",
                "direction": "Final Estimation",
                "samples_in_leaf": leaf_samples,
                "leaf_prediction": predicted_val
            })
            text_narrative.append(
                f"Step {step_num} (Terminal Leaf #{node_id}): Reached leaf representing "
                f"{leaf_samples} training vehicles. Predicted resale value = ₹{predicted_val:,.0f}."
            )
        else:
            feature_idx = int(tree.feature[node_id])
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
            threshold = float(tree.threshold[node_id])
            feature_value = float(sample_vector[0, feature_idx])

            # In CART, left branch is <= threshold, right branch is > threshold
            went_left = feature_value <= threshold
            direction = "LEFT (<=)" if went_left else "RIGHT (>)"

            # Prettify condition for user
            cond_str = f"{feature_name} <= {threshold:.2f}"
            eval_str = f"Actual {feature_value:.2f} <= {threshold:.2f} is {went_left}"

            steps.append({
                "step": step_num,
                "node_id": int(node_id),
                "is_leaf": False,
                "feature": feature_name,
                "feature_value": feature_value,
                "threshold": threshold,
                "went_left": went_left,
                "direction": direction,
                "samples_at_node": int(tree.n_node_samples[node_id]),
                "node_impurity_mse": float(tree.impurity[node_id])
            })

            dir_text = "went LEFT (meets condition)" if went_left else "went RIGHT (exceeds condition)"
            text_narrative.append(
                f"Step {step_num} (Node #{node_id}): Evaluated '{feature_name}'. "
                f"Observed value was {feature_value:.2f} compared to threshold of {threshold:.2f}. "
                f"Decision: {dir_text}."
            )

    leaf_samples = int(tree.n_node_samples[leaf_id])
    leaf_pred = float(tree.value[leaf_id][0][0])

    return {
        "leaf_id": leaf_id,
        "leaf_samples": leaf_samples,
        "predicted_price": leaf_pred,
        "path_length": len(node_indices),
        "steps": steps,
        "narrative": text_narrative,
        "summary": (
            f"Prediction of ₹{leaf_pred:,.0f} was reached after {len(node_indices) - 1} binary decisions. "
            f"The final estimate is the mean price of {leaf_samples} similar vehicles in the training leaf."
        )
    }
