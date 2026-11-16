"""
Unit tests for CART Regression Model training, evaluation, and persistence.
"""

import pytest
import numpy as np
from pathlib import Path
from src.cart_model import create_cart_regressor, evaluate_cart_performance, extract_tree_properties
from src.preprocessing import save_pipeline, load_pipeline


def test_cart_regressor_training():
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    # Target linearly related to features with noise
    y_train = (X_train[:, 0] * 500000 + X_train[:, 1] * 200000 + 100000 + np.random.randn(100) * 10000)

    X_val = np.random.rand(30, 10)
    y_val = (X_val[:, 0] * 500000 + X_val[:, 1] * 200000 + 100000 + np.random.randn(30) * 10000)

    # Max depth constraint test
    max_d = 4
    model = create_cart_regressor(max_depth=max_d, random_state=42)
    model.fit(X_train, y_train)

    assert model.get_depth() <= max_d

    props = extract_tree_properties(model)
    assert props["max_depth"] <= max_d
    assert props["total_nodes"] > props["leaf_nodes"]

    metrics = evaluate_cart_performance(model, X_train, y_train, X_val, y_val)
    assert metrics["train_r2"] > 0.5
    assert metrics["val_r2"] > 0.4
    assert metrics["train_rmse"] > 0
    assert metrics["val_rmse"] > 0

    # Predictions finite and non-negative
    preds = model.predict(X_val)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)


def test_model_persistence(tmp_path):
    model = create_cart_regressor(max_depth=3, random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([100.0, 200.0, 300.0, 400.0])
    model.fit(X, y)

    save_path = tmp_path / "test_model.joblib"
    save_pipeline(model, save_path)
    assert save_path.exists()

    loaded_model = load_pipeline(save_path)
    pred_orig = model.predict(X)
    pred_loaded = loaded_model.predict(X)
    np.testing.assert_array_almost_equal(pred_orig, pred_loaded)
