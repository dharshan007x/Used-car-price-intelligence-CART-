"""
Unit tests for Prediction Pipeline, Decision Path Explainability, and Novelty Features.
"""

import pytest
import numpy as np
from src.prediction import format_inr, evaluate_prediction_reliability, compute_market_positioning
from src.explainability import explain_prediction_path
from src.cart_model import create_cart_regressor


def test_format_inr():
    assert format_inr(45000) == "₹45,000"
    assert format_inr(545000) == "₹5.45 Lakh"
    assert format_inr(15000000) == "₹1.50 Cr"
    assert format_inr(0) == "₹0"


def test_market_positioning():
    estimated = 500000.0

    # Underpriced (> 10% lower)
    pos_under = compute_market_positioning(estimated, 400000.0, tolerance_pct=10.0)
    assert "Underpriced" in pos_under["positioning"]
    assert pos_under["diff_pct"] == -20.0

    # Fair range (+/- 10%)
    pos_fair = compute_market_positioning(estimated, 520000.0, tolerance_pct=10.0)
    assert "Within Estimated" in pos_fair["positioning"]
    assert pos_fair["diff_pct"] == 4.0

    # Overpriced (> 10% higher)
    pos_over = compute_market_positioning(estimated, 650000.0, tolerance_pct=10.0)
    assert "Overpriced" in pos_over["positioning"]
    assert pos_over["diff_pct"] == 30.0


def test_explain_prediction_path():
    np.random.seed(42)
    X = np.array([
        [2.0, 30000.0, 1200.0],
        [4.0, 50000.0, 1500.0],
        [7.0, 80000.0, 1995.0],
        [10.0, 120000.0, 2200.0]
    ])
    y = np.array([600000.0, 500000.0, 350000.0, 200000.0])

    model = create_cart_regressor(max_depth=3, random_state=42)
    model.fit(X, y)

    feature_names = ["vehicle_age", "km_driven", "engine_cc"]
    sample = np.array([3.0, 40000.0, 1300.0])
    raw_dict = {"vehicle_age": 3, "km_driven": 40000, "engine_cc": 1300}

    explanation = explain_prediction_path(model, feature_names, sample, raw_dict)

    assert "leaf_id" in explanation
    assert "leaf_samples" in explanation
    assert explanation["leaf_samples"] >= 1
    assert len(explanation["steps"]) > 0
    assert len(explanation["narrative"]) > 0
    assert explanation["steps"][-1]["is_leaf"] is True


def test_evaluate_prediction_reliability():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randn(50) * 10000 + 500000

    model = create_cart_regressor(max_depth=3, random_state=42)
    model.fit(X, y)

    sample = X[0]
    user_inputs = {"brand": "Maruti", "km_driven": 45000, "vehicle_age": 4, "engine_cc": 1200}
    rel = evaluate_prediction_reliability(model, sample, user_inputs, known_brands={"Maruti", "Hyundai"})

    assert "support_level" in rel
    assert "margin_of_error" in rel
    assert rel["is_out_of_distribution"] is False

    # Test extreme outlier
    outlier_inputs = {"brand": "UnknownSupercar", "km_driven": 500000, "vehicle_age": 30, "engine_cc": 7000}
    rel_outlier = evaluate_prediction_reliability(model, sample, outlier_inputs, known_brands={"Maruti", "Hyundai"})
    assert rel_outlier["is_out_of_distribution"] is True
    assert len(rel_outlier["ood_flags"]) >= 3
