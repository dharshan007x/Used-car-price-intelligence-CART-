"""
Unit tests for Cost-Complexity Pruning.
"""

import pytest
import numpy as np
from src.pruning import compute_pruning_path, select_candidate_alphas, run_pruning_experiments


def test_pruning_path_computation():
    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = X[:, 0] * 100 + X[:, 1] * 50 + np.random.randn(80) * 10

    alphas, impurities, base_tree = compute_pruning_path(X, y, random_state=42)
    assert len(alphas) > 0
    assert np.all(alphas >= 0.0)
    assert np.all(np.diff(alphas) >= 0.0)  # Monotonically increasing


def test_candidate_alpha_selection():
    alphas = np.linspace(0, 100, 50)
    candidates = select_candidate_alphas(alphas, max_candidates=10)
    assert len(candidates) <= 10
    assert 0.0 in candidates
    assert candidates[-1] == 100.0


def test_run_pruning_experiments():
    np.random.seed(42)
    X_train = np.random.randn(100, 6)
    y_train = X_train[:, 0] * 500 + X_train[:, 1] * 300 + np.random.randn(100) * 20
    X_val = np.random.randn(30, 6)
    y_val = X_val[:, 0] * 500 + X_val[:, 1] * 300 + np.random.randn(30) * 20

    res_df, interp, best_m = run_pruning_experiments(
        X_train, y_train, X_val, y_val, random_state=42
    )
    assert not res_df.empty
    assert "alpha" in res_df.columns
    assert "total_nodes" in res_df.columns
    # Check that as alpha increases, total nodes generally non-increases
    assert res_df["total_nodes"].iloc[0] >= res_df["total_nodes"].iloc[-1]
    assert interp["best_alpha"] >= 0.0
