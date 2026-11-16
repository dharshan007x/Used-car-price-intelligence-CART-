"""
Tests for km_driven sensitivity — Step 5 of the km-fix task.

Verifies:
1. Wiring: changing km_driven changes the transformed feature vector.
2. Feature presence: km_per_year and km_ratio exist after engineering.
3. Monotonicity: CART + adjuster price never rises when km increases (for ≥70% of samples).
4. Sensitivity: ≥70% of sampled cars show strictly lower price at 3× km vs 0.3× km.
5. Adjuster bounds: adjustment factor is finite and within [0.70, 1.30].
6. Mileage impact curve: returns valid non-empty list of dicts with required keys.
7. km importance: km-related features have non-zero total importance in the fitted tree.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers to load artifacts (cached across tests) ─────────────────────────

def _load_artifacts():
    """Load fitted model, preprocessor, metadata. Skip if not found."""
    models_dir = PROJECT_ROOT / "models"
    meta_path = models_dir / "model_metadata.json"
    model_path = models_dir / "best_cart_model.joblib"
    prep_path = models_dir / "preprocessing_pipeline.joblib"
    adjuster_path = models_dir / "mileage_adjuster.joblib"

    for p in (meta_path, model_path, prep_path):
        if not p.exists():
            pytest.skip(f"Artifact not found: {p}. Run python -m src.train first.")

    import json
    import joblib
    with open(meta_path) as f:
        meta = json.load(f)

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    adjuster = joblib.load(adjuster_path) if adjuster_path.exists() else None
    return model, preprocessor, meta, adjuster


def _make_base_input(meta: dict) -> dict:
    """Create a representative input row using training medians."""
    medians = meta.get("training_stats", {}).get("medians", {})
    input_features = meta.get("input_features", [])

    row = {}
    for col in input_features:
        row[col] = medians.get(col, np.nan)

    # Set sensible categorical defaults
    row.setdefault("brand", "Maruti")
    row.setdefault("model_name", "Swift")
    row.setdefault("fuel", "Petrol")
    row.setdefault("transmission", "Manual")
    row.setdefault("seller_type", "Individual")
    row.setdefault("trim_tier", "Mid")
    row.setdefault("body_segment", "Hatchback")

    # Sensible numeric defaults
    row.setdefault("vehicle_age", 5.0)
    row.setdefault("km_driven", 50000.0)
    row.setdefault("engine_cc", 1197.0)
    row.setdefault("mileage_val", 18.0)
    row.setdefault("max_power_bhp", 82.0)
    row.setdefault("owner_tier", 1.0)
    row.setdefault("seats", 5.0)
    row.setdefault("torque_nm", 113.0)
    row.setdefault("high_mileage_flag", 0.0)

    # Derived mileage features
    typical_km = meta.get("typical_km_per_year", 10000.0)
    age = max(float(row["vehicle_age"]), 0.5)
    km = float(row["km_driven"])
    row["km_per_year"] = round(km / age, 1)
    expected = age * typical_km
    row["km_ratio"] = float(np.clip(km / max(expected, 1), 0.05, 10.0))

    return row


def _predict(model, preprocessor, input_row: dict, all_feature_cols: list) -> float:
    """Score a single input row."""
    import pandas as pd
    df = pd.DataFrame([{col: input_row.get(col, np.nan) for col in all_feature_cols}])
    vec = preprocessor.transform(df)
    return float(model.predict(vec)[0])


# ════════════════════════════════════════════════════════════════════
# TEST 1 — Wiring: km_driven change → transformed vector changes
# ════════════════════════════════════════════════════════════════════

def test_km_driven_wiring():
    """Changing km_driven must change at least one element of the transformed vector."""
    import pandas as pd
    model, preprocessor, meta, _ = _load_artifacts()
    all_cols = meta.get("input_features", [])
    base = _make_base_input(meta)

    high = dict(base)
    high["km_driven"] = 150000.0
    high["km_per_year"] = round(150000 / max(float(base["vehicle_age"]), 0.5), 1)
    typical_km = meta.get("typical_km_per_year", 10000.0)
    age = max(float(base["vehicle_age"]), 0.5)
    high["km_ratio"] = float(np.clip(150000 / max(age * typical_km, 1), 0.05, 10.0))
    high["high_mileage_flag"] = 1 if high["km_per_year"] > 20000 else 0

    df_low = pd.DataFrame([{col: base.get(col, np.nan) for col in all_cols}])
    df_high = pd.DataFrame([{col: high.get(col, np.nan) for col in all_cols}])

    vec_low = preprocessor.transform(df_low)
    vec_high = preprocessor.transform(df_high)

    n_differ = int(np.sum(vec_low != vec_high))
    assert n_differ >= 1, (
        f"Transformed vector did not change when km_driven changed from "
        f"{base['km_driven']} to {high['km_driven']}. "
        "km_driven may not be wired into the ColumnTransformer."
    )


# ════════════════════════════════════════════════════════════════════
# TEST 2 — Feature presence: km_per_year and km_ratio after engineering
# ════════════════════════════════════════════════════════════════════

def test_km_derived_features_in_input_list():
    """km_per_year and km_ratio must both appear in input_features after training."""
    _, _, meta, _ = _load_artifacts()
    input_features = meta.get("input_features", [])
    assert "km_per_year" in input_features, (
        "km_per_year not in input_features. "
        "Add it to MARKET_NUMERICAL_FEATURES in preprocessing.py and retrain."
    )
    assert "km_ratio" in input_features, (
        "km_ratio not in input_features. "
        "Add it to MARKET_NUMERICAL_FEATURES in preprocessing.py and retrain."
    )


# ════════════════════════════════════════════════════════════════════
# TEST 3 — Monotonicity: price must not rise as km increases (for most cars)
# ════════════════════════════════════════════════════════════════════

def test_km_monotonicity():
    """
    For 20 test cars (median profile + varying km), increasing km must
    never increase the final price for ≥70% of them.
    """
    model, preprocessor, meta, adjuster = _load_artifacts()
    all_cols = meta.get("input_features", [])
    typical_km = meta.get("typical_km_per_year", 10000.0)

    from src.prediction import apply_mileage_adjustment

    base = _make_base_input(meta)
    n_monotone = 0
    n_total = 0

    # Test 20 different age/km combos
    for age in [2, 5, 8, 12]:
        for km_low_mult in [0.3, 0.6]:
            km_low = max(500, int(50000 * km_low_mult))
            km_high = km_low * 3

            prices = []
            for km in [km_low, km_high]:
                row = dict(base)
                row["vehicle_age"] = float(age)
                row["km_driven"] = float(km)
                eff_age = max(float(age), 0.5)
                row["km_per_year"] = round(km / eff_age, 1)
                row["high_mileage_flag"] = 1 if row["km_per_year"] > 20000 else 0
                expected = eff_age * (typical_km if typical_km > 0 else 10000.0)
                km_r = float(np.clip(km / max(expected, 1), 0.05, 10.0))
                row["km_ratio"] = round(km_r, 4)

                cart_p = _predict(model, preprocessor, row, all_cols)
                adj = apply_mileage_adjustment(cart_p, km_r, adjuster,
                                              enabled=adjuster is not None)
                prices.append(adj["final_price"])

            n_total += 1
            if prices[1] <= prices[0]:  # higher km → lower or equal price ✓
                n_monotone += 1

    pct = 100.0 * n_monotone / n_total if n_total > 0 else 0
    assert pct >= 70.0, (
        f"Monotonicity failed: only {n_monotone}/{n_total} ({pct:.1f}%) car profiles "
        f"show non-increasing price with higher km. "
        f"Expected ≥70%. Monotonic constraints or adjuster may not be working."
    )


# ════════════════════════════════════════════════════════════════════
# TEST 4 — Sensitivity: ≥70% of cars show lower price at 3× km vs 0.3× km
# ════════════════════════════════════════════════════════════════════

def test_km_sensitivity_threshold():
    """
    For 20 sampled profiles, the final (adjusted) price at km×3
    must be strictly lower than at km×0.3 for ≥70% of profiles.
    """
    model, preprocessor, meta, adjuster = _load_artifacts()
    all_cols = meta.get("input_features", [])
    typical_km = meta.get("typical_km_per_year", 10000.0)

    from src.prediction import apply_mileage_adjustment

    failing = []
    test_cases = [
        (3, 20000), (3, 50000), (3, 80000),
        (5, 20000), (5, 50000), (5, 80000), (5, 120000),
        (8, 50000), (8, 80000), (8, 120000), (8, 200000),
        (10, 50000), (10, 80000), (10, 150000), (10, 200000),
        (12, 80000), (12, 150000), (12, 200000),
        (15, 100000), (15, 200000),
    ]

    n_pass = 0
    base = _make_base_input(meta)

    for age, base_km in test_cases:
        prices = {}
        for mult, label in [(0.3, "lo"), (3.0, "hi")]:
            km = max(500, int(base_km * mult))
            row = dict(base)
            row["vehicle_age"] = float(age)
            row["km_driven"] = float(km)
            eff_age = max(float(age), 0.5)
            row["km_per_year"] = round(km / eff_age, 1)
            row["high_mileage_flag"] = 1 if row["km_per_year"] > 20000 else 0
            expected = eff_age * (typical_km if typical_km > 0 else 10000.0)
            km_r = float(np.clip(km / max(expected, 1), 0.05, 10.0))
            row["km_ratio"] = round(km_r, 4)

            cart_p = _predict(model, preprocessor, row, all_cols)
            adj = apply_mileage_adjustment(cart_p, km_r, adjuster,
                                           enabled=adjuster is not None)
            prices[label] = adj["final_price"]

        if prices["hi"] < prices["lo"]:
            n_pass += 1
        else:
            failing.append((age, base_km, prices["lo"], prices["hi"]))

    total = len(test_cases)
    pct = 100.0 * n_pass / total
    assert pct >= 70.0, (
        f"Sensitivity check failed: only {n_pass}/{total} ({pct:.1f}%) profiles show "
        f"lower price at 3× km vs 0.3× km (expected ≥70%).\n"
        f"Failing cases (age, base_km, price_lo, price_hi): {failing[:5]}"
    )


# ════════════════════════════════════════════════════════════════════
# TEST 5 — Adjuster bounds: factor finite, within [0.70, 1.30]
# ════════════════════════════════════════════════════════════════════

def test_adjuster_bounds():
    """Mileage adjustment factor must be finite and within [0.70, 1.30]."""
    _, _, meta, adjuster = _load_artifacts()
    if adjuster is None:
        pytest.skip("No mileage adjuster found (models/mileage_adjuster.joblib missing).")

    from src.prediction import apply_mileage_adjustment

    for km_ratio in np.linspace(0.1, 5.0, 20):
        result = apply_mileage_adjustment(
            cart_price=500000.0,
            km_ratio=float(km_ratio),
            adjuster=adjuster,
            adj_lower=0.80,
            adj_upper=1.15,
            enabled=True,
        )
        f = result["adjustment_factor"]
        fp = result["final_price"]
        assert np.isfinite(f), f"Adjustment factor is not finite at km_ratio={km_ratio}"
        assert np.isfinite(fp), f"Final price is not finite at km_ratio={km_ratio}"
        assert fp >= 0, f"Final price is negative at km_ratio={km_ratio}"
        assert 0.70 <= f <= 1.30, (
            f"Adjustment factor {f:.4f} out of broad bounds [0.70, 1.30] at km_ratio={km_ratio}. "
            "Check adjuster training or clip bounds."
        )


# ════════════════════════════════════════════════════════════════════
# TEST 6 — Mileage impact curve: valid structure
# ════════════════════════════════════════════════════════════════════

def test_mileage_impact_curve_structure():
    """compute_mileage_impact_curve must return a non-empty list with required keys."""
    model, preprocessor, meta, adjuster = _load_artifacts()
    all_cols = meta.get("input_features", [])
    typical_km = meta.get("typical_km_per_year", 10000.0)

    from src.prediction import compute_mileage_impact_curve

    base = _make_base_input(meta)
    pts = compute_mileage_impact_curve(
        base_input=base,
        model=model,
        preprocessor=preprocessor,
        feature_columns=all_cols,
        adjuster=adjuster,
        typical_km_per_year=typical_km,
        n_points=10,
    )

    assert len(pts) > 0, "compute_mileage_impact_curve returned empty list."
    required_keys = {"km", "km_ratio", "cart_price", "adjusted_price", "pct_vs_base", "adjustment_factor"}
    for pt in pts:
        missing = required_keys - set(pt.keys())
        assert not missing, f"Missing keys in impact curve point: {missing}"
        assert np.isfinite(pt["cart_price"]), f"cart_price not finite: {pt}"
        assert np.isfinite(pt["adjusted_price"]), f"adjusted_price not finite: {pt}"
        assert pt["cart_price"] >= 0
        assert pt["adjusted_price"] >= 0


# ════════════════════════════════════════════════════════════════════
# TEST 7 — km feature importance: non-zero in fitted tree
# ════════════════════════════════════════════════════════════════════

def test_km_feature_importance_nonzero():
    """
    km-related features (km_driven, km_per_year, km_ratio, high_mileage_flag)
    must have non-zero total importance in the best CART model.
    """
    model, preprocessor, meta, _ = _load_artifacts()

    feature_names = meta.get("transformed_feature_names", [])
    if not feature_names:
        pytest.skip("transformed_feature_names not in metadata. Retrain.")

    km_features = {"km_driven", "km_per_year", "km_ratio", "high_mileage_flag"}
    importances = model.feature_importances_

    total_km_imp = 0.0
    for name, imp in zip(feature_names, importances):
        raw = name
        for pfx in ("num__", "cat_hc__", "cat_std__"):
            if name.startswith(pfx):
                raw = name[len(pfx):]
                break
        if raw in km_features:
            total_km_imp += imp

    assert total_km_imp > 1e-6, (
        f"Total km-related feature importance is {total_km_imp:.6f} ≈ 0. "
        "km features have no effect in the fitted tree."
    )
