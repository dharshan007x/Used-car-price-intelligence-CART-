"""Quick verification that the mileage adjuster produces km-sensitive prices."""
import json
import joblib
import numpy as np

from src.prediction import apply_mileage_adjustment, _build_input_df

meta  = json.load(open("models/model_metadata.json"))
model = joblib.load("models/best_cart_model.joblib")
pre   = joblib.load("models/preprocessing_pipeline.joblib")
adj   = joblib.load("models/mileage_adjuster.joblib")

cols = meta["input_features"]
tkm  = meta["typical_km_per_year"]

base = {
    "vehicle_age": 5, "engine_cc": 1197, "mileage_val": 18.0,
    "max_power_bhp": 82.0, "owner_tier": 1, "seats": 5.0,
    "torque_nm": 113.0, "high_mileage_flag": 0,
    "brand": "Maruti", "model_name": "Swift", "fuel": "Petrol",
    "transmission": "Manual", "seller_type": "Individual",
    "trim_tier": "Mid", "body_segment": "Hatchback",
}

print(f"typical_km_per_year = {tkm:.0f} km/yr")
print(f"{'km':>10}  {'km_ratio':>8}  {'CART base':>12}  {'Adjusted':>12}  {'Adj%':>7}")
print("-" * 60)

prev_adj = None
n_changed = 0
test_kms = [10000, 30000, 50000, 80000, 120000, 200000, 300000]

for km in test_kms:
    row = dict(base)
    row["km_driven"] = km
    eff = max(float(base["vehicle_age"]), 0.5)
    row["km_per_year"] = round(km / eff, 1)
    row["high_mileage_flag"] = 1 if row["km_per_year"] > 20000 else 0
    expected = eff * (tkm if tkm > 0 else 10000.0)
    km_r = float(np.clip(km / max(expected, 1), 0.05, 10.0))
    row["km_ratio"] = round(km_r, 4)

    df = _build_input_df(row, cols)
    vec = pre.transform(df)
    cart_p = max(0.0, float(model.predict(vec)[0]))
    res = apply_mileage_adjustment(cart_p, km_r, adj, enabled=True)
    pct = (res["adjustment_factor"] - 1) * 100
    adj_p = res["final_price"]

    if prev_adj is not None and adj_p != prev_adj:
        n_changed += 1

    prev_adj = adj_p
    print(f"{km:>10,}  {km_r:>8.3f}  {cart_p:>12,.0f}  {adj_p:>12,.0f}  {pct:>+7.1f}%")

print("-" * 60)
print(f"\nDistinct adjusted prices across {len(test_kms)} km values: {n_changed+1 if n_changed else 1}")
lo_p = None
hi_p = None
for km, mult in [(15000, "0.3x"), (150000, "3x")]:
    row = dict(base)
    row["km_driven"] = km
    eff = max(float(base["vehicle_age"]), 0.5)
    row["km_per_year"] = round(km / eff, 1)
    row["high_mileage_flag"] = 1 if row["km_per_year"] > 20000 else 0
    expected = eff * (tkm if tkm > 0 else 10000.0)
    km_r = float(np.clip(km / max(expected, 1), 0.05, 10.0))
    row["km_ratio"] = round(km_r, 4)
    df = _build_input_df(row, cols)
    vec = pre.transform(df)
    cart_p = max(0.0, float(model.predict(vec)[0]))
    res = apply_mileage_adjustment(cart_p, km_r, adj, enabled=True)
    print(f"  km={km:,} ({mult}): CART={cart_p:,.0f}  Adjusted={res['final_price']:,.0f}")
    if mult == "0.3x":
        lo_p = res["final_price"]
    else:
        hi_p = res["final_price"]

if lo_p and hi_p:
    drop = (hi_p / lo_p - 1) * 100
    print(f"\nPrice drop from 0.3x km to 3x km: {drop:+.1f}%")
    if hi_p < lo_p:
        print("PASS: Higher km -> lower adjusted price")
    else:
        print("WARN: Adjuster did not reduce price for higher km")
