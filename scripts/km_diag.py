"""
Step 0: Full km sensitivity diagnostic.
Run with: .\venv\Scripts\python.exe scripts\km_diag.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

ROOT = Path(r"c:\Users\DHARSHAN\Documents\Antigravity\ML PBL3")
sys.path.insert(0, str(ROOT))

from src.feature_engineering import extract_brand_and_model
from src.prediction import _build_input_df

# ── Load artifacts ──────────────────────────────────────────────
meta = json.load(open(ROOT / "models/model_metadata.json"))
model = joblib.load(ROOT / "models/best_cart_model.joblib")
prep  = joblib.load(ROOT / "models/preprocessing_pipeline.joblib")
df    = pd.read_csv(ROOT / "data/processed/cleaned_car_data.csv")
all_cols = meta["input_features"]
num_cols  = meta["numerical_features"]

print("=" * 65)
print("DIAGNOSTIC 1 — WIRING CHECK")
print("=" * 65)
print(f"input_features in metadata: {all_cols}")
print(f"km_driven in input_features: {'km_driven' in all_cols}")
print(f"km_per_year in input_features: {'km_per_year' in all_cols}")

# Check ColumnTransformer
print("\nColumnTransformer transformers:")
for name, trans, cols in prep.transformers_:
    print(f"  [{name}] cols={cols}")
    if isinstance(cols, list) and "km_driven" in cols:
        print("    ^^^ km_driven IS in this transformer")

# Manual transform test
sample = {c: df[c].median() if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else
           (df[c].mode()[0] if c in df.columns else "Unknown")
           for c in all_cols}
sample["km_driven"] = 50000
df1 = _build_input_df(sample, all_cols)
v1 = prep.transform(df1)

sample["km_driven"] = 150000
df2 = _build_input_df(sample, all_cols)
v2 = prep.transform(df2)

km_col_idx = all_cols.index("km_driven") if "km_driven" in all_cols else -1
print(f"\nkm_driven column index in input: {km_col_idx}")
print(f"Transformed vector changes: {(v1 != v2).sum()} elements differ")
if km_col_idx >= 0:
    print(f"  v1[km_idx]={v1[0, km_col_idx]:.1f}  v2[km_idx]={v2[0, km_col_idx]:.1f}")

p1 = float(model.predict(v1)[0])
p2 = float(model.predict(v2)[0])
print(f"\nPrice at km=50,000:  ₹{p1:,.0f}")
print(f"Price at km=150,000: ₹{p2:,.0f}")
print(f"Difference: ₹{p2-p1:,.0f}  ({(p2-p1)/p1*100:.1f}%)")

print("\n" + "=" * 65)
print("DIAGNOSTIC 2 — TREE USAGE OF km_driven / km_per_year")
print("=" * 65)
from sklearn.tree import _tree
from src.preprocessing import get_feature_names_from_preprocessor
feat_names = get_feature_names_from_preprocessor(prep)

tree = model.tree_
km_related = [i for i, f in enumerate(feat_names)
              if "km_driven" in f or "km_per_year" in f or "high_mileage" in f]
print(f"km-related feature indices in transformed space: {km_related}")
print(f"km-related feature names: {[feat_names[i] for i in km_related]}")

# Count nodes using km features
total_splits = int((tree.feature >= 0).sum())
km_splits = sum(1 for i in km_related if i in tree.feature)

# More precise: count splits per feature
from collections import Counter
split_features = [tree.feature[n] for n in range(tree.node_count) if tree.feature[n] >= 0]
feat_split_counts = Counter(split_features)
print(f"\nTotal split nodes: {total_splits}")
km_node_count = sum(feat_split_counts.get(i, 0) for i in km_related)
print(f"Splits using km features: {km_node_count}")
print(f"Percentage: {km_node_count/total_splits*100:.2f}%")

# Depths of km splits
depths = []
def get_depth(node_id, depth=0):
    if tree.feature[node_id] in km_related:
        depths.append(depth)
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    if left != _tree.TREE_LEAF:
        get_depth(left, depth+1)
        get_depth(right, depth+1)
get_depth(0)
if depths:
    print(f"Depths of km splits: min={min(depths)} max={max(depths)} mean={np.mean(depths):.1f}")
else:
    print("km features appear in ZERO splits!")

# Feature importances
importances = model.feature_importances_
km_importance = sum(importances[i] for i in km_related if i < len(importances))
print(f"\nkm-related feature importance: {km_importance:.4f} ({km_importance*100:.2f}%)")
print(f"Top 10 features by importance:")
top_idx = np.argsort(importances)[::-1][:10]
for idx in top_idx:
    fn = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
    print(f"  {fn:50s}: {importances[idx]:.4f}")

print("\n" + "=" * 65)
print("DIAGNOSTIC 3 — PATH CHECK (20 random cars)")
print("=" * 65)
df_sample = df.dropna(subset=["km_driven", "selling_price"]).sample(20, random_state=42)
df_sample = df_sample[[c for c in all_cols if c in df_sample.columns]].copy()

n_no_change = 0
n_total = 0
pct_changes = []
for _, row in df_sample.iterrows():
    base = row.to_dict()
    base_km = base.get("km_driven", 50000)
    preds = set()
    for mult in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        r = dict(base)
        r["km_driven"] = base_km * mult
        try:
            inp = _build_input_df(r, all_cols)
            v = prep.transform(inp)
            preds.add(round(float(model.predict(v)[0]), -3))  # round to nearest 1000
        except Exception:
            pass
    n_total += 1
    if len(preds) == 1:
        n_no_change += 1
    else:
        # pct change from 0.1x to 3.0x
        lo_r = dict(base); lo_r["km_driven"] = base_km * 0.1
        hi_r = dict(base); hi_r["km_driven"] = base_km * 3.0
        try:
            vlo = prep.transform(_build_input_df(lo_r, all_cols))
            vhi = prep.transform(_build_input_df(hi_r, all_cols))
            plo = float(model.predict(vlo)[0])
            phi = float(model.predict(vhi)[0])
            pct_changes.append((phi - plo) / plo * 100)
        except Exception:
            pass

print(f"Cars with ZERO prediction change across km range: {n_no_change}/{n_total} ({n_no_change/n_total*100:.0f}%)")
if pct_changes:
    print(f"Among cars that DO change: avg % change (lo->hi km): {np.mean(pct_changes):.1f}%")

print("\n" + "=" * 65)
print("DIAGNOSTIC 4 — HYPERPARAMETERS")
print("=" * 65)
print(f"model type: {type(model).__name__}")
print(f"max_depth: {model.max_depth}")
print(f"min_samples_leaf: {model.min_samples_leaf}")
print(f"min_samples_split: {model.min_samples_split}")
print(f"ccp_alpha: {model.ccp_alpha}")
print(f"tree actual depth: {model.get_depth()}")
print(f"tree n_leaves: {model.get_n_leaves()}")
print(f"monotonic_cst: {getattr(model, 'monotonic_cst', 'NOT SUPPORTED')}")

print("\n" + "=" * 65)
print("DIAGNOSTIC 5 — DATA CHECK (km vs price within cohorts)")
print("=" * 65)
df_full = df.copy()
if "brand" in df_full.columns and "model_name" in df_full.columns and "km_driven" in df_full.columns:
    overall_corr, pval = spearmanr(df_full["km_driven"].dropna(), df_full["selling_price"].dropna())
    print(f"Overall Spearman(km, price): r={overall_corr:.3f}  p={pval:.4f}")

    # Within same brand+model+age_band cohorts
    df_full["age_band"] = pd.cut(df_full.get("vehicle_age", 2024 - df_full["year"]),
                                  bins=[0, 3, 7, 12, 30], labels=["0-3y", "3-7y", "7-12y", "12+y"])
    cohort_corrs = []
    for (brand, model_n, age_b), grp in df_full.groupby(["brand", "model_name", "age_band"]):
        if len(grp) >= 10:
            r, p = spearmanr(grp["km_driven"], grp["selling_price"])
            cohort_corrs.append({"brand": brand, "model": model_n, "age_band": str(age_b),
                                  "n": len(grp), "spearman_r": round(r, 3)})
    cdf = pd.DataFrame(cohort_corrs)
    if not cdf.empty:
        print(f"Cohorts with >= 10 rows: {len(cdf)}")
        print(f"Median Spearman r within cohorts: {cdf['spearman_r'].median():.3f}")
        print(f"Fraction with negative r (expected): {(cdf['spearman_r'] < 0).mean()*100:.1f}%")
        print(f"\nTop 10 cohorts with strongest negative r:")
        print(cdf.sort_values("spearman_r").head(10).to_string(index=False))

# Save summary
summary = {
    "km_in_input_features": "km_driven" in all_cols,
    "km_transformer_elements_change": int((v1 != v2).sum()),
    "price_at_50k": float(p1),
    "price_at_150k": float(p2),
    "price_diff": float(p2 - p1),
    "km_split_nodes": int(km_node_count),
    "km_importance_pct": float(km_importance * 100),
    "pct_cars_no_change_across_km_range": float(n_no_change / n_total * 100),
    "model_max_depth": model.max_depth,
    "model_min_samples_leaf": model.min_samples_leaf,
    "model_ccp_alpha": model.ccp_alpha,
    "overall_spearman_km_price": float(overall_corr),
}
(ROOT / "reports").mkdir(exist_ok=True)
json.dump(summary, open(ROOT / "reports/km_sensitivity_summary.json", "w"), indent=2)
print("\nSaved reports/km_sensitivity_summary.json")
print("\nDIAGNOSTIC COMPLETE")
