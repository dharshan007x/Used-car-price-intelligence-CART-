"""
Generates the comprehensive Google Colab / Jupyter notebook:
notebooks/used_car_price_intelligence_colab.ipynb
and updates notebooks/exploratory_analysis.ipynb with Colab compatibility.
"""
import json
from pathlib import Path

def create_colab_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚗 Used Car Price Intelligence – CART Regression System\n",
                    "### Batch 12 | Machine Learning Academic Project\n",
                    "\n",
                    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dharshan007x/Used-car-price-intelligence-CART-/blob/main/notebooks/used_car_price_intelligence_colab.ipynb)\n",
                    "![Python](https://img.shields.io/badge/Python-3.11-blue.svg)\n",
                    "![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)\n",
                    "![Tests](https://img.shields.io/badge/Tests-49%20Passing-brightgreen.svg)\n",
                    "\n",
                    "---\n",
                    "### Notebook Overview\n",
                    "This end-to-end notebook implements the complete **CART Regression Intelligence System** for pre-owned car valuation in the Indian automotive market:\n",
                    "1. **Google Colab Environment Auto-Setup**: Automatic repository cloning and dependency setup.\n",
                    "2. **Data Acquisition & Leak-Free Cleaning**: Validation and parsing of real CarDekho listings.\n",
                    "3. **Domain Feature Engineering**: Multi-word brand/model extraction, torque parsing, trim tier, body segments, and usage intensity.\n",
                    "4. **Exploratory Data Analysis**: Price distributions, non-linear depreciation patterns, and factor interactions.\n",
                    "5. **CART Tree Depth & Bias-Variance Analysis**: Characterizing underfitting vs overfitting across depths 1 to 20.\n",
                    "6. **Minimal Cost-Complexity Pruning (CCP)**: Pruning trajectory along effective $\\alpha$ penalties.\n",
                    "7. **Monotonic Constraints & Mileage Sensitivity**: Enforcing $km \\to \\text{Price}$ monotonicity with an Isotonic post-processing layer.\n",
                    "8. **Transparent Decision-Path Explainability**: Deconstructing individual predictions node-by-node."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment & Colab Auto-Setup\n",
                    "If running in Google Colab, this cell automatically clones the repository, navigates into the project directory, and installs required packages."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Detect Colab environment and clone repository\n",
                    "import os\n",
                    "import sys\n",
                    "\n",
                    "IN_COLAB = 'google.colab' in sys.modules or ('get_ipython' in globals() and 'google.colab' in str(get_ipython()))\n",
                    "\n",
                    "if IN_COLAB:\n",
                    "    print(\"🚀 Running in Google Colab environment.\")\n",
                    "    if not os.path.exists('Used-car-price-intelligence-CART-'):\n",
                    "        !git clone https://github.com/dharshan007x/Used-car-price-intelligence-CART-.git\n",
                    "    %cd Used-car-price-intelligence-CART-\n",
                    "    !pip install -q -r requirements.txt\n",
                    "    if '.' not in sys.path:\n",
                    "        sys.path.insert(0, '.')\n",
                    "else:\n",
                    "    print(\"💻 Running in local Python/Jupyter environment.\")\n",
                    "    if '..' not in sys.path and os.path.exists('../src'):\n",
                    "        sys.path.insert(0, '..')\n",
                    "    elif '.' not in sys.path and os.path.exists('src'):\n",
                    "        sys.path.insert(0, '.')\n",
                    "\n",
                    "# Standard data science stack imports\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from pathlib import Path\n",
                    "import json\n",
                    "import joblib\n",
                    "\n",
                    "# Set plot styling\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "plt.rcParams['figure.figsize'] = (10, 6)\n",
                    "plt.rcParams['font.size'] = 11\n",
                    "print(\"✅ Setup complete. Libraries ready.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Data Acquisition & Validation\n",
                    "Load authentic dataset of 8,128 CarDekho vehicle listings and run schema validation and cleaning without train-test leakage."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def resolve_path(rel_path):\n",
                    "    p1 = Path(rel_path)\n",
                    "    if p1.exists():\n",
                    "        return str(p1)\n",
                    "    p2 = Path(\"..\") / rel_path\n",
                    "    if p2.exists():\n",
                    "        return str(p2)\n",
                    "    return rel_path\n",
                    "\n",
                    "from src.data_loader import load_raw_dataset\n",
                    "from src.data_validation import perform_data_validation\n",
                    "from src.feature_engineering import engineer_features\n",
                    "\n",
                    "raw_path = resolve_path(\"data/raw/car_details_v3.csv\")\n",
                    "df_raw = load_raw_dataset(raw_path)\n",
                    "print(f\"📊 Raw Records Loaded: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns\")\n",
                    "\n",
                    "df_clean, audit_report = perform_data_validation(df_raw, reference_year=2024)\n",
                    "print(f\"🧹 Validated Records: {audit_report['final_record_count']:,} (Removed {audit_report['initial_record_count'] - audit_report['final_record_count']} invalid/duplicate rows)\")\n",
                    "df_clean.head(4)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Market-Grade Feature Engineering\n",
                    "We derive:\n",
                    "- **Brand & Model**: Multi-word alias-aware extraction (e.g. \"Maruti Swift Dzire\", \"Land Rover Range Rover\")\n",
                    "- **Annual Usage Intensity (`km_per_year`)** and **Age-Normalised Usage (`km_ratio`)**\n",
                    "- **Ownership Tier** (ordinal 1 to 5)\n",
                    "- **Engine Torque (Nm)** parsed from free-text\n",
                    "- **Trim Tier** (Base, Mid, Top) and **Body Segment** (Hatchback, Sedan, SUV, MUV, Van)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Feature Engineering with typical_km_per_year = 6250.0 (leak-free training median)\n",
                    "df_eng = engineer_features(df_clean, reference_year=2024, typical_km_per_year=6250.0)\n",
                    "print(f\"⚙️ Engineered Dataset Dimensions: {df_eng.shape[0]:,} rows × {df_eng.shape[1]} features\")\n",
                    "\n",
                    "display_cols = ['brand', 'model_name', 'vehicle_age', 'km_driven', 'km_per_year', 'km_ratio', 'owner_tier', 'trim_tier', 'body_segment', 'selling_price']\n",
                    "df_eng[display_cols].head(8)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Exploratory Data Analysis & Visualizations\n",
                    "Analyzing price distribution and non-linear depreciation across vehicle age and mileage."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
                    "\n",
                    "# Price Distribution\n",
                    "sns.histplot(df_eng['selling_price'] / 100000, kde=True, bins=45, color='#1f77b4', ax=axes[0])\n",
                    "axes[0].set_title('Used Car Resale Price Distribution (INR)', fontweight='bold')\n",
                    "axes[0].set_xlabel('Selling Price (₹ Lakhs)')\n",
                    "axes[0].set_ylabel('Listing Count')\n",
                    "axes[0].set_xlim(0, 35)\n",
                    "\n",
                    "# Age vs Price Boxplot\n",
                    "sns.boxplot(x='vehicle_age', y=df_eng['selling_price'] / 100000, data=df_eng[df_eng['vehicle_age'] <= 15],\n",
                    "            palette='Blues_r', ax=axes[1])\n",
                    "axes[1].set_title('Non-Linear Depreciation: Vehicle Age vs Resale Price', fontweight='bold')\n",
                    "axes[1].set_xlabel('Vehicle Age (Years)')\n",
                    "axes[1].set_ylabel('Selling Price (₹ Lakhs)')\n",
                    "axes[1].set_ylim(0, 30)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Scatter plot: km_driven vs Selling Price\n",
                    "plt.figure(figsize=(10, 5.5))\n",
                    "sample_df = df_eng[df_eng['km_driven'] <= 250000].sample(n=min(1200, len(df_eng)), random_state=42)\n",
                    "sns.scatterplot(x='km_driven', y=sample_df['selling_price']/100000, hue='fuel', data=sample_df, alpha=0.6, palette='Dark2')\n",
                    "plt.title('Odometer Kilometres vs Selling Price by Fuel Type', fontweight='bold')\n",
                    "plt.xlabel('Kilometres Driven')\n",
                    "plt.ylabel('Selling Price (₹ Lakhs)')\n",
                    "plt.ylim(0, 35)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Preprocessing & Partitioning\n",
                    "Constructing the scikit-learn `ColumnTransformer` with median imputation for numeric attributes and `OneHotEncoder` with `min_frequency=10` for high-cardinality brand and model names."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.model_selection import train_test_split\n",
                    "from src.preprocessing import build_preprocessor, get_feature_names_from_preprocessor, MARKET_NUMERICAL_FEATURES, MARKET_CATEGORICAL_FEATURES, TARGET_COLUMN\n",
                    "\n",
                    "num_cols = [c for c in MARKET_NUMERICAL_FEATURES if c in df_eng.columns]\n",
                    "cat_cols = [c for c in MARKET_CATEGORICAL_FEATURES if c in df_eng.columns]\n",
                    "all_feature_cols = num_cols + cat_cols\n",
                    "\n",
                    "valid_data = df_eng.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)\n",
                    "X = valid_data[all_feature_cols]\n",
                    "y = valid_data[TARGET_COLUMN].values\n",
                    "\n",
                    "# 80% Train, 20% Test; then 75% sub-train, 25% validation\n",
                    "X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42)\n",
                    "X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)\n",
                    "\n",
                    "print(f\"📊 Split Sizes: Sub-Train={len(X_train_sub):,}, Validation={len(X_val):,}, Test={len(X_test):,}\")\n",
                    "\n",
                    "preprocessor = build_preprocessor(num_features=num_cols, cat_features=cat_cols)\n",
                    "X_train_sub_trans = preprocessor.fit_transform(X_train_sub)\n",
                    "X_val_trans = preprocessor.transform(X_val)\n",
                    "feature_names = get_feature_names_from_preprocessor(preprocessor)\n",
                    "print(f\"📐 Transformed Feature Space: {len(feature_names)} columns\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. CART Depth Experimentation & Bias-Variance Analysis\n",
                    "Testing max depths $[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, \\text{None}]$ to evaluate the bias-variance tradeoff."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.depth_analysis import run_depth_experiments\n",
                    "\n",
                    "depth_df, depth_interp = run_depth_experiments(X_train_sub_trans, y_train_sub, X_val_trans, y_val, random_state=42)\n",
                    "best_d = depth_interp['best_depth']\n",
                    "print(f\"🏆 Optimal Tree Depth based on Validation RMSE: Depth = {best_d}\")\n",
                    "\n",
                    "# Plot Depth Curve\n",
                    "plt.figure(figsize=(10, 5))\n",
                    "plt.plot(depth_df['depth_label'], depth_df['train_rmse']/100000, 'o--', color='gray', label='Train RMSE')\n",
                    "plt.plot(depth_df['depth_label'], depth_df['val_rmse']/100000, 'o-', color='crimson', linewidth=2.5, label='Validation RMSE')\n",
                    "plt.title('Tree Depth vs Prediction Error (Bias-Variance Curve)', fontweight='bold')\n",
                    "plt.xlabel('Tree Max Depth')\n",
                    "plt.ylabel('RMSE (₹ Lakhs)')\n",
                    "plt.legend()\n",
                    "plt.xticks(rotation=45)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Cost-Complexity Pruning & Minimal Effective Alpha\n",
                    "Extracting the cost-complexity pruning path $$R_\\alpha(T) = R(T) + \\alpha |T|$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.pruning import run_pruning_experiments\n",
                    "\n",
                    "pruning_df, pruning_interp, _ = run_pruning_experiments(X_train_sub_trans, y_train_sub, X_val_trans, y_val, random_state=42)\n",
                    "best_alpha = pruning_interp['best_alpha']\n",
                    "print(f\"✂️ Best Pruning Alpha: {best_alpha:.2e}\")\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(10, 5))\n",
                    "ax.semilogx(pruning_df['alpha'], pruning_df['val_rmse']/100000, 'r-', linewidth=2, label='Validation RMSE')\n",
                    "ax.semilogx(pruning_df['alpha'], pruning_df['train_rmse']/100000, 'k--', label='Train RMSE')\n",
                    "ax.set_title('Cost-Complexity Pruning Penalty (α) vs Prediction Error', fontweight='bold')\n",
                    "ax.set_xlabel('Alpha Penalty (Log Scale)')\n",
                    "ax.set_ylabel('RMSE (₹ Lakhs)')\n",
                    "ax.legend()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Final Benchmark Evaluation on Untouched Test Set\n",
                    "Retraining on full 80% partition with monotonic constraints, evaluating on the untouched 20% test partition, and comparing against a Linear Regression baseline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.cart_model import create_cart_regressor, build_monotonic_cst\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from src.evaluation import build_model_comparison_table\n",
                    "\n",
                    "preprocessor_full = build_preprocessor(num_features=num_cols, cat_features=cat_cols)\n",
                    "X_train_full_trans = preprocessor_full.fit_transform(X_train_full)\n",
                    "X_test_final_trans = preprocessor_full.transform(X_test)\n",
                    "full_feature_names = get_feature_names_from_preprocessor(preprocessor_full)\n",
                    "\n",
                    "# Build monotonic constraints (km_driven, km_per_year, km_ratio, vehicle_age, owner_tier → -1)\n",
                    "mono_cst = build_monotonic_cst(full_feature_names)\n",
                    "\n",
                    "unrestricted_model = create_cart_regressor(max_depth=None, random_state=42, monotonic_cst=mono_cst)\n",
                    "unrestricted_model.fit(X_train_full_trans, y_train_full)\n",
                    "\n",
                    "depth_restricted_model = create_cart_regressor(max_depth=best_d, random_state=42, monotonic_cst=mono_cst)\n",
                    "depth_restricted_model.fit(X_train_full_trans, y_train_full)\n",
                    "\n",
                    "pruned_model = create_cart_regressor(ccp_alpha=best_alpha, random_state=42, monotonic_cst=mono_cst)\n",
                    "pruned_model.fit(X_train_full_trans, y_train_full)\n",
                    "\n",
                    "linear_model = LinearRegression()\n",
                    "linear_model.fit(X_train_full_trans, y_train_full)\n",
                    "\n",
                    "candidate_models = {\n",
                    "    \"Unrestricted CART Tree\": unrestricted_model,\n",
                    "    f\"Depth-Restricted CART (Depth={best_d})\": depth_restricted_model,\n",
                    "    \"Cost-Complexity Pruned CART\": pruned_model,\n",
                    "    \"Linear Regression (Baseline)\": linear_model,\n",
                    "}\n",
                    "\n",
                    "comparison_df = build_model_comparison_table(candidate_models, X_test_final_trans, y_test)\n",
                    "display(comparison_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 9. Mileage Sensitivity & Isotonic Adjustment Layer\n",
                    "CART regression trees are piecewise constant step functions. To ensure real-world sensitivity across the odometer mileage range, an **Isotonic Regression Adjustment Layer** is fitted on held-out validation residuals."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.prediction import fit_mileage_adjuster, apply_mileage_adjustment, compute_mileage_impact_curve, format_inr\n",
                    "\n",
                    "# Load pre-saved adjuster or fit fresh\n",
                    "adj_path = resolve_path(\"models/mileage_adjuster.joblib\")\n",
                    "if Path(adj_path).exists():\n",
                    "    mileage_adjuster = joblib.load(adj_path)\n",
                    "else:\n",
                    "    _val_preds = depth_restricted_model.predict(preprocessor_full.transform(X_val))\n",
                    "    val_km_r = X_val[\"km_ratio\"].fillna(1.0).values\n",
                    "    mileage_adjuster = fit_mileage_adjuster(val_km_r, y_val, _val_preds)\n",
                    "\n",
                    "# Representative vehicle test case: 5-year-old Maruti Swift\n",
                    "test_car = {\n",
                    "    \"brand\": \"Maruti\",\n",
                    "    \"model_name\": \"Swift\",\n",
                    "    \"vehicle_age\": 5.0,\n",
                    "    \"km_driven\": 45000.0,\n",
                    "    \"km_per_year\": 9000.0,\n",
                    "    \"km_ratio\": 9000.0 / 6250.0,\n",
                    "    \"engine_cc\": 1197.0,\n",
                    "    \"mileage_val\": 21.2,\n",
                    "    \"max_power_bhp\": 82.0,\n",
                    "    \"owner_tier\": 1,\n",
                    "    \"seats\": 5.0,\n",
                    "    \"torque_nm\": 113.0,\n",
                    "    \"high_mileage_flag\": 0,\n",
                    "    \"fuel\": \"Petrol\",\n",
                    "    \"transmission\": \"Manual\",\n",
                    "    \"seller_type\": \"Individual\",\n",
                    "    \"trim_tier\": \"Mid\",\n",
                    "    \"body_segment\": \"Hatchback\",\n",
                    "}\n",
                    "\n",
                    "impact_pts = compute_mileage_impact_curve(\n",
                    "    base_input=test_car,\n",
                    "    model=depth_restricted_model,\n",
                    "    preprocessor=preprocessor_full,\n",
                    "    feature_columns=all_feature_cols,\n",
                    "    adjuster=mileage_adjuster,\n",
                    "    typical_km_per_year=6250.0,\n",
                    "    n_points=25,\n",
                    ")\n",
                    "\n",
                    "km_x = [p['km'] for p in impact_pts]\n",
                    "p_cart = [p['cart_price']/100000 for p in impact_pts]\n",
                    "p_adj = [p['adjusted_price']/100000 for p in impact_pts]\n",
                    "\n",
                    "plt.figure(figsize=(10, 5))\n",
                    "plt.plot(km_x, p_cart, 'b--', linewidth=1.8, label='CART Base (Piecewise Constant Tree)')\n",
                    "plt.plot(km_x, p_adj, color='#ff7f0e', linewidth=2.5, label='Market-Grade Adjusted (Isotonic Layer)')\n",
                    "plt.axvline(test_car['km_driven'], color='green', linestyle=':', label=f\"Sample Car ({int(test_car['km_driven']):,} km)\")\n",
                    "plt.title(\"Odometer Mileage Sensitivity: Resale Value vs Kilometres\", fontweight='bold')\n",
                    "plt.xlabel(\"Kilometres Driven\")\n",
                    "plt.ylabel(\"Resale Price (₹ Lakhs)\")\n",
                    "plt.legend()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 10. Single Car Valuation & Decision-Path Explainability\n",
                    "Evaluating an individual vehicle and printing its clinical decision path from root to leaf."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.prediction import compute_price_range, _build_input_df\n",
                    "from src.explainability import explain_prediction_path\n",
                    "\n",
                    "car_df = _build_input_df(test_car, all_feature_cols)\n",
                    "car_vec = preprocessor_full.transform(car_df)\n",
                    "raw_pred = float(depth_restricted_model.predict(car_vec)[0])\n",
                    "adj_pred = apply_mileage_adjustment(raw_pred, test_car['km_ratio'], mileage_adjuster)['final_price']\n",
                    "price_band = compute_price_range(adj_pred)\n",
                    "\n",
                    "print(\"🚘 Vehicle Profile:\")\n",
                    "print(f\"   {test_car['brand']} {test_car['model_name']} ({int(2024-test_car['vehicle_age'])}) | {test_car['fuel']} {test_car['transmission']} | {int(test_car['km_driven']):,} km\")\n",
                    "print(\"\\n💰 Valuation Result:\")\n",
                    "print(f\"   • CART Base Prediction:      {format_inr(raw_pred)}\")\n",
                    "print(f\"   • Fair Market Value:         {format_inr(adj_pred)}\")\n",
                    "print(f\"   • Empirical 10%-90% Range:   {format_inr(price_band['low'])} – {format_inr(price_band['high'])}\")\n",
                    "\n",
                    "print(\"\\n🧭 Exact CART Decision Path Traversal:\")\n",
                    "explanation = explain_prediction_path(depth_restricted_model, full_feature_names, car_vec, test_car)\n",
                    "print(f\"Summary: {explanation['summary']}\\n\")\n",
                    "for i, step in enumerate(explanation['narrative'], 1):\n",
                    "    print(f\"  {i}. {step}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 11. (Optional) Run Streamlit Dashboard inside Colab\n",
                    "You can launch the full interactive web application inside Google Colab using `localtunnel`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# To run Streamlit in Colab, uncomment and execute:\n",
                    "# !npm install -g localtunnel\n",
                    "# !streamlit run app.py & npx localtunnel --port 8501"
                ]
            }
        ],
        "metadata": {
            "colab": {
                "name": "used_car_price_intelligence_colab.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = Path(__file__).resolve().parent.parent / "notebooks" / "used_car_price_intelligence_colab.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated Colab notebook at {out_path}")

def update_exploratory_notebook():
    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "exploratory_analysis.ipynb"
    if not nb_path.exists():
        return
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # Check if first cell already has badge
    first_cell_text = "".join(nb["cells"][0]["source"])
    if "Open In Colab" not in first_cell_text:
        badge = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dharshan007x/Used-car-price-intelligence-CART-/blob/main/notebooks/exploratory_analysis.ipynb)\n\n"
        nb["cells"][0]["source"].insert(0, badge)
    
    # Add Colab auto-setup in code cell 1 if not present
    second_cell = nb["cells"][1]
    second_cell_text = "".join(second_cell["source"])
    if "google.colab" not in second_cell_text:
        setup_code = [
            "# Setup for Google Colab environment\n",
            "import os, sys\n",
            "if 'google.colab' in sys.modules or ('get_ipython' in globals() and 'google.colab' in str(get_ipython())):\n",
            "    if not os.path.exists('Used-car-price-intelligence-CART-'):\n",
            "        !git clone https://github.com/dharshan007x/Used-car-price-intelligence-CART-.git\n",
            "    %cd Used-car-price-intelligence-CART-\n",
            "    !pip install -q -r requirements.txt\n",
            "    sys.path.insert(0, '.')\n",
            "else:\n",
            "    sys.path.append('..')\n\n"
        ]
        # Replace sys.path.append('..') if present
        new_source = []
        for line in second_cell["source"]:
            if "sys.path.append('..')" in line:
                continue
            new_source.append(line)
        second_cell["source"] = setup_code + new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Updated {nb_path} with Colab badge and setup.")

if __name__ == "__main__":
    create_colab_notebook()
    update_exploratory_notebook()
