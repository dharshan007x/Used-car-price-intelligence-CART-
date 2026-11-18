# Used Car Price Intelligence – CART Regression System
### Batch 12 | Machine Learning Academic Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dharshan007x/Used-car-price-intelligence-CART-/blob/main/notebooks/used_car_price_intelligence_colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/Tests-49%20Passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 1. Project Title
**Used Car Price Prediction Using Regression Trees (CART)**  
*An intelligent resale price estimation platform powered by Classification and Regression Trees, featuring depth constraint analysis, minimal cost-complexity pruning, and decision-path explainability.*

---

## 2. Abstract
Accurate valuation of pre-owned automobiles is a significant economic challenge due to severe non-linear depreciation patterns, multi-way interactions between physical specifications, and high brand variance. While linear models fail to capture non-linear depreciation thresholds and black-box ensembles obscure interpretability, Classification and Regression Trees (CART) offer a mathematically rigorous, fully interpretable non-parametric framework.

This project implements an end-to-end Machine Learning intelligence system utilizing CART (`DecisionTreeRegressor` with squared-error reduction). We systematically investigate the bias-variance tradeoff across varying tree depths ($1 \le \text{depth} \le 27$) and minimal cost-complexity pruning paths ($R_\alpha(T) = R(T) + \alpha |T|$). Trained on an authentic dataset of 8,128 vehicle listings from the Indian automotive market, the optimal depth-restricted CART model achieves a test $R^2$ of **0.8612** and Mean Absolute Error (MAE) of **₹97,665**, outperforming a Linear Regression baseline ($R^2 = 0.6915$). Furthermore, cost-complexity pruning achieves a **93.8% reduction in node complexity** (from 3,879 nodes to 241 nodes) while preserving an $R^2$ of **0.8592**, demonstrating extraordinary parsimony and clinical interpretability.

---

## 3. Problem Statement
> *"Develop a system to estimate used car prices using vehicle age, mileage, fuel type, engine capacity, transmission, and brand. Apply a CART regression tree and investigate how tree depth and pruning affect prediction accuracy."*

Pre-owned vehicle buyers and sellers in India frequently face asymmetric market information. Traditional manual valuation methods rely on subjective appraisal or coarse linear depreciation heuristics, neglecting complex multi-feature interactions (e.g., automatic diesel luxury SUVs depreciate at markedly different rates than manual petrol compact hatchbacks). This project develops an interpretable, data-driven regression tree system that provides accurate valuation, transparent reasoning paths, empirical reliability metrics, and market price positioning.

---

## 4. Project Objectives
1. **Accurate Resale Price Estimation**: Implement a machine learning regression system to predict vehicle market prices in Indian Rupees (₹).
2. **Implementation of CART**: Construct a DecisionTreeRegressor utilizing recursive binary splitting with mean squared error reduction.
3. **Exploratory Data Analysis**: Statistically model and visualize the relationships between vehicle attributes and resale values.
4. **Tree Depth Experimentation**: Evaluate trees across depths $[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, \text{None}]$ to characterize underfitting and overfitting.
5. **Cost-Complexity Pruning**: Extract the minimal cost-complexity pruning path ($ccp\_\alpha$) and evaluate effective penalties.
6. **Benchmark Comparison**: Statistically compare unpruned, depth-restricted, pruned, and linear baseline models on an untouched test set.
7. **Transparent Explainability**: Trace and decode the exact decision path ($Root \to Split \to Leaf$) for individual vehicle predictions.
8. **Novelty Features**: Deliver honest prediction reliability intervals, out-of-distribution detection, and asking price market positioning.
9. **Interactive Dashboard**: Deploy a full-stack, responsive Streamlit dashboard with interactive Plotly visualizations.

---

## 5. Existing System vs. Proposed System

| Dimension | Existing Heuristic / Linear Systems | Proposed CART Intelligence System |
| :--- | :--- | :--- |
| **Model Structure** | Linear / Parametric (Assumes linear depreciation) | Non-parametric Hierarchical CART Binary Tree |
| **Non-Linear Thresholds** | Cannot model step-changes (e.g., warranty expiry drop) | Captures sharp non-linear boundaries naturally |
| **Feature Interactions** | Requires manual combinatorial interaction terms | Automatically discovers multi-way feature interactions |
| **Interpretability** | Global coefficients only; obscure local effects | Complete white-box step-by-step decision path |
| **Complexity Control** | $L_1 / L_2$ regularization on global slope | Cost-Complexity Pruning ($R_\alpha(T) = R(T) + \alpha |T|$) |
| **Decision Transparency** | Black-box or coarse scorecards | Exact node trace with sample counts and MSE reduction |

---

## 6. Dataset Description
The system is built upon the **CarDekho Indian Used Car Dataset**, consisting of 8,128 authentic market observations across 31 major automotive manufacturers.

- **Primary Source**: CarDekho pre-owned automobile transactions.
- **Currency**: Indian Rupees (₹).
- **Target Variable**: `selling_price` (Continuous target in INR).
- **Initial Schema**:
  - `name`: Vehicle make, model, and variant string.
  - `year`: Vehicle manufacturing year ($1985 \le year \le 2024$).
  - `selling_price`: Transaction or listing price in INR.
  - `km_driven`: Total odometer reading in kilometers.
  - `fuel`: Fuel type (`Diesel`, `Petrol`, `CNG`, `LPG`).
  - `seller_type`: Seller classification (`Individual`, `Dealer`, `Trustmark Dealer`).
  - `transmission`: Gearbox transmission (`Manual`, `Automatic`).
  - `owner`: Prior ownership tier (`First Owner`, `Second Owner`, etc.).
  - `mileage`: Fuel economy specification (e.g., `23.4 kmpl` or `17.3 km/kg`).
  - `engine`: Engine displacement (e.g., `1248 CC`).
  - `max_power`: Maximum brake horsepower (e.g., `74 bhp`).
  - `seats`: Seating capacity.

---

## 7. Feature Descriptions & Engineering

| Raw Attribute | Cleaned / Engineered Feature | Data Type | Role | Transformation & Domain Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `name` | `brand` | Categorical | Predictor | Extracted manufacturer token (`Maruti`, `Hyundai`, `BMW`, etc.). |
| `name` | `model` | Categorical | Informational | Extracted model series (`Swift`, `City`, `Fortuner`). |
| `year` | `vehicle_age` | Numerical (Years) | Mandatory Predictor | $\text{Age} = \text{Reference Year} (2024) - \text{Manufacturing Year}$. Eliminates raw temporal drift. |
| `km_driven` | `km_driven` | Numerical (km) | Mandatory Predictor | Odometer reading validated and clipped ($500 \le km \le 800,000$). |
| `km_driven`, `year` | `km_per_year` | Numerical (km/yr) | Informational | Usage intensity metric: $km\_driven / \max(age, 0.5)$. |
| `engine` | `engine_cc` | Numerical (CC) | Mandatory Predictor | Regex parsed displacement; filtered between $600 \text{ CC}$ and $6500 \text{ CC}$. |
| `mileage` | `mileage_val` | Numerical (kmpl) | Predictor | Regex parsed fuel efficiency; imputed via median. |
| `max_power` | `max_power_bhp` | Numerical (bhp) | Predictor | Regex parsed power output; imputed via median. |
| `fuel` | `fuel` | Categorical | Mandatory Predictor | One-hot encoded into `Diesel`, `Petrol`, `CNG`, `LPG`. |
| `transmission` | `transmission` | Categorical | Mandatory Predictor | One-hot encoded into `Manual`, `Automatic`. |
| `selling_price` | `selling_price` | Numerical (₹) | Target Variable | Validated between ₹25,000 and ₹5,00,00,000. |

---

## 8. Technology Stack

- **Core Programming Language**: Python 3.11.9
- **Machine Learning & Preprocessing**: `scikit-learn` (1.4+), `scipy` (1.11+)
- **Data Manipulation**: `pandas` (2.2+), `numpy` (1.26+)
- **Interactive Visualization**: `plotly` (5.18+), `matplotlib` (3.8+), `seaborn` (0.13+)
- **Web Application & UI/UX**: `streamlit` (1.28+)
- **Model Persistence**: `joblib` (1.3+)
- **Automated Testing & QA**: `pytest` (7.4+)

---

## 9. System Architecture

```
used-car-price-prediction/
│
├── app.py                             # Main Streamlit Dashboard Application
├── requirements.txt                   # Production dependency specifications
├── README.md                          # Comprehensive academic & technical documentation
├── .gitignore                         # Version control exclusions
│
├── data/
│   ├── raw/
│   │   └── car_details_v3.csv         # Authentic CarDekho dataset (8,128 records)
│   ├── processed/
│   │   └── cleaned_car_data.csv       # Cleaned & feature-engineered dataset
│   └── README.md
│
├── models/
│   ├── best_cart_model.joblib         # Fitted primary CART model
│   ├── preprocessing_pipeline.joblib  # Fitted scikit-learn ColumnTransformer
│   ├── unrestricted_cart_model.joblib # Baseline unrestricted tree
│   ├── depth_restricted_cart_model.joblib # Optimal depth-constrained tree
│   ├── pruned_cart_model.joblib       # Cost-complexity pruned tree
│   ├── linear_baseline_model.joblib   # Linear regression benchmark
│   └── model_metadata.json            # Model parameters, splits, and metrics
│
├── notebooks/
│   └── exploratory_analysis.ipynb     # Jupyter Notebook for exploratory analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Remote/local data loading & mirror fallback
│   ├── data_validation.py             # Schema validation, bounds, audit logging
│   ├── feature_engineering.py         # Brand/model parsing, vehicle age derivation
│   ├── preprocessing.py               # Leakage-free ColumnTransformer pipeline
│   ├── cart_model.py                  # CART instantiation & metric extraction
│   ├── depth_analysis.py              # Max depth grid experiments & diagnosis
│   ├── pruning.py                     # Minimal cost-complexity pruning path
│   ├── evaluation.py                  # MAE, RMSE, R2, MAPE, and segment analytics
│   ├── prediction.py                  # Inference, empirical intervals, market positioning
│   ├── explainability.py              # Exact decision_path() extraction & narrative
│   └── visualization.py               # Publication-grade Plotly & Matplotlib charts
│
├── pages/
│   ├── 1_Dataset_Explorer.py          # Interactive dataset filtering & export
│   ├── 2_Exploratory_Analysis.py      # Bivariate distributions & correlation maps
│   ├── 3_Price_Prediction.py          # Prediction form with decision path trail
│   ├── 4_Tree_Depth_Analysis.py       # Underfitting vs overfitting analysis
│   ├── 5_Pruning_Analysis.py          # Cost-complexity pruning curves
│   ├── 6_Model_Evaluation.py          # Test diagnostics & segment performance
│   └── 7_Tree_Explorer.py             # Structural tree plot & node inspection
│
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py        # Schema checks, bound tests, string cleaning
│   ├── test_eda_and_features.py       # Brand extraction, age logic, summary metrics
│   ├── test_model.py                  # Training convergence, persistence, constraints
│   ├── test_preprocessing.py          # ColumnTransformer, unseen category handling
│   ├── test_pruning.py                # CCP path, monotonicity, alpha selection
│   └── test_prediction.py             # Predictions, intervals, market positioning
│
└── reports/
    ├── model_comparison.csv           # Benchmark comparison on test set
    ├── depth_analysis.csv             # Full depth grid experimentation results
    ├── pruning_analysis.csv           # Cost-complexity pruning trajectory
    ├── segment_performance.csv        # Performance broken down by price tier
    └── evaluation_summary.json        # Structured JSON execution report
```

---

## 10. Data Preprocessing & Leakage Prevention

The preprocessing pipeline is strictly encapsulated within a scikit-learn `ColumnTransformer`. To guarantee **zero train-test data leakage**, the preprocessor is fitted **exclusively on the training split**:

1. **Numerical Pipeline**:
   - Features: `vehicle_age`, `km_driven`, `engine_cc`, `mileage_val`, `max_power_bhp`.
   - Imputation: Median imputation computed strictly from training samples.
   - Scaling: Explicitly omitted. CART regression trees are invariant to monotonic feature scaling; preserving raw units maintains direct interpretability of split thresholds (e.g., $km\_driven \le 45000$).
2. **Categorical Pipeline**:
   - Features: `brand`, `fuel`, `transmission`.
   - Imputation: Most-frequent value strategy.
   - Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. Unseen brands or categories in testing/production are mapped to all-zero indicator vectors without throwing exceptions.
3. **Partition Strategy**:
   - Master Split: 80% Training ($N=5,538$), 20% Untouched Test ($N=1,385$).
   - Hyperparameter Partition: Training split is subdivided into Sub-Train ($75\% = 4,153$) and Validation ($25\% = 1,385$) to select optimal depth and pruning penalty $\alpha$ without touching the test set.

---

## 11. CART Regression Tree Methodology

### Mathematical Formulation
A Classification and Regression Tree (CART) partitions the $p$-dimensional feature space into $M$ distinct, non-overlapping hyper-rectangles $R_1, R_2, \dots, R_M$.

1. **Terminal Node Prediction**:
   Within any terminal leaf node $R_m$, the predicted vehicle price $\hat{c}_m$ is the empirical mean of the training target values belonging to that region:
   $$\hat{c}_m = \text{mean}(y_i \mid x_i \in R_m) = \frac{1}{N_m} \sum_{x_i \in R_m} y_i$$

2. **Recursive Binary Splitting**:
   CART considers a splitting variable $j$ and split point $s$ that divides region $R$ into two half-planes:
   $$R_1(j, s) = \{X \mid X_j \le s\} \quad \text{and} \quad R_2(j, s) = \{X \mid X_j > s\}$$

3. **Impurity Reduction (Squared Error Criterion)**:
   The splitting criterion searches for $(j^*, s^*)$ that minimizes the weighted sum of squared errors in child nodes:
   $$\min_{j, s} \left[ \sum_{x_i \in R_1(j, s)} (y_i - \hat{c}_1)^2 + \sum_{x_i \in R_2(j, s)} (y_i - \hat{c}_2)^2 \right]$$
   Equivalently, the algorithm maximizes impurity reduction $\Delta Q(j, s)$:
   $$\Delta Q(j, s) = Q(R) - \left( \frac{N_1}{N} Q(R_1) + \frac{N_2}{N} Q(R_2) \right)$$
   where $Q(R_m) = \frac{1}{N_m} \sum_{x_i \in R_m} (y_i - \hat{c}_m)^2$.

---

## 12. Tree Depth Experimentation & Bias-Variance Analysis

We trained separate CART models across depth limits $[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, \text{None}]$ on the sub-training partition and evaluated both training and validation performance:

### Actual Experimental Results
| Depth Constraint | Total Nodes | Leaf Nodes | Train RMSE (₹) | Train $R^2$ | Validation RMSE (₹) | Validation $R^2$ | Generalization Gap (₹) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Depth 1** | 3 | 2 | ₹3,90,592 | 0.3671 | ₹4,42,939 | 0.3617 | ₹52,348 |
| **Depth 2** | 7 | 4 | ₹3,26,625 | 0.5574 | ₹3,87,153 | 0.5123 | ₹60,528 |
| **Depth 3** | 15 | 8 | ₹2,60,502 | 0.7185 | ₹3,09,470 | 0.6884 | ₹48,968 |
| **Depth 4** | 31 | 16 | ₹2,20,847 | 0.7977 | ₹2,67,982 | 0.7664 | ₹47,135 |
| **Depth 5** | 61 | 31 | ₹1,86,285 | 0.8560 | ₹2,59,761 | 0.7805 | ₹73,476 |
| **Depth 6** | 113 | 57 | ₹1,64,563 | 0.8876 | ₹2,39,847 | 0.8128 | ₹75,284 |
| **Depth 8** | 329 | 165 | ₹1,35,228 | 0.9241 | ₹2,25,402 | 0.8347 | ₹90,174 |
| **Depth 10** | 767 | 384 | ₹1,19,927 | 0.9403 | ₹2,26,588 | 0.8330 | ₹1,06,662 |
| **Depth 12 (Optimal)** | **1,371** | **686** | **₹1,10,938** | **0.9489** | **₹2,23,292** | **0.8378** | **₹1,12,354** |
| **Depth 15** | 2,343 | 1,172 | ₹1,04,636 | 0.9546 | ₹2,24,317 | 0.8363 | ₹1,19,680 |
| **Depth 20** | 2,865 | 1,433 | ₹1,02,688 | 0.9563 | ₹2,31,346 | 0.8259 | ₹1,28,658 |
| **Unrestricted (Depth 24)** | 2,925 | 1,463 | ₹1,02,583 | 0.9563 | ₹2,28,351 | 0.8303 | ₹1,25,768 |

### Theoretical Interpretation
- **Underfitting Zone ($\text{Depth} \le 3$)**: Extreme bias. A tree of depth 1 partitions the entire automotive market into merely two price bands, producing a validation $R^2$ of only 0.3617.
- **Overfitting Zone ($\text{Depth} \ge 15$)**: High variance. While training RMSE drops to ₹1,02,583 ($R^2 = 0.9563$), the validation RMSE rebounds up to ₹2,31,346, producing a widening generalization gap ($\Delta \text{RMSE} = ₹1,28,658$).
- **Optimal Tradeoff ($\text{Depth} = 12$)**: Reaches the minimum validation error ($\text{RMSE} = ₹2,23,292$ and validation $R^2 = 0.8378$).

---

## 13. Minimal Cost-Complexity Pruning (CCP)

### Formulation
Cost-complexity pruning defines a penalized cost function over all subtrees $T \subseteq T_{\max}$:
$$R_\alpha(T) = R(T) + \alpha |T|$$
where:
- $R(T) = \sum_{m=1}^{|T|} N_m Q_m(T)$ is the total residual sum of squared errors of the terminal nodes.
- $|T|$ is the number of terminal leaf nodes.
- $\alpha \ge 0$ is the complexity penalty parameter.

When $\alpha = 0$, the penalty vanishes and $T_\alpha = T_{\max}$. As $\alpha \to \infty$, the penalty dominates, collapsing the tree to the single root node.

### Pruning Trajectory
Using `cost_complexity_pruning_path()`, we evaluated effective candidate alphas across several orders of magnitude:

| Effective $\alpha$ | Tree Depth | Total Nodes | Leaves | Train RMSE (₹) | Val RMSE (₹) | Val $R^2$ | Generalization Gap (₹) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.00$ (Unpruned) | 24 | 2,925 | 1,463 | ₹1,02,583 | ₹2,28,351 | 0.8303 | ₹1,25,768 |
| $2.15 \times 10^6$ | 20 | 1,409 | 705 | ₹1,05,106 | ₹2,27,322 | 0.8319 | ₹1,22,216 |
| $7.39 \times 10^6$ | 16 | 729 | 365 | ₹1,11,660 | ₹2,26,126 | 0.8336 | ₹1,14,465 |
| $1.42 \times 10^7$ | 14 | 483 | 242 | ₹1,17,241 | ₹2,24,318 | 0.8363 | ₹1,07,076 |
| **$5.77 \times 10^7$ (Optimal)** | **12** | **231** | **116** | **₹1,32,083** | **₹2,23,744** | **0.8371** | **₹91,661** |
| $1.97 \times 10^8$ | 9 | 111 | 56 | ₹1,54,146 | ₹2,25,887 | 0.8340 | ₹71,741 |
| $1.46 \times 10^9$ | 6 | 47 | 24 | ₹1,87,322 | ₹2,38,719 | 0.8146 | ₹51,397 |

### Pruning Impact
Selecting $\alpha^* = 5.77 \times 10^7$ collapses the tree from **2,925 nodes down to 231 nodes** (a **92.1% complexity reduction**) while concurrently **lowering validation RMSE** from ₹2,28,351 to ₹2,23,744. This demonstrates the power of cost-complexity pruning in removing noisy, over-specialized leaf partitions.

---

## 14. Final Test Set Evaluation & Model Benchmark

Each candidate model was retrained on the full 80% training partition ($N=5,538$) and evaluated **once on the completely untouched 20% test partition** ($N=1,385$):

### Master Benchmark Table
| Model | Depth | Nodes | Leaves | Test MAE (₹) | Test RMSE (₹) | Test $R^2$ Score | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unrestricted CART Tree** | 27 | 3,879 | 1,940 | ₹98,621 | **₹2,09,753** | **0.8627** | 21.22% |
| **Depth-Restricted CART ($\text{depth}=12$)** | 12 | 1,571 | 786 | **₹97,665** | ₹2,10,928 | 0.8612 | **20.47%** |
| **Cost-Complexity Pruned CART ($\alpha^*$)** | 11 | **241** | **121** | ₹1,03,950 | ₹2,12,419 | 0.8592 | 25.30% |
| **Linear Regression (Baseline)** | N/A | N/A | N/A | ₹1,54,618 | ₹3,14,433 | 0.6915 | 49.82% |

### Performance Partitioned by Price Tier
| Price Segment | Test Sample Count | MAE (₹) | RMSE (₹) | $R^2$ Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Budget (< ₹3L)** | 400 | ₹48,459 | ₹67,294 | 0.4468 | 24.31% |
| **Mid-Range (₹3L - ₹7L)** | 622 | ₹68,529 | ₹97,713 | 0.3812 | 14.39% |
| **Premium (₹7L - ₹15L)** | 288 | ₹1,43,059 | ₹2,07,678 | 0.3475 | 15.34% |
| **Luxury (> ₹15L)** | 75 | ₹4,30,974 | ₹6,80,676 | 0.5898 | 15.71% |

---

## 15. Novelty Features

### A. Intelligent Price Estimator
Provides a streamlined prediction interface with real-time field validation, standard Indian Lakh/Crore formatting, and instant sub-second inference.

### B. Explain My Prediction (Exact Decision Path)
Unlike black-box models, every prediction provides a verifiable audit trail using `decision_path()` and `apply()`. The system outputs:
- Node-by-node decisions from Root to Terminal Leaf.
- Split variable and numerical split threshold.
- Direction taken (`LEFT <=` or `RIGHT >`).
- Sample count and mean target value at the reached leaf node.
- A dynamic natural language narrative translating mathematical splits into intuitive vehicle explanations.

### C. Tree Visualization Explorer
An interactive visual viewer using Matplotlib and Graphviz integration. Features dynamic depth restriction (Levels 1 to 4) to ensure rapid browser rendering without crashing on large trees.

### D. Used Car Market Positioning
Evaluates buyer listings against model estimates within a configurable $\pm 10\%$ tolerance corridor:
- **Potentially Underpriced**: Listing price below fair market corridor (strong buyer opportunity or warrants inspection).
- **Within Estimated Fair Range**: Listing price closely matches learned valuation.
- **Potentially Overpriced**: Listing exceeds typical market expectations (room for negotiation).

### E. Honest Prediction Reliability & OOD Detection
Rather than fabricating synthetic confidence percentages, the system computes defensible statistical reliability indicators:
1. **Leaf Sample Support**: Quantifies the empirical observation density in the matched terminal leaf (High Support $\ge 20$, Moderate Support $7-19$, Sparse $< 7$).
2. **Empirical Error Intervals**: Constructs an 85% residual margin based on held-out calibration residuals.
3. **Out-of-Distribution Flags**: Detects unseen brands, extreme odometer readings ($> 250,000 \text{ km}$), or rare engine displacements ($> 4,500 \text{ CC}$).

---

## 16. Installation and Setup Instructions

### ⚡ Quick Launch in Google Colab (Zero Local Setup)
Run the entire end-to-end CART intelligence pipeline directly in your browser without installing anything locally:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dharshan007x/Used-car-price-intelligence-CART-/blob/main/notebooks/used_car_price_intelligence_colab.ipynb)

- **Master Colab Notebook**: [used_car_price_intelligence_colab.ipynb](https://colab.research.google.com/github/dharshan007x/Used-car-price-intelligence-CART-/blob/main/notebooks/used_car_price_intelligence_colab.ipynb)
- **EDA & Pruning Notebook**: [exploratory_analysis.ipynb](https://colab.research.google.com/github/dharshan007x/Used-car-price-intelligence-CART-/blob/main/notebooks/exploratory_analysis.ipynb)

---

### Prerequisites (For Local Setup)
- Python 3.11.x (or compatible Python 3.10+)
- Git
- Command Prompt / PowerShell (Windows) or Terminal (macOS/Linux)

### Step 1: Clone the Repository
```bash
git clone https://github.com/dharshan007x/Used-car-price-intelligence-CART-.git
cd Used-car-price-intelligence-CART-
```

### Step 2: Set Up Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 17. Execution Workflow

### Step 4: Run the Master Training & Experimentation Pipeline
```bash
python -m src.train
```
This executes data loading, schema validation, feature engineering, tree depth experiments, cost-complexity pruning, test set evaluation, report generation, and Joblib model persistence.

### Step 5: Execute Automated Test Suite
```bash
pytest tests -v
```
Verifies all 49 unit and integration tests across data validation, preprocessing, model training, pruning paths, and explainability.

### Step 6: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the full interactive interface.

---

## 18. Limitations & Future Enhancements

### Limitations
1. **Axis-Aligned Splits**: CART splits perpendicular to feature axes; oblique linear boundaries require multiple staircase splits.
2. **Step Function Extrapolation**: Decision trees cannot extrapolate continuous trends beyond the minimum or maximum observed target values.
3. **Regional Price Variation**: The dataset does not currently capture city-specific road taxes, state registration costs, or regional market demand.
4. **Piecewise-Constant km Response**: CART is inherently piecewise-constant; the mileage adjustment layer (isotonic regression) smooths the km response post-prediction. See §20 for details.

### Future Enhancements
1. **Spatial Price Modeling**: Incorporate state and metro city regional indices.
2. **Vehicle Condition Grading**: Integrate exterior/interior inspection scores and maintenance service history.
3. **Ensemble Comparison**: Add optional Gradient Boosted Trees (XGBoost / LightGBM) as secondary benchmark tabs while preserving CART as the central interpretable core.

---

## 20. km Sensitivity Fix — Root Cause & Solution

### Root Cause (Step 0 Diagnostic)
Before the fix, **80% of test cars showed zero price change** when km_driven varied from 10% to 300% of its actual value. The diagnostic found:
- `km_driven` was wired into the ColumnTransformer correctly (rank 2 of 9 numerical features).
- However, 20 of 20 random cars landed in the same decision-tree leaf regardless of km change.
- km splits appeared at **mean depth 8.1** (max_depth was 10), making them unreachable for most paths dominated by torque/power splits near the root.
- Dataset confirms genuine km–price signal: Spearman r = −0.276 within cohorts, 81% of cohorts show negative r.
- **Conclusion:** CART's greedy split selection consistently preferred torque/power/age at upper nodes, leaving km splits only in sparsely-reached deep subtrees.

### Fix (Steps 1–3)

#### Step 1 — New Mileage Features
Added two new numerical features (computed leak-free from training partition only):

| Feature | Formula | Purpose |
|---|---|---|
| `km_per_year` | `km_driven / max(vehicle_age, 0.5)` | Annual usage intensity |
| `km_ratio` | `km_driven / (vehicle_age × typical_km_per_year)` | Age-normalised usage vs fleet median |

`typical_km_per_year` = **6,250 km/yr** (median of training partition only, stored in `model_metadata.json`).

#### Step 2 — Monotonic Constraints
Applied `monotonic_cst` (sklearn ≥ 1.4) to enforce price-decreasing relationships:

| Feature | Constraint |
|---|---|
| `km_driven` | −1 (higher → lower price) |
| `km_per_year` | −1 |
| `km_ratio` | −1 |
| `vehicle_age` | −1 |
| `owner_tier` | −1 |

This guarantees the tree can never assign higher price to higher mileage cars within the same decision path.

#### Step 3 — Isotonic Mileage Adjustment Layer
Because CART is piecewise-constant (many cars still land in the same leaf), a post-prediction correction is applied:

1. Fit `IsotonicRegression(increasing=False)` on validation set: maps `km_ratio → actual/cart_pred` ratio.
2. At prediction time: `final_price = cart_base × clip(isotonic(km_ratio), 0.80, 1.15)`.
3. Adjuster saved to `models/mileage_adjuster.joblib`.

### Before vs After

| Metric | Before | After |
|---|---|---|
| Cars with zero km response (20 random) | **80%** (16/20) | Adjuster gives 4+ distinct price points across km range |
| Price at 15,000 km (Maruti Swift, 5yr) | ₹3.94L | ₹5.54L |
| Price at 150,000 km (same car) | ₹3.94L (identical) | ₹5.41L (−2.3%) |
| Price drop from 0.3× → 3× km | 0% | −2.3% |
| km-related feature importance | 4.87% (CART only) | 4.50% CART + isotonic adjustment |
| Test suite | 42 passed | **49 passed** (7 new km-sensitivity tests) |

### Acknowledged Limitation
CART is a step function — the adjustment layer smooths km response but does not create a continuous curve from the tree alone. The orange "Adjusted" line in the Mileage Impact panel shows the corrected prices; the dotted "CART Base" line shows the raw tree output. For fully continuous km response, an ensemble method (e.g. gradient boosting) would be required, but that is outside scope for this academic CART project.

---

---

