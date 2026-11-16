"""
Visualization Module for Used Car Price Intelligence.

Generates publication-quality, interactive Plotly and Matplotlib figures
for EDA, Depth Analysis, Pruning trajectories, Evaluation diagnostics, and CART Tree structures.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Theme color palette
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ACCENT_COLOR = "#2ca02c"
GRID_COLOR = "#f0f2f6"


def plot_price_distribution(df: pd.DataFrame, target_col: str = "selling_price") -> go.Figure:
    """Generates an interactive histogram with median and quantile indicators."""
    fig = px.histogram(
        df,
        x=target_col,
        nbins=60,
        marginal="box",
        title="Distribution of Used Car Selling Prices",
        labels={target_col: "Selling Price (₹)"},
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    median_val = df[target_col].median()
    mean_val = df[target_col].mean()

    fig.add_vline(x=median_val, line_dash="dash", line_color="green", annotation_text=f"Median: ₹{median_val:,.0f}")
    fig.add_vline(x=mean_val, line_dash="dot", line_color="red", annotation_text=f"Mean: ₹{mean_val:,.0f}")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Selling Price (₹)",
        yaxis_title="Vehicle Count",
        hovermode="x unified"
    )
    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "selling_price",
    title: Optional[str] = None,
    color_col: Optional[str] = None
) -> go.Figure:
    """Creates scatter plot exploring bivariate relationships against selling price."""
    title_text = title or f"{x_col.replace('_', ' ').title()} vs Selling Price"
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline="ols",
        title=title_text,
        labels={x_col: x_col.replace("_", " ").title(), y_col: "Selling Price (₹)"},
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_brand_analysis(df: pd.DataFrame, top_n: int = 15) -> Tuple[go.Figure, go.Figure]:
    """Generates brand frequency and brand median resale price bar charts."""
    brand_counts = df["brand"].value_counts().head(top_n).reset_index()
    brand_counts.columns = ["Brand", "Listing Count"]

    fig_counts = px.bar(
        brand_counts,
        x="Brand",
        y="Listing Count",
        title=f"Top {top_n} Vehicle Brands by Listing Volume",
        color="Listing Count",
        color_continuous_scale="Blues"
    )
    fig_counts.update_layout(template="plotly_white", xaxis_tickangle=-45)

    brand_prices = df.groupby("brand")["selling_price"].agg(["median", "count"])
    brand_prices = brand_prices[brand_prices["count"] >= 10].sort_values("median", ascending=False).head(top_n).reset_index()
    brand_prices.columns = ["Brand", "Median Price", "Count"]

    fig_prices = px.bar(
        brand_prices,
        x="Brand",
        y="Median Price",
        title=f"Top {top_n} Brands by Median Resale Value (Min. 10 listings)",
        color="Median Price",
        color_continuous_scale="Viridis"
    )
    fig_prices.update_layout(template="plotly_white", xaxis_tickangle=-45, yaxis_title="Median Selling Price (₹)")

    return fig_counts, fig_prices


def plot_categorical_comparison(df: pd.DataFrame, cat_col: str) -> go.Figure:
    """Generates box plots and median price bars across categories (Fuel, Transmission, etc.)."""
    summary = df.groupby(cat_col)["selling_price"].median().reset_index()
    summary = summary.sort_values("selling_price", ascending=False)

    fig = px.box(
        df,
        x=cat_col,
        y="selling_price",
        color=cat_col,
        title=f"Selling Price Distribution by {cat_col.replace('_', ' ').title()}",
        labels={"selling_price": "Selling Price (₹)", cat_col: cat_col.replace("_", " ").title()},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plots correlation matrix of numerical vehicle attributes."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix (Numerical Attributes)",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_depth_experiment_curves(depth_results_df: pd.DataFrame) -> go.Figure:
    """
    Plots training vs validation error (RMSE) and R² score across max_depth values.
    Demonstrates underfitting at low depths and overfitting at high/unrestricted depths.
    """
    df = depth_results_df.copy()
    df["display_depth"] = df["depth_param"].apply(lambda d: "None" if d == -1 else str(d))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Max Tree Depth vs RMSE (Prediction Error)", "Max Tree Depth vs R² Score (Goodness of Fit)")
    )

    # Subplot 1: RMSE
    fig.add_trace(
        go.Scatter(
            x=df["display_depth"], y=df["train_rmse"],
            mode="lines+markers", name="Train RMSE", line=dict(color="blue", dash="dash")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["display_depth"], y=df["val_rmse"],
            mode="lines+markers", name="Validation RMSE", line=dict(color="red", width=3)
        ),
        row=1, col=1
    )

    # Subplot 2: R2
    fig.add_trace(
        go.Scatter(
            x=df["display_depth"], y=df["train_r2"],
            mode="lines+markers", name="Train R²", line=dict(color="blue", dash="dash"),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df["display_depth"], y=df["val_r2"],
            mode="lines+markers", name="Validation R²", line=dict(color="green", width=3)
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Max Tree Depth Parameter", row=1, col=1)
    fig.update_xaxes(title_text="Max Tree Depth Parameter", row=1, col=2)
    fig.update_yaxes(title_text="RMSE (₹)", row=1, col=1)
    fig.update_yaxes(title_text="R² Score", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        title_text="Tree Depth Analysis: Underfitting, Overfitting & Generalization",
        hovermode="x unified"
    )
    return fig


def plot_pruning_curves(pruning_df: pd.DataFrame) -> go.Figure:
    """
    Visualizes cost-complexity pruning curves:
    - Alpha vs Total Nodes & Leaf Count
    - Alpha vs Train and Validation RMSE / R²
    """
    df = pruning_df.copy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Alpha vs Tree Complexity (Leaves & Nodes)", "Alpha vs Validation Error & Fit")
    )

    # Complexity
    fig.add_trace(
        go.Scatter(x=df["alpha"], y=df["total_nodes"], mode="lines+markers", name="Total Nodes", line=dict(color="purple")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["alpha"], y=df["leaf_nodes"], mode="lines+markers", name="Leaf Nodes", line=dict(color="orange")),
        row=1, col=1
    )

    # Performance
    fig.add_trace(
        go.Scatter(x=df["alpha"], y=df["val_rmse"], mode="lines+markers", name="Val RMSE (₹)", line=dict(color="red")),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df["alpha"], y=df["train_rmse"], mode="lines+markers", name="Train RMSE (₹)", line=dict(color="blue", dash="dash")),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Complexity Parameter (alpha)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Complexity Parameter (alpha)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="RMSE (₹)", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        title_text="Cost-Complexity Pruning Trajectory: R_alpha(T) = R(T) + alpha * |T|",
        hovermode="x unified"
    )
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "CART Model") -> go.Figure:
    """Plots actual vs predicted prices with 45-degree reference line."""
    df_eval = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    max_val = max(float(y_true.max()), float(y_pred.max()))
    
    fig = px.scatter(
        df_eval,
        x="Actual",
        y="Predicted",
        title=f"Actual vs Predicted Resale Prices ({model_name})",
        labels={"Actual": "Actual Selling Price (₹)", "Predicted": "Predicted Selling Price (₹)"},
        opacity=0.6,
        color_discrete_sequence=[PRIMARY_COLOR]
    )

    # Add 45-degree ideal prediction line
    fig.add_shape(
        type="line", line=dict(dash="dash", color="red", width=2),
        x0=0, y0=0, x1=max_val, y1=max_val
    )

    fig.update_layout(template="plotly_white")
    return fig


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plots residual distribution and residuals vs predicted values."""
    residuals = y_true - y_pred
    df_res = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residual Distribution (Errors)", "Residuals vs Predicted Prices")
    )

    fig.add_trace(
        go.Histogram(x=residuals, nbins=50, name="Residuals", marker_color=PRIMARY_COLOR),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df_res["Predicted"], y=df_res["Residual"], mode="markers", opacity=0.5, name="Residuals", marker_color=SECONDARY_COLOR),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    fig.update_xaxes(title_text="Prediction Residual (₹)", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Price (₹)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Residual (Actual - Pred) (₹)", row=1, col=2)

    fig.update_layout(template="plotly_white", title_text="Residual Diagnostics & Error Analysis")
    return fig


def render_decision_tree_matplotlib(
    tree_model: DecisionTreeRegressor,
    feature_names: List[str],
    max_depth_to_display: int = 3,
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Renders the CART decision tree with clean styling, feature names, and threshold nodes.
    Restricts displayed depth to avoid crashing or illegible text on large trees.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    plot_tree(
        tree_model,
        max_depth=max_depth_to_display,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        precision=2,
        fontsize=9,
        ax=ax
    )
    ax.set_title(
        f"CART Regression Tree Structure (Showing Top {max_depth_to_display} Levels)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    plt.tight_layout()
    return fig
