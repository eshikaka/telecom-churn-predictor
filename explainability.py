"""
=============================================================================
Explainability Module — SHAP Global & Local Interpretations
=============================================================================
Computes global feature importance summaries (pre-computed at training time)
and local per-customer SHAP explanations (computed on-demand in the UI).

Usage:
    # During training (global, saved as artifacts):
    shap_values, explainer = compute_global_shap(model, X_train, cfg)
    save_shap_artifacts(shap_values, explainer, X_train, cfg)

    # During inference (local, on-demand):
    explanation = compute_local_shap(model, single_row_df, cfg)
=============================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global SHAP (Pre-computed during training)
# ---------------------------------------------------------------------------
def compute_global_shap(
    model: Any,
    X_train: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[Any, Any]:
    """
    Compute SHAP values on a background sample of the training data.

    Returns
    -------
    (shap_values, explainer)
    """
    if not HAS_SHAP:
        raise ImportError("SHAP is not installed. Run: pip install shap")

    explain_cfg = cfg.get("explainability", {})
    n_background = explain_cfg.get("shap_background_samples", 100)

    n_background = min(n_background, len(X_train))
    background = X_train.sample(n=n_background, random_state=42)

    model_type = type(model).__name__

    if model_type in ("XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model, data=background)
    else:
        explainer = shap.LinearExplainer(model, masker=background)

    shap_values = explainer.shap_values(X_train)

    # For binary classification, some explainers return a list of two arrays
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    logger.info(
        "Global SHAP computed: %s explainer on %d samples -> %s",
        type(explainer).__name__, len(X_train),
        shap_values.shape if hasattr(shap_values, "shape") else type(shap_values),
    )
    return shap_values, explainer


def save_shap_artifacts(
    shap_values: Any,
    explainer: Any,
    X_train: pd.DataFrame,
    cfg: Dict[str, Any],
) -> None:
    """
    Save SHAP explainer, values, and a pre-rendered global summary plot.
    """
    artifacts_dir = cfg["data"]["artifacts_directory"]
    explain_cfg = cfg.get("explainability", {})
    max_display = explain_cfg.get("shap_max_display", 15)

    os.makedirs(artifacts_dir, exist_ok=True)

    # Save explainer and values
    joblib.dump(explainer, os.path.join(artifacts_dir, "shap_explainer.joblib"))
    joblib.dump(shap_values, os.path.join(artifacts_dir, "shap_values_global.joblib"))

    # Save a background sample for the explainer (needed for force plots)
    n_bg = min(100, len(X_train))
    background = X_train.sample(n=n_bg, random_state=42)
    joblib.dump(background, os.path.join(artifacts_dir, "shap_background.joblib"))

    # Pre-render global summary bar plot
    # Note: shap.summary_plot creates its own figure internally — do NOT call
    # plt.subplots() before it, that orphaned figure is never used and the
    # SHAP figure gets no size/padding adjustments.
    plt.close("all")
    shap.summary_plot(
        shap_values, X_train,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    plt.tight_layout()
    summary_path = os.path.join(artifacts_dir, "shap_global_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close("all")
    logger.info("Saved global SHAP summary plot to %s", summary_path)

    # Pre-render beeswarm plot
    plt.close("all")
    shap.summary_plot(
        shap_values, X_train,
        plot_type="dot",
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    plt.tight_layout()
    beeswarm_path = os.path.join(artifacts_dir, "shap_global_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close("all")
    logger.info("Saved global SHAP beeswarm plot to %s", beeswarm_path)

    # Compute and save mean absolute SHAP importances
    if isinstance(shap_values, np.ndarray):
        importance = pd.DataFrame({
            "feature": X_train.columns.tolist(),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(
            os.path.join(artifacts_dir, "shap_feature_importance.csv"), index=False
        )


# ---------------------------------------------------------------------------
# Local SHAP (On-demand for single customer)
# ---------------------------------------------------------------------------
def compute_local_shap(
    model: Any,
    single_row: pd.DataFrame,
    cfg: Dict[str, Any],
    explainer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute SHAP explanation for a single customer prediction.

    Returns
    -------
    dict with keys:
        - 'shap_values': array of SHAP values for each feature
        - 'base_value': expected model output
        - 'feature_names': list of feature names
        - 'feature_values': list of input feature values
        - 'explanation': shap.Explanation object (for waterfall plots)
    """
    if not HAS_SHAP:
        raise ImportError("SHAP is not installed.")

    artifacts_dir = cfg["data"]["artifacts_directory"]

    if explainer is None:
        explainer_path = os.path.join(artifacts_dir, "shap_explainer.joblib")
        if os.path.exists(explainer_path):
            explainer = joblib.load(explainer_path)
        else:
            bg_path = os.path.join(artifacts_dir, "shap_background.joblib")
            if os.path.exists(bg_path):
                background = joblib.load(bg_path)
            else:
                background = single_row
            model_type = type(model).__name__
            if model_type in ("XGBClassifier", "RandomForestClassifier"):
                explainer = shap.TreeExplainer(model, data=background)
            else:
                explainer = shap.LinearExplainer(model, masker=background)

    sv = explainer.shap_values(single_row)

    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]

    if hasattr(sv, "shape") and sv.ndim > 1:
        sv_flat = sv[0]
    else:
        sv_flat = sv

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    explanation = shap.Explanation(
        values=sv_flat,
        base_values=base_value,
        data=single_row.values[0] if hasattr(single_row, "values") else single_row,
        feature_names=list(single_row.columns) if hasattr(single_row, "columns") else None,
    )

    return {
        "shap_values": sv_flat,
        "base_value": float(base_value),
        "feature_names": list(single_row.columns),
        "feature_values": single_row.values[0].tolist(),
        "explanation": explanation,
    }


def render_waterfall_plot(explanation: Any, max_display: int = 15) -> plt.Figure:
    """Render a SHAP waterfall plot and return the figure object."""
    plt.close("all")  # clear any existing figures before SHAP creates its own
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    return fig


def render_force_plot_html(explanation_dict: Dict[str, Any]) -> str:
    """Render a SHAP force plot as an HTML string for embedding in Streamlit."""
    if not HAS_SHAP:
        return "<p>SHAP not installed.</p>"

    force = shap.force_plot(
        base_value=explanation_dict["base_value"],
        shap_values=explanation_dict["shap_values"],
        features=explanation_dict["feature_values"],
        feature_names=explanation_dict["feature_names"],
        matplotlib=False,
    )
    return shap.getjs() + force.html()
