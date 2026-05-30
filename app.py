"""
=============================================================================
Streamlit Dashboard — Churn Analytics & Single-Customer Risk Predictor
=============================================================================
Two-view dashboard:
  View 1 (Macro):  Executive churn insights, KPI cards, interactive charts.
  View 2 (Micro):  Single-customer churn probability with SHAP explanation
                   and automated retention strategy.

Launch:
    streamlit run app.py
=============================================================================
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from data_pipeline import load_config, run_pipeline
from feature_engineering import (
    compute_arpu,
    compute_autopay_indicator,
    compute_service_density,
    engineer_features,
)
from explainability import compute_local_shap, render_waterfall_plot

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Caching
# ---------------------------------------------------------------------------
CONFIG_PATH = "config.yaml"


@st.cache_data
def get_config() -> Dict[str, Any]:
    return load_config(CONFIG_PATH)


@st.cache_resource
def load_model(artifacts_dir: str):
    return joblib.load(os.path.join(artifacts_dir, "best_model.joblib"))


@st.cache_resource
def load_preprocessor(artifacts_dir: str):
    return joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))


@st.cache_data
def load_training_results(artifacts_dir: str) -> List[Dict]:
    path = os.path.join(artifacts_dir, "training_results.json")
    if os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh)
    return []


@st.cache_data
def load_raw_data(_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load and join the raw dataset for macro analytics."""
    features, target, cust_ids = run_pipeline(_cfg)
    df = features.copy()
    df.insert(0, "Customer_ID", cust_ids.values)
    if target is not None:
        df["Churn"] = target.values
    return df


@st.cache_data
def load_feature_importance(artifacts_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(artifacts_dir, "shap_feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_optimal_threshold(artifacts_dir: str) -> float:
    """Load the F1-optimised decision threshold saved by train.py."""
    path = os.path.join(artifacts_dir, "optimal_threshold.joblib")
    if os.path.exists(path):
        return float(joblib.load(path))
    return 0.5


def check_artifacts(artifacts_dir: str) -> Dict[str, bool]:
    """Return presence status for every artifact the inference path requires."""
    required = [
        "best_model.joblib",
        "preprocessor.joblib",
        "feature_names.joblib",
        "clip_bounds.joblib",
        "rare_mappings.joblib",
        "freq_maps.joblib",
        "knn_imputer.joblib",
        "column_medians.joblib",
    ]
    return {f: os.path.exists(os.path.join(artifacts_dir, f)) for f in required}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def display_metric_card(label: str, value: str, delta: Optional[str] = None):
    """Render a styled metric."""
    st.metric(label=label, value=value, delta=delta)


def generate_retention_strategy(
    cfg: Dict[str, Any],
    customer_data: Dict[str, Any],
    churn_probability: float,
    top_features: List[str],
) -> str:
    """
    Generate a conditional retention strategy based on the customer's profile
    and the top SHAP-driving features.
    """
    strategies = cfg.get("dashboard", {}).get("retention_strategies", [])
    default = cfg.get("dashboard", {}).get("default_strategy", "Schedule a proactive check-in.")

    triggered: List[str] = []

    for rule in strategies:
        cond = rule.get("condition", {})
        threshold = rule.get("risk_threshold", 0.5)
        feature_name = cond.get("feature", "")

        if churn_probability < threshold:
            continue

        # Check feature value match
        if "value" in cond:
            val = customer_data.get(feature_name, "")
            if isinstance(val, str) and val.lower() == str(cond["value"]).lower():
                triggered.append(rule["recommendation"])
            elif val == cond["value"]:
                triggered.append(rule["recommendation"])

        if "value_lt" in cond:
            val = customer_data.get(feature_name, float("inf"))
            try:
                if float(val) < cond["value_lt"]:
                    triggered.append(rule["recommendation"])
            except (TypeError, ValueError):
                pass

    if not triggered:
        return default.strip()

    return "\n\n".join(f"**Strategy {i+1}:** {s.strip()}" for i, s in enumerate(triggered))


# ---------------------------------------------------------------------------
# VIEW 1: Macro Executive Insights
# ---------------------------------------------------------------------------
def render_macro_view(cfg: Dict[str, Any]):
    st.header("Executive Churn Insights")

    df = load_raw_data(cfg)
    artifacts_dir = cfg["data"]["artifacts_directory"]

    if "Churn" not in df.columns:
        st.error("Target column 'Churn' not found in dataset.")
        return

    # --- KPI Cards ---
    total_customers = len(df)
    churned = df["Churn"].sum()
    churn_rate = churned / total_customers if total_customers > 0 else 0
    retained = total_customers - churned

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_card("Total Customers", f"{total_customers:,}")
    with col2:
        display_metric_card("Churned", f"{int(churned):,}")
    with col3:
        display_metric_card("Retained", f"{int(retained):,}")
    with col4:
        display_metric_card("Churn Rate", f"{churn_rate:.1%}")

    st.divider()

    # --- Interactive Charts ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Churn Rate by Contract Type")
        if "Contract" in df.columns:
            contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(contract_churn.index.astype(str), contract_churn.values, color=["#e74c3c", "#f39c12", "#2ecc71"])
            ax.set_ylabel("Churn Rate")
            ax.set_ylim(0, 1)
            for bar, val in zip(bars, contract_churn.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.1%}", ha="center", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Contract column not available.")

    with chart_col2:
        st.subheader("Churn Rate by Internet Type")
        if "Internet Type" in df.columns:
            inet_churn = df.groupby("Internet Type")["Churn"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(inet_churn.index.astype(str), inet_churn.values, color="#3498db")
            ax.set_xlabel("Churn Rate")
            ax.set_xlim(0, 1)
            for bar, val in zip(bars, inet_churn.values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{val:.1%}", va="center", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Internet Type column not available.")

    st.divider()

    # --- Churn by Tenure Segments ---
    st.subheader("Churn Rate by Tenure Segment")
    if "Tenure in Months" in df.columns:
        bins = [0, 6, 12, 24, 48, 72, 200]
        labels_t = ["0-6m", "6-12m", "1-2y", "2-4y", "4-6y", "6y+"]
        # Use a local copy so the cached df is never mutated in-place
        _tenure = df[["Tenure in Months", "Churn"]].copy()
        _tenure["Tenure_Segment"] = pd.cut(_tenure["Tenure in Months"], bins=bins, labels=labels_t, right=False)
        tenure_churn = _tenure.groupby("Tenure_Segment", observed=False)["Churn"].agg(["mean", "count"])
        fig, ax1 = plt.subplots(figsize=(10, 4))
        color1 = "#e74c3c"
        color2 = "#95a5a6"
        ax1.bar(tenure_churn.index.astype(str), tenure_churn["count"], color=color2, alpha=0.6, label="Customer Count")
        ax1.set_ylabel("Customer Count", color=color2)
        ax2 = ax1.twinx()
        ax2.plot(tenure_churn.index.astype(str), tenure_churn["mean"], "o-", color=color1, linewidth=2, label="Churn Rate")
        ax2.set_ylabel("Churn Rate", color=color1)
        ax2.set_ylim(0, 1)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # --- Global SHAP Summary ---
    st.subheader("Global Feature Importance (SHAP)")
    shap_img_path = os.path.join(artifacts_dir, "shap_global_summary.png")
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, use_container_width=True)
    else:
        importance_df = load_feature_importance(artifacts_dir)
        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_n = importance_df.head(15)
            ax.barh(top_n["feature"], top_n["mean_abs_shap"], color="#8e44ad")
            ax.set_xlabel("Mean |SHAP Value|")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Run train.py first to generate SHAP artifacts.")

    # --- Model Performance ---
    st.subheader("Model Comparison")
    results = load_training_results(artifacts_dir)
    if results:
        perf_data = []
        for r in results:
            perf_data.append({
                "Model": r.get("model_name", "Unknown"),
                "Accuracy": r.get("accuracy", 0),
                "Precision": r.get("precision", 0),
                "Recall": r.get("recall", 0),
                "F1-Score": r.get("f1_score", 0),
                "ROC-AUC": r.get("roc_auc", 0),
            })
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    # --- Batch Export ---
    st.divider()
    st.subheader("Batch Risk Predictions Export")
    if st.button("Generate Batch Predictions CSV"):
        with st.spinner("Computing predictions for all customers..."):
            try:
                model = load_model(artifacts_dir)
                preprocessor = load_preprocessor(artifacts_dir)
                feature_names = joblib.load(os.path.join(artifacts_dir, "feature_names.joblib"))

                # Re-run feature engineering in inference-like mode on full dataset
                raw_df = df.drop(columns=["Churn", "Customer_ID"], errors="ignore")
                X_all, _, _ = engineer_features(cfg, raw_df, fit=False)
                probas = model.predict_proba(X_all)[:, 1]

                export_df = pd.DataFrame({
                    "Customer_ID": df["Customer_ID"].values,
                    "Churn_Probability": probas.round(4),
                    "Risk_Level": pd.cut(
                        probas,
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=["Low", "Medium", "High"],
                    ),
                })
                if "Churn" in df.columns:
                    export_df["Actual_Churn"] = df["Churn"].values

                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="churn_predictions_batch.csv",
                    mime="text/csv",
                )
                st.success(f"Generated predictions for {len(export_df):,} customers.")
            except Exception as e:
                st.error(f"Error generating batch predictions: {e}")


# ---------------------------------------------------------------------------
# Dynamic Form Builder — reads entirely from config.yaml
# ---------------------------------------------------------------------------
def render_dynamic_form(cfg: Dict[str, Any]) -> tuple:
    """
    Build and render the single-customer predictor form entirely from the
    ``dashboard.form_fields`` section of config.yaml.

    Supports four field types:
      select  — st.selectbox  with an explicit options list
      binary  — st.selectbox  with Yes / No  (converted to 1/0 later)
      number  — st.number_input  (min, max, step, default required)
      slider  — st.slider  (min, max, default; integer only)

    An optional ``label`` key overrides the UI display name without changing
    the column name passed to the pipeline.

    Returns
    -------
    (customer_raw : dict, submitted : bool)
      customer_raw  — {column_name: raw_value} for every form field
      submitted     — True when the user clicked the predict button
    """
    dash_cfg   = cfg.get("dashboard", {})
    form_cfg   = dash_cfg.get("form_fields", {})
    sections   = form_cfg.get("sections", [])
    n_layout   = min(form_cfg.get("layout_columns", 3), max(len(sections), 1))

    if not sections:
        st.warning(
            "No `dashboard.form_fields.sections` found in config.yaml. "
            "Add field definitions to enable the predictor."
        )
        return {}, False

    collected: Dict[str, Any] = {}

    with st.form("customer_form"):
        cols = st.columns(n_layout)

        for sec_idx, section in enumerate(sections):
            col_slot = cols[sec_idx % n_layout]
            with col_slot:
                st.markdown(f"**{section.get('title', f'Section {sec_idx + 1}')}**")

                for field in section.get("fields", []):
                    name    = field["name"]
                    label   = field.get("label", name)   # display label (optional override)
                    ftype   = field.get("type", "number")
                    key_id  = f"field__{name.replace(' ', '_')}"

                    # ── select ─────────────────────────────────────────────
                    if ftype == "select":
                        options = field.get("options", [])
                        default = field.get("default", options[0] if options else None)
                        idx = options.index(default) if default in options else 0
                        collected[name] = st.selectbox(label, options, index=idx, key=key_id)

                    # ── binary (Yes / No) ───────────────────────────────────
                    elif ftype == "binary":
                        default = field.get("default", "No")
                        idx = 0 if str(default).lower() in ("yes", "1", "true") else 1
                        collected[name] = st.selectbox(label, ["Yes", "No"], index=idx, key=key_id)

                    # ── slider ──────────────────────────────────────────────
                    elif ftype == "slider":
                        collected[name] = st.slider(
                            label,
                            min_value=int(field.get("min", 1)),
                            max_value=int(field.get("max", 10)),
                            value=int(field.get("default", 5)),
                            key=key_id,
                        )

                    # ── number (default) ────────────────────────────────────
                    else:
                        min_v     = field.get("min", 0)
                        max_v     = field.get("max", 9999)
                        step_v    = field.get("step", 1)
                        default_v = field.get("default", min_v)
                        collected[name] = st.number_input(
                            label,
                            min_value=min_v,
                            max_value=max_v,
                            value=default_v,
                            step=step_v,
                            key=key_id,
                        )

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    return collected, submitted


def _apply_computed_fields(
    customer_raw: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add derived fields (e.g. Total Revenue = sum of billing fields) to the
    customer dict after the form is submitted.

    Supported operations: "sum"
    Configured under dashboard.computed_fields in config.yaml.
    """
    for comp in cfg.get("dashboard", {}).get("computed_fields", []):
        op      = comp.get("operation", "sum")
        name    = comp.get("name", "")
        sources = comp.get("sources", [])
        if op == "sum" and name:
            customer_raw[name] = sum(float(customer_raw.get(s, 0)) for s in sources)
    return customer_raw


def _standardise_inference_input(
    input_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply the same text standardisation and binary/gender encoding that
    data_pipeline.run_pipeline() applies during training, so that
    single-row inference inputs match the trained preprocessor's expectations.
    """
    preproc_cfg = cfg.get("preprocessing", {})

    # 1. Boolean text → 0/1  (Yes/No, True/False)
    bool_map_lower = {str(k).lower().strip(): v
                      for k, v in preproc_cfg.get("boolean_map", {}).items()}
    for col in input_df.select_dtypes(include=["object", "string"]).columns:
        vals = set(input_df[col].dropna().astype(str).str.lower().str.strip().unique())
        if vals and vals <= set(bool_map_lower.keys()):
            input_df[col] = (
                input_df[col].astype(str).str.lower().str.strip()
                .map(bool_map_lower).fillna(0).astype(np.int8)
            )
        else:
            # Lowercase non-binary text to match training-time standardisation
            input_df[col] = input_df[col].astype(str).str.lower().str.strip()

    # 2. Gender encoding (configurable map, e.g. Male→1 / Female→0)
    gender_map_cfg = preproc_cfg.get("gender_map", {})
    if gender_map_cfg:
        g_map = {str(k).lower().strip(): v for k, v in gender_map_cfg.items()}
        str_cols = input_df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            vals = set(input_df[col].dropna().astype(str).unique())
            if vals and vals <= set(g_map.keys()):
                input_df[col] = (
                    input_df[col].astype(str).str.lower().str.strip()
                    .map(g_map).fillna(0).astype(np.int8)
                )

    return input_df


# ---------------------------------------------------------------------------
# VIEW 2: Micro Customer Predictor
# ---------------------------------------------------------------------------
def render_micro_view(cfg: Dict[str, Any]):
    st.header("Single-Customer Churn Risk Predictor")

    artifacts_dir = cfg["data"]["artifacts_directory"]
    model_path    = os.path.join(artifacts_dir, "best_model.joblib")

    if not os.path.exists(model_path):
        st.error("No trained model found. Please run `python train.py` first.")
        return

    model = load_model(artifacts_dir)

    # --- Build form entirely from config.yaml ---
    st.subheader("Customer Profile Input")

    if st.button("Reset to Defaults"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    customer_raw, submitted = render_dynamic_form(cfg)

    if submitted:
        with st.spinner("Computing churn risk..."):
            # Add any fields computed from other fields (e.g. Total Revenue)
            customer_raw = _apply_computed_fields(customer_raw, cfg)

            input_df = pd.DataFrame([customer_raw])

            try:
                # Standardise input to match training-time encodings
                input_df = _standardise_inference_input(input_df, cfg)

                # Run through feature engineering (inference mode)
                X_processed, _, _ = engineer_features(cfg, input_df, fit=False)

                # Predict using F1-optimised threshold (falls back to 0.5 if not available)
                proba = model.predict_proba(X_processed)[:, 1][0]
                threshold = load_optimal_threshold(artifacts_dir)
                prediction = int(proba >= threshold)

                # --- Display Results ---
                st.divider()
                st.subheader("Prediction Results")

                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    risk_color = "🔴" if proba >= 0.6 else ("🟡" if proba >= 0.3 else "🟢")
                    risk_label = "HIGH" if proba >= 0.6 else ("MEDIUM" if proba >= 0.3 else "LOW")
                    st.metric("Churn Probability", f"{proba:.1%}")
                with res_col2:
                    st.metric("Risk Level", f"{risk_color} {risk_label}")
                with res_col3:
                    st.metric("Prediction", "Will Churn" if prediction else "Will Stay")

                # --- SHAP Explanation ---
                st.divider()
                st.subheader("Why This Prediction? (SHAP Explanation)")

                if HAS_SHAP:
                    try:
                        explanation_dict = compute_local_shap(model, X_processed, cfg)

                        fig = render_waterfall_plot(
                            explanation_dict["explanation"],
                            max_display=cfg.get("explainability", {}).get("shap_max_display", 15),
                        )
                        st.pyplot(fig)
                        plt.close("all")

                        # Top contributing features
                        sv = explanation_dict["shap_values"]
                        fn = explanation_dict["feature_names"]
                        importance_order = np.argsort(np.abs(sv))[::-1]
                        top_features = [fn[i] for i in importance_order[:5]]

                        st.markdown("**Top Contributing Factors:**")
                        for i, idx in enumerate(importance_order[:5]):
                            direction = "increases" if sv[idx] > 0 else "decreases"
                            st.markdown(
                                f"{i+1}. **{fn[idx]}** (SHAP: {sv[idx]:+.4f}) — "
                                f"{direction} churn risk"
                            )

                    except Exception as e:
                        st.warning(f"SHAP explanation unavailable: {e}")
                        top_features = []
                else:
                    st.info("Install SHAP (`pip install shap`) for model explanations.")
                    top_features = []

                # --- Retention Strategy ---
                st.divider()
                st.subheader("Automated Retention Strategy")

                strategy = generate_retention_strategy(
                    cfg, customer_raw, proba, top_features
                )
                if proba >= 0.6:
                    st.error(strategy)
                elif proba >= 0.3:
                    st.warning(strategy)
                else:
                    st.success(strategy)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.exception("Prediction error")


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def main():
    cfg = get_config()
    dash_cfg = cfg.get("dashboard", {})

    st.set_page_config(
        page_title=dash_cfg.get("title", "Churn Analytics"),
        page_icon=dash_cfg.get("page_icon", "📊"),
        layout=dash_cfg.get("layout", "wide"),
    )

    st.title(dash_cfg.get("title", "Customer Churn Analytics & Prediction"))

    # Sidebar navigation
    view = st.sidebar.radio(
        "Navigation",
        ["Macro: Executive Insights", "Micro: Customer Predictor"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown("**System Info**")

    artifacts_dir = cfg["data"]["artifacts_directory"]
    model_exists = os.path.exists(os.path.join(artifacts_dir, "best_model.joblib"))
    st.sidebar.markdown(f"Model trained: {'Yes' if model_exists else 'No'}")

    if model_exists:
        results = load_training_results(artifacts_dir)
        if results:
            best = max(results, key=lambda r: r.get("roc_auc", 0))
            st.sidebar.markdown(f"Best model: **{best.get('model_name', 'N/A')}**")
            st.sidebar.markdown(f"ROC-AUC: **{best.get('roc_auc', 0):.4f}**")
        opt_thresh = load_optimal_threshold(artifacts_dir)
        st.sidebar.markdown(
            f"Churn threshold: **{opt_thresh:.2f}** "
            f"({'F1-optimised' if opt_thresh != 0.5 else 'default'})"
        )
        artifact_status = check_artifacts(artifacts_dir)
        missing = [f for f, ok in artifact_status.items() if not ok]
        if missing:
            st.sidebar.warning(f"Missing artifacts: {', '.join(missing)}")
        else:
            st.sidebar.markdown("Artifacts: all present")

    if view == "Macro: Executive Insights":
        render_macro_view(cfg)
    else:
        render_micro_view(cfg)


if __name__ == "__main__":
    main()
