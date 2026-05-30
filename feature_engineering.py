"""
=============================================================================
Feature Engineering — Behavioural Metrics, Transformations & Encoding
=============================================================================
Computes derived features (Service Density, ARPU, Autopay Indicator),
handles outliers via Winsorization, groups rare categories, and prepares
the final feature matrix for modelling.

All transformers are fit on training data only and serialised to artifacts/
so that inference data flows through the identical pipeline without leakage.

Usage:
    from feature_engineering import engineer_features
    X_train, X_test = engineer_features(cfg, X_train_raw, X_test_raw, fit=True)
=============================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Behavioural Feature Construction
# ---------------------------------------------------------------------------
def compute_service_density(
    df: pd.DataFrame, service_cols: List[str]
) -> pd.DataFrame:
    """
    Count of active services per customer.
    A service is 'active' if the column value is 1 (binary) or a truthy
    string like 'yes'. Columns that are missing are silently skipped.
    """
    available = [c for c in service_cols if c in df.columns]
    if not available:
        logger.warning("No service columns found for Service Density calculation.")
        df["Service_Density"] = 0
        return df

    numeric_subset = df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["Service_Density"] = (numeric_subset > 0).sum(axis=1).astype(np.int8)
    logger.info("Computed Service_Density from %d service columns.", len(available))
    return df


def compute_arpu(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    zero_tenure_default: float = 0.0,
) -> pd.DataFrame:
    """
    Average Monthly Revenue Per User = Total Charges / Tenure.
    Zero-tenure guard: explicitly returns zero_tenure_default instead of
    allowing a ZeroDivisionError.
    """
    if numerator_col not in df.columns or denominator_col not in df.columns:
        logger.warning(
            "ARPU columns missing (%s / %s). Setting ARPU = 0.",
            numerator_col, denominator_col,
        )
        df["ARPU"] = 0.0
        return df

    num = pd.to_numeric(df[numerator_col], errors="coerce").fillna(0.0)
    den = pd.to_numeric(df[denominator_col], errors="coerce").fillna(0.0)

    df["ARPU"] = np.where(den == 0, zero_tenure_default, num / den)
    n_zero = (den == 0).sum()
    if n_zero > 0:
        logger.info(
            "ARPU: %d rows with zero tenure set to default %.1f.",
            n_zero, zero_tenure_default,
        )
    logger.info("Computed ARPU (%s / %s).", numerator_col, denominator_col)
    return df


def compute_autopay_indicator(
    df: pd.DataFrame,
    payment_col: str,
    autopay_keywords: List[str],
) -> pd.DataFrame:
    """
    Binary flag: 1 if the payment method contains any autopay keyword.
    """
    if payment_col not in df.columns:
        logger.warning("Payment column '%s' not found. Autopay_Indicator = 0.", payment_col)
        df["Autopay_Indicator"] = 0
        return df

    col_lower = df[payment_col].astype(str).str.lower()
    pattern = "|".join(k.lower() for k in autopay_keywords)
    df["Autopay_Indicator"] = col_lower.str.contains(pattern, na=False).astype(np.int8)
    logger.info("Computed Autopay_Indicator from '%s'.", payment_col)
    return df


# ---------------------------------------------------------------------------
# 2. Outlier Treatment — IQR-Based Winsorization
# ---------------------------------------------------------------------------
def winsorize_columns(
    df: pd.DataFrame,
    columns: List[str],
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
    fit: bool = True,
    clip_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Cap values at the given percentiles (Winsorization).

    Parameters
    ----------
    fit : bool
        If True, compute percentile bounds from this data. If False, use
        pre-computed clip_bounds (inference mode).
    clip_bounds : dict or None
        Pre-computed {col: (lower, upper)} from training data.

    Returns
    -------
    (df, clip_bounds_dict)
    """
    if clip_bounds is None:
        clip_bounds = {}

    available = [c for c in columns if c in df.columns]
    total_capped = 0

    for col in available:
        if fit:
            p_low = df[col].quantile(lower_pct)
            p_high = df[col].quantile(upper_pct)
            clip_bounds[col] = (p_low, p_high)
        elif col not in clip_bounds:
            continue

        low, high = clip_bounds[col]
        n_capped = ((df[col] < low) | (df[col] > high)).sum()
        df[col] = df[col].clip(lower=low, upper=high)
        total_capped += n_capped

    logger.info("Winsorized %d columns: %d values capped.", len(available), total_capped)
    return df, clip_bounds


# ---------------------------------------------------------------------------
# 3. Rare-Label Grouping
# ---------------------------------------------------------------------------
def group_rare_labels(
    df: pd.DataFrame,
    threshold: float = 0.01,
    fit: bool = True,
    rare_mappings: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    For each categorical column, labels appearing in less than `threshold`
    fraction of rows are collapsed into 'other'.

    Parameters
    ----------
    fit : bool
        If True, compute rare labels from this data.
    rare_mappings : dict or None
        Pre-computed {col: [rare_labels]} from training data.

    Returns
    -------
    (df, rare_mappings_dict)
    """
    if rare_mappings is None:
        rare_mappings = {}

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in cat_cols:
        if fit:
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < threshold].index.tolist()
            if rare:
                rare_mappings[col] = rare
        else:
            rare = rare_mappings.get(col, [])

        if rare:
            df.loc[df[col].isin(rare), col] = "other"

    logger.info("Rare-label grouping applied to %d columns.", len(rare_mappings))
    return df, rare_mappings


# ---------------------------------------------------------------------------
# 4. Frequency Encoding for High-Cardinality Columns
# ---------------------------------------------------------------------------
def frequency_encode(
    df: pd.DataFrame,
    columns: List[str],
    fit: bool = True,
    freq_maps: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Replace high-cardinality string columns with their frequency ratio.

    Returns
    -------
    (df, freq_maps_dict)
    """
    if freq_maps is None:
        freq_maps = {}

    for col in columns:
        if col not in df.columns:
            continue

        if fit:
            fmap = df[col].value_counts(normalize=True).to_dict()
            freq_maps[col] = fmap
        else:
            fmap = freq_maps.get(col, {})

        new_col = f"{col.replace(' ', '_')}_Frequency"
        df[new_col] = df[col].map(fmap).fillna(0.0).astype(np.float32)
        df = df.drop(columns=[col])
        logger.info("Frequency-encoded '%s' -> '%s'.", col, new_col)

    return df, freq_maps


# ---------------------------------------------------------------------------
# 5. Sklearn ColumnTransformer Pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """
    Build a ColumnTransformer that scales numerics and one-hot-encodes
    categoricals. Returns (transformer, num_cols, cat_cols, pass_cols).

    Columns not listed in config are silently dropped to avoid schema drift.
    """
    train_cfg = cfg["training"]
    num_features = [c for c in train_cfg.get("numeric_features", []) if c in df.columns]
    cat_features = [c for c in train_cfg.get("categorical_features", []) if c in df.columns]
    pass_features = [c for c in train_cfg.get("passthrough_features", []) if c in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                cat_features,
            ),
            ("pass", "passthrough", pass_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "ColumnTransformer built: %d numeric, %d categorical, %d passthrough.",
        len(num_features), len(cat_features), len(pass_features),
    )
    return preprocessor, num_features, cat_features, pass_features


# ---------------------------------------------------------------------------
# 6. Master Feature Engineering Function
# ---------------------------------------------------------------------------
def engineer_features(
    cfg: Dict[str, Any],
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[ColumnTransformer]]:
    """
    End-to-end feature engineering pipeline.

    Parameters
    ----------
    cfg : dict
        Parsed config.yaml.
    df_train : DataFrame
        Training feature matrix (raw, post data_pipeline).
    df_test : DataFrame or None
        Test feature matrix (same schema as train). None in inference mode.
    fit : bool
        If True, fit all encoders/scalers and save artifacts.
        If False, load pre-fitted artifacts (inference mode).

    Returns
    -------
    (X_train_processed, X_test_processed_or_None, fitted_preprocessor)
    """
    fe_cfg = cfg["feature_engineering"]
    artifacts_dir = cfg["data"]["artifacts_directory"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # Read derived-feature sub-configs (each has its own `enabled` flag)
    sd_cfg       = fe_cfg.get("service_density",    {})
    arpu_cfg     = fe_cfg.get("arpu",               {})
    autopay_cfg  = fe_cfg.get("autopay_indicator",  {})

    # --- Compute Derived Features (only when enabled: true in config) -------
    # Processing both train and test in one loop to keep logic DRY.
    # Because compute_* functions modify the DataFrame in-place, changes
    # propagate back to df_train / df_test through the reference.
    all_dfs = [df_train] + ([df_test] if df_test is not None else [])
    for _df in all_dfs:

        # Feature 1: Service Density
        if sd_cfg.get("enabled", True):
            compute_service_density(_df, sd_cfg.get("columns", []))
        else:
            logger.info("Service Density skipped (enabled: false).")

        # Feature 2: ARPU
        if arpu_cfg.get("enabled", True):
            compute_arpu(
                _df,
                numerator_col=arpu_cfg.get("numerator", "Total Charges"),
                denominator_col=arpu_cfg.get("denominator", "Tenure in Months"),
                zero_tenure_default=arpu_cfg.get("zero_tenure_default", 0.0),
            )
        else:
            logger.info("ARPU skipped (enabled: false).")

        # Feature 3: Autopay Indicator
        if autopay_cfg.get("enabled", True):
            compute_autopay_indicator(
                _df,
                payment_col=autopay_cfg.get("payment_method_column", "Payment Method"),
                autopay_keywords=autopay_cfg.get("keywords", []),
            )
        else:
            logger.info("Autopay Indicator skipped (enabled: false).")

    # --- Outlier Treatment (Winsorization) ---
    outlier_cfg = fe_cfg.get("outlier", {})
    cont_cols = outlier_cfg.get("continuous_columns", [])

    if fit:
        df_train, clip_bounds = winsorize_columns(
            df_train, cont_cols,
            lower_pct=outlier_cfg.get("lower_percentile", 0.01),
            upper_pct=outlier_cfg.get("upper_percentile", 0.99),
            fit=True,
        )
        joblib.dump(clip_bounds, os.path.join(artifacts_dir, "clip_bounds.joblib"))
    else:
        clip_bounds = joblib.load(os.path.join(artifacts_dir, "clip_bounds.joblib"))
        df_train, _ = winsorize_columns(df_train, cont_cols, fit=False, clip_bounds=clip_bounds)

    if df_test is not None:
        df_test, _ = winsorize_columns(df_test, cont_cols, fit=False, clip_bounds=clip_bounds)

    # --- Rare-Label Grouping ---
    threshold = fe_cfg.get("rare_label_threshold", 0.01)
    if fit:
        df_train, rare_mappings = group_rare_labels(df_train, threshold=threshold, fit=True)
        joblib.dump(rare_mappings, os.path.join(artifacts_dir, "rare_mappings.joblib"))
    else:
        rare_mappings = joblib.load(os.path.join(artifacts_dir, "rare_mappings.joblib"))
        df_train, _ = group_rare_labels(df_train, fit=False, rare_mappings=rare_mappings)

    if df_test is not None:
        df_test, _ = group_rare_labels(df_test, fit=False, rare_mappings=rare_mappings)

    # --- Frequency Encoding ---
    freq_cols = fe_cfg.get("frequency_encode_columns", [])
    if fit:
        df_train, freq_maps = frequency_encode(df_train, freq_cols, fit=True)
        joblib.dump(freq_maps, os.path.join(artifacts_dir, "freq_maps.joblib"))
    else:
        freq_maps = joblib.load(os.path.join(artifacts_dir, "freq_maps.joblib"))
        df_train, _ = frequency_encode(df_train, freq_cols, fit=False, freq_maps=freq_maps)

    if df_test is not None:
        if fit:
            df_test, _ = frequency_encode(df_test, freq_cols, fit=False, freq_maps=freq_maps)
        else:
            df_test, _ = frequency_encode(df_test, freq_cols, fit=False, freq_maps=freq_maps)

    # --- Sklearn Preprocessing Pipeline ---
    def _to_float_df(arr: np.ndarray, columns: List[str], index: Any) -> pd.DataFrame:
        """Convert ColumnTransformer output to a float64 DataFrame."""
        df_out = pd.DataFrame(arr, columns=columns, index=index)
        for c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
        return df_out.astype(np.float64)

    if fit:
        preprocessor, num_cols, cat_cols, pass_cols = build_preprocessor(cfg, df_train)
        X_train = preprocessor.fit_transform(df_train)
        feature_names = preprocessor.get_feature_names_out().tolist()
        X_train = _to_float_df(X_train, feature_names, df_train.index)

        joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
        joblib.dump(feature_names, os.path.join(artifacts_dir, "feature_names.joblib"))
        logger.info("Fitted and saved preprocessor (%d output features).", len(feature_names))
    else:
        preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
        feature_names = joblib.load(os.path.join(artifacts_dir, "feature_names.joblib"))
        X_train = preprocessor.transform(df_train)
        X_train = _to_float_df(X_train, feature_names, df_train.index)

    X_test = None
    if df_test is not None:
        X_test = preprocessor.transform(df_test)
        X_test = _to_float_df(X_test, feature_names, df_test.index)

    logger.info(
        "Feature engineering complete: train=%s, test=%s",
        X_train.shape, X_test.shape if X_test is not None else "N/A",
    )
    return X_train, X_test, preprocessor
