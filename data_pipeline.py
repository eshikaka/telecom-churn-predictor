"""
=============================================================================
Data Pipeline — Generic Ingestion, Joining & Preprocessing
=============================================================================
Reads raw files from a source directory, joins them on configurable keys,
performs defensive type casting, imputation, and text standardisation.

All behaviour is driven by config.yaml — no hardcoded column names.

Usage:
    python data_pipeline.py                     # Uses default config.yaml
    python data_pipeline.py --config alt.yaml   # Custom config path
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import KNNImputer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------
def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Configuration loaded from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Schema Alignment — Normalise Column Names
# ---------------------------------------------------------------------------
def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names and normalise minor variations:
    underscores vs spaces, inconsistent casing on the *join key* side.
    Original display names are preserved for downstream readability.
    """
    df.columns = df.columns.str.strip()
    return df


def align_join_key(df: pd.DataFrame, expected_key: str) -> pd.DataFrame:
    """
    If the expected join key is not present, attempt to find a close match
    (case-insensitive, underscore/space agnostic) and rename it.
    """
    if expected_key in df.columns:
        return df

    normalised_expected = re.sub(r"[\s_]+", "", expected_key.lower())
    for col in df.columns:
        normalised_col = re.sub(r"[\s_]+", "", col.lower())
        if normalised_col == normalised_expected:
            logger.warning(
                "Join key '%s' not found — matched '%s' and renamed.", expected_key, col
            )
            return df.rename(columns={col: expected_key})

    logger.warning(
        "Join key '%s' not found in columns: %s. Returning DataFrame unchanged.",
        expected_key,
        list(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# File Loading
# ---------------------------------------------------------------------------
def load_file(filepath: str) -> pd.DataFrame:
    """Load a single CSV or Excel file with defensive column stripping."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df = standardise_column_names(df)
    logger.info("Loaded %-50s  %d rows x %d cols", path.name, len(df), df.shape[1])
    return df


def load_all_sources(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load every file declared in config['data']['files']."""
    src_dir = cfg["data"]["source_directory"]
    frames: Dict[str, pd.DataFrame] = {}
    for name, filename in cfg["data"]["files"].items():
        full_path = os.path.join(src_dir, filename)
        frames[name] = load_file(full_path)
    return frames


# ---------------------------------------------------------------------------
# Safe Merge (duplicate-column aware)
# ---------------------------------------------------------------------------
def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "left",
    tag: str = "",
) -> pd.DataFrame:
    """
    Merge two DataFrames, automatically dropping pandas-generated suffix
    columns (_x / _y) when the values are identical — i.e. true duplicates.
    """
    cols_before = set(left.columns)
    merged = left.merge(right, on=on, how=how, suffixes=("", f"__{tag}"))

    suffix = f"__{tag}"
    new_cols = set(merged.columns) - cols_before - {on}
    dupes_to_drop: List[str] = []

    for col in new_cols:
        if col.endswith(suffix):
            original = col[: -len(suffix)]
            if original in merged.columns:
                match = (
                    merged[col].fillna("__NULL__") == merged[original].fillna("__NULL__")
                ).all()
                if match:
                    dupes_to_drop.append(col)

    if dupes_to_drop:
        merged.drop(columns=dupes_to_drop, inplace=True)
        logger.debug("Dropped duplicate columns after merge [%s]: %s", tag, dupes_to_drop)

    logger.info(
        "Merged %-15s on %-15s -> %d rows x %d cols (dropped %d dupes)",
        tag, on, len(merged), merged.shape[1], len(dupes_to_drop),
    )
    return merged


# ---------------------------------------------------------------------------
# Join Strategy
# ---------------------------------------------------------------------------
def join_all_sources(
    frames: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> pd.DataFrame:
    """
    Join all loaded DataFrames according to configuration.
    Uses the primary join_key for most tables and secondary_joins where specified.
    """
    schema = cfg["schema"]
    join_key = schema["join_key"]
    admin_cols = set(schema.get("admin_columns", []))
    secondary = schema.get("secondary_joins", {})

    # Align join keys across all frames
    for name in frames:
        frames[name] = align_join_key(frames[name], join_key)

    # Drop admin columns from every frame
    for name, df in frames.items():
        to_drop = [c for c in admin_cols if c in df.columns]
        if to_drop:
            frames[name] = df.drop(columns=to_drop)

    # Build join order: demographics is the base
    base_name = "demographics"
    if base_name not in frames:
        base_name = list(frames.keys())[0]
    master = frames[base_name].copy()
    logger.info("Base table: %s (%d rows x %d cols)", base_name, len(master), master.shape[1])

    join_order = [k for k in frames if k != base_name]

    for name in join_order:
        right = frames[name]

        # Determine join key for this table
        on_key = secondary.get(name, join_key)
        right = align_join_key(right, on_key)

        # Special handling for churn_extra: only keep selected columns
        if name == "churn_extra":
            keep = cfg["schema"].get("churn_extra_keep", [])
            available_keep = [c for c in keep if c in right.columns]
            if available_keep:
                right = right[available_keep]

        master = safe_merge(master, right, on=on_key, tag=name)

    logger.info("Master DataFrame: %d rows x %d cols", len(master), master.shape[1])
    return master


# ---------------------------------------------------------------------------
# Type Casting & Numeric Coercion
# ---------------------------------------------------------------------------
def coerce_numeric_columns(
    df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Force specified columns to numeric dtype. Handles whitespace strings
    (e.g. ' ' in Total Charges) by coercing to NaN.
    """
    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not in DataFrame for numeric coercion.", col)
            continue
        before_null = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_null = df[col].isnull().sum()
        new_nulls = after_null - before_null
        if new_nulls > 0:
            logger.info(
                "Coerced '%s' to numeric: %d new NaN from non-numeric values.",
                col, new_nulls,
            )
    return df


# ---------------------------------------------------------------------------
# Boolean Mapping
# ---------------------------------------------------------------------------
def map_binary_text(
    df: pd.DataFrame, bool_map: Dict[str, int]
) -> pd.DataFrame:
    """
    Dynamically detect columns whose unique values are a subset of the
    boolean map keys, and convert them to 0/1.
    """
    map_keys_lower = {k.lower(): v for k, v in bool_map.items()}
    converted: List[str] = []

    for col in df.select_dtypes(include=["object", "string"]).columns:
        uniques = set(df[col].dropna().str.strip().str.lower().unique())
        if uniques and uniques <= set(map_keys_lower.keys()):
            df[col] = df[col].str.strip().str.lower().map(map_keys_lower).fillna(0).astype(np.int8)
            converted.append(col)

    if converted:
        logger.info("Boolean-mapped %d columns: %s", len(converted), converted)
    return df


# ---------------------------------------------------------------------------
# Text Standardisation
# ---------------------------------------------------------------------------
def standardise_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace, lowercase, and collapse multi-spaces in all
    object/string columns to fix categorical variations.
    """
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col]
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
    logger.info("Standardised text in %d categorical columns.", len(df.select_dtypes(include=["object", "string"]).columns))
    return df


# ---------------------------------------------------------------------------
# Schema Verification — Missing Expected Columns
# ---------------------------------------------------------------------------
def verify_and_fill_missing_columns(
    df: pd.DataFrame,
    expected_columns: List[str],
    training_stats: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Check that all expected feature columns exist. If a column is missing,
    fill it with the training-time mean (if available) or 0.
    """
    for col in expected_columns:
        if col not in df.columns:
            fill_val = 0
            if training_stats and col in training_stats:
                fill_val = training_stats[col]
            logger.warning(
                "Expected column '%s' missing — filling with default value %s.",
                col, fill_val,
            )
            df[col] = fill_val
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------
def apply_conditional_imputation(
    df: pd.DataFrame, rules: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    """Apply conditional fill rules (e.g. Offer NaN -> 'None')."""
    for col, rule in rules.items():
        if col in df.columns and df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(rule["fill_value"])
            logger.info(
                "Conditional imputation: '%s' — %d NaN -> '%s' (%s)",
                col, n_missing, rule["fill_value"], rule.get("reason", ""),
            )
    return df


def apply_knn_imputation(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    weights: str = "distance",
    fit: bool = True,
    imputer: Optional[KNNImputer] = None,
) -> Tuple[pd.DataFrame, KNNImputer]:
    """
    KNN-impute remaining numeric NaN values.

    Parameters
    ----------
    fit : bool
        If True, fit a new imputer. If False, use the provided imputer (inference mode).
    imputer : KNNImputer or None
        Pre-fitted imputer for inference mode.

    Returns
    -------
    (DataFrame, fitted KNNImputer)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    null_counts = df[numeric_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()

    if not cols_with_nulls:
        logger.info("No numeric NaN remaining — KNN imputation skipped.")
        if imputer is None:
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imputer.fit(df[numeric_cols])
        return df, imputer

    if fit:
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df[numeric_cols] = pd.DataFrame(
            imputer.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index,
        )
        total_filled = null_counts[cols_with_nulls].sum()
        logger.info(
            "KNN imputation (k=%d): filled %d values across %d columns.",
            n_neighbors, total_filled, len(cols_with_nulls),
        )
    else:
        if imputer is None:
            raise ValueError("Inference mode requires a pre-fitted imputer.")
        # For single-row inference, KNN may fail — fall back to column medians
        if len(df) == 1:
            logger.warning("Single-row inference: using stored medians instead of KNN.")
            return df, imputer
        df[numeric_cols] = pd.DataFrame(
            imputer.transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index,
        )

    return df, imputer


# ---------------------------------------------------------------------------
# Leakage & Target Separation
# ---------------------------------------------------------------------------
def separate_target_and_leakage(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Remove the target column and leakage columns from the feature matrix.

    Returns
    -------
    (features_df, target_series, leakage_df)
    """
    schema = cfg["schema"]
    target_col = schema["target"]["column"]
    leakage_cols = [c for c in schema.get("leakage_columns", []) if c in df.columns]
    target_aliases = [c for c in schema.get("target_aliases", []) if c in df.columns]

    # Extract leakage
    leakage_df = df[leakage_cols].copy() if leakage_cols else pd.DataFrame()
    df = df.drop(columns=leakage_cols, errors="ignore")

    # Extract target
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    target = df[target_col].copy()
    df = df.drop(columns=[target_col], errors="ignore")

    # Drop target aliases
    df = df.drop(columns=target_aliases, errors="ignore")

    logger.info(
        "Separated target '%s' and quarantined %d leakage columns.",
        target_col, len(leakage_cols),
    )
    return df, target, leakage_df


# ---------------------------------------------------------------------------
# Drop Non-Feature Columns
# ---------------------------------------------------------------------------
def drop_non_features(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove identifier and geospatial columns that are not features.
    Returns (features_df, customer_ids).
    """
    schema = cfg["schema"]
    id_cols = schema.get("id_columns", [])
    geo_cols = schema.get("geo_drop_columns", [])

    # Preserve Customer ID for downstream tracking
    join_key = schema["join_key"]
    customer_ids = df[join_key].copy() if join_key in df.columns else pd.Series(dtype="object")

    all_drop = set(id_cols + geo_cols)
    to_drop = [c for c in all_drop if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")

    logger.info("Dropped %d non-feature columns: %s", len(to_drop), to_drop)
    return df, customer_ids


# ---------------------------------------------------------------------------
# Redundancy Checks
# ---------------------------------------------------------------------------
def drop_redundant_columns(
    df: pd.DataFrame, checks: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Drop columns that are perfectly derivable from other columns."""
    for check in checks:
        src = check.get("source", "")
        if src not in df.columns:
            continue

        dup_of = check.get("duplicate_of", "")
        if dup_of and dup_of in df.columns:
            if (df[src].fillna("__NULL__") == df[dup_of].fillna("__NULL__")).all():
                df = df.drop(columns=[src])
                logger.info("Dropped redundant column '%s' (duplicate of '%s').", src, dup_of)
                continue

        derived_from = check.get("derived_from", "")
        if derived_from and derived_from in df.columns:
            df = df.drop(columns=[src], errors="ignore")
            logger.info("Dropped derivable column '%s'.", src)

    return df


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    cfg: Dict[str, Any],
    inference_mode: bool = False,
    inference_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Execute the full data pipeline.

    Parameters
    ----------
    cfg : dict
        Parsed config.yaml.
    inference_mode : bool
        If True, skip target separation and use pre-fitted artifacts.
    inference_df : DataFrame or None
        Single-row (or batch) DataFrame for inference. If provided, bypasses
        file loading and joining.

    Returns
    -------
    (features_df, target_series_or_None, customer_ids_or_None)
    """
    preproc = cfg["preprocessing"]
    schema = cfg["schema"]
    artifacts_dir = cfg["data"]["artifacts_directory"]

    # --- Load & Join ---
    if inference_df is not None:
        master = inference_df.copy()
        master = standardise_column_names(master)
    else:
        frames = load_all_sources(cfg)
        master = join_all_sources(frames, cfg)

    # --- Numeric Coercion ---
    master = coerce_numeric_columns(master, preproc.get("force_numeric_columns", []))

    # --- Boolean Mapping ---
    master = map_binary_text(master, preproc.get("boolean_map", {}))

    # --- Text Standardisation ---
    master = standardise_text_columns(master)

    # --- Ordinal Maps ---
    for col, mapping in preproc.get("ordinal_maps", {}).items():
        if col in master.columns:
            lower_mapping = {k.lower().strip(): v for k, v in mapping.items()}
            master[col] = master[col].map(lower_mapping)
            logger.info("Ordinal-mapped '%s'.", col)

    # --- Gender Encoding (explicit, dtype-agnostic) ---
    gender_map = preproc.get("gender_map", {})
    if gender_map:
        lower_map = {str(k).lower().strip(): v for k, v in gender_map.items()}
        str_cols = master.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            uniques = set(master[col].dropna().astype(str).str.lower().str.strip().unique())
            if uniques and uniques <= set(lower_map.keys()):
                master[col] = (
                    master[col].astype(str).str.lower().str.strip()
                    .map(lower_map).fillna(0).astype(np.int8)
                )
                logger.info("Gender-encoded '%s': %s", col, lower_map)

    # --- Redundancy Checks ---
    master = drop_redundant_columns(master, schema.get("redundant_checks", []))

    # --- Conditional Imputation ---
    master = apply_conditional_imputation(master, preproc.get("conditional_imputation", {}))

    # --- Separate Target & Leakage (training only) ---
    target = None
    if not inference_mode:
        master, target, _leakage = separate_target_and_leakage(master, cfg)

    # --- Drop Non-Features ---
    master, customer_ids = drop_non_features(master, cfg)

    # --- Resolve Tenure Duplicates ---
    if "Tenure" in master.columns and "Tenure in Months" in master.columns:
        master.drop(columns=["Tenure"], inplace=True)
        logger.info("Dropped duplicate 'Tenure' column (kept 'Tenure in Months').")

    # --- KNN Imputation ---
    knn_cfg = preproc.get("knn_imputer", {})
    if inference_mode:
        imputer_path = os.path.join(artifacts_dir, "knn_imputer.joblib")
        if os.path.exists(imputer_path):
            imputer = joblib.load(imputer_path)
            master, _ = apply_knn_imputation(
                master, fit=False, imputer=imputer,
                n_neighbors=knn_cfg.get("n_neighbors", 5),
                weights=knn_cfg.get("weights", "distance"),
            )
        else:
            # Fallback: fill with training-time column medians if available, else 0
            medians_path = os.path.join(artifacts_dir, "column_medians.joblib")
            if os.path.exists(medians_path):
                medians = joblib.load(medians_path)
                numeric_cols = master.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    master[col] = master[col].fillna(medians.get(col, 0))
                logger.info("Inference fallback: filled NaN with training-time medians.")
            else:
                master = master.fillna(0)
                logger.warning("No medians artifact found; filled NaN with 0 (run train.py first).")
    else:
        master, imputer = apply_knn_imputation(
            master,
            n_neighbors=knn_cfg.get("n_neighbors", 5),
            weights=knn_cfg.get("weights", "distance"),
        )
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(imputer, os.path.join(artifacts_dir, "knn_imputer.joblib"))
        logger.info("Saved KNN imputer to %s", artifacts_dir)

        # Save column medians for single-row inference fallback
        medians = master.select_dtypes(include=[np.number]).median().to_dict()
        joblib.dump(medians, os.path.join(artifacts_dir, "column_medians.joblib"))

    logger.info(
        "Pipeline complete: %d rows x %d cols | Target: %s",
        len(master), master.shape[1], "present" if target is not None else "N/A",
    )
    return master, target, customer_ids


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Churn Data Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["data"]["output_directory"]
    os.makedirs(out_dir, exist_ok=True)

    features, target, customer_ids = run_pipeline(cfg)

    # Save outputs
    features.insert(0, "Customer_ID", customer_ids.values)
    features.to_csv(os.path.join(out_dir, "features.csv"), index=False)

    if target is not None:
        target_df = pd.DataFrame({"Customer_ID": customer_ids.values, "Churn": target.values})
        target_df.to_csv(os.path.join(out_dir, "target.csv"), index=False)

    logger.info("Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
