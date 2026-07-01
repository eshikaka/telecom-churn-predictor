"""
=============================================================================
Model Training — Hyperparameter Tuning, Cross-Validation & Evaluation
=============================================================================
Trains Logistic Regression, XGBoost, and Random Forest classifiers with
Stratified K-Fold CV. Computes comprehensive evaluation metrics, ROC/PR
curves, and exports the best model + SHAP artifacts.

Usage:
    python train.py                     # Uses default config.yaml
    python train.py --config alt.yaml   # Custom config
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from data_pipeline import load_config, run_pipeline
from feature_engineering import engineer_features
from explainability import compute_global_shap, save_shap_artifacts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
def build_model_grid(
    cfg: Dict[str, Any], class_ratio: float
) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """
    Build a list of (model_name, estimator, param_grid) tuples from config.

    Parameters
    ----------
    class_ratio : float
        Ratio of negative to positive class counts (for scale_pos_weight).
    """
    models_cfg = cfg["training"]["models"]
    model_list: List[Tuple[str, Any, Dict[str, Any]]] = []

    # Logistic Regression
    lr_cfg = models_cfg.get("logistic_regression", {})
    if lr_cfg.get("enabled", False):
        model_list.append((
            "LogisticRegression",
            LogisticRegression(random_state=cfg["training"]["random_state"]),
            lr_cfg.get("param_grid", {}),
        ))

    # XGBoost
    xgb_cfg = models_cfg.get("xgboost", {})
    if xgb_cfg.get("enabled", False) and HAS_XGB:
        grid = xgb_cfg.get("param_grid", {}).copy()
        if "scale_pos_weight" in grid:
            grid["scale_pos_weight"] = [
                class_ratio if v == "auto" else v for v in grid["scale_pos_weight"]
            ]
        model_list.append((
            "XGBClassifier",
            XGBClassifier(
                random_state=cfg["training"]["random_state"],
                use_label_encoder=False,
                verbosity=0,
            ),
            grid,
        ))
    elif xgb_cfg.get("enabled", False) and not HAS_XGB:
        logger.warning("XGBoost not installed — skipping.")

    # Random Forest
    rf_cfg = models_cfg.get("random_forest", {})
    if rf_cfg.get("enabled", False):
        grid = rf_cfg.get("param_grid", {}).copy()
        if "max_depth" in grid:
            grid["max_depth"] = [None if v is None else v for v in grid["max_depth"]]
        model_list.append((
            "RandomForest",
            RandomForestClassifier(random_state=cfg["training"]["random_state"]),
            grid,
        ))

    # LightGBM
    lgbm_cfg = models_cfg.get("lightgbm", {})
    if lgbm_cfg.get("enabled", False) and HAS_LGBM:
        grid = lgbm_cfg.get("param_grid", {}).copy()
        if "scale_pos_weight" in grid:
            grid["scale_pos_weight"] = [
                class_ratio if v == "auto" else v for v in grid["scale_pos_weight"]
            ]
        model_list.append((
            "LightGBM",
            LGBMClassifier(
                random_state=cfg["training"]["random_state"],
                verbosity=-1,
            ),
            grid,
        ))
    elif lgbm_cfg.get("enabled", False) and not HAS_LGBM:
        logger.warning("LightGBM not installed — skipping.")

    return model_list


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    artifacts_dir: str,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics and save plots.

    Returns a dictionary with all metrics + paths to saved plots.
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    # Core metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "confusion_matrix": {
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    roc_path = os.path.join(artifacts_dir, f"roc_curve_{model_name}.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    metrics["roc_curve_path"] = roc_path

    # --- Precision-Recall Curve ---
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_vals, prec_vals, label=f"{model_name} (AP={metrics['avg_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    pr_path = os.path.join(artifacts_dir, f"pr_curve_{model_name}.png")
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    metrics["pr_curve_path"] = pr_path

    # --- Confusion Matrix Heatmap ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    cm_path = os.path.join(artifacts_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    metrics["confusion_matrix_path"] = cm_path

    # --- Calibration Curve ---
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_pred, frac_pos, "s-", label=f"{model_name}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    cal_path = os.path.join(artifacts_dir, f"calibration_curve_{model_name}.png")
    fig.savefig(cal_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    metrics["calibration_curve_path"] = cal_path

    # --- Optimal Threshold (maximises F1 on the test set) ---
    thresh_range = np.linspace(0.05, 0.95, 91)
    f1_by_thresh = [f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresh_range]
    best_thresh_idx = int(np.argmax(f1_by_thresh))
    metrics["optimal_threshold"] = float(thresh_range[best_thresh_idx])
    metrics["optimal_threshold_f1"] = float(f1_by_thresh[best_thresh_idx])

    logger.info(
        "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f | OptThresh: %.2f",
        model_name, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1_score"], metrics["roc_auc"],
        metrics["optimal_threshold"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_and_select(
    cfg: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Train all configured models via GridSearchCV with Stratified K-Fold,
    evaluate on the holdout set, and return the best model.

    Returns
    -------
    (best_model, best_metrics, all_results)
    """
    train_cfg = cfg["training"]
    artifacts_dir = cfg["data"]["artifacts_directory"]
    os.makedirs(artifacts_dir, exist_ok=True)

    n_folds = train_cfg.get("cv_folds", 5)
    scoring = train_cfg.get("scoring_metric", "roc_auc")
    random_state = train_cfg.get("random_state", 42)

    # Class imbalance ratio for scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_ratio = round(n_neg / max(n_pos, 1), 2)
    logger.info("Class distribution: %d negative, %d positive (ratio %.2f)", n_neg, n_pos, class_ratio)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    model_list = build_model_grid(cfg, class_ratio)

    # ── SMOTE oversampling (applied once, before all model training) ──
    smote_cfg = train_cfg.get("smote", {})
    if smote_cfg.get("enabled", False) and HAS_SMOTE:
        smote = SMOTE(
            sampling_strategy=smote_cfg.get("sampling_strategy", "auto"),
            k_neighbors=smote_cfg.get("k_neighbors", 5),
            random_state=random_state,
        )
        n_before = len(X_train)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE: %d → %d samples (minority oversampled to %.0f%%)",
            n_before, len(X_train),
            (y_train == 1).sum() / len(y_train) * 100,
        )
    elif smote_cfg.get("enabled", False) and not HAS_SMOTE:
        logger.warning("imbalanced-learn not installed — skipping SMOTE.")

    all_results: List[Dict[str, Any]] = []
    best_score = -1.0
    best_model = None
    best_metrics: Dict[str, Any] = {}
    best_name = ""

    for name, estimator, param_grid in model_list:
        logger.info("Training %s with %d-fold Stratified CV...", name, n_folds)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)

        logger.info(
            "%s — Best CV %s: %.4f | Params: %s",
            name, scoring, search.best_score_, search.best_params_,
        )

        # Evaluate on holdout
        metrics = evaluate_model(search.best_estimator_, X_test, y_test, name, artifacts_dir)
        metrics["best_cv_score"] = round(search.best_score_, 4)
        metrics["best_params"] = search.best_params_
        all_results.append(metrics)

        # Track best
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = search.best_estimator_
            best_metrics = metrics
            best_name = name

    logger.info("Best model: %s (ROC-AUC = %.4f)", best_name, best_score)

    # Save best model
    model_path = os.path.join(artifacts_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    logger.info("Saved best model to %s", model_path)

    # Save the F1-optimised decision threshold for this model
    opt_thresh = best_metrics.get("optimal_threshold", 0.5)
    joblib.dump(opt_thresh, os.path.join(artifacts_dir, "optimal_threshold.joblib"))
    logger.info("Saved optimal threshold: %.3f (F1=%.4f)", opt_thresh, best_metrics.get("optimal_threshold_f1", 0))

    # Save all results
    results_path = os.path.join(artifacts_dir, "training_results.json")
    serializable_results = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.floating)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serializable_results.append(sr)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(serializable_results, fh, indent=2, default=str)

    return best_model, best_metrics, all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Churn Model Training")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, metavar="N",
        help="Extra random seeds for split-variance analysis (e.g. --seeds 0 1 2 3 4).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = cfg["data"]["artifacts_directory"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # ---- Data Pipeline ----
    logger.info("=" * 70)
    logger.info("STAGE 1: DATA PIPELINE")
    logger.info("=" * 70)
    features, target, customer_ids = run_pipeline(cfg)

    if target is None:
        raise ValueError("Target column not found. Cannot train.")

    # ---- Train/Test Split ----
    train_cfg = cfg["training"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features, target,
        test_size=train_cfg.get("test_size", 0.20),
        random_state=train_cfg.get("random_state", 42),
        stratify=target,
    )
    logger.info(
        "Train/Test split: %d / %d (stratified)",
        len(X_train_raw), len(X_test_raw),
    )

    # Save raw column list for inference
    joblib.dump(list(features.columns), os.path.join(artifacts_dir, "raw_feature_columns.joblib"))

    # ---- Feature Engineering (fit on train only) ----
    logger.info("=" * 70)
    logger.info("STAGE 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    X_train, X_test, preprocessor = engineer_features(
        cfg, X_train_raw, X_test_raw, fit=True
    )

    # ---- Naive Baselines (academic reference point) ----
    logger.info("=" * 70)
    logger.info("STAGE 2b: NAIVE BASELINES")
    logger.info("=" * 70)
    majority_label = int(y_train.value_counts().idxmax())
    majority_preds = np.full(len(y_test), majority_label)
    logger.info(
        "Majority-class baseline (always class %d) — Acc: %.4f | F1: %.4f",
        majority_label,
        accuracy_score(y_test, majority_preds),
        f1_score(y_test, majority_preds, zero_division=0),
    )
    dummy_strat = DummyClassifier(strategy="stratified", random_state=train_cfg.get("random_state", 42))
    dummy_strat.fit(X_train, y_train)
    dummy_proba = dummy_strat.predict_proba(X_test)[:, 1]
    logger.info(
        "Stratified-random baseline — ROC-AUC: %.4f",
        roc_auc_score(y_test, dummy_proba),
    )
    logger.info(
        "These baselines set the floor — every trained model must substantially exceed them."
    )

    # ---- Model Training ----
    logger.info("=" * 70)
    logger.info("STAGE 3: MODEL TRAINING & EVALUATION")
    logger.info("=" * 70)
    best_model, best_metrics, all_results = train_and_select(
        cfg, X_train, y_train, X_test, y_test
    )

    # ---- SHAP Explanations (pre-compute global) ----
    if cfg.get("explainability", {}).get("save_global_shap", True):
        logger.info("=" * 70)
        logger.info("STAGE 4: GLOBAL SHAP EXPLANATIONS")
        logger.info("=" * 70)
        try:
            shap_values, explainer = compute_global_shap(
                best_model, X_train, cfg
            )
            save_shap_artifacts(shap_values, explainer, X_train, cfg)
            logger.info("Global SHAP artifacts saved.")
        except Exception as e:
            logger.error("SHAP computation failed: %s", e)

    # ---- Summary ----
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Best Model      : %s", best_metrics.get("model_name"))
    logger.info("ROC-AUC         : %.4f", best_metrics.get("roc_auc", 0))
    logger.info("F1-Score        : %.4f", best_metrics.get("f1_score", 0))
    logger.info("Precision       : %.4f", best_metrics.get("precision", 0))
    logger.info("Recall          : %.4f", best_metrics.get("recall", 0))
    logger.info("Artifacts saved to: %s", artifacts_dir)

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['model_name']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['roc_auc']:>10.4f}"
        )
    print("=" * 80)

    # ---- Optional: Split-Variance Analysis ----
    if args.seeds:
        logger.info("=" * 70)
        logger.info("STAGE 5: SPLIT-VARIANCE ANALYSIS ACROSS %d EXTRA SEEDS", len(args.seeds))
        logger.info("=" * 70)
        logger.warning(
            "Using the preprocessor fitted on seed=%d (not independently re-fit per seed). "
            "This estimates variance from random splitting, not full nested CV.",
            train_cfg.get("random_state", 42),
        )

        # Combine train + test back into one processed pool
        X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
        y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

        best_name_v = best_metrics.get("model_name", "")
        best_params_v = best_metrics.get("best_params", {})
        n_neg_f = int((y_full == 0).sum())
        n_pos_f = int((y_full == 1).sum())
        model_list_v = build_model_grid(cfg, round(n_neg_f / max(n_pos_f, 1), 2))
        template_v = next((e for n, e, _ in model_list_v if n == best_name_v), None)

        var_aucs: List[float] = [best_metrics.get("roc_auc", 0.0)]
        var_f1s:  List[float] = [best_metrics.get("f1_score", 0.0)]

        for seed in args.seeds:
            Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(
                X_full, y_full,
                test_size=train_cfg.get("test_size", 0.20),
                random_state=seed,
                stratify=y_full,
            )
            if template_v is not None:
                vmodel = clone(template_v)
                vmodel.set_params(**best_params_v)
                vmodel.fit(Xv_tr, yv_tr)
                vp = vmodel.predict_proba(Xv_te)[:, 1]
                v_auc = roc_auc_score(yv_te, vp)
                v_f1 = f1_score(yv_te, (vp >= 0.5).astype(int), zero_division=0)
                var_aucs.append(v_auc)
                var_f1s.append(v_f1)
                logger.info("  seed=%d  ROC-AUC=%.4f  F1=%.4f", seed, v_auc, v_f1)

        if len(var_aucs) > 1:
            logger.info(
                "  Summary (%d evaluations): ROC-AUC = %.4f ± %.4f | F1 = %.4f ± %.4f",
                len(var_aucs),
                float(np.mean(var_aucs)), float(np.std(var_aucs)),
                float(np.mean(var_f1s)),  float(np.std(var_f1s)),
            )


if __name__ == "__main__":
    main()
