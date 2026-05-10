"""
EEG PoemType Classification — Full Feature Selection Pipeline
=============================================================

Author  : Pramit Biswas
Version : v2.0

Overview
--------
This pipeline performs EEG-based PoemType classification using
ROI-wise bandpower features extracted from multiple subjects.

The workflow combines:
    - Correlation-based feature pruning
    - Leave-One-Subject-Out Cross-Validation (LOSO-CV)
    - Multi-model evaluation
    - XGBoost feature importance analysis
    - SHAP analysis for model interpretability
    - Top-N feature subset optimization
    - Temporal delta feature engineering
    - ROI × frequency-band interpretability analysis
    - Subject-level z-score normalization

The primary goal is:
    Predict PoemType labels from EEG-derived ROI bandpower features
    while ensuring subject-independent generalization.

--------------------------------------------------------------------
Models
--------------------------------------------------------------------
1. Logistic Regression (Elastic Net)
    - Linear model with L1 + L2 regularization
    - Performs implicit feature selection
    - Baseline interpretable classifier

2. Random Forest
    - Ensemble of decision trees
    - Captures nonlinear interactions
    - Provides feature importance scores
    - Robust to overfitting

3. XGBoost
    - Gradient boosted decision trees
    - Handles nonlinear interactions
    - Provides feature importance scores
    - Used for feature ranking and selection

--------------------------------------------------------------------
Task
--------------------------------------------------------------------
Multi-class classification of:
    TARGET = "PoemType"

Input EEG features consist of:
    - ROI bandpower features
    - Multiple frequency bands
    - Early and late temporal windows
    - Absolute / relative / ratio / asymmetry features

Optional temporal-delta features are later added:
    delta_feature = late_feature − early_feature

--------------------------------------------------------------------
Pipeline Stages
--------------------------------------------------------------------

[1] Data Loading
----------------
- Load all subject parquet files
- Each parquet file corresponds to one subject
- Combine all subjects into a global dataframe

Expected file pattern:
    *_roi_bpfeatures.parquet

--------------------------------------------------------------------

[2] Feature Selection
---------------------
Automatically select valid numeric EEG feature columns.

Excluded columns include:
    - PoemType
    - Subject identifiers
    - Metadata columns
    - Trial information

Only numeric EEG-derived features are retained.

--------------------------------------------------------------------

[3] Subject-Level Z-Score Normalization
----------------------------------------
Each subject's features are normalized independently:

    x_norm = (x - μ_subject) / σ_subject

Benefits:
    - Removes between-subject amplitude differences
    - Preserves within-subject EEG patterns
    - Improves cross-subject generalization

--------------------------------------------------------------------

[4] Correlation Pruning
-----------------------
Highly correlated features are removed to reduce redundancy.

Method:
    - Compute feature correlation matrix
    - Remove one feature from each pair where:

        |r| > 0.90

Benefits:
    - Reduces multicollinearity
    - Improves model stability
    - Reduces overfitting
    - Improves interpretability

--------------------------------------------------------------------

[5] Leave-One-Subject-Out Cross-Validation (LOSO-CV)
----------------------------------------------------
Main evaluation strategy.

Procedure:
    - Hold out one subject for testing
    - Train on all remaining subjects
    - Repeat for every subject
    - Aggregate metrics across folds

Purpose:
    Evaluate subject-independent EEG generalization.

This prevents:
    - Subject leakage
    - Identity memorization
    - Artificially inflated performance

--------------------------------------------------------------------

[6] Performance Metrics
-----------------------
For every LOSO fold and model, compute:

    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1-score (macro)
    - ROC-AUC (OvR macro)
    - Confusion Matrix

Macro averaging ensures:
    - Equal contribution from each class
    - Robust evaluation under class imbalance

--------------------------------------------------------------------

[7] XGBoost Feature Importance
------------------------------
During LOSO-CV:
    - XGBoost feature importances are accumulated
      across all folds

Importance type:
    - Gain-based importance

Purpose:
    Identify stable EEG biomarkers across subjects.

Feature rankings are later used for:
    - Feature subset selection
    - ROI-band interpretation
    - Visualization

--------------------------------------------------------------------

[8] SHAP Analysis
-----------------
After final model training:
    - Compute SHAP values for XGBoost model
    - Generate summary plots
    - Identify feature contributions per class

Benefits:
    - Local and global interpretability
    - Feature directionality analysis
    - Class-specific patterns

--------------------------------------------------------------------

[9] Top-N Feature Subset Optimization
-------------------------------------
The pipeline evaluates multiple feature subset sizes:

    TOP_N_CANDIDATES = [30, 50, 80]

Procedure:
    - Select top-N features using accumulated
      XGBoost importance
    - Re-run LOSO-CV
    - Compare model performance

Selection criterion:
    Highest mean ROC-AUC

Purpose:
    Find optimal feature dimensionality while
    minimizing redundancy and overfitting.

--------------------------------------------------------------------

[10] Temporal Delta Feature Engineering
---------------------------------------
For every matching early/late feature pair:

    delta = late − early

Example:
    delta_alpha_CL1 =
        late_alpha_CL1 − early_alpha_CL1

Purpose:
    Capture temporal EEG dynamics rather than
    static bandpower alone.

These features may encode:
    - Cognitive transitions
    - Temporal processing changes
    - Dynamic neural responses

--------------------------------------------------------------------

[11] Final LOSO-CV Re-evaluation
-------------------------------
After adding temporal delta features:
    - Re-run full LOSO-CV
    - Recompute metrics
    - Recompute feature importance

Purpose:
    Evaluate whether temporal dynamics improve
    classification performance.

--------------------------------------------------------------------

[12] Visualization and Interpretability
---------------------------------------

A. Top Feature Importance Bar Plot
----------------------------------
Displays:
    - Top-20 EEG features
    - Importance magnitude
    - Feature-type colour coding

Feature categories:
    - Absolute power
    - Relative power
    - Ratios
    - Asymmetry
    - Temporal delta

--------------------------------------------------------------------

B. ROI × Frequency Band Heatmap
-------------------------------
Visualizes importance distribution across:
    - Brain ROIs
    - Frequency bands

Helps identify:
    - Dominant neural regions
    - Relevant EEG frequency bands
    - Spatial-temporal EEG signatures

--------------------------------------------------------------------

C. Confusion Matrices
---------------------
Generated for each model.

Purpose:
    - Analyze class-specific performance
    - Identify commonly confused poem categories

--------------------------------------------------------------------

D. SHAP Summary Plots
---------------------
Global interpretability:
    - SHAP summary (beeswarm) plot
    - SHAP bar plot
    - Class-specific SHAP analysis

--------------------------------------------------------------------

Expected Input
--------------
Input directory should contain:
    51 subject parquet files

Each file should contain:
    - EEG ROI bandpower feature columns
    - Target column:
            'PoemType'
    - Optional metadata columns

Example:
    sub-021_roi_bpfeatures.parquet

--------------------------------------------------------------------

Expected Feature Types
----------------------
Typical feature naming conventions include:

    early_alpha_CL1_abs
    late_beta_CL5_rel
    gamma_ratio_CL2
    alpha_asym_CL4
    delta_theta_CL3

Supported EEG bands:
    - Delta
    - Theta
    - Alpha
    - Beta
    - Gamma

Supported ROIs:
    - CL1_lf
    - CL2_lft
    - CL3_lpo
    - CL4_rf
    - CL5_rft
    - CL6_rpo

--------------------------------------------------------------------

Outputs
-------
The pipeline automatically saves:

1. Subject-wise LOSO metrics
2. Aggregated performance summaries
3. Feature importance rankings
4. Selected feature lists
5. Top-feature plots
6. ROI × band heatmaps
7. Confusion matrices
8. SHAP analysis plots

Example outputs:
    stage1_summary.csv
    final_summary.csv
    final_feature_importances.csv
    top20_features_final.png
    importance_heatmap_final.png
    cm_final_XGBoost.png
    shap_summary_plot.png
    shap_bar_plot.png

--------------------------------------------------------------------

Scientific Significance
-----------------------
This pipeline is designed for:
    - Subject-independent EEG classification
    - EEG biomarker discovery
    - Spatial-frequency interpretability
    - Temporal neural dynamics analysis

The framework combines:
    - Robust validation
    - Feature engineering
    - Interpretable ML
    - SHAP-based explainability

making it suitable for:
    - EEG decoding studies
    - Cognitive neuroscience
    - Computational neuroscience research
    - Publication-oriented ML pipelines

--------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = (BASE_DIR / "../../data").resolve()
INPUT_DIR  = DATA_DIR / "features/roi_ftrs2"
OUTPUT_DIR = BASE_DIR / "../results/ml_pipeline"
PLOT_DIR   = OUTPUT_DIR / "plots"

TARGET        = "PoemType"
RANDOM_STATE  = 42
FILE_PATTERN  = "*_roi_bpfeatures.parquet"
CORR_THRESH   = 0.90
TOP_N_CANDIDATES = [30, 50, 80]

for d in [OUTPUT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# PLOTTING STYLE
# ─────────────────────────────────────────────
PALETTE = {
    "primary"   : "#5C6BC0",
    "secondary" : "#26A69A",
    "accent"    : "#EF6C00",
    "bg"        : "#FAFAFA",
    "grid"      : "#E8E8E8",
    "text"      : "#1A1A2E",
    "muted"     : "#7B7B9A",
}

# Feature-type colour map (for bar chart)
TYPE_COLORS = {
    "abs"   : "#5C6BC0",
    "rel"   : "#26A69A",
    "ratio" : "#EF6C00",
    "asym"  : "#EC407A",
    "delta" : "#AB47BC",
}

plt.rcParams.update({
    "figure.facecolor" : PALETTE["bg"],
    "axes.facecolor"   : PALETTE["bg"],
    "axes.edgecolor"   : PALETTE["grid"],
    "axes.grid"        : True,
    "grid.color"       : PALETTE["grid"],
    "grid.linewidth"   : 0.3,
    "text.color"       : PALETTE["text"],
    "axes.labelcolor"  : PALETTE["text"],
    "xtick.color"      : PALETTE["text"],
    "ytick.color"      : PALETTE["text"],
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
})


# ─────────────────────────────────────────────
# HELPERS — FEATURE TYPING
# ─────────────────────────────────────────────
def feature_type(name: str) -> str:
    """Infer feature type from column name prefix."""
    n = name.lower()
    if "asym"  in n: return "asym"
    if "delta" in n: return "delta"
    if "ratio" in n: return "ratio"
    if "_rel_" in n: return "rel"
    if "_abs_" in n: return "abs"
    return "other"


def feature_band(name: str) -> str:
    for b in ["delta", "theta", "alpha", "beta", "gamma"]:
        if b in name.lower():
            return b
    return "other"


def feature_roi(name: str) -> str:
    for r in ["CL1_lf", "CL2_lft", "CL3_lpo", "CL4_rf", "CL5_rft", "CL6_rpo"]:
        if r.lower() in name.lower():
            return r
    return "other"


def feature_window(name: str) -> str:
    n = name.lower()
    if n.startswith("early"): return "early"
    if n.startswith("late"):  return "late"
    if n.startswith("delta"): return "delta"
    return "other"


# ─────────────────────────────────────────────
# FEATURE SELECTION UTILITIES
# ─────────────────────────────────────────────
def subject_normalize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Apply z-score normalization per subject."""
    df_norm = df.copy()
    scaler = StandardScaler()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_norm


def prune_correlated_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = CORR_THRESH,
) -> list[str]:
    """Drop one column from each highly correlated pair (|r| > threshold)."""
    corr = df[feature_cols].corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = {col for col in upper.columns if any(upper[col] > threshold)}
    pruned  = [c for c in feature_cols if c not in to_drop]
    print(f"  Correlation pruning: {len(feature_cols):>4} → {len(pruned):>4} features "
          f"(removed {len(to_drop)} at |r|>{threshold})")
    return pruned


def add_temporal_delta(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Add (late − early) delta features for every early/late pair."""
    df = df.copy()
    delta_cols = []
    early_cols = [c for c in feature_cols if c.startswith("early_")]
    for ecol in early_cols:
        lcol = ecol.replace("early_", "late_", 1)
        if lcol in df.columns:
            dcol = ecol.replace("early_", "delta_", 1)
            df[dcol] = df[lcol] - df[ecol]
            delta_cols.append(dcol)
    return df, delta_cols


def get_top_features(
    importance_dict: dict[str, float],
    n: int,
) -> list[str]:
    ranked = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:n]]


# ─────────────────────────────────────────────
# FEATURE COLUMNS SELECTOR
# ─────────────────────────────────────────────
_EXCLUDE = {
    TARGET, "trial_index", "Block", "AA", "Creativity", "Imagery",
    "Moved", "Originality", "Subject", "subject", "sub",
}

def get_feature_columns(df: pd.DataFrame, extra_exclude: set | None = None) -> list[str]:
    exclude = _EXCLUDE | (extra_exclude or set())
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def get_models(random_state: int = RANDOM_STATE) -> dict:
    return {
        "LR_ElasticNet": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(
                penalty="elasticnet", solver="saga",
                l1_ratio=0.5, C=1.0, max_iter=5000,
                class_weight="balanced", random_state=random_state,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     RandomForestClassifier(
                n_estimators=300, max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                class_weight="balanced", random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=random_state, tree_method="hist", device="cuda",
            )),
        ]),
    }


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, class_labels) -> dict:
    m = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall"   : recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1"       : f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        yb = label_binarize(y_true, classes=class_labels)
        m["auc"] = roc_auc_score(yb, y_prob, average="macro", multi_class="ovr")
    except Exception:
        m["auc"] = np.nan
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_labels)
    return m


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_all_subjects(data_dir: Path) -> tuple[dict, list[str]]:
    files = sorted(data_dir.glob(FILE_PATTERN))
    print(f"Found {len(files)} subject files")
    subject_data: dict[str, pd.DataFrame] = {}
    for f in files:
        sid = f.stem.split("_")[0]
        df  = pd.read_parquet(f)
        if TARGET not in df.columns:
            print(f"  Skipping {sid}: '{TARGET}' not found")
            continue
        subject_data[sid] = df
    return subject_data, sorted(subject_data.keys())


# ─────────────────────────────────────────────
# CORE LOSO-CV RUNNER
# ─────────────────────────────────────────────
def run_loso_cv(
    subject_data : dict[str, pd.DataFrame],
    subject_ids  : list[str],
    feature_cols : list[str],
    le           : LabelEncoder,
    class_labels : np.ndarray,
    tag          : str = "",
    accumulate_importance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Run LOSO-CV and return:
        all_metrics_df, summary_df, group_cm, importance_dict
    """
    all_metrics : list[dict] = []
    group_cm    : dict[str, list] = defaultdict(list)
    importance  : dict[str, float] = defaultdict(float)

    models      = get_models()
    total_steps = len(subject_ids) * len(models)

    with tqdm(total=total_steps, desc=f"LOSO-CV [{tag}]") as pbar:
        for test_sub in subject_ids:
            # Prepare training data (all except test subject)
            train_dfs = []
            for s in subject_ids:
                if s != test_sub:
                    df_norm = subject_normalize(subject_data[s], feature_cols)
                    train_dfs.append(df_norm)
            train_df = pd.concat(train_dfs, ignore_index=True)
            
            # Normalize test subject
            test_df = subject_normalize(subject_data[test_sub], feature_cols)

            # Ensure delta cols exist in both if added
            shared_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

            X_train = train_df[shared_cols].values
            y_train = le.transform(train_df[TARGET].values)
            X_test  = test_df[shared_cols].values
            y_test  = le.transform(test_df[TARGET].values)

            for mname, model in models.items():
                clf = clone(model)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)

                scores = compute_metrics(y_test, y_pred, y_prob, class_labels)
                group_cm[mname].append(scores["confusion_matrix"])

                all_metrics.append({
                    "TestSubject": test_sub,
                    "Model"      : mname,
                    "Accuracy"   : scores["accuracy"],
                    "Precision"  : scores["precision"],
                    "Recall"     : scores["recall"],
                    "F1"         : scores["f1"],
                    "AUC"        : scores["auc"],
                })

                # Accumulate XGBoost importances
                if accumulate_importance and mname == "XGBoost":
                    if hasattr(clf.named_steps["clf"], "feature_importances_"):
                        imp = clf.named_steps["clf"].feature_importances_
                        for fname, fval in zip(shared_cols, imp):
                            importance[fname] += float(fval)

                pbar.update(1)
                pbar.set_postfix({"Sub": test_sub, "Model": mname})

    all_metrics_df = pd.DataFrame(all_metrics)
    summary_df = (
        all_metrics_df
        .groupby("Model")[["Accuracy", "Precision", "Recall", "F1", "AUC"]]
        .agg(["mean", "std"])
        .round(4)
    )
    return all_metrics_df, summary_df, dict(group_cm), dict(importance)


# ─────────────────────────────────────────────
# SHAP ANALYSIS
# ─────────────────────────────────────────────
def perform_shap_analysis(
    subject_data: dict[str, pd.DataFrame],
    subject_ids: list[str],
    feature_cols: list[str],
    le: LabelEncoder,
    class_labels: np.ndarray,
    save_dir: Path,
):
    """
    Train final XGBoost model on all data and generate SHAP plots.
    """
    print("\n[SHAP] Running SHAP analysis...")
    
    # Combine all subjects and normalize
    all_dfs = []
    for sid in subject_ids:
        df_norm = subject_normalize(subject_data[sid], feature_cols)
        all_dfs.append(df_norm)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    shared_cols = [c for c in feature_cols if c in full_df.columns]
    
    X = full_df[shared_cols]
    y = le.transform(full_df[TARGET])
    
    # -----------------------------------------
    # Train final interpretation model
    # -----------------------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        device="cuda",
    )
    
    model.fit(X, y)
    
    # -----------------------------------------
    # SHAP values
    # -----------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For multi-class, shap_values is a list of arrays
    # Convert to 3D array: (n_classes, n_samples, n_features)
    shap_values_3d = np.array(shap_values)
    
    # -----------------------------------------
    # SHAP Summary Plot (combined) - using absolute values
    # -----------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Compute mean absolute SHAP across all classes for summary
    mean_abs_shap = np.mean(np.abs(shap_values_3d), axis=0)
    
    shap.summary_plot(
        mean_abs_shap,
        X,
        show=False,
        max_display=25,
        feature_names=shared_cols,
    )
    
    plt.tight_layout()
    summary_path = save_dir / "shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {summary_path.name}")
    
    # -----------------------------------------
    # SHAP Bar Plot (combined)
    # -----------------------------------------
    plt.figure(figsize=(10, 7))
    
    shap.summary_plot(
        mean_abs_shap,
        X,
        plot_type="bar",
        show=False,
        max_display=25,
        feature_names=shared_cols,
    )
    
    plt.tight_layout()
    bar_path = save_dir / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {bar_path.name}")
    
    # -----------------------------------------
    # Class-specific SHAP plots
    # -----------------------------------------
    class_names = le.classes_
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            shap_values_3d[i],
            X,
            show=False,
            max_display=25,
            feature_names=shared_cols,
        )
        
        plt.title(f"SHAP for class: {class_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        class_path = save_dir / f"shap_summary_{class_name}.png"
        plt.savefig(class_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {class_path.name}")
    
    # -----------------------------------------
    # Save SHAP values to CSV
    # -----------------------------------------
    n_classes = len(class_names)
    n_samples = X.shape[0]
    n_features = len(shared_cols)
    
    # Compute mean absolute SHAP for each feature across all classes and samples
    mean_shap_abs = np.zeros(n_features)
    for class_idx in range(n_classes):
        mean_shap_abs += np.mean(np.abs(shap_values_3d[class_idx]), axis=0)
    mean_shap_abs /= n_classes
    
    shap_df = pd.DataFrame({
        "feature": shared_cols,
        "mean_abs_shap": mean_shap_abs,
        "type": [feature_type(f) for f in shared_cols],
        "band": [feature_band(f) for f in shared_cols],
        "roi": [feature_roi(f) for f in shared_cols],
        "window": [feature_window(f) for f in shared_cols],
    }).sort_values("mean_abs_shap", ascending=False)
    
    shap_df.to_csv(save_dir / "shap_values.csv", index=False)
    print(f"  Saved: shap_values.csv")
    
    # Save class-specific mean SHAP values
    class_shap_dfs = []
    for i, class_name in enumerate(class_names):
        class_df = pd.DataFrame({
            "feature": shared_cols,
            f"mean_shap_{class_name}": np.mean(shap_values_3d[i], axis=0),
            f"mean_abs_shap_{class_name}": np.mean(np.abs(shap_values_3d[i]), axis=0),
        })
        class_shap_dfs.append(class_df)
    
    # Merge all class-specific dataframes
    class_shap_combined = class_shap_dfs[0]
    for class_df in class_shap_dfs[1:]:
        class_shap_combined = class_shap_combined.merge(class_df, on="feature")
    
    class_shap_combined.to_csv(save_dir / "shap_class_specific.csv", index=False)
    print(f"  Saved: shap_class_specific.csv")
    
    return shap_values_3d, shap_df


# ─────────────────────────────────────────────
# AGGREGATE CONFUSION MATRICES
# ─────────────────────────────────────────────
def aggregate_cm(group_cm: dict) -> dict:
    return {k: np.sum(v, axis=0) for k, v in group_cm.items()}


# ─────────────────────────────────────────────
# ── PLOTTING FUNCTIONS ──────────────────────
# ─────────────────────────────────────────────

def plot_top_features_bar(
    importance_dict : dict[str, float],
    top_n           : int = 20,
    title           : str = "Top features by XGBoost gain (averaged across LOSO folds)",
    save_path       : Path | None = None,
):
    """
    Horizontal bar chart of top-N features coloured by feature type.
    """
    ranked  = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names   = [r[0] for r in ranked]
    scores  = [r[1] for r in ranked]
    types   = [feature_type(n) for n in names]
    colors  = [TYPE_COLORS.get(t, "#999") for t in types]

    # Shorten names for display
    def shorten(n):
        n = n.replace("early_", "E_").replace("late_", "L_").replace("delta_", "D_")
        n = n.replace("_abs_", ".abs.").replace("_rel_", ".rel.").replace("_ratio_", ".rt.")
        n = n.replace("_asym_abs_", ".asym.")
        return n

    short_names = [shorten(n) for n in names]

    fig, ax = plt.subplots(figsize=(12, top_n * 0.32 + 1.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.barh(range(top_n), scores[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.4, height=0.72)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(short_names[::-1], fontsize=8.5)
    ax.set_xlabel("Cumulative XGBoost gain (sum over folds)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.85, edgecolor=PALETTE["grid"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path.name}")
    plt.close()


def plot_importance_heatmap(
    importance_dict : dict[str, float],
    top_n           : int = 60,
    title           : str = "Feature importance — ROI × Band heatmap",
    save_path       : Path | None = None,
):
    """
    Heatmap: rows = band, columns = ROI (early abs only for clarity).
    One panel per window + feature type combination.
    """
    ranked = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for name, score in ranked:
        rows.append({
            "name"   : name,
            "score"  : score,
            "type"   : feature_type(name),
            "band"   : feature_band(name),
            "roi"    : feature_roi(name),
            "window" : feature_window(name),
        })
    df_feat = pd.DataFrame(rows)

    # One heatmap per (window, type) combo that has data
    combos = df_feat.groupby(["window", "type"]).size().reset_index()
    combos = combos[combos[0] >= 3]  # skip combos with too few features

    n_combos = len(combos)
    if n_combos == 0:
        print("  Not enough data for heatmap.")
        return

    cols_per_row = min(3, n_combos)
    n_rows = (n_combos + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(cols_per_row * 5, n_rows * 3.5 + 0.8),
                             squeeze=False)
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    rois  = ["CL1_lf", "CL2_lft", "CL3_lpo", "CL4_rf", "CL5_rft", "CL6_rpo"]

    for idx, row in combos.iterrows():
        ax_row = (list(combos.index).index(idx)) // cols_per_row
        ax_col = (list(combos.index).index(idx)) % cols_per_row
        ax = axes[ax_row][ax_col]

        subset = df_feat[(df_feat["window"] == row["window"]) & (df_feat["type"] == row["type"])]

        heat = pd.DataFrame(0.0, index=bands, columns=rois)
        for _, r in subset.iterrows():
            if r["band"] in heat.index and r["roi"] in heat.columns:
                heat.loc[r["band"], r["roi"]] = r["score"]

        sns.heatmap(
            heat, ax=ax,
            cmap="YlOrRd", linewidths=0.4, linecolor=PALETTE["grid"],
            annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{row['window']} · {row['type']}", fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0,  labelsize=8)

    # Hide unused axes
    total_axes = n_rows * cols_per_row
    for i in range(n_combos, total_axes):
        axes[i // cols_per_row][i % cols_per_row].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path.name}")
    plt.close()


def plot_subset_comparison(
    results : dict[str, dict],   # {label: {"acc": float, "auc": float}}
    save_path: Path | None = None,
):
    """Bar chart comparing accuracy + AUC across feature subsets."""
    labels  = list(results.keys())
    accs    = [results[l]["acc"] for l in labels]
    aucs    = [results[l]["auc"] for l in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    b1 = ax.bar(x - w/2, accs, w, label="Accuracy (XGBoost)",
                color=PALETTE["primary"], alpha=0.88)
    b2 = ax.bar(x + w/2, aucs, w, label="AUC (XGBoost)",
                color=PALETTE["secondary"], alpha=0.88)

    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("XGBoost performance across feature subsets", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path.name}")
    plt.close()


def plot_confusion_matrix(
    cm         : np.ndarray,
    class_names: list[str],
    title      : str,
    save_path  : Path | None = None,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(PALETTE["bg"])

    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        cmap="Blues", linewidths=0.5, linecolor=PALETTE["grid"],
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path.name}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("EEG PoemType Classification — Full Feature Selection Pipeline v2.0")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n[1] Loading subjects...")
    subject_data, subject_ids = load_all_subjects(INPUT_DIR)

    # Pool all training data to compute correlations (no leakage risk for
    # global correlation structure; we still do LOSO for metrics)
    all_df = pd.concat(subject_data.values(), ignore_index=True)

    # ── 2. Base feature columns ───────────────────────────────────────
    print("\n[2] Selecting base feature columns...")
    base_cols = get_feature_columns(all_df)
    print(f"  Total numeric columns before pruning: {len(base_cols)}")

    # ── 3. Correlation pruning ────────────────────────────────────────
    print("\n[3] Correlation pruning...")
    pruned_cols = prune_correlated_features(all_df, base_cols, threshold=CORR_THRESH)

    # ── 4. Label encoder ──────────────────────────────────────────────
    all_labels  = all_df[TARGET].values
    le          = LabelEncoder().fit(all_labels)
    class_labels = np.arange(len(le.classes_))
    print(f"\n  Classes: {le.classes_}")

    # ── 5. Stage-1 LOSO-CV (pruned features, accumulate importances) ──
    print("\n[4] Stage-1 LOSO-CV on pruned features (accumulating XGB importances)...")
    m_df1, sum1, cm1, importance = run_loso_cv(
        subject_data, subject_ids, pruned_cols, le, class_labels,
        tag="pruned", accumulate_importance=True,
    )

    print("\n── Stage-1 Summary ──")
    print(sum1)

    # Save stage-1 results
    m_df1.to_csv(OUTPUT_DIR / "stage1_subject_metrics.csv", index=False)
    sum1.to_csv(OUTPUT_DIR / "stage1_summary.csv")

    # ── 6. Plot: top-20 feature importance bar chart ──────────────────
    print("\n[5] Plotting top-20 feature importance...")
    plot_top_features_bar(
        importance, top_n=20,
        title="Top-20 features — XGBoost gain (stage 1, all pruned features)",
        save_path=PLOT_DIR / "top20_features_bar.png",
    )
    plot_importance_heatmap(
        importance, top_n=60,
        title="Feature importance heatmap — ROI × Band (stage 1)",
        save_path=PLOT_DIR / "importance_heatmap_stage1.png",
    )

    # ── 7. Top-N subset search ────────────────────────────────────────
    print("\n[6] Searching over top-N feature subsets...")
    subset_results: dict[str, dict] = {}

    # Baseline from stage 1
    xgb_s1 = sum1.loc["XGBoost"]
    subset_results["all_pruned"] = {
        "acc": xgb_s1["Accuracy"]["mean"],
        "auc": xgb_s1["AUC"]["mean"],
        "cols": pruned_cols,
    }

    for n in TOP_N_CANDIDATES:
        top_cols = get_top_features(importance, n)
        _, sum_n, _, _ = run_loso_cv(
            subject_data, subject_ids, top_cols, le, class_labels,
            tag=f"top{n}",
        )
        xgb_n = sum_n.loc["XGBoost"]
        subset_results[f"top_{n}"] = {
            "acc" : xgb_n["Accuracy"]["mean"],
            "auc" : xgb_n["AUC"]["mean"],
            "cols": top_cols,
        }
        print(f"  top-{n:>3}: acc={xgb_n['Accuracy']['mean']:.4f}  "
              f"auc={xgb_n['AUC']['mean']:.4f}")

    # Pick best subset by AUC
    best_key = max(subset_results, key=lambda k: subset_results[k]["auc"])
    best_cols = subset_results[best_key]["cols"]
    print(f"\n  Best subset: {best_key}  "
          f"(auc={subset_results[best_key]['auc']:.4f}, "
          f"n={len(best_cols)} features)")

    # ── 8. Add temporal-delta features on best subset ─────────────────
    print("\n[7] Adding temporal-delta features to best subset...")
    subject_data_delta: dict[str, pd.DataFrame] = {}
    for sid, df in subject_data.items():
        df_aug, delta_cols = add_temporal_delta(df, best_cols)
        subject_data_delta[sid] = df_aug

    delta_feature_cols = best_cols + delta_cols
    print(f"  Feature count with delta: {len(delta_feature_cols)}")

    _, sum_delta, cm_delta, imp_delta = run_loso_cv(
        subject_data_delta, subject_ids, delta_feature_cols, le, class_labels,
        tag="best+delta", accumulate_importance=True,
    )

    xgb_delta = sum_delta.loc["XGBoost"]
    subset_results["best+delta"] = {
        "acc" : xgb_delta["Accuracy"]["mean"],
        "auc" : xgb_delta["AUC"]["mean"],
        "cols": delta_feature_cols,
    }

    print("\n── Final Summary (best subset + delta) ──")
    print(sum_delta)

    sum_delta.to_csv(OUTPUT_DIR / "final_summary.csv")

    # ── 9. SHAP Analysis ──────────────────────────────────────────────
    print("\n[8] Performing SHAP analysis...")
    perform_shap_analysis(
        subject_data_delta, subject_ids, delta_feature_cols,
        le, class_labels, PLOT_DIR
    )

    # ── 10. Plots: subset comparison, final top-20, final heatmap ─────
    print("\n[9] Generating final plots...")

    plot_subset_comparison(
        {k: v for k, v in subset_results.items()},
        save_path=PLOT_DIR / "subset_comparison.png",
    )

    plot_top_features_bar(
        imp_delta, top_n=20,
        title="Top-20 features — XGBoost gain (final: best subset + delta)",
        save_path=PLOT_DIR / "top20_features_final.png",
    )

    plot_importance_heatmap(
        imp_delta, top_n=60,
        title="Feature importance heatmap — ROI × Band (final)",
        save_path=PLOT_DIR / "importance_heatmap_final.png",
    )

    # Confusion matrices for final run
    agg_cm_final = aggregate_cm(cm_delta)
    for mname, cm in agg_cm_final.items():
        plot_confusion_matrix(
            cm, list(le.classes_),
            title=f"Confusion matrix — {mname} (final)",
            save_path=PLOT_DIR / f"cm_final_{mname}.png",
        )

    # ── 11. Save feature list & importance ────────────────────────────
    print("\n[10] Saving final feature list and importances...")
    final_top = get_top_features(imp_delta, len(delta_feature_cols))
    pd.DataFrame({
        "feature"   : final_top,
        "importance": [imp_delta.get(f, 0.0) for f in final_top],
        "type"      : [feature_type(f) for f in final_top],
        "band"      : [feature_band(f) for f in final_top],
        "roi"       : [feature_roi(f)  for f in final_top],
        "window"    : [feature_window(f) for f in final_top],
    }).to_csv(OUTPUT_DIR / "final_feature_importances.csv", index=False)

    json_path = OUTPUT_DIR / "best_feature_cols.json"
    json_path.write_text(json.dumps(delta_feature_cols, indent=2))

    # ── 12. Final console report ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nBest feature subset : {best_key}  ({len(best_cols)} features)")
    print(f"After delta features: {len(delta_feature_cols)} features\n")
    print("Final XGBoost performance:")
    print(f"  Accuracy  : {xgb_delta['Accuracy']['mean']:.4f} ± {xgb_delta['Accuracy']['std']:.4f}")
    print(f"  F1 (macro): {xgb_delta['F1']['mean']:.4f} ± {xgb_delta['F1']['std']:.4f}")
    print(f"  AUC       : {xgb_delta['AUC']['mean']:.4f} ± {xgb_delta['AUC']['std']:.4f}")
    
    # Print Random Forest performance
    rf_delta = sum_delta.loc["RandomForest"]
    print("\nFinal Random Forest performance:")
    print(f"  Accuracy  : {rf_delta['Accuracy']['mean']:.4f} ± {rf_delta['Accuracy']['std']:.4f}")
    print(f"  F1 (macro): {rf_delta['F1']['mean']:.4f} ± {rf_delta['F1']['std']:.4f}")
    print(f"  AUC       : {rf_delta['AUC']['mean']:.4f} ± {rf_delta['AUC']['std']:.4f}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Plots saved to  : {PLOT_DIR}")
    print("\nFiles:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()