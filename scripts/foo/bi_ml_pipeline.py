"""
EEG PoemType Classification — Binary Classification Pipeline
============================================================

Binary Labels:
    0: Control (C)
    1: Poetry (Haiku + Senryu)
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
import shap

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = (BASE_DIR / "../../data").resolve()
INPUT_DIR  = DATA_DIR / "features/roi_ftrs2"
OUTPUT_DIR = BASE_DIR / "../../results/ml_pipeline_binary1"
PLOT_DIR   = OUTPUT_DIR / "plots"

TARGET        = "PoemType"
RANDOM_STATE  = 42
FILE_PATTERN  = "*_roi_bpfeatures.parquet"
CORR_THRESH   = 0.90
TOP_N_CANDIDATES = [30, 100]

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

# Feature-type colour map
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
    # "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
})


# ─────────────────────────────────────────────
# BINARY LABEL MAPPING
# ─────────────────────────────────────────────
def convert_to_binary(original_labels: pd.Series) -> pd.Series:
    """
    Convert multi-class poem types to binary labels.
    
    Mapping:
        0: Control (C)
        1: Poetry (Haiku + Senryu)
    """
    poetry_poems = {"H", "S"}
    return original_labels.apply(lambda x: 1 if x in poetry_poems else 0)


def get_class_names():
    """Return class names for binary classification."""
    return ["Control (C)", "Poetry (Haiku/Senryu)"]


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
# _EXCLUDE = {
#     TARGET, "trial_index", "Block", "AA", "Imagery", "Creativity",
#     "Moved", "Originality", "subject",
# }
_EXCLUDE = {
    TARGET, "trial_index", "Block", "AA", "Imagery","Moved", "Originality", "subject",
}

def get_feature_columns(df: pd.DataFrame, extra_exclude: set | None = None) -> list[str]:
    exclude = _EXCLUDE | (extra_exclude or set())
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# ─────────────────────────────────────────────
# SIMPLIFIED MODELS
# ─────────────────────────────────────────────
def get_models(random_state: int = RANDOM_STATE) -> dict:
    return {
        "ElasticNet": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",  # Elastic net requires saga
                l1_ratio=0.5,  # 0.5 = equal mix; 1.0 = pure L1; 0 = pure L2
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        # "SVM_Linear": Pipeline([
        #     ("imputer", SimpleImputer(strategy="median")),
        #     ("scaler",  StandardScaler()),
        #     ("clf",     SVC(
        #         kernel="linear", C=1.0,
        #         probability=True, class_weight="balanced",
        #         random_state=random_state,
        #     )),
        # ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                random_state=random_state, tree_method="hist", device="cuda",
            )),
        ]),
    }


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    m = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        m["auc"] = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        m["auc"] = np.nan
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return m


# ─────────────────────────────────────────────
# DATA LOADING AND CONVERSION
# ─────────────────────────────────────────────
def load_and_convert_subjects(data_dir: Path) -> tuple[dict, list[str]]:
    """Load subjects and convert to binary labels."""
    files = sorted(data_dir.glob(FILE_PATTERN))
    print(f"Found {len(files)} subject files")
    subject_data: dict[str, pd.DataFrame] = {}
    
    for f in files:
        sid = f.stem.split("_")[0]
        df  = pd.read_parquet(f)
        
        if TARGET not in df.columns:
            print(f"  Skipping {sid}: '{TARGET}' not found")
            continue
        
        # Print original class distribution
        print(f"\n  {sid} - Original classes:")
        original_counts = df[TARGET].value_counts()
        for cls, count in original_counts.items():
            print(f"      {cls}: {count}")
        
        # Convert to binary labels
        df["BinaryLabel"] = convert_to_binary(df[TARGET])
        
        # Print binary class distribution
        binary_counts = df["BinaryLabel"].value_counts()
        print(f"  {sid} - Binary classes (0=Control, 1=Poetry):")
        print(f"      Control (0): {binary_counts.get(0, 0)}")
        print(f"      Poetry (1): {binary_counts.get(1, 0)}")
        
        subject_data[sid] = df
    
    # Verify we have both classes across subjects
    if not subject_data:
        raise ValueError("No valid subject data loaded")
    
    all_binary = pd.concat([df["BinaryLabel"] for df in subject_data.values()])
    unique_classes = all_binary.unique()
    
    if len(unique_classes) < 2:
        print(f"\nWARNING: Only one class found in data: {unique_classes}")
        print("Check if your data contains both Control (C) and Poetry (Haiku/Senryu) samples")
        
        # Show which subjects have which classes
        for sid, df in subject_data.items():
            classes_present = df["BinaryLabel"].unique()
            print(f"  {sid}: classes {classes_present}")
        
        raise ValueError(f"Binary conversion failed: only one class present in data. "
                        f"Found classes: {unique_classes}. "
                        f"Expected both 0 (Control) and 1 (Poetry).")
    
    print(f"\nTotal samples: {len(all_binary)}")
    print(f"  Control (0): {(all_binary == 0).sum()}")
    print(f"  Poetry (1): {(all_binary == 1).sum()}")
    
    return subject_data, sorted(subject_data.keys())


# ─────────────────────────────────────────────
# CORE LOSO-CV RUNNER
# ─────────────────────────────────────────────
def run_loso_cv(
    subject_data : dict[str, pd.DataFrame],
    subject_ids  : list[str],
    feature_cols : list[str],
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
            train_df = pd.concat(
                [subject_data[s] for s in subject_ids if s != test_sub],
                ignore_index=True,
            )
            test_df = subject_data[test_sub]

            # Ensure columns exist in both
            shared_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

            X_train = train_df[shared_cols].values
            y_train = train_df["BinaryLabel"].values
            X_test  = test_df[shared_cols].values
            y_test  = test_df["BinaryLabel"].values

            for mname, model in models.items():
                clf = clone(model)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)

                scores = compute_metrics(y_test, y_pred, y_prob)
                group_cm[mname].append(scores["confusion_matrix"])

                all_metrics.append({
                    "TestSubject": test_sub,
                    "Model"      : mname,
                    "Accuracy"   : scores["accuracy"],
                    "BalancedAccuracy": scores["balanced_accuracy"],
                    "Precision"  : scores["precision"],
                    "Recall"     : scores["recall"],
                    "F1"         : scores["f1"],
                    "AUC"        : scores["auc"],
                })

                # Accumulate XGBoost importances
                if accumulate_importance and mname == "XGBoost":
                    imp = clf.named_steps["clf"].feature_importances_
                    for fname, fval in zip(shared_cols, imp):
                        importance[fname] += float(fval)

                pbar.update(1)
                pbar.set_postfix({"Sub": test_sub, "Model": mname})

    all_metrics_df = pd.DataFrame(all_metrics)
    summary_df = (
        all_metrics_df
        .groupby("Model")[[
            "Accuracy",
            "BalancedAccuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC"
        ]]
        .agg(["mean", "std"])
        .round(4)
    )
    return all_metrics_df, summary_df, dict(group_cm), dict(importance)


# ─────────────────────────────────────────────
# AGGREGATE CONFUSION MATRICES
# ─────────────────────────────────────────────
def aggregate_cm(group_cm: dict) -> dict:
    return {k: np.sum(v, axis=0) for k, v in group_cm.items()}


# ─────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────
def plot_top_features_bar(
    importance_dict : dict[str, float],
    top_n           : int = 40,
    title           : str = "Top features by XGBoost gain (averaged across LOSO folds)",
    save_path       : Path | None = None,
):
    """Horizontal bar chart of top-N features coloured by feature type."""
    if not importance_dict:
        print("  Warning: No importance scores to plot")
        return
        
    ranked  = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names   = [r[0] for r in ranked]
    scores  = [r[1] for r in ranked]
    types   = [feature_type(n) for n in names]
    colors  = [TYPE_COLORS.get(t, "#999") for t in types]

    def shorten(n):
        n = n.replace("early_", "E_").replace("late_", "L_").replace("delta_", "D_")
        n = n.replace("_abs_", ".abs.").replace("_rel_", ".rel.").replace("_ratio_", ".rt.")
        n = n.replace("_asym_abs_", ".asym.")
        return n

    short_names = [shorten(n) for n in names]

    fig, ax = plt.subplots(figsize=(12, min(top_n, len(names)) * 0.32 + 1.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.barh(range(len(scores)), scores[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.4, height=0.72)

    ax.set_yticks(range(len(scores)))
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
    title           : str = "Feature importance — ROI × Band heatmap (Binary Classification)",
    save_path       : Path | None = None,
):
    """Heatmap: rows = band, columns = ROI."""
    if not importance_dict:
        print("  Warning: No importance scores for heatmap")
        return
        
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

    # One heatmap per (window, type) combo
    combos = df_feat.groupby(["window", "type"]).size().reset_index()
    combos = combos[combos[0] >= 3]

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
    results : dict[str, dict],
    save_path: Path | None = None,
):
    """Bar chart comparing metrics across feature subsets."""
    if not results:
        print("  Warning: No results to plot")
        return
        
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
    ax.set_title("XGBoost performance across feature subsets (Binary Classification)", 
                 fontsize=12, fontweight="bold")
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
    """Plot confusion matrix for binary classification."""
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

def run_shap_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    save_dir: Path,
):
    """
    Train final XGBoost model on all data and generate SHAP plots.
    """

    print("\n[SHAP] Running SHAP analysis...")

    X = df[feature_cols]
    y = df["BinaryLabel"]

    # -----------------------------------------
    # Train final interpretation model
    # -----------------------------------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
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

    # -----------------------------------------
    # SHAP Summary Plot
    # -----------------------------------------
    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        max_display=25,
    )

    plt.tight_layout()

    summary_path = save_dir / "shap_summary.png"

    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {summary_path.name}")

    # -----------------------------------------
    # SHAP Bar Plot
    # -----------------------------------------
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False,
        max_display=25,
    )

    plt.tight_layout()

    bar_path = save_dir / "shap_bar.png"

    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {bar_path.name}")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("EEG PoemType Classification — Binary Classification Pipeline")
    print("Binary Labels: 0 = Control (C), 1 = Poetry (Haiku/Senryu)")
    print("=" * 70)

    # ── 1. Load and convert data ──────────────────────────────────────
    print("\n[1] Loading subjects and converting to binary labels...")
    
    try:
        subject_data, subject_ids = load_and_convert_subjects(INPUT_DIR)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure your data contains both Control (C) and Poetry (Haiku/Senryu) samples.")
        print("Check the column names in your parquet files.")
        return

    # Pool all training data for correlation analysis
    all_df = pd.concat(subject_data.values(), ignore_index=True)

    # ── 2. Base feature columns ───────────────────────────────────────
    print("\n[2] Selecting base feature columns...")
    base_cols = get_feature_columns(all_df, extra_exclude={"BinaryLabel"})
    print(f"  Total numeric columns: {len(base_cols)}")

    # ── 3. Correlation pruning ────────────────────────────────────────
    print("\n[3] Correlation pruning...")
    pruned_cols = prune_correlated_features(all_df, base_cols, threshold=CORR_THRESH)

    # ── 4. Stage-1 LOSO-CV (pruned features) ──────────────────────────
    print("\n[4] Stage-1 LOSO-CV on pruned features...")
    m_df1, sum1, cm1, importance = run_loso_cv(
        subject_data, subject_ids, pruned_cols,
        tag="pruned", accumulate_importance=True,
    )

    print("\n── Stage-1 Summary ──")
    print(sum1)

    # Save stage-1 results
    m_df1.to_csv(OUTPUT_DIR / "stage1_subject_metrics.csv", index=False)
    sum1.to_csv(OUTPUT_DIR / "stage1_summary.csv")

    # ── 5. Plot top feature importance ────────────────────────────────
    print("\n[5] Plotting feature importance...")
    if importance:
        plot_top_features_bar(
            importance, top_n=40,
            title="Top-40 features — XGBoost gain (stage 1, binary classification)",
            save_path=PLOT_DIR / "top40_features_bar.png",
        )
        plot_importance_heatmap(
            importance, top_n=60,
            title="Feature importance heatmap — ROI × Band (binary classification)",
            save_path=PLOT_DIR / "importance_heatmap_stage1.png",
        )
    else:
        print("  No importance scores to plot")

    # ── 6. Top-N subset search ────────────────────────────────────────
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
        if importance:
            top_cols = get_top_features(importance, n)
            _, sum_n, _, _ = run_loso_cv(
                subject_data, subject_ids, top_cols, tag=f"top{n}",
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

    # ── 7. Add temporal-delta features ────────────────────────────────
    print("\n[7] Adding temporal-delta features to best subset...")
    subject_data_delta: dict[str, pd.DataFrame] = {}
    for sid, df in subject_data.items():
        df_aug, delta_cols = add_temporal_delta(df, best_cols)
        subject_data_delta[sid] = df_aug

    delta_feature_cols = best_cols + delta_cols
    print(f"  Feature count with delta: {len(delta_feature_cols)}")

    _, sum_delta, cm_delta, imp_delta = run_loso_cv(
        subject_data_delta, subject_ids, delta_feature_cols,
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

    # ── 8. Generate final plots ───────────────────────────────────────
    print("\n[8] Generating final plots...")

    plot_subset_comparison(
        {k: v for k, v in subset_results.items()},
        save_path=PLOT_DIR / "subset_comparison.png",
    )

    if imp_delta:
        plot_top_features_bar(
            imp_delta, top_n=40,
            title="Top-40 features — XGBoost gain (final: best subset + delta, binary)",
            save_path=PLOT_DIR / "top40_features_final.png",
        )

        plot_importance_heatmap(
            imp_delta, top_n=60,
            title="Feature importance heatmap — ROI × Band (final, binary)",
            save_path=PLOT_DIR / "importance_heatmap_final.png",
        )

    # Confusion matrices for final run
    agg_cm_final = aggregate_cm(cm_delta)
    class_names = get_class_names()
    for mname, cm in agg_cm_final.items():
        plot_confusion_matrix(
            cm, class_names,
            title=f"Confusion matrix — {mname} (final, binary)",
            save_path=PLOT_DIR / f"cm_final_{mname}.png",
        )

    # ── SHAP ANALYSIS ────────────────────────────────────────────────
    run_shap_analysis(
        all_df,
        pruned_cols,
        PLOT_DIR,
    )

    # ── 9. Save feature list and importances ──────────────────────────
    print("\n[9] Saving final feature list and importances...")
    if imp_delta:
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

    # ── 10. Final report ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nBest feature subset : {best_key}  ({len(best_cols)} features)")
    print(f"After delta features: {len(delta_feature_cols)} features\n")
    print("Final XGBoost performance (Binary Classification):")
    print(f"  Accuracy  : {xgb_delta['Accuracy']['mean']:.4f} ± {xgb_delta['Accuracy']['std']:.4f}")
    print(f"  Precision : {xgb_delta['Precision']['mean']:.4f} ± {xgb_delta['Precision']['std']:.4f}")
    print(f"  Recall    : {xgb_delta['Recall']['mean']:.4f} ± {xgb_delta['Recall']['std']:.4f}")
    print(f"  F1        : {xgb_delta['F1']['mean']:.4f} ± {xgb_delta['F1']['std']:.4f}")
    print(f"  AUC       : {xgb_delta['AUC']['mean']:.4f} ± {xgb_delta['AUC']['std']:.4f}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Plots saved to  : {PLOT_DIR}")
    print("\nFiles:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()