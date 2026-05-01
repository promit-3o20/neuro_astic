"""
PoemType Classification (Within-Subject CV)
==========================================
Models
------
1. Logistic Regression (Elastic Net)
2. SVM (RBF)
3. XGBoost

Task
----
Predict PoemType using ROI band-power features (60 ROI features).

Evaluation
----------
Within-subject stratified CV for each subject independently, then aggregate:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- ROC-AUC (OvR macro)
- Confusion Matrix

Expected input
--------------
A single parquet file containing:
- 60 ROI feature columns
- target column: 'PoemType'
- optional metadata columns (Block, ratings, etc.)

Author : Pramit Biswas
Version: v0.0
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

INPUT_DIR = DATA_DIR/ "features/roi_ftrs"  # input folder (47 subject parquet files)
OUTPUT_DIR = BASE_DIR/ "../../results/ml"             # output folder

TARGET = "PoemType"
N_SPLITS = 5
RANDOM_STATE = 42
FILE_PATTERN = "*_roi_bpfeatures.parquet"

# create output folder if not present
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "PoemType"
N_SPLITS = 5
RANDOM_STATE = 42

# Optional: if filenames are like sub-001_roi_bpfeatures.parquet
FILE_PATTERN = "*_roi_bpfeatures.parquet"


# ---------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame, target: str = "PoemType") -> list[str]:
    """
    Select only ROI bandpower features.

    Excludes metadata / labels / behavioral columns.
    Assumes ROI features are numeric and target/meta columns are not used.
    """
    exclude_cols = {
        target,
        "trial_index",
        "Block",
        "AA",
        "Imagery",
        "Moved",
        "Originality",
        "Creativity",
        "Subject",
        "subject",
        "sub",
    }

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    return feature_cols


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
def get_models(random_state: int = 42) -> dict:
    """
    Return classification models.
    """
    models = {
        "LR_ElasticNet": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                multi_class="multinomial",
                random_state=random_state,
            )),
        ]),

        "SVM_RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }

    return models


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_labels: np.ndarray,
) -> dict:
    """
    Compute classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        y_true_bin = label_binarize(y_true, classes=class_labels)
        metrics["auc"] = roc_auc_score(
            y_true_bin,
            y_prob,
            average="macro",
            multi_class="ovr"
        )
    except Exception:
        metrics["auc"] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels
    )

    return metrics


# ---------------------------------------------------------------------
# WITHIN-SUBJECT CV
# ---------------------------------------------------------------------
def run_within_subject_cv(
    df: pd.DataFrame,
    subject_id: str,
    target: str = "PoemType",
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Run within-subject CV for one subject.

    Returns
    -------
    metrics_df : fold-level metrics for all models
    cm_store   : confusion matrices per model
    """
    df = df.copy()

    feature_cols = get_feature_columns(df, target=target)
    X = df[feature_cols].values
    y_raw = df[target].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_labels = np.arange(len(le.classes_))

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    models = get_models(random_state=random_state)

    rows = []
    cm_store = {name: [] for name in models.keys()}

    for model_name, model in models.items():
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[tr_idx], X[te_idx]
            y_train, y_test = y[tr_idx], y[te_idx]

            clf = clone(model)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            scores = compute_metrics(y_test, y_pred, y_prob, class_labels)
            cm_store[model_name].append(scores["confusion_matrix"])

            rows.append({
                "Subject": subject_id,
                "Model": model_name,
                "Fold": fold,
                "Accuracy": scores["accuracy"],
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1": scores["f1"],
                "AUC": scores["auc"],
            })

    metrics_df = pd.DataFrame(rows)
    return metrics_df, cm_store


# ---------------------------------------------------------------------
# ALL SUBJECTS
# ---------------------------------------------------------------------
def run_group_analysis(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run within-subject CV for all subjects and aggregate results.
    """
    all_metrics = []
    group_cm = defaultdict(list)

    files = sorted(data_dir.glob(FILE_PATTERN))
    print(f"Found {len(files)} subject files")

    for file in files:
        subject_id = file.stem.split("_")[0]   # e.g., sub-021
        print(f"Processing {subject_id} ...")

        df = pd.read_parquet(file)

        if TARGET not in df.columns:
            print(f"Skipping {subject_id}: '{TARGET}' not found")
            continue

        metrics_df, cm_store = run_within_subject_cv(
            df=df,
            subject_id=subject_id,
            target=TARGET,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )

        all_metrics.append(metrics_df)

        for model_name, cms in cm_store.items():
            group_cm[model_name].extend(cms)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)

    summary_df = (
        all_metrics_df
        .groupby("Model")[["Accuracy", "Precision", "Recall", "F1", "AUC"]]
        .agg(["mean", "std"])
        .round(4)
    )

    return all_metrics_df, summary_df, group_cm


# ---------------------------------------------------------------------
# CONFUSION MATRIX AGGREGATION
# ---------------------------------------------------------------------
def aggregate_confusion_matrices(group_cm: dict) -> dict:
    """
    Sum fold-level confusion matrices across all subjects.
    """
    agg_cm = {}
    for model_name, cms in group_cm.items():
        agg_cm[model_name] = np.sum(cms, axis=0)
    return agg_cm


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    all_metrics_df, summary_df, group_cm = run_group_analysis(INPUT_DIR)
    agg_cm = aggregate_confusion_matrices(group_cm)

    print("\n" + "=" * 80)
    print("WITHIN-SUBJECT CV SUMMARY (47 Subjects)")
    print("=" * 80)
    print(summary_df)

    print("\n" + "=" * 80)
    print("AGGREGATED CONFUSION MATRICES")
    print("=" * 80)
    for model_name, cm in agg_cm.items():
        print(f"\n{model_name}")
        print(cm)

    # Save outputs
    all_metrics_df.to_csv(OUTPUT_DIR / "poemtype_within_subject_fold_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "poemtype_within_subject_summary.csv")

    for model_name, cm in agg_cm.items():
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_{model_name}.csv", index=False)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'poemtype_within_subject_fold_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'poemtype_within_subject_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix_<model>.csv'}")