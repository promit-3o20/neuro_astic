"""
PoemType Classification (LOSO-CV)

Author: Pramit
Version: v0.0
================================
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
Leave-One-Subject-Out Cross-Validation (LOSO-CV):
- Train on all subjects except one
- Test on held-out subject
- Repeat for all subjects, then aggregate:
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1-score (macro)
    - ROC-AUC (OvR macro)
    - Confusion Matrix

Expected input
--------------
47 subject parquet files, each containing:
- 60 ROI feature columns
- target column: 'PoemType'
- optional metadata columns

Author : Pramit Biswas
Version: v0.1
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
from tqdm import tqdm


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

INPUT_DIR = DATA_DIR / "features/roi_ftrs"
OUTPUT_DIR = BASE_DIR / "../../results/ml_loso"

TARGET = "PoemType"
RANDOM_STATE = 42
FILE_PATTERN = "*_roi_bpfeatures.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------
def get_feature_columns(
    df: pd.DataFrame,
    target: str = "PoemType",
    time_window: str | None = None,   # "early", "late", or None
) -> list[str]:
    exclude_cols = {
        target,
        "trial_index",
        "Block",
        "AA",
        "Imagery",
        "Moved",
        "Originality",
        # "Creativity",
        "Subject",
        "subject",
        "sub",
    }

    cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if time_window is not None:
        time_window = time_window.lower()
        cols = [c for c in cols if c.lower().startswith(f"{time_window}_")]

    return cols


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
def get_models(random_state: int = 42) -> dict:
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
                tree_method="hist",
                device="cuda",
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
        y_true, y_pred, labels=class_labels
    )

    return metrics


# ---------------------------------------------------------------------
# LOAD ALL SUBJECT FILES
# ---------------------------------------------------------------------
def load_all_subjects(data_dir: Path) -> tuple[dict, list[str]]:
    subject_data = {}
    files = sorted(data_dir.glob(FILE_PATTERN))

    print(f"Found {len(files)} subject files")

    for file in files:
        subject_id = file.stem.split("_")[0]
        df = pd.read_parquet(file)

        if TARGET not in df.columns:
            print(f"Skipping {subject_id}: '{TARGET}' not found")
            continue

        subject_data[subject_id] = df

    return subject_data, sorted(subject_data.keys())


# ---------------------------------------------------------------------
# LOSO-CV
# ---------------------------------------------------------------------
def run_loso_cv(data_dir: Path):
    subject_data, subject_ids = load_all_subjects(data_dir)

    all_metrics = []
    group_cm = defaultdict(list)

    first_df = next(iter(subject_data.values()))
    feature_cols = get_feature_columns(first_df, target=TARGET, time_window="early")
    # print(feature_cols)
    # Global label encoder across all subjects
    all_labels = pd.concat([df[TARGET] for df in subject_data.values()], axis=0).values
    le = LabelEncoder()
    le.fit(all_labels)
    class_labels = np.arange(len(le.classes_))

    models = get_models(random_state=RANDOM_STATE)

    total_steps = len(subject_ids) * len(models)

    with tqdm(total=total_steps, desc="Running LOSO-CV") as pbar:

        for test_subject in subject_ids:

            train_dfs = [subject_data[s] for s in subject_ids if s != test_subject]
            test_df = subject_data[test_subject]

            train_df = pd.concat(train_dfs, ignore_index=True)

            X_train = train_df[feature_cols].values
            y_train = le.transform(train_df[TARGET].values)

            X_test = test_df[feature_cols].values
            y_test = le.transform(test_df[TARGET].values)

            for model_name, model in models.items():
                clf = clone(model)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)

                scores = compute_metrics(y_test, y_pred, y_prob, class_labels)
                group_cm[model_name].append(scores["confusion_matrix"])

                all_metrics.append({
                    "TestSubject": test_subject,
                    "Model": model_name,
                    "Accuracy": scores["accuracy"],
                    "Precision": scores["precision"],
                    "Recall": scores["recall"],
                    "F1": scores["f1"],
                    "AUC": scores["auc"],
                })

                pbar.update(1)
                pbar.set_postfix({
                    "Subject": test_subject,
                    "Model": model_name
                })

    all_metrics_df = pd.DataFrame(all_metrics)

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
    agg_cm = {}
    for model_name, cms in group_cm.items():
        agg_cm[model_name] = np.sum(cms, axis=0)
    return agg_cm


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    all_metrics_df, summary_df, group_cm = run_loso_cv(INPUT_DIR)
    agg_cm = aggregate_confusion_matrices(group_cm)

    print("\n" + "=" * 80)
    print("LOSO-CV SUMMARY")
    print("=" * 80)
    print(summary_df)

    print("\n" + "=" * 80)
    print("AGGREGATED CONFUSION MATRICES")
    print("=" * 80)
    for model_name, cm in agg_cm.items():
        print(f"\n{model_name}")
        print(cm)

    # Save outputs
    all_metrics_df.to_csv(OUTPUT_DIR / "poemtype_loso_subject_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "poemtype_loso_summary.csv")

    for model_name, cm in agg_cm.items():
        pd.DataFrame(cm).to_csv(
            OUTPUT_DIR / f"confusion_matrix_{model_name}.csv",
            index=False
        )

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'poemtype_loso_subject_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'poemtype_loso_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix_<model>.csv'}")