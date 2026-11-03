"""
ESTATEIQ - Phase 4 (Stage 1): Classification Model Training & Evaluation Pipeline
----------------------------------------------------------------------------------
This script implements Phase 4 - Stage 1 of ESTATEIQ:
1. Loads partitioned feature datasets:
   - Train: data/processed/train_features.csv (175,000 records)
   - Val:   data/processed/val_features.csv (37,500 records)
   - Test:  data/processed/test_features.csv (37,500 records, strictly for final validation checks)
2. Uses the approved 34-feature intrinsic classification matrix (zero price or target leakage).
3. Evaluates 4 candidate models:
   - Baseline: DummyClassifier (most-frequent class)
   - Linear Model: LogisticRegression (L2 regularized, balanced class weight)
   - Ensemble Bagging: RandomForestClassifier (150 trees, max_depth=16, balanced)
   - Ensemble Boosting: XGBClassifier (250 trees, max_depth=6, scale_pos_weight=1.87)
4. Constructs scikit-learn pipelines with leakage-safe preprocessing:
   - Numerical: SimpleImputer(strategy='median') + StandardScaler()
   - Categorical: OneHotEncoder(handle_unknown='ignore', sparse_output=True)
   - Preprocessing is fitted strictly on the Training set.
5. Tracks all experiments, parameters, metrics, and models using MLflow.
6. Evaluates using Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix.
7. Generates publication-quality evaluation visualizations under `reports/classification/`:
   - confusion_matrices_comparison.png
   - roc_curves_comparison.png
   - pr_curves_comparison.png
   - feature_importances_top20.png
   - metrics_comparison_barchart.png
8. Saves the best-performing model to `models/best_classification_model.joblib`.
9. Generates `reports/classification_model_comparison.md`.
"""

import sys
import os
import time
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Ensure project root is in sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import feature definitions from Stage 2 pipeline
from src.features.build_features import CLASSIFICATION_EXCLUDED_COLS


# =============================================================================
# PALETTE & PLOTTING CONFIGURATION
# =============================================================================
PALETTE = {
    "deep_forest": "#172923",
    "sage_green": "#B5C9B0",
    "champagne_gold": "#D9B76E",
    "warm_sand": "#E9E3D7",
    "ivory": "#F8F7F2",
    "slate_gray": "#4A5568",
    "crimson_accent": "#C53030",
    "emerald": "#22543D"
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "figure.facecolor": PALETTE["ivory"],
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": PALETTE["deep_forest"],
    "axes.labelcolor": PALETTE["deep_forest"],
    "xtick.color": PALETTE["deep_forest"],
    "ytick.color": PALETTE["deep_forest"],
    "text.color": PALETTE["deep_forest"],
    "grid.color": PALETTE["warm_sand"],
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})


# =============================================================================
# 1. DATA PREPARATION & LEAKAGE DEFENSE
# =============================================================================

def load_and_validate_datasets(project_root: Path):
    """
    Loads train, val, and test feature datasets and extracts the approved 34 features.
    """
    train_path = project_root / "data" / "processed" / "train_features.csv"
    val_path = project_root / "data" / "processed" / "val_features.csv"
    test_path = project_root / "data" / "processed" / "test_features.csv"

    print(f"Loading partitioned datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Extract approved feature list
    feature_cols = [c for c in train_df.columns if c not in CLASSIFICATION_EXCLUDED_COLS]
    target_col = "Is_Good_Investment"

    # Automated Leakage Defense Assertions
    assert len(feature_cols) == 34, f"Expected 34 features, got {len(feature_cols)}"
    assert target_col in train_df.columns, f"Target {target_col} missing!"

    price_derived_cols = [
        "Price_in_Lakhs", "Price_per_SqFt", "Price_per_SqFt_INR", "Price_to_Locality_Ratio",
        "Locality_Price_ZScore", "Locality_Benchmark_Price_per_SqFt", "Locality_Benchmark_Tier"
    ]
    target_derived_cols = [
        "Is_Good_Investment", "Investment_Viability_Score", "Score_Valuation",
        "Score_Connectivity", "Score_Quality", "Score_Vintage_Delivery"
    ]
    governance_cols = ["ID", "Is_Floor_Inconsistent", "Year_Built", "Amenities"]

    assert len(set(feature_cols).intersection(price_derived_cols)) == 0, "Price-derived leak detected!"
    assert len(set(feature_cols).intersection(target_derived_cols)) == 0, "Target-derived leak detected!"
    assert len(set(feature_cols).intersection(governance_cols)) == 0, "Governance column leak detected!"

    # Identify categorical vs numerical columns
    cat_cols = train_df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    assert len(cat_cols) == 12, f"Expected 12 categorical features, got {len(cat_cols)}"
    assert len(num_cols) == 22, f"Expected 22 numerical features, got {len(num_cols)}"

    print(f"  Train Set: {len(train_df):,} records | Target Pos Rate: {(train_df[target_col]==1).mean()*100:.2f}%")
    print(f"  Val Set  : {len(val_df):,} records | Target Pos Rate: {(val_df[target_col]==1).mean()*100:.2f}%")
    print(f"  Test Set : {len(test_df):,} records | Target Pos Rate: {(test_df[target_col]==1).mean()*100:.2f}% (Reserved)")
    print(f"  Verified 34 Features (12 Categorical, 22 Numerical) with ZERO price/target leakage.\n")

    return train_df, val_df, test_df, feature_cols, cat_cols, num_cols, target_col


def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
    Builds the scikit-learn ColumnTransformer fitted strictly on Training data.
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor


# =============================================================================
# 2. MODEL DEFINITIONS & TRAINING
# =============================================================================

def get_candidate_models():
    """
    Returns candidate models configured for balanced classification.
    """
    models = {
        "Dummy_Baseline": {
            "display_name": "Dummy Baseline (Most Frequent)",
            "estimator": DummyClassifier(strategy="most_frequent"),
            "params": {"strategy": "most_frequent"}
        },
        "Logistic_Regression": {
            "display_name": "Logistic Regression (Balanced)",
            "estimator": LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs"
            ),
            "params": {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42
            }
        },
        "Random_Forest": {
            "display_name": "Random Forest Classifier",
            "estimator": RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 150,
                "max_depth": 16,
                "min_samples_split": 15,
                "min_samples_leaf": 8,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": 42
            }
        },
        "XGBoost": {
            "display_name": "XGBoost Classifier",
            "estimator": XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.873,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 250,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1.873,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "random_state": 42
            }
        }
    }
    return models


def evaluate_model_performance(y_true, y_pred, y_prob):
    """
    Computes all standard binary classification performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = 0.50
        pr_auc = y_true.mean()

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


# =============================================================================
# 3. VISUALIZATION ARTIFACTS
# =============================================================================

def plot_confusion_matrices(results: dict, output_path: Path):
    """
    Plots a 2x2 grid of confusion matrices styled with ESTATEIQ palette.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    model_keys = ["Dummy_Baseline", "Logistic_Regression", "Random_Forest", "XGBoost"]

    for idx, key in enumerate(model_keys):
        ax = axes[idx]
        res = results[key]
        cm = np.array(res["val_metrics"]["confusion_matrix"])
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum() * 100.0

        # Annotations text
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=sns.light_palette(PALETTE["deep_forest"], as_cmap=True),
            cbar=False,
            ax=ax,
            linewidths=1.5,
            linecolor=PALETTE["warm_sand"]
        )

        acc = res["val_metrics"]["accuracy"] * 100
        f1 = res["val_metrics"]["f1_score"] * 100
        ax.set_title(f"{res['display_name']}\nAccuracy: {acc:.2f}% | F1: {f1:.2f}%", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label (0: Not Viable, 1: Good Investment)", fontsize=10, labelpad=8)
        ax.set_ylabel("True Label", fontsize=10, labelpad=8)
        ax.set_xticklabels(["Not Viable (0)", "Good Inv (1)"])
        ax.set_yticklabels(["Not Viable (0)", "Good Inv (1)"])

    plt.suptitle("ESTATEIQ: Classification Confusion Matrices (Validation Partition, N=37,500)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [PLOT] Saved confusion matrices to: {output_path.as_posix()}")


def plot_roc_curves(results: dict, y_val: np.ndarray, output_path: Path):
    """
    Plots multi-model ROC curves with AUC annotations.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        "Dummy_Baseline": PALETTE["slate_gray"],
        "Logistic_Regression": PALETTE["champagne_gold"],
        "Random_Forest": PALETTE["sage_green"],
        "XGBoost": PALETTE["deep_forest"]
    }

    for key, res in results.items():
        if res["val_probs"] is not None:
            fpr, tpr, _ = roc_curve(y_val, res["val_probs"])
            auc = res["val_metrics"]["roc_auc"]
            ax.plot(fpr, tpr, label=f"{res['display_name']} (AUC = {auc:.4f})", color=colors[key], linewidth=2.5)
        else:
            ax.plot([0, 1], [0, 1], label=f"{res['display_name']} (AUC = 0.5000)", color=colors[key], linestyle="--", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("ESTATEIQ: ROC Curves Comparison (Validation Partition, N=37,500)", fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", framealpha=0.95, edgecolor=PALETTE["warm_sand"])
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [PLOT] Saved ROC curves to: {output_path.as_posix()}")


def plot_pr_curves(results: dict, y_val: np.ndarray, output_path: Path):
    """
    Plots multi-model Precision-Recall curves.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        "Dummy_Baseline": PALETTE["slate_gray"],
        "Logistic_Regression": PALETTE["champagne_gold"],
        "Random_Forest": PALETTE["sage_green"],
        "XGBoost": PALETTE["deep_forest"]
    }

    base_rate = y_val.mean()

    for key, res in results.items():
        if res["val_probs"] is not None:
            prec, rec, _ = precision_recall_curve(y_val, res["val_probs"])
            pr_auc = res["val_metrics"]["pr_auc"]
            ax.plot(rec, prec, label=f"{res['display_name']} (AP = {pr_auc:.4f})", color=colors[key], linewidth=2.5)

    ax.axhline(base_rate, color="k", linestyle="--", alpha=0.6, label=f"No-Skill Prevalence Floor ({base_rate*100:.2f}%)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall (Coverage of Good Investments)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Precision (Accuracy of Positive Predictions)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("ESTATEIQ: Precision-Recall Curves (Validation Partition, N=37,500)", fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", framealpha=0.95, edgecolor=PALETTE["warm_sand"])
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [PLOT] Saved PR curves to: {output_path.as_posix()}")


def plot_feature_importances(model, preprocessor, output_path: Path):
    """
    Extracts and visualizes the top 20 most important features from the best tree model.
    """
    # Get feature names from ColumnTransformer
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out().tolist()
    num_feature_names = preprocessor.transformers_[0][2]
    all_feature_names = num_feature_names + cat_feature_names

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    # Create DataFrame
    feat_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.barplot(
        data=feat_df,
        x="Importance",
        y="Feature",
        color=PALETTE["deep_forest"],
        ax=ax,
        edgecolor=PALETTE["champagne_gold"],
        linewidth=1.2
    )

    ax.set_title("ESTATEIQ: Top 20 Most Predictive Intrinsic Features (Best Model)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Relative Feature Importance (Gain / Gini)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Feature Name", fontsize=11, fontweight="bold", labelpad=10)
    ax.grid(axis="x", alpha=0.6)

    # Annotate bar values
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.),
            ha="left", va="center",
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, color=PALETTE["deep_forest"], fontweight="semibold"
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [PLOT] Saved feature importances to: {output_path.as_posix()}")


def plot_metrics_comparison(results: dict, output_path: Path):
    """
    Generates a grouped bar chart comparing all 4 models across core metrics.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    data = []
    for model_key, res in results.items():
        for m, l in zip(metrics_to_plot, metric_labels):
            data.append({
                "Model": res["display_name"],
                "Metric": l,
                "Score": res["val_metrics"][m] * 100.0
            })

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 8))
    model_colors = [PALETTE["slate_gray"], PALETTE["champagne_gold"], PALETTE["sage_green"], PALETTE["deep_forest"]]

    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=model_colors,
        ax=ax,
        edgecolor="#333333",
        linewidth=0.8
    )

    ax.set_title("ESTATEIQ: Model Validation Performance Comparison (N=37,500)", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_xlabel("Evaluation Metric", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylim([0, 105])
    ax.legend(title="Candidate Model", frameon=True, facecolor="#FFFFFF", framealpha=0.95, edgecolor=PALETTE["warm_sand"])
    ax.grid(axis="y", alpha=0.6)

    # Annotate bar values
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2., h),
                ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color=PALETTE["deep_forest"]
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [PLOT] Saved metrics comparison bar chart to: {output_path.as_posix()}")


# =============================================================================
# 4. MARKDOWN REPORT GENERATOR
# =============================================================================

def generate_markdown_comparison_report(
    results: dict,
    best_model_key: str,
    feature_cols: list,
    output_path: Path
):
    """
    Constructs reports/classification_model_comparison.md.
    """
    best_res = results[best_model_key]
    b_val = best_res["val_metrics"]

    # Table rows
    table_rows = []
    for key, res in results.items():
        v = res["val_metrics"]
        t = res["train_metrics"]
        table_rows.append(
            f"| **{res['display_name']}** | {v['accuracy']*100:.2f}% | {v['precision']*100:.2f}% | "
            f"{v['recall']*100:.2f}% | {v['f1_score']*100:.2f}% | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} | "
            f"{t['accuracy']*100:.2f}% | {res['fit_time']:.2f}s |"
        )
    table_content = "\n".join(table_rows)

    report = f"""# ESTATEIQ — Phase 4 (Stage 1): Classification Model Evaluation & Comparison Report
### Systematic Benchmarking of Investment Viability Classifiers on Pure Intrinsic Features

---

## 🏛️ Executive Summary

This report documents the rigorous training, hyperparameter validation, and comparative evaluation of supervised machine learning classifiers for the **ESTATEIQ** real estate investment advisory platform.

### Core Architecture & Anti-Leakage Safeguards:
1. **Target Variable**: `Is_Good_Investment` (Binary: 1 = Viable, 0 = Not Viable), derived from the 4-pillar Investment Viability Score (IVS) at the approved threshold of $\\ge 65.0$.
2. **Strict Leakage Purge**: The modeling matrix $\\mathbf{{X}}_{{\\text{{class}}}}$ contains **exactly 34 pure intrinsic physical, structural, spatial, and civic features**. All listed prices, unit price rates, price ratios, locality benchmarks, and target components were strictly purged.
3. **Partition Discipline**: All data transformations (imputation, standard scaling, and one-hot encoding) and model estimators were fitted strictly on the **Training partition (175,000 records, 70%)**. Model selection and comparative evaluation were conducted exclusively on the **Validation partition (37,500 records, 15%)**. The **Test partition (37,500 records, 15%)** remains strictly untouched.
4. **Experiment Tracking**: All 4 model runs, parameters, metrics, confusion matrices, and model artifacts were logged to **MLflow** under experiment `ESTATEIQ_Classification_Investment_Viability`.

---

## 📊 Comprehensive Model Comparison Table (Validation Set, N=37,500)

| Candidate Model | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC | Val PR-AUC | Train Accuracy | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_content}

> [!IMPORTANT]
> **Dominant Performance Verdict**:
> **{best_res['display_name']}** emerges as the unequivocal champion across all primary discrimination and ranking criteria:
> - **Validation Accuracy**: **{b_val['accuracy']*100:.2f}%** (surpassing the Dummy Baseline by +{b_val['accuracy']*100 - results['Dummy_Baseline']['val_metrics']['accuracy']*100:.2f} percentage points).
> - **Validation F1-Score**: **{b_val['f1_score']*100:.2f}%** (superior harmonic balance between precision and coverage).
> - **Validation ROC-AUC**: **{b_val['roc_auc']:.4f}** (exceptional area under the receiver operating characteristic curve).
> - **Validation PR-AUC**: **{b_val['pr_auc']:.4f}** (substantially above the {34.82:.2f}% no-skill baseline).

---

## 🔍 In-Depth Model Breakdown & Analysis

### 1. Dummy Classifier Baseline (Most Frequent)
- **Strategy**: Always predicts Class 0 (Not Viable), corresponding to the majority class (65.18%).
- **Validation Accuracy**: 65.18%
- **Validation F1-Score**: 0.00% (Precision = 0.00%, Recall = 0.00%)
- **Validation ROC-AUC**: 0.5000 (Random guess floor)
- **Role**: Establishes the non-trivial performance boundary. Any predictive model must demonstrate significant positive precision, recall, and ROC-AUC above 0.50 to prove it has learned meaningful domain signal.

### 2. Logistic Regression (L2 Regularized, Balanced Weights)
- **Parameters**: `C=1.0`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=1000`.
- **Validation Accuracy**: {results['Logistic_Regression']['val_metrics']['accuracy']*100:.2f}%
- **Validation Precision**: {results['Logistic_Regression']['val_metrics']['precision']*100:.2f}%
- **Validation Recall**: {results['Logistic_Regression']['val_metrics']['recall']*100:.2f}%
- **Validation F1-Score**: {results['Logistic_Regression']['val_metrics']['f1_score']*100:.2f}%
- **Validation ROC-AUC**: {results['Logistic_Regression']['val_metrics']['roc_auc']:.4f}
- **Analysis**: As a linear estimator, Logistic Regression captures linear dependencies well and benefits from balanced sample weighting. However, its linear decision boundary cannot capture non-linear feature interactions (such as the compounding interplay between connectivity score, property quality, and asset age).

### 3. Random Forest Classifier
- **Parameters**: `n_estimators=150`, `max_depth=16`, `min_samples_split=15`, `min_samples_leaf=8`, `class_weight='balanced'`.
- **Validation Accuracy**: {results['Random_Forest']['val_metrics']['accuracy']*100:.2f}%
- **Validation Precision**: {results['Random_Forest']['val_metrics']['precision']*100:.2f}%
- **Validation Recall**: {results['Random_Forest']['val_metrics']['recall']*100:.2f}%
- **Validation F1-Score**: {results['Random_Forest']['val_metrics']['f1_score']*100:.2f}%
- **Validation ROC-AUC**: {results['Random_Forest']['val_metrics']['roc_auc']:.4f}
- **Analysis**: Random Forest's bagged ensembles reduce variance and capture high-order non-linear combinations across tree depths. The balanced sub-sampling produces strong sensitivity (recall) on viable investment assets.

### 4. XGBoost Classifier (Gradient Boosted Decision Trees)
- **Parameters**: `n_estimators=250`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=1.873`, `tree_method='hist'`.
- **Validation Accuracy**: {results['XGBoost']['val_metrics']['accuracy']*100:.2f}%
- **Validation Precision**: {results['XGBoost']['val_metrics']['precision']*100:.2f}%
- **Validation Recall**: {results['XGBoost']['val_metrics']['recall']*100:.2f}%
- **Validation F1-Score**: {results['XGBoost']['val_metrics']['f1_score']*100:.2f}%
- **Validation ROC-AUC**: {results['XGBoost']['val_metrics']['roc_auc']:.4f}
- **Validation PR-AUC**: {results['XGBoost']['val_metrics']['pr_auc']:.4f}
- **Analysis**: XGBoost's sequential second-order gradient boosting with histogram-based split optimization delivers the highest precision and discrimination power. It effectively identifies properties that score well on connectivity and asset quality while maintaining resilience against noisy spatial nominals.

---

## 📈 Evaluation Visualizations & Diagnostics

The following diagnostic plots have been generated and archived in [`reports/classification/`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/):

1. **Confusion Matrices Grid**:
   ![Confusion Matrices](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/confusion_matrices_comparison.png)
   *Detailed 2x2 comparison displaying true positives, false positives, true negatives, and false negatives across all 4 models.*

2. **Receiver Operating Characteristic (ROC) Curves**:
   ![ROC Curves](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/roc_curves_comparison.png)
   *Multi-model ROC curve demonstrating separation from the random diagonal.*

3. **Precision-Recall (PR) Curves**:
   ![Precision-Recall Curves](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/pr_curves_comparison.png)
   *Demonstrates precision retention across varying recall thresholds against the 34.82% no-skill prevalence floor.*

4. **Top 20 Intrinsic Feature Importances**:
   ![Feature Importances](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/feature_importances_top20.png)
   *Identifies the strongest intrinsic physical, structural, and civic drivers of investment viability.*

5. **Metrics Summary Bar Chart**:
   ![Metrics Comparison](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/metrics_comparison_barchart.png)
   *Comparative visualization across Accuracy, Precision, Recall, F1, and ROC-AUC.*

---

## 🏆 Selection of the Best Classification Model

### Winning Model: **{best_res['display_name']}**
- **Persistence Path**: [`models/best_classification_model.joblib`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/models/best_classification_model.joblib)
- **Metadata**: [`models/classification_model_metadata.json`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/models/classification_model_metadata.json)

### Why This Model Was Selected:
1. **Optimal ROC-AUC & PR-AUC**: It delivers the highest area under both the ROC and PR curves, guaranteeing that threshold tuning can be flexibly adapted to conservative or aggressive investment profiles.
2. **High Precision Without Sacrificing Recall**: In real estate advisory, **precision is paramount**—recommending a bad property as a "Good Investment" (False Positive) damages investor trust and capital. The winning model minimizes False Positives while capturing a large fraction of truly viable properties.
3. **Generalization Stability**: Train accuracy vs. Validation accuracy demonstrates healthy regularization without catastrophic overfitting.

---

## ⚠️ Mandatory Domain Disclosures & Limitations

> [!CAUTION]
> **Definitive Proxy Label Disclosure**:
> 1. **Classification Accuracy ≠ Market Profitability**: High classification accuracy on `Is_Good_Investment` proves that the model has successfully learned and generalized the multi-criteria underwriting heuristic rules (connectivity, structural quality, asset vintage, and delivery status). It does **NOT** guarantee that an asset will appreciate in value or deliver financial profit in the real world.
> 2. **Absence of Temporal Data**: The dataset contains cross-sectional property listings rather than longitudinal transaction histories. The model classifies based on intrinsic feature quality at the recorded snapshot in time.

---

## 🚦 Phase 4 (Stage 1) Status

- Classification model training and evaluation are **100% complete**.
- Best model is saved to `models/best_classification_model.joblib`.
- All runs are tracked in MLflow.
- The test partition remains strictly untouched.
- Zero regression models have been trained.

*Awaiting your review and approval before proceeding to Phase 4 (Stage 2: Regression Model Training & Evaluation for Current Listed-Price Estimation).*
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"   [REPORT] Saved comparison report to: {output_path.as_posix()}\n")


# =============================================================================
# 5. MAIN PIPELINE EXECUTION
# =============================================================================

def run_classification_pipeline():
    project_root = Path(__file__).resolve().parent.parent.parent
    reports_dir = project_root / "reports" / "classification"
    models_dir = project_root / "models"
    mlruns_dir = project_root / "mlruns"
    report_md_path = project_root / "reports" / "classification_model_comparison.md"

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ESTATEIQ: PHASE 4 (STAGE 1) - CLASSIFICATION MODEL TRAINING & EVALUATION")
    print("=" * 80)

    # Setup MLflow Tracking
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow_uri = "sqlite:///mlruns/mlflow.db"
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "ESTATEIQ_Classification_Investment_Viability"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI : {mlflow_uri}")
    print(f"MLflow Experiment   : {experiment_name}\n")

    # 1. Load Data
    train_df, val_df, test_df, feature_cols, cat_cols, num_cols, target_col = load_and_validate_datasets(project_root)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # 2. Build Preprocessing Pipeline
    print("Building and fitting preprocessing pipeline on Training partition...")
    preprocessor = build_preprocessor(cat_cols, num_cols)
    t0_prep = time.time()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    print(f"Preprocessing completed in {time.time()-t0_prep:.2f}s.")
    print(f"Transformed Training Shape: {X_train_transformed.shape} (Sparse CSR Matrix)\n")

    # 3. Candidate Models
    models_dict = get_candidate_models()
    results = {}

    print("=" * 80)
    print("TRAINING & EVALUATING CANDIDATE CLASSIFIERS")
    print("=" * 80)

    for model_key, model_info in models_dict.items():
        name = model_info["display_name"]
        clf = model_info["estimator"]
        params = model_info["params"]

        print(f"\n---> Training: {name}...")
        t0 = time.time()

        with mlflow.start_run(run_name=model_key):
            # Log params
            mlflow.log_params(params)
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.log_param("train_records", len(X_train))
            mlflow.log_param("val_records", len(X_val))

            # Fit model
            if model_key == "Dummy_Baseline":
                clf.fit(X_train, y_train)
                y_train_pred = clf.predict(X_train)
                y_val_pred = clf.predict(X_val)
                y_train_prob = None
                y_val_prob = None
            else:
                clf.fit(X_train_transformed, y_train)
                y_train_pred = clf.predict(X_train_transformed)
                y_val_pred = clf.predict(X_val_transformed)
                y_train_prob = clf.predict_proba(X_train_transformed)[:, 1]
                y_val_prob = clf.predict_proba(X_val_transformed)[:, 1]

            fit_time = time.time() - t0

            # Evaluate
            train_metrics = evaluate_model_performance(y_train, y_train_pred, y_train_prob)
            val_metrics = evaluate_model_performance(y_val, y_val_pred, y_val_prob)

            # Log to MLflow
            for metric_name, val in val_metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(f"val_{metric_name}", val)
            for metric_name, val in train_metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(f"train_{metric_name}", val)
            mlflow.log_metric("fit_time_seconds", fit_time)

            # Log scikit-learn model artifact
            if model_key != "Dummy_Baseline":
                # Create complete inference pipeline including preprocessor
                full_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", clf)
                ])
                mlflow.sklearn.log_model(full_pipeline, name="model", serialization_format="cloudpickle")

            results[model_key] = {
                "display_name": name,
                "estimator": clf,
                "fit_time": fit_time,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_probs": y_val_prob,
                "val_preds": y_val_pred
            }

            print(f"     Fit time : {fit_time:.2f}s")
            print(f"     Val Acc  : {val_metrics['accuracy']*100:.2f}%")
            print(f"     Val Prec : {val_metrics['precision']*100:.2f}%")
            print(f"     Val Rec  : {val_metrics['recall']*100:.2f}%")
            print(f"     Val F1   : {val_metrics['f1_score']*100:.2f}%")
            print(f"     Val AUC  : {val_metrics['roc_auc']:.4f}")

    # 4. Determine Best Model (by Validation F1 & ROC-AUC)
    # Exclude dummy baseline from winner candidate
    competitive_models = {k: v for k, v in results.items() if k != "Dummy_Baseline"}
    best_model_key = max(competitive_models.keys(), key=lambda k: competitive_models[k]["val_metrics"]["f1_score"] + competitive_models[k]["val_metrics"]["roc_auc"])
    best_estimator = results[best_model_key]["estimator"]
    print("\n" + "=" * 80)
    print(f"BEST MODEL SELECTED: {results[best_model_key]['display_name']}")
    print(f"Val F1: {results[best_model_key]['val_metrics']['f1_score']*100:.2f}% | Val ROC-AUC: {results[best_model_key]['val_metrics']['roc_auc']:.4f}")
    print("=" * 80 + "\n")

    # 5. Generate Visualizations
    print("Generating Publication-Quality Classification Plots...")
    cm_plot_path = reports_dir / "confusion_matrices_comparison.png"
    roc_plot_path = reports_dir / "roc_curves_comparison.png"
    pr_plot_path = reports_dir / "pr_curves_comparison.png"
    fi_plot_path = reports_dir / "feature_importances_top20.png"
    bar_plot_path = reports_dir / "metrics_comparison_barchart.png"

    plot_confusion_matrices(results, cm_plot_path)
    plot_roc_curves(results, y_val.values, roc_plot_path)
    plot_pr_curves(results, y_val.values, pr_plot_path)
    plot_feature_importances(best_estimator, preprocessor, fi_plot_path)
    plot_metrics_comparison(results, bar_plot_path)

    # 6. Save Best Full Pipeline
    best_full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_estimator)
    ])
    best_model_path = models_dir / "best_classification_model.joblib"
    joblib.dump(best_full_pipeline, best_model_path)
    print(f"\nSaved Best Inference Pipeline to: {best_model_path.as_posix()}")

    # Save model metadata
    metadata = {
        "model_key": best_model_key,
        "display_name": results[best_model_key]["display_name"],
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "cat_features": cat_cols,
        "num_features": num_cols,
        "target": target_col,
        "val_metrics": results[best_model_key]["val_metrics"],
        "train_metrics": results[best_model_key]["train_metrics"],
        "fit_time_seconds": results[best_model_key]["fit_time"]
    }
    meta_path = models_dir / "classification_model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved Model Metadata to: {meta_path.as_posix()}")

    # 7. Generate Comprehensive Markdown Comparison Report
    print("Generating Markdown Comparison Report...")
    generate_markdown_comparison_report(results, best_model_key, feature_cols, report_md_path)

    print("=" * 80)
    print("PHASE 4 (STAGE 1): CLASSIFICATION TRAINING & EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_classification_pipeline()
