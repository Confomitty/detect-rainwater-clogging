"""
evaluate.py
===========
Computes and prints evaluation metrics for a trained classifier.

Metrics used
------------
Accuracy        : fraction of correct predictions (avoid relying on this
                  alone — misleading when classes are imbalanced).
Precision       : of all roads predicted as waterlogged, how many truly were?
                  (Low precision → many false alarms → wasted resources.)
Recall          : of all roads that actually waterlogged, how many did we catch?
                  (Low recall → missed floods → higher risk to people.)
F1-score        : harmonic mean of precision and recall.
                  Good single summary when you care about both.
Confusion Matrix: shows TP, FP, FN, TN counts.

Why F1 and recall matter here
------------------------------
For a waterlogging early-warning system it is usually *worse* to miss a
real flood (false negative) than to give an unnecessary alert (false positive).
So recall for class=1 is typically the most important metric — but your
stakeholders at Silk Board should confirm this priority.

Usage
-----
    from src.evaluate import evaluate_model, print_report

    metrics = evaluate_model(model, X_test, y_test, model_name="Random Forest")
    print_report(metrics)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate *model* on the test set and return a metrics dictionary.

    Parameters
    ----------
    model      : fitted sklearn estimator
    X_test     : preprocessed test features
    y_test     : true labels (0 or 1)
    model_name : human-readable name shown in printed output

    Returns
    -------
    metrics : dict with keys:
        model_name, accuracy, precision, recall, f1,
        roc_auc (if model supports predict_proba),
        confusion_matrix, classification_report
    """
    y_pred = model.predict(X_test)

    metrics: dict = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["not waterlogged (0)", "waterlogged (1)"],
            zero_division=0,
        ),
    }

    # ROC-AUC only when the model can output probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
        except ValueError:
            # Only one class in y_test — can't compute AUC
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def print_report(metrics: dict) -> None:
    """
    Pretty-print a metrics dict returned by evaluate_model().
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Evaluation Report — {metrics['model_name']}")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  "
          "(of roads flagged, how many truly flooded?)")
    print(f"  Recall    : {metrics['recall']:.4f}  "
          "(of actual floods, how many did we catch?)")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Per-class Report:")
    print(metrics["classification_report"])


def plot_confusion_matrix(
    metrics: dict,
    save_path: str | None = None,
) -> None:
    """
    Plot and optionally save the confusion matrix as a heatmap.

    Parameters
    ----------
    metrics   : dict from evaluate_model()
    save_path : if provided, save the figure to this path instead of showing
    """
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted: Dry", "Predicted: Flooded"],
        yticklabels=["Actual: Dry", "Actual: Flooded"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {metrics['model_name']}")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def compare_models(all_metrics: list[dict]) -> None:
    """
    Print a comparison table for multiple models.

    Parameters
    ----------
    all_metrics : list of dicts, each from evaluate_model()
    """
    print("\n" + "=" * 70)
    print("  Model Comparison")
    print("=" * 70)
    header = f"  {'Model':<25} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>6} {'AUC':>7}"
    print(header)
    print("  " + "-" * 66)
    for m in all_metrics:
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  N/A "
        print(
            f"  {m['model_name']:<25} "
            f"{m['accuracy']:>8.4f} "
            f"{m['precision']:>9.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['f1']:>6.4f} "
            f"{auc:>7}"
        )
    print("=" * 70)

