"""Evaluate the dummy model on test features and save JSON metrics/plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:  # Supports both `python -m dummy.evaluate` and `python dummy/evaluate.py`.
    from dummy.common import TARGET_COLUMNS, features_for, jsonable, labels_for, load_archive, load_pickle
except ModuleNotFoundError:  # pragma: no cover - only used for direct script execution
    from common import TARGET_COLUMNS, features_for, jsonable, labels_for, load_archive, load_pickle

DEFAULT_TEST_FEATURES = Path("features/test.npz")
DEFAULT_MODEL = Path("models/dummy.pkl")
DEFAULT_RESULTS = Path("dummy/results")


def evaluate(
    test_features: str | Path = DEFAULT_TEST_FEATURES,
    model_path: str | Path = DEFAULT_MODEL,
    results_dir: str | Path = DEFAULT_RESULTS,
) -> dict:
    data = load_archive(test_features)
    bundle = load_pickle(model_path)
    x_test = features_for(data)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"model": bundle.get("model_type", "unknown"), "targets": {}}

    for target in TARGET_COLUMNS:
        y_true = labels_for(data, target)
        estimator = bundle["estimators"][target]
        y_pred = estimator.predict(x_test).astype(str)
        labels = [str(label) for label in estimator.classes_]
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        target_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "classification_report": report,
            "classes": labels,
        }
        metrics["targets"][target] = target_metrics
        _save_confusion_matrix(y_true, y_pred, labels, target, results_dir)
        _save_roc(estimator, x_test, y_true, target, results_dir, metrics["targets"][target])

    output = results_dir / "metrics.json"
    output.write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    print(f"Saved metrics and plots to {results_dir}")
    return metrics


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], target: str, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    # Keep small matrices compact, while giving larger class sets enough room
    # for every class name on both axes (including articleType's many classes).
    class_count = len(labels)
    figure_side = max(4.5, min(14, 3.5 + class_count * 0.075))
    label_size = max(4, min(9, 180 / max(class_count, 1)))
    figure, axis = plt.subplots(figsize=(figure_side, figure_side))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(title=f"Confusion matrix: {target}", xlabel="Predicted label", ylabel="True label")
    # Always show the class names; hiding them makes the matrix ambiguous.
    positions = np.arange(class_count)
    axis.set_xticks(positions, labels, rotation=90, fontsize=label_size)
    axis.set_yticks(positions, labels, fontsize=label_size)
    axis.tick_params(axis="x", pad=1)
    axis.tick_params(axis="y", pad=1)
    figure.tight_layout()
    figure.savefig(output_dir / f"confusion_matrix_{target}.png", dpi=150)
    plt.close(figure)


def _save_roc(estimator, features: np.ndarray, y_true: np.ndarray, target: str, output_dir: Path, target_metrics: dict) -> None:
    """Save a one-vs-rest ROC plot when the test set supports it."""
    classes = np.asarray(estimator.classes_).astype(str)
    probabilities = estimator.predict_proba(features)
    figure, axis = plt.subplots(figsize=(5, 4))
    plotted = False
    auc_values = {}
    for index, label in enumerate(classes):
        truth = (y_true == label).astype(int)
        if truth.min() == truth.max():
            continue
        false_positive, true_positive, _ = roc_curve(truth, probabilities[:, index])
        auc_value = roc_auc_score(truth, probabilities[:, index])
        axis.plot(false_positive, true_positive, label=f"{label} (AUC={auc_value:.3f})")
        auc_values[label] = auc_value
        plotted = True
    if plotted:
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set(title=f"ROC curve: {target}", xlabel="False positive rate", ylabel="True positive rate")
        axis.legend(fontsize="small", loc="lower right")
        target_metrics["roc_auc"] = auc_values
    else:
        axis.text(0.5, 0.5, "ROC unavailable: test set has one class", ha="center", va="center")
        axis.set_axis_off()
        target_metrics["roc_auc"] = None
    figure.tight_layout()
    figure.savefig(output_dir / f"roc_{target}.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    evaluate()
