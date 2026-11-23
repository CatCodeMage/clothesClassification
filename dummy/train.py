"""Train one most-frequent dummy classifier for each planned target."""

from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

try:  # Supports both `python -m dummy.train` and `python dummy/train.py`.
    from dummy.common import TARGET_COLUMNS, features_for, labels_for, load_archive, save_pickle
except ModuleNotFoundError:  # pragma: no cover - only used for direct script execution
    from common import TARGET_COLUMNS, features_for, labels_for, load_archive, save_pickle

DEFAULT_TRAIN_FEATURES = Path("features/train.npz")
DEFAULT_VALIDATION_FEATURES = Path("features/validation.npz")
DEFAULT_MODEL = Path("models/dummy.pkl")


def train_model(
    train_features: str | Path = DEFAULT_TRAIN_FEATURES,
    validation_features: str | Path = DEFAULT_VALIDATION_FEATURES,
    model_path: str | Path = DEFAULT_MODEL,
) -> Path:
    """Fit the classifiers and save the model bundle.

    Validation is deliberately checked here so a bad/misaligned feature archive
    fails before the model is saved.
    """
    train_data = load_archive(train_features)
    validation_data = load_archive(validation_features)
    x_train = features_for(train_data)
    x_validation = features_for(validation_data)
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Train and validation feature dimensions do not match")

    estimators = {}
    validation_scores = {}
    for target in TARGET_COLUMNS:
        y_train = labels_for(train_data, target)
        y_validation = labels_for(validation_data, target)
        if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
            raise ValueError(f"Feature/label length mismatch for {target}")
        estimator = DummyClassifier(strategy="most_frequent")
        estimator.fit(x_train, y_train)
        estimators[target] = estimator
        validation_scores[target] = float(accuracy_score(y_validation, estimator.predict(x_validation)))

    bundle = {
        "model_type": "dummy_most_frequent",
        "targets": TARGET_COLUMNS,
        "feature_dimension": int(x_train.shape[1]),
        "validation_accuracy": validation_scores,
        "estimators": estimators,
    }
    output = save_pickle(bundle, model_path)
    scores = ", ".join(f"{target}={score:.3f}" for target, score in validation_scores.items())
    print(f"Validation accuracy: {scores}")
    print(f"Saved dummy model for {', '.join(TARGET_COLUMNS)} to {output}")
    return output


if __name__ == "__main__":
    train_model()
