"""Run the trained dummy classifiers on an image feature vector or image file."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

try:  # Supports both `python -m dummy.infer` and `python dummy/infer.py`.
    from dummy.common import TARGET_COLUMNS, load_pickle
except ModuleNotFoundError:  # pragma: no cover - only used for direct script execution
    from common import TARGET_COLUMNS, load_pickle

from feature_extractor import extract_single_feature

DEFAULT_MODEL = Path("models/dummy.pkl")
TEST_IMG = Path("dataset/test/images/16988.jpg")


def predict_features(features: np.ndarray, model_path: str | Path = DEFAULT_MODEL) -> dict[str, str]:
    """Return predictions for the three targets from extracted features."""
    bundle = load_pickle(model_path)
    if not isinstance(bundle, Mapping) or "estimators" not in bundle:
        raise ValueError(f"Unsupported model artifact: {model_path}")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != 1:
        raise ValueError("predict_features expects one feature vector")
    expected = bundle.get("feature_dimension")
    if expected is not None and matrix.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got {matrix.shape[1]}")
    return {
        target: str(bundle["estimators"][target].predict(matrix)[0])
        for target in TARGET_COLUMNS
    }


def predict_image(image_path: str | Path, model_path: str | Path = DEFAULT_MODEL, device: str | None = None) -> dict[str, str]:
    """Extract ResNet features for an image and predict all targets."""
    return predict_features(extract_single_feature(image_path, device=device), model_path)


if __name__ == "__main__":
    print(predict_image(TEST_IMG, DEFAULT_MODEL))
