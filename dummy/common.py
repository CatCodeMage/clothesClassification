"""Shared utilities for the dummy model scripts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np

TARGET_COLUMNS = ("subCategory", "articleType", "baseColour")
TARGET_ALIASES = {"color": "baseColour", "type": "subCategory", "subtype": "articleType"}


def load_archive(path: str | Path) -> dict[str, np.ndarray]:
    """Load an extractor archive and return ordinary numpy arrays."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def target_key(data: Mapping[str, np.ndarray], target: str) -> str:
    """Find a target column under its canonical name or extractor alias."""
    candidates = (f"target_{target}", target)
    candidates += tuple(alias for alias, canonical in TARGET_ALIASES.items() if canonical == target)
    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Target {target!r} is missing from archive; tried {candidates}")


def labels_for(data: Mapping[str, np.ndarray], target: str) -> np.ndarray:
    values = np.asarray(data[target_key(data, target)]).reshape(-1)
    return values.astype(str)


def features_for(data: Mapping[str, np.ndarray]) -> np.ndarray:
    if "features" not in data:
        raise KeyError("Feature archive must contain a 'features' array")
    features = np.asarray(data["features"])
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {features.shape}")
    return features


def save_pickle(value: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: str | Path) -> object:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def jsonable(value: object) -> object:
    """Convert numpy scalar/array values to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
