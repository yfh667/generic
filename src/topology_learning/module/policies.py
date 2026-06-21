from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class NearestNeighborPolicy:
    """Small dependency-free classifier for smoke-testing exported datasets."""

    k: int = 1
    train_features: np.ndarray | None = None
    train_labels: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "NearestNeighborPolicy":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        if features.ndim != 2:
            raise ValueError("features must be 2D")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must be 1D and aligned with features")
        self.train_features = features
        self.train_labels = labels
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.train_features is None or self.train_labels is None:
            raise RuntimeError("policy must be fitted before predict")
        features = np.asarray(features, dtype=np.float32)
        train = np.asarray(self.train_features, dtype=np.float32)
        labels = np.asarray(self.train_labels, dtype=np.int32)
        k = max(1, min(int(self.k), int(train.shape[0])))
        preds = np.empty(features.shape[0], dtype=np.int32)
        for row, x in enumerate(features):
            dist = np.sum((train - x[None, :]) ** 2, axis=1)
            nearest = np.argpartition(dist, k - 1)[:k]
            counts = Counter(int(labels[idx]) for idx in nearest)
            preds[row] = int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])
        return preds


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shape")
    correct = int(np.count_nonzero(y_true == y_pred))
    total = int(y_true.size)
    unique_true = set(int(x) for x in np.unique(y_true))
    unique_pred = set(int(x) for x in np.unique(y_pred))
    return {
        "total": total,
        "correct": correct,
        "accuracy": float(correct / total) if total else 0.0,
        "num_true_classes": int(len(unique_true)),
        "num_pred_classes": int(len(unique_pred)),
    }


def action_value_report(
    *,
    action_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Mapping[str, float]:
    values = np.asarray(action_values, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rows = np.arange(values.shape[0])
    true_values = values[rows, y_true]
    pred_values = values[rows, y_pred]
    regret = pred_values - true_values
    finite = np.isfinite(regret)
    return {
        "mean_teacher_value": float(np.mean(true_values)),
        "mean_pred_value": float(np.mean(pred_values[np.isfinite(pred_values)])) if np.any(np.isfinite(pred_values)) else float("inf"),
        "mean_regret": float(np.mean(regret[finite])) if np.any(finite) else float("inf"),
        "max_regret": float(np.max(regret[finite])) if np.any(finite) else float("inf"),
        "invalid_pred_count": int(np.count_nonzero(~np.isfinite(pred_values))),
        "finite_regret_count": int(np.count_nonzero(finite)),
    }
