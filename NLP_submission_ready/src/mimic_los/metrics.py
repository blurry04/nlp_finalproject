from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .constants import LABELS


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(LABELS, f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS)))))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }
    if y_proba is not None:
        try:
            metrics["weighted_ovr_auroc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["weighted_ovr_auroc"] = None
    else:
        metrics["weighted_ovr_auroc"] = None
    return metrics
