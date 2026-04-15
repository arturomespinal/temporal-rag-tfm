"""
utils/metrics.py
================
Evaluation metrics for the Temporal RAG system (TFM §7).

This module provides:
- Forecasting metrics: MAE, RMSE, MAPE
- Anomaly detection metrics: Precision, Recall, F1-score, AUC-ROC
- A point-adjust evaluation protocol (standard in the anomaly detection
  literature to handle label propagation in temporal windows)

References
----------
Wu, H., & Keogh, E. (2021). Current Time Series Anomaly Detection Benchmarks
    are Flawed and are Creating Impossible Experimental Conditions.
    *IEEE TKDE*, 35(3), 2421–2429.
Audibert, J., et al. (2020). USAD: Unsupervised anomaly detection on
    multivariate time series. *KDD*, 3395–3404.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Forecasting metrics
# ---------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (in %)."""
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def forecasting_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Aggregate forecasting metrics.

    Returns
    -------
    dict with keys: ``mae``, ``rmse``, ``mape``
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Anomaly detection metrics
# ---------------------------------------------------------------------------


def point_adjust(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Point-Adjust protocol for anomaly detection evaluation.

    In temporal anomaly detection, a model may trigger only on one point
    within an anomalous segment but detect the whole segment conceptually.
    The point-adjust protocol considers a segment detected if **any**
    point within it crosses the threshold, and labels all points in that
    segment as detected.

    This is the standard evaluation protocol used by USAD, Anomaly
    Transformer, and other SOTA methods (Wu & Keogh, 2021).

    Parameters
    ----------
    y_true : np.ndarray of shape (N,), dtype=int
        Ground-truth binary labels (1=anomaly).
    y_scores : np.ndarray of shape (N,)
        Continuous anomaly scores.
    threshold : float
        Score threshold above which a point is flagged as anomalous.

    Returns
    -------
    y_pred_adjusted : np.ndarray of shape (N,), dtype=int
        Point-adjusted binary predictions.
    """
    y_pred = (y_scores >= threshold).astype(int)
    y_pred_adj = y_pred.copy()

    # Identify contiguous anomalous segments in y_true
    in_anomaly = False
    seg_start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            seg_start = i
        elif y_true[i] == 0 and in_anomaly:
            # Segment [seg_start, i)
            if y_pred[seg_start:i].any():
                y_pred_adj[seg_start:i] = 1
            in_anomaly = False
    # Handle last segment
    if in_anomaly and y_pred[seg_start:].any():
        y_pred_adj[seg_start:] = 1

    return y_pred_adj


def best_f1_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 200,
    point_adjust_protocol: bool = True,
) -> tuple[float, float]:
    """
    Find the threshold that maximises F1-score on the evaluation set.

    Parameters
    ----------
    y_true : np.ndarray
    y_scores : np.ndarray
    n_thresholds : int
        Number of candidate thresholds to evaluate.
    point_adjust_protocol : bool
        Whether to apply the point-adjust correction.

    Returns
    -------
    best_threshold : float
    best_f1 : float
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)
    best_f1 = 0.0
    best_thr = thresholds[0]

    for thr in thresholds:
        if point_adjust_protocol:
            y_pred = point_adjust(y_true, y_scores, thr)
        else:
            y_pred = (y_scores >= thr).astype(int)

        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


def anomaly_detection_report(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
    point_adjust_protocol: bool = True,
) -> dict:
    """
    Comprehensive anomaly detection evaluation report.

    If ``threshold`` is None, the threshold that maximises F1 is used.

    Returns
    -------
    dict with keys:
        ``precision``, ``recall``, ``f1``, ``auc_roc``,
        ``threshold``, ``auc_pr``
    """
    from typing import Optional  # local import to avoid circular

    if threshold is None:
        threshold, _ = best_f1_threshold(
            y_true, y_scores, point_adjust_protocol=point_adjust_protocol
        )

    if point_adjust_protocol:
        y_pred = point_adjust(y_true, y_scores, threshold)
    else:
        y_pred = (y_scores >= threshold).astype(int)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    auc_pr = auc(recall_vals, precision_vals)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_scores) if y_true.sum() > 0 else float("nan"),
        "auc_pr": float(auc_pr),
        "threshold": float(threshold),
    }
