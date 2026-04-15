"""
loader/nab_loader.py
====================
Stage 1 of the Temporal RAG pipeline: data ingestion and preprocessing.

This module handles loading of the Numenta Anomaly Benchmark (NAB) dataset,
applying local Z-score normalization and producing labelled sliding-window
tensors ready for the encoder stage.

Theoretical context (TFM §5 / §6 Phase 1)
------------------------------------------
Raw time series are non-stationary signals. Direct distance comparisons in the
original value space are misleading because absolute magnitude differences
dominate cosine/Euclidean similarity, masking the true **dynamic shape**
similarity that we wish to capture in the latent space. Local Z-score
normalization (per-window mean subtraction and standard-deviation scaling)
ensures that geometric proximity in the embedding space reflects *pattern*
similarity rather than scale similarity.

    x_norm(t) = (x(t) - μ_w) / (σ_w + ε)

where μ_w, σ_w are the mean and std of the window, and ε prevents
division-by-zero on flat signals.

References
----------
Lavin, A., & Ahmad, S. (2015). Evaluating real-time anomaly detection
    algorithms – the Numenta Anomaly Benchmark. *Proceedings of ICMLA*, 38–44.
Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey.
    *ACM Computing Surveys, 41*(3), 1–58.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAB_LABELS_FILENAME = "combined_windows.json"
_DEFAULT_NAB_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "NAB"


class NABLoader:
    """
    Load a single NAB series, apply per-window Z-score normalization and
    return sliding-window arrays with binary anomaly labels.

    Parameters
    ----------
    series_name : str
        Name of the NAB series file (without extension), e.g.
        ``"ambient_temperature_system_failure"``.
    data_dir : str or Path, optional
        Root directory containing the NAB ``data/`` and ``labels/``
        subdirectories. Defaults to ``<repo_root>/data/NAB``.
    label_buffer_secs : int
        Half-width (in seconds) of the label window around each anomaly
        interval. NAB labels are interval-based; this pads point labels.
    epsilon : float
        Small constant added to std to avoid division by zero in per-window
        normalization.

    Attributes
    ----------
    df : pd.DataFrame
        Raw loaded series (columns: ``timestamp``, ``value``).
    anomaly_windows : list[tuple[str, str]]
        List of (start_iso, end_iso) anomaly intervals from the label file.

    Examples
    --------
    >>> loader = NABLoader("ambient_temperature_system_failure")
    >>> windows, labels = loader.load_windows(window_size=64, step=1)
    >>> windows.shape  # (N, 64)
    >>> labels.shape   # (N,)
    """

    def __init__(
        self,
        series_name: str,
        data_dir: Optional[str | Path] = None,
        label_buffer_secs: int = 0,
        epsilon: float = 1e-8,
    ) -> None:
        self.series_name = series_name
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_NAB_DATA_DIR
        self.label_buffer_secs = label_buffer_secs
        self.epsilon = epsilon

        self.df: Optional[pd.DataFrame] = None
        self.anomaly_windows: list[tuple[str, str]] = []

        self._load_raw()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_series_path(self) -> Path:
        """Recursively locate the CSV for *series_name* under data_dir."""
        matches = list(self.data_dir.rglob(f"{self.series_name}.csv"))
        if not matches:
            raise FileNotFoundError(
                f"Series '{self.series_name}' not found under {self.data_dir}. "
                "Download NAB from https://github.com/numenta/NAB"
            )
        return matches[0]

    def _load_labels(self) -> list[tuple[str, str]]:
        """Load anomaly interval labels from the NAB JSON label file."""
        label_path = self.data_dir / "labels" / NAB_LABELS_FILENAME
        if not label_path.exists():
            logger.warning(
                "NAB labels file not found at %s. "
                "All labels will be set to 0.",
                label_path,
            )
            return []

        with open(label_path) as fh:
            all_labels: dict = json.load(fh)

        # Keys are relative paths, e.g. "data/realKnownCause/ambient_..."
        for key, intervals in all_labels.items():
            if self.series_name in key:
                return [(iv[0], iv[1]) for iv in intervals]
        return []

    def _load_raw(self) -> None:
        """Read the CSV and label file into instance attributes."""
        csv_path = self._find_series_path()
        self.df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        self.df.sort_values("timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.anomaly_windows = self._load_labels()
        logger.info(
            "Loaded '%s' — %d points, %d anomaly intervals.",
            self.series_name,
            len(self.df),
            len(self.anomaly_windows),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_point_labels(self) -> np.ndarray:
        """
        Convert interval-based anomaly labels to per-point binary labels.

        Returns
        -------
        labels : np.ndarray of shape (T,), dtype=int8
            1 if the timestamp falls within (or near) an anomaly interval,
            0 otherwise.
        """
        labels = np.zeros(len(self.df), dtype=np.int8)
        ts = self.df["timestamp"]

        for start_iso, end_iso in self.anomaly_windows:
            start = pd.Timestamp(start_iso) - pd.Timedelta(
                seconds=self.label_buffer_secs
            )
            end = pd.Timestamp(end_iso) + pd.Timedelta(
                seconds=self.label_buffer_secs
            )
            mask = (ts >= start) & (ts <= end)
            labels[mask] = 1

        return labels

    def load_windows(
        self,
        window_size: int = 64,
        step: int = 1,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment the series into overlapping windows and compute per-window
        Z-score normalization.

        Parameters
        ----------
        window_size : int
            Number of consecutive observations per window. Should be chosen
            based on the dominant seasonality detected in the EDA (§5.3).
        step : int
            Stride between consecutive windows. ``step=1`` gives maximum
            overlap; ``step=window_size`` gives non-overlapping chunks.
        normalize : bool
            Whether to apply per-window Z-score normalization.

        Returns
        -------
        windows : np.ndarray of shape (N, window_size)
            Normalized (or raw) sliding windows.
        labels : np.ndarray of shape (N,), dtype=int8
            Binary label for each window: 1 if **any** point in the window
            is anomalous, 0 otherwise. This is a conservative choice that
            maximises recall at the expense of precision — consistent with
            the NAB scoring philosophy.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call _load_raw() first.")

        values = self.df["value"].to_numpy(dtype=np.float32)
        point_labels = self.get_point_labels()

        n_windows = (len(values) - window_size) // step + 1
        windows = np.lib.stride_tricks.sliding_window_view(values, window_size)[
            ::step
        ].copy()  # shape (N, window_size)

        # Per-window Z-score normalization
        if normalize:
            mu = windows.mean(axis=1, keepdims=True)
            sigma = windows.std(axis=1, keepdims=True)
            windows = (windows - mu) / (sigma + self.epsilon)

        # Window-level label: 1 if any point is anomalous
        label_windows = np.lib.stride_tricks.sliding_window_view(
            point_labels, window_size
        )[::step]
        labels = label_windows.any(axis=1).astype(np.int8)

        logger.info(
            "Generated %d windows (size=%d, step=%d). "
            "Anomalous windows: %d (%.1f%%)",
            n_windows,
            window_size,
            step,
            labels.sum(),
            100 * labels.mean(),
        )
        return windows, labels
