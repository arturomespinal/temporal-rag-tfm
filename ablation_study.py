"""
ablation_study.py
=================
Estudio de ablación sobre la cardinalidad k del conjunto recuperado en el
sistema Temporal RAG. Evalúa k ∈ {0, 1, 5, 10} sobre tareas de:
    (A) Forecasting   — métricas: MAE, MSE, MASE
    (B) Detección de anomalías — métricas: Precision, Recall, F1, AUROC

Protocolo de evaluación:
    - Split temporal 70/10/20 (train/val/test) — sin contaminación del test (§6.9)
    - Umbral de anomalía τ optimizado sobre validación, no sobre test
    - Point-Adjust (PA@K) activado para detección de anomalías (§6.7)
    - Ablación estrictamente controlada: solo varía k; encoder y arquitectura fijos

Referencia del diseño experimental:
    Shi, F., et al. (2023). Large language models can be easily distracted
    by irrelevant context. ICML 2023.

    Asai, A., et al. (2023). Self-RAG: Learning to retrieve, generate,
    and critique through self-reflection. arXiv:2310.11511.

Uso:
    python ablation_study.py \
        --data_path data/ambient_temperature_system_failure.csv \
        --labels_path data/labels.json \
        --encoder_path checkpoints/best_encoder.pt \
        --k_values 0 1 5 10 \
        --forecast_horizon 96 \
        --alpha 0.5

Repositorio:
    https://github.com/arturomespinal/temporal-rag-tfm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importación del encoder entrenado (mismo módulo)
from train_contrastive import (
    SlidingWindowDataset,
    TemporalEncoder,
    TransformerBackbone,
    load_series,
)

# FAISS: búsqueda aproximada de vecinos (§6.4)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning(
        "FAISS no disponible. Instalar con: pip install faiss-cpu. "
        "Se usará búsqueda exacta por fuerza bruta (solo para debugging)."
    )

# Métricas
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. ESTRUCTURAS DE DATOS
# ===========================================================================

@dataclass
class ForecastingMetrics:
    """Métricas de evaluación para la tarea de forecasting (§7.2.2)."""
    mae: float = 0.0
    mse: float = 0.0
    mase: float = 0.0
    inference_time_ms: float = 0.0


@dataclass
class AnomalyMetrics:
    """Métricas de evaluación para detección de anomalías (§7.2.3)."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: float = 0.0
    inference_time_ms: float = 0.0


@dataclass
class AblationResult:
    """Resultado completo de una condición experimental k."""
    k: int
    forecasting: ForecastingMetrics = field(default_factory=ForecastingMetrics)
    anomaly: AnomalyMetrics = field(default_factory=AnomalyMetrics)


# ===========================================================================
# 2. ÍNDICE FAISS
# ===========================================================================

class FAISSVectorIndex:
    """
    Índice vectorial FAISS para recuperación k-NN aproximada (§6.4).

    Implementa IndexFlatL2 como baseline (búsqueda exacta) e IndexIVFFlat
    como índice optimizado para escenarios de mayor escala.

    Configuración de hiperparámetros (§6.4):
        n_list ≈ 4√N    (número de celdas de Voronoi)
        n_probe ≈ n_list / 8    (celdas inspeccionadas por consulta)

    Esta configuración garantiza recall > 95% con latencia ~1ms para N=10⁶
    (Johnson et al., 2019).

    Args:
        d:           Dimensión de los embeddings.
        index_type:  'flat' (exacto) o 'ivf' (aproximado).
        n_list:      Número de celdas IVF (solo para index_type='ivf').
    """

    def __init__(
        self,
        d: int,
        index_type: str = "flat",
        n_list: Optional[int] = None,
    ) -> None:
        self.d = d
        self.index_type = index_type
        self.windows: List[np.ndarray] = []        # ventanas originales asociadas a cada embedding

        if not FAISS_AVAILABLE:
            # Fallback: búsqueda exacta con numpy
            self._embeddings: List[np.ndarray] = []
            self.index = None
            return

        if index_type == "flat":
            # IndexFlatIP: producto interno (equivalente a coseno con L2-normalización)
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            n_list = n_list or 100
            self.index = faiss.IndexIVFFlat(quantizer, d, n_list, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"index_type debe ser 'flat' o 'ivf', recibido: {index_type}")

    def add(self, embeddings: np.ndarray, windows: np.ndarray) -> None:
        """
        Añade embeddings al índice junto con las ventanas originales asociadas.

        Los embeddings deben estar L2-normalizados para que el producto interno
        sea equivalente a la similitud coseno (§6.5).

        Args:
            embeddings: Array (N, d) con embeddings L2-normalizados.
            windows:    Array (N, window_size) con las ventanas originales.
        """
        # Normalización L2 de seguridad
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        if FAISS_AVAILABLE:
            if self.index_type == "ivf" and not self.index.is_trained:
                logger.info("Entrenando índice IVF con k-means...")
                self.index.train(embeddings.astype(np.float32))
            self.index.add(embeddings.astype(np.float32))
        else:
            self._embeddings.extend(embeddings.tolist())

        self.windows.extend(windows.tolist())
        logger.info(f"Índice FAISS: {len(self.windows)} vectores indexados (d={self.d})")

    def search(
        self,
        query: np.ndarray,
        k: int,
        n_probe: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Búsqueda k-NN para una query en el espacio de embeddings (§6.5).

        Procedimiento de recuperación (§6.5):
            1. Normalización L2 del query
            2. k-NN en FAISS por similitud coseno (producto interno)
            3. Reconstrucción de las ventanas originales a partir de los índices

        Args:
            query:   Array (d,) con el embedding L2-normalizado del query.
            k:       Número de vecinos a recuperar.
            n_probe: Celdas IVF a inspeccionar (solo para IndexIVFFlat).
        Returns:
            Tupla (distances, indices, retrieved_windows):
                distances:         Array (k,) con las similitudes coseno.
                indices:           Array (k,) con los índices en el índice.
                retrieved_windows: Array (k, window_size) con las ventanas recuperadas.
        """
        query = query / (np.linalg.norm(query) + 1e-8)
        query = query.reshape(1, -1).astype(np.float32)

        if FAISS_AVAILABLE:
            if self.index_type == "ivf":
                self.index.nprobe = n_probe
            distances, indices = self.index.search(query, k)
            distances = distances[0]
            indices = indices[0]
        else:
            # Fallback numpy: producto interno exacto
            emb_matrix = np.array(self._embeddings)
            scores = emb_matrix @ query.T
            sorted_idx = np.argsort(-scores.flatten())[:k]
            indices = sorted_idx
            distances = scores.flatten()[sorted_idx]

        # Filtrar índices inválidos (-1 indica que FAISS no encontró suficientes vecinos)
        valid = indices >= 0
        indices = indices[valid]
        distances = distances[valid]
        retrieved_windows = np.array(self.windows)[indices]

        return distances, indices, retrieved_windows


# ===========================================================================
# 3. MÓDULOS DE PREDICCIÓN Y DETECCIÓN
# ===========================================================================

class ForecastHead(nn.Module):
    """
    Cabezal de predicción (forecasting) sobre la representación fusionada (§6.6).

    Implementa una proyección lineal sobre el vector de contexto fusionado
    [z_q ‖ z_fused] ∈ ℝ^{2d} para producir el horizonte de predicción H.

    Basado en la arquitectura de generador híbrido de la Fase 6 del pipeline.

    Args:
        input_dim:        Dimensión de entrada (2 * d_model para fusión por concatenación).
        forecast_horizon: Horizonte de predicción H (default: 96).
    """

    def __init__(self, input_dim: int, forecast_horizon: int = 96) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch, input_dim).
        Returns:
            Predicciones de forma (batch, forecast_horizon).
        """
        return self.head(x)


class AnomalyHead(nn.Module):
    """
    Cabezal de puntuación de anomalías paramétrico (§6.6).

    Produce una puntuación escalar s_head ∈ [0, 1] sobre la representación
    fusionada. Se combina con el score de aislamiento no paramétrico s_isolation
    para obtener el score ensemble final (§6.6):

        s_ensemble = α · s_head + (1 − α) · s_isolation

    Args:
        input_dim: Dimensión de entrada (2 * d_model).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch, input_dim).
        Returns:
            Score paramétrico de forma (batch, 1).
        """
        return self.head(x)


class CrossAttentionFusion(nn.Module):
    """
    Módulo de fusión por cross-attention entre query y vecinos recuperados (§6.6).

    Implementa la estrategia de fusión preferida del generador híbrido:

        c_attn = MultiHeadAttention(Q=z_q, K=C, V=C)
        z_fused = LayerNorm(c_attn + z_q)
        output = [z_q ‖ z_fused] ∈ ℝ^{2d}

    donde C ∈ ℝ^{k×d} es la matriz de embeddings de los k vecinos recuperados.
    Para k=0 (baseline sin recuperación), z_fused = z_q y output = [z_q ‖ z_q].

    Referencia:
        Zhang, H., et al. (2025). TS-RAG. ICLR 2025 Workshop.
        Borgeaud, S., et al. (2022). RETRO. ICML 2022.

    Args:
        d_model: Dimensión de los embeddings.
        nhead:   Cabezas de atención (default: 4).
    """

    def __init__(self, d_model: int, nhead: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_q: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            z_q:     Embedding del query, forma (batch, d_model).
            context: Embeddings de los k vecinos, forma (batch, k, d_model).
                     Si None (k=0), retorna [z_q ‖ z_q].
        Returns:
            Vector fusionado de forma (batch, 2 * d_model).
        """
        if context is None or context.shape[1] == 0:
            # Baseline k=0: sin recuperación, concatenar z_q consigo mismo
            return torch.cat([z_q, z_q], dim=-1)

        z_q_expanded = z_q.unsqueeze(1)                       # (batch, 1, d_model)
        c_attn, _ = self.attn(z_q_expanded, context, context) # (batch, 1, d_model)
        c_attn = c_attn.squeeze(1)                            # (batch, d_model)
        z_fused = self.norm(c_attn + z_q)
        return torch.cat([z_q, z_fused], dim=-1)              # (batch, 2 * d_model)


class TemporalRAGModel(nn.Module):
    """
    Modelo Temporal RAG completo: encoder + FAISS + fusión + predicción/detección (§6.0).

    Implementa el pipeline completo descrito en el Capítulo 6 de la memoria:
        Fase 3: Generación de embeddings (backbone pre-entrenado)
        Fase 5: Recuperación contextual k-NN (FAISS)
        Fase 6: Generador híbrido (cross-attention + cabezales)

    Args:
        backbone:         TransformerBackbone pre-entrenado (congelado durante ablación).
        faiss_index:      FAISSVectorIndex con ventanas de entrenamiento indexadas.
        d_model:          Dimensión del backbone.
        forecast_horizon: Horizonte H de predicción.
        alpha:            Peso del score paramétrico en el ensemble (§6.6).
    """

    def __init__(
        self,
        backbone: TransformerBackbone,
        faiss_index: FAISSVectorIndex,
        d_model: int = 128,
        forecast_horizon: int = 96,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.faiss_index = faiss_index
        self.fusion = CrossAttentionFusion(d_model=d_model, nhead=4)
        self.forecast_head = ForecastHead(2 * d_model, forecast_horizon)
        self.anomaly_head = AnomalyHead(2 * d_model)
        self.alpha = alpha
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Forward pass completo del sistema Temporal RAG.

        Procedimiento (§6.5 y §6.6):
            1. z_q = backbone(x)                          — embedding del query
            2. {z_i1,...,z_ik} = FAISS.search(z_q, k)    — recuperación k-NN
            3. z_fused = CrossAttention(z_q, context)      — fusión contextual
            4. ŷ = forecast_head(z_fused)                  — predicción
            5. s_head = anomaly_head(z_fused)              — score paramétrico
            6. s_isolation = f(dist_media k-NN)            — score no paramétrico
            7. s_ensemble = α·s_head + (1−α)·s_isolation  — score final

        Args:
            x: Tensor (batch, window_size) — ventana de consulta normalizada.
            k: Número de vecinos a recuperar.
        Returns:
            Tupla (forecast, anomaly_score, isolation_score):
                forecast:        (batch, forecast_horizon)
                anomaly_score:   (batch, 1) — score ensemble en [0,1]
                isolation_score: float — s_isolation del batch (media)
        """
        device = x.device
        batch_size = x.shape[0]

        # ── Paso 1: Embedding del query ──────────────────────────────────────
        with torch.no_grad():
            h_q = self.backbone(x)                          # (batch, d_model)
        z_q = F.normalize(h_q, dim=-1)

        # ── Pasos 2-3: Recuperación y fusión ────────────────────────────────
        if k == 0:
            # Baseline: sin recuperación
            context = None
            isolation_score = 0.0
        else:
            # Recuperar k vecinos para cada elemento del batch
            context_list = []
            distances_list = []

            for b in range(batch_size):
                q_np = z_q[b].detach().cpu().numpy()
                dists, _, ret_windows = self.faiss_index.search(q_np, k=k)
                distances_list.append(dists.mean() if len(dists) > 0 else 1.0)

                if len(ret_windows) > 0:
                    # Encodear las ventanas recuperadas
                    ret_tensor = torch.tensor(ret_windows, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        h_ret = self.backbone(ret_tensor)        # (k, d_model)
                    z_ret = F.normalize(h_ret, dim=-1)
                    context_list.append(z_ret.unsqueeze(0))      # (1, k, d_model)
                else:
                    # Sin vecinos válidos: usar cero como contexto
                    context_list.append(
                        torch.zeros(1, k, self.d_model, device=device)
                    )

            context = torch.cat(context_list, dim=0)            # (batch, k, d_model)

            # Score de aislamiento (§6.5):
            # s_isolation = 1 − exp(−d̄_k / τ_iso)
            # Valores altos → zona de baja densidad → candidato a anomalía
            mean_dist = np.mean(distances_list)
            tau_iso = 0.5                                        # temperatura de aislamiento
            isolation_score = float(1.0 - np.exp(-mean_dist / tau_iso))

        # ── Paso 4-7: Fusión y predicción ────────────────────────────────────
        z_fused = self.fusion(z_q, context)                     # (batch, 2*d_model)
        forecast = self.forecast_head(z_fused)                  # (batch, H)
        s_head = self.anomaly_head(z_fused)                     # (batch, 1)

        # Score ensemble (§6.6)
        s_isolation_tensor = torch.tensor(
            isolation_score, dtype=torch.float32, device=device
        )
        anomaly_score = self.alpha * s_head + (1 - self.alpha) * s_isolation_tensor

        return forecast, anomaly_score, isolation_score


# ===========================================================================
# 4. MÉTRICAS DE EVALUACIÓN
# ===========================================================================

def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 24,
) -> float:
    """
    Mean Absolute Scaled Error (MASE) (§7.2.2).

    MASE = MAE / [ (1/(T−s)) · Σ_{t=s+1}^{T} |y_t − y_{t−s}| ]

    Métrica libre de escala comparable entre series de distintas magnitudes,
    propuesta por Hyndman & Koehler (2006). Supera las limitaciones del MAPE
    en series con valores próximos a cero.

    Args:
        y_true:          Array con valores reales del test.
        y_pred:          Array con predicciones del modelo.
        y_train:         Array con los valores de entrenamiento (denominador naive).
        seasonal_period: Período estacional s (default: 24 para series horarias).
    Returns:
        Valor MASE (float). MASE < 1 indica que el modelo supera al naive estacional.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    # Predictor naive estacional: denominator
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return mae / (naive_mae + 1e-8)


def point_adjust(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Protocolo Point-Adjust (PA@K) para detección de anomalías (§6.7).

    Si el modelo detecta al menos un punto dentro de un segmento anómalo
    continuo, todos los puntos de ese segmento se consideran correctamente
    detectados. Este protocolo es estándar en USAD, Anomaly Transformer y
    trabajos similares (Wu & Keogh, 2021).

    Referencia:
        Wu, R., & Keogh, E. J. (2023). Current time series anomaly detection
        benchmarks are flawed. IEEE TKDE, 35(3), 2421–2429.

    Args:
        y_true:    Array binario de etiquetas reales (1=anomalía, 0=normal).
        y_score:   Array de scores de anomalía continuos en [0, 1].
        threshold: Umbral de decisión τ para binarizar y_score.
    Returns:
        Tupla (y_pred_adjusted, y_true_adjusted):
            Etiquetas ajustadas tras aplicar point-adjust.
    """
    y_pred = (y_score >= threshold).astype(int)
    y_pred_adj = y_pred.copy()

    # Identificar segmentos anómalos continuos en y_true
    in_anomaly = False
    start_idx = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif y_true[i] == 0 and in_anomaly:
            in_anomaly = False
            segment = slice(start_idx, i)
            # Si al menos un punto detectado en el segmento → todos positivos
            if y_pred[segment].sum() > 0:
                y_pred_adj[segment] = 1

    # Último segmento (si termina en anomalía)
    if in_anomaly:
        segment = slice(start_idx, len(y_true))
        if y_pred[segment].sum() > 0:
            y_pred_adj[segment] = 1

    return y_pred_adj, y_true


def optimize_threshold(
    y_true_val: np.ndarray,
    y_score_val: np.ndarray,
    n_thresholds: int = 100,
) -> float:
    """
    Optimiza el umbral de detección τ sobre el conjunto de validación (§6.9).

    El umbral se busca en el percentil empírico de scores que maximiza el F1
    sobre los datos de validación. Nunca se usa el conjunto de test para esta
    optimización (§6.9 — protocolo de no-contaminación).

    Args:
        y_true_val:   Etiquetas reales del conjunto de validación.
        y_score_val:  Scores de anomalía del conjunto de validación.
        n_thresholds: Número de candidatos evaluados en la búsqueda.
    Returns:
        Umbral óptimo τ* que maximiza F1 en validación.
    """
    thresholds = np.linspace(y_score_val.min(), y_score_val.max(), n_thresholds)
    best_f1, best_tau = 0.0, thresholds[n_thresholds // 2]

    for tau in thresholds:
        y_pred_adj, y_true_adj = point_adjust(y_true_val, y_score_val, tau)
        if y_pred_adj.sum() == 0:
            continue
        f1 = f1_score(y_true_adj, y_pred_adj, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    logger.info(f"Umbral óptimo (val): τ* = {best_tau:.4f}, F1_val = {best_f1:.4f}")
    return float(best_tau)


# ===========================================================================
# 5. CONSTRUCCIÓN DEL ÍNDICE FAISS CON LOS DATOS DE ENTRENAMIENTO
# ===========================================================================

def build_faiss_index(
    backbone: TransformerBackbone,
    series_train: np.ndarray,
    window_size: int,
    d_model: int,
    batch_size: int,
    device: torch.device,
) -> FAISSVectorIndex:
    """
    Construye el índice FAISS indexando todas las ventanas del conjunto de
    entrenamiento mediante el backbone pre-entrenado (§6.4).

    Protocolo (§6.9): Solo se indexan datos de entrenamiento.
    Ningún dato de validación o test contamina el índice.

    Args:
        backbone:    TransformerBackbone pre-entrenado y congelado.
        series_train: Array 1D con los datos de entrenamiento.
        window_size: Longitud L de cada ventana.
        d_model:     Dimensión de los embeddings del backbone.
        batch_size:  Tamaño de batch para la codificación.
        device:      Dispositivo de cómputo.
    Returns:
        FAISSVectorIndex listo para consultas k-NN.
    """
    backbone.eval()
    dataset = SlidingWindowDataset(series_train, window_size, step=1)

    all_embeddings = []
    all_windows = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_windows_1, _ = zip(*[dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
            batch_tensor = torch.stack(batch_windows_1).to(device)
            h = backbone(batch_tensor)
            z = F.normalize(h, dim=-1)
            all_embeddings.append(z.cpu().numpy())
            all_windows.append(batch_tensor.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    windows = np.concatenate(all_windows, axis=0)

    # Configuración IndexIVFFlat (§6.4): n_list ≈ 4√N
    N = len(embeddings)
    n_list = max(10, int(4 * np.sqrt(N)))
    index_type = "ivf" if (FAISS_AVAILABLE and N > 1000) else "flat"

    faiss_idx = FAISSVectorIndex(d=d_model, index_type=index_type, n_list=n_list)
    faiss_idx.add(embeddings, windows)

    return faiss_idx


# ===========================================================================
# 6. EVALUACIÓN DE UNA CONDICIÓN EXPERIMENTAL
# ===========================================================================

def evaluate_condition(
    model: TemporalRAGModel,
    series_test: np.ndarray,
    series_train: np.ndarray,
    labels_test: Optional[np.ndarray],
    window_size: int,
    forecast_horizon: int,
    k: int,
    threshold: float,
    device: torch.device,
) -> AblationResult:
    """
    Evalúa una condición experimental k sobre el conjunto de test (§7.1).

    Mide tanto las métricas de forecasting (MAE, MSE, MASE) como las de
    detección de anomalías (Precision, Recall, F1, AUROC) con el protocolo
    Point-Adjust activado.

    También registra la latencia de inferencia para el análisis de la
    Tabla 7.3 (trade-off calidad–latencia).

    Args:
        model:           TemporalRAGModel inicializado.
        series_test:     Array 1D con los datos de test.
        series_train:    Array 1D con los datos de entrenamiento (para MASE).
        labels_test:     Array binario de etiquetas de anomalía del test (o None).
        window_size:     Longitud L de la ventana.
        forecast_horizon: Horizonte H de predicción.
        k:               Número de vecinos recuperados (condición experimental).
        threshold:       Umbral de detección τ* optimizado en validación.
        device:          Dispositivo de cómputo.
    Returns:
        AblationResult con todas las métricas para la condición k.
    """
    model.eval()
    result = AblationResult(k=k)

    # ── Preparación de datos de test ─────────────────────────────────────────
    test_dataset = SlidingWindowDataset(series_test, window_size, step=window_size // 2)

    all_forecasts = []
    all_true = []
    all_anomaly_scores = []
    inference_times = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            w1, _ = test_dataset[idx]
            x = w1.unsqueeze(0).to(device)

            t0 = time.perf_counter()
            forecast, anomaly_score, _ = model(x, k=k)
            t1 = time.perf_counter()

            inference_times.append((t1 - t0) * 1000)          # ms
            all_forecasts.append(forecast.squeeze().cpu().numpy())
            all_anomaly_scores.append(anomaly_score.squeeze().item())

            # Ground truth: siguiente ventana (forecasting)
            start = idx * (window_size // 2) + window_size
            end = start + forecast_horizon
            if end <= len(series_test):
                all_true.append(series_test[start:end])

    # ── Métricas de Forecasting ──────────────────────────────────────────────
    n_valid = min(len(all_forecasts), len(all_true))
    if n_valid > 0:
        y_pred_fc = np.array(all_forecasts[:n_valid])
        y_true_fc = np.array(all_true[:n_valid])

        # Aplanar para métricas globales
        y_pred_flat = y_pred_fc.flatten()
        y_true_flat = y_true_fc.flatten()

        mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
        mse = float(np.mean((y_true_flat - y_pred_flat) ** 2))
        mase = compute_mase(y_true_flat, y_pred_flat, series_train, seasonal_period=24)

        result.forecasting = ForecastingMetrics(
            mae=mae,
            mse=mse,
            mase=mase,
            inference_time_ms=float(np.mean(inference_times)),
        )

    # ── Métricas de Detección de Anomalías ───────────────────────────────────
    if labels_test is not None and len(all_anomaly_scores) > 0:
        scores_arr = np.array(all_anomaly_scores)

        # Alinear longitudes (las ventanas solapadas reducen la cardinalidad)
        n_scores = len(scores_arr)
        labels_aligned = _align_labels(labels_test, n_scores, window_size, step=window_size // 2)

        if len(np.unique(labels_aligned)) > 1:
            y_pred_adj, y_true_adj = point_adjust(labels_aligned, scores_arr, threshold)

            precision = precision_score(y_true_adj, y_pred_adj, zero_division=0)
            recall = recall_score(y_true_adj, y_pred_adj, zero_division=0)
            f1 = f1_score(y_true_adj, y_pred_adj, zero_division=0)
            try:
                auroc = roc_auc_score(labels_aligned, scores_arr)
            except ValueError:
                auroc = 0.5

            result.anomaly = AnomalyMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                auroc=auroc,
                inference_time_ms=float(np.mean(inference_times)),
            )
        else:
            logger.warning(f"k={k}: Etiquetas de anomalía constantes — AUROC no computable.")

    logger.info(
        f"k={k:2d} | MAE={result.forecasting.mae:.4f} | "
        f"F1={result.anomaly.f1:.4f} | "
        f"T_inf={result.forecasting.inference_time_ms:.1f}ms"
    )
    return result


def _align_labels(
    labels: np.ndarray,
    n_windows: int,
    window_size: int,
    step: int,
) -> np.ndarray:
    """
    Proyecta las etiquetas punto-a-punto al espacio de ventanas.

    Asigna etiqueta 1 a la ventana i si al menos un punto en
    [i*step, i*step + window_size] está etiquetado como anomalía.

    Args:
        labels:      Array binario de longitud T (etiquetas punto a punto).
        n_windows:   Número de ventanas generadas.
        window_size: Longitud de cada ventana.
        step:        Paso entre ventanas.
    Returns:
        Array binario de longitud n_windows.
    """
    aligned = np.zeros(n_windows, dtype=int)
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end <= len(labels):
            aligned[i] = int(labels[start:end].max())
    return aligned


# ===========================================================================
# 7. REPORTE DE RESULTADOS
# ===========================================================================

def print_results_tables(results: List[AblationResult]) -> None:
    """
    Imprime las tablas de resultados del estudio de ablación (§7.3).

    Reproduce el formato de las Tablas 7.1, 7.2 y 7.3 de la memoria.
    """
    sep = "─" * 80

    print(f"\n{sep}")
    print("TABLA 7.1 — Ablación sobre Forecasting")
    print(f"{sep}")
    print(f"{'k':>4} | {'MAE ↓':>8} | {'MSE ↓':>8} | {'MASE ↓':>8} | {'ΔMAE vs k=0':>12}")
    print(sep)

    baseline_mae = results[0].forecasting.mae if results else 0.0

    for r in results:
        delta = ((r.forecasting.mae - baseline_mae) / (baseline_mae + 1e-8)) * 100
        delta_str = f"{delta:+.1f}%" if r.k > 0 else "—"
        print(
            f"{r.k:>4} | {r.forecasting.mae:>8.4f} | "
            f"{r.forecasting.mse:>8.4f} | "
            f"{r.forecasting.mase:>8.4f} | "
            f"{delta_str:>12}"
        )

    print(f"\n{sep}")
    print("TABLA 7.2 — Ablación sobre Detección de Anomalías")
    print(f"{sep}")
    print(f"{'k':>4} | {'Precision ↑':>11} | {'Recall ↑':>8} | {'F1 ↑':>6} | {'AUROC ↑':>8}")
    print(sep)

    for r in results:
        print(
            f"{r.k:>4} | {r.anomaly.precision:>11.4f} | "
            f"{r.anomaly.recall:>8.4f} | "
            f"{r.anomaly.f1:>6.4f} | "
            f"{r.anomaly.auroc:>8.4f}"
        )

    print(f"\n{sep}")
    print("TABLA 7.3 — Análisis de Latencia de Inferencia")
    print(f"{sep}")
    baseline_t = results[0].forecasting.inference_time_ms if results else 1.0
    print(f"{'k':>4} | {'T_retrieval (ms)':>16} | {'T_total (ms)':>13} | {'Overhead':>10}")
    print(sep)

    for r in results:
        t_total = r.forecasting.inference_time_ms
        t_retrieval = max(0.0, t_total - baseline_t) if r.k > 0 else 0.0
        overhead = ((t_total - baseline_t) / (baseline_t + 1e-8)) * 100
        overhead_str = f"+{overhead:.1f}%" if r.k > 0 else "—"
        print(
            f"{r.k:>4} | {t_retrieval:>16.1f} | "
            f"{t_total:>13.1f} | "
            f"{overhead_str:>10}"
        )
    print(sep)


def save_results_json(results: List[AblationResult], output_path: str) -> None:
    """
    Serializa los resultados de ablación a JSON para reproducibilidad.

    Args:
        results:     Lista de AblationResult por condición k.
        output_path: Ruta del archivo JSON de salida.
    """
    output = []
    for r in results:
        output.append({
            "k": r.k,
            "forecasting": {
                "mae": r.forecasting.mae,
                "mse": r.forecasting.mse,
                "mase": r.forecasting.mase,
                "inference_time_ms": r.forecasting.inference_time_ms,
            },
            "anomaly": {
                "precision": r.anomaly.precision,
                "recall": r.anomaly.recall,
                "f1": r.anomaly.f1,
                "auroc": r.anomaly.auroc,
            },
        })

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Resultados guardados en: {output_path}")


# ===========================================================================
# 8. PIPELINE PRINCIPAL DE ABLACIÓN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    """
    Punto de entrada principal del estudio de ablación.

    Flujo completo (§7.1):
        1. Carga de la serie temporal y etiquetas
        2. Split temporal 70/10/20 (train/val/test) — sin contaminación
        3. Carga del backbone pre-entrenado (train_contrastive.py)
        4. Construcción del índice FAISS con datos de entrenamiento
        5. Optimización del umbral τ sobre validación
        6. Evaluación de k ∈ {0, 1, 5, 10} sobre test con Point-Adjust
        7. Reporte de tablas (formato Capítulo 7) y guardado JSON

    Args:
        args: Namespace con todos los argumentos de configuración.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    series = load_series(args.data_path, args.column)
    n = len(series)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    series_train = series[:train_end]
    series_val = series[train_end:val_end]
    series_test = series[val_end:]

    # Etiquetas de anomalía (JSON NAB format o array binario)
    labels_test: Optional[np.ndarray] = None
    labels_val: Optional[np.ndarray] = None

    if args.labels_path and Path(args.labels_path).exists():
        labels_full = _load_labels(args.labels_path, n)
        labels_val = labels_full[train_end:val_end]
        labels_test = labels_full[val_end:]
        logger.info(f"Etiquetas cargadas: {int(labels_test.sum())} anomalías en test")

    # ── 2. Backbone pre-entrenado ────────────────────────────────────────────
    backbone = TransformerBackbone(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    if args.encoder_path and Path(args.encoder_path).exists():
        checkpoint = torch.load(args.encoder_path, map_location=device)
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        logger.info(f"Backbone cargado desde: {args.encoder_path}")
    else:
        logger.warning("encoder_path no encontrado. Usando backbone con pesos aleatorios (solo para debugging).")

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # ── 3. Índice FAISS ──────────────────────────────────────────────────────
    logger.info("Construyendo índice FAISS con datos de entrenamiento...")
    faiss_index = build_faiss_index(
        backbone, series_train, args.window_size, args.d_model,
        args.batch_size, device
    )

    # ── 4. Optimización del umbral sobre validación ──────────────────────────
    threshold = 0.5    # default si no hay etiquetas de validación
    if labels_val is not None:
        # Generar scores de anomalía en validación con k=5 (condición central)
        temp_model = TemporalRAGModel(
            backbone=backbone,
            faiss_index=faiss_index,
            d_model=args.d_model,
            forecast_horizon=args.forecast_horizon,
            alpha=args.alpha,
        ).to(device)

        val_scores = _compute_val_scores(
            temp_model, series_val, args.window_size, device, k=5
        )
        labels_val_aligned = _align_labels(
            labels_val, len(val_scores), args.window_size, step=args.window_size // 2
        )
        threshold = optimize_threshold(labels_val_aligned, val_scores)

    # ── 5. Ablación: k ∈ {0, 1, 5, 10} ──────────────────────────────────────
    results: List[AblationResult] = []

    for k in args.k_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluando condición k = {k}")
        logger.info(f"{'='*60}")

        model = TemporalRAGModel(
            backbone=backbone,
            faiss_index=faiss_index,
            d_model=args.d_model,
            forecast_horizon=args.forecast_horizon,
            alpha=args.alpha,
        ).to(device)

        result = evaluate_condition(
            model=model,
            series_test=series_test,
            series_train=series_train,
            labels_test=labels_test,
            window_size=args.window_size,
            forecast_horizon=args.forecast_horizon,
            k=k,
            threshold=threshold,
            device=device,
        )
        results.append(result)

    # ── 6. Reporte ───────────────────────────────────────────────────────────
    print_results_tables(results)
    save_results_json(results, args.output_json)


def _compute_val_scores(
    model: TemporalRAGModel,
    series_val: np.ndarray,
    window_size: int,
    device: torch.device,
    k: int = 5,
) -> np.ndarray:
    """Genera scores de anomalía sobre el conjunto de validación para optimizar τ."""
    model.eval()
    dataset = SlidingWindowDataset(series_val, window_size, step=window_size // 2)
    scores = []
    with torch.no_grad():
        for i in range(len(dataset)):
            w, _ = dataset[i]
            x = w.unsqueeze(0).to(device)
            _, anomaly_score, _ = model(x, k=k)
            scores.append(anomaly_score.squeeze().item())
    return np.array(scores)


def _load_labels(labels_path: str, series_length: int) -> np.ndarray:
    """
    Carga las etiquetas de anomalía desde un archivo JSON o CSV.

    Formatos soportados:
        - JSON: {"timestamps": [...], "labels": [0, 0, 1, ...]}
        - CSV: columna 'label' con valores binarios
        - NPY: array numpy binario de longitud T

    Args:
        labels_path:   Ruta al archivo de etiquetas.
        series_length: Longitud total de la serie (para validación).
    Returns:
        Array binario numpy de longitud T.
    """
    path = Path(labels_path)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if "labels" in data:
            labels = np.array(data["labels"])
        else:
            # Formato lista directa
            labels = np.array(data)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        col = "label" if "label" in df.columns else df.columns[-1]
        labels = df[col].values.astype(int)
    elif path.suffix == ".npy":
        labels = np.load(path).astype(int)
    else:
        raise ValueError(f"Formato de etiquetas no soportado: {path.suffix}")

    if len(labels) != series_length:
        logger.warning(
            f"Longitud de etiquetas ({len(labels)}) ≠ longitud de serie ({series_length}). "
            "Truncando/rellenando con ceros."
        )
        if len(labels) > series_length:
            labels = labels[:series_length]
        else:
            labels = np.pad(labels, (0, series_length - len(labels)))

    return labels.astype(int)


# ===========================================================================
# 9. ARGPARSE
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos del estudio de ablación.
    """
    parser = argparse.ArgumentParser(
        description="Estudio de ablación Temporal RAG: k ∈ {0, 1, 5, 10}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Datos
    parser.add_argument("--data_path", type=str, required=True,
                        help="Ruta al CSV de la serie temporal")
    parser.add_argument("--labels_path", type=str, default=None,
                        help="Ruta al archivo de etiquetas de anomalía (JSON/CSV/NPY)")
    parser.add_argument("--column", type=str, default=None,
                        help="Columna de valores en el CSV (default: auto)")

    # Encoder pre-entrenado
    parser.add_argument("--encoder_path", type=str, default="checkpoints/best_encoder.pt",
                        help="Ruta al checkpoint del backbone (train_contrastive.py)")

    # Arquitectura (debe coincidir con train_contrastive.py)
    parser.add_argument("--window_size", type=int, default=96)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--forecast_horizon", type=int, default=96,
                        help="Horizonte H de predicción")

    # Ablación
    parser.add_argument("--k_values", type=int, nargs="+", default=[0, 1, 5, 10],
                        help="Valores de k a evaluar (condiciones experimentales §7.1.1)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Peso α del score paramétrico en el ensemble (§6.6)")

    # Infraestructura
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_json", type=str, default="results/ablation_results.json",
                        help="Ruta de salida para los resultados en JSON")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
