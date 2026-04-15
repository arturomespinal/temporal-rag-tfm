"""
Temporal RAG — Experimento de Ablación (k=0, 1, 5, 10)
=======================================================
Script que ejecuta el experimento de ablación completo sobre el sistema
Temporal RAG, variando la cardinalidad k del conjunto recuperado.
Genera las tablas de resultados en formato CSV y LaTeX para la memoria del TFM.

Diseño del experimento:
    - k=0  → Baseline sin recuperación (solo memoria paramétrica)
    - k=1  → Recuperación mínima (1 vecino más cercano)
    - k=5  → Recuperación moderada (5 vecinos)
    - k=10 → Recuperación máxima (10 vecinos)

Autor: Arturo Miguel Espinal Reyes
TFM: RAG Temporal — Máster en Ciencia de Datos, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import faiss
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

logger = logging.getLogger("temporal_rag.ablation")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24,
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Escala el MAE por el error naive estacional in-sample, produciendo
    una métrica libre de escala comparable entre series.

    Args:
        y_true:     Valores reales del horizonte de predicción.
        y_pred:     Predicciones del modelo.
        y_train:    Serie de entrenamiento (para calcular el denominador).
        seasonality: Período estacional s (default 24 para datos horarios).

    Returns:
        Valor MASE escalar.
    """
    naive_error = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    return float(np.mean(np.abs(y_true - y_pred)) / (naive_error + 1e-8))


def point_adjust_f1(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Calcula Precision, Recall y F1 con el protocolo Point-Adjust (PA@K).

    En el protocolo PA@K, si cualquier punto dentro de un segmento anómalo
    contiguo es detectado, todos los puntos del segmento se marcan como
    correctamente detectados. Esto evita penalizar detecciones temporalmente
    desplazadas dentro del mismo evento anómalo.

    Referencia: Wu & Keogh (2023). "Current time series anomaly detection
    benchmarks are flawed and are creating the illusion of progress."
    IEEE TKDE.

    Args:
        y_true:         Etiquetas binarias (1=anomalía, 0=normal).
        anomaly_scores: Puntuaciones de anomalía del modelo.
        threshold:      Umbral de decisión.

    Returns:
        Diccionario con precision, recall, f1.
    """
    y_pred = (anomaly_scores > threshold).astype(int)

    # Point-adjust: propagar detecciones dentro de segmentos anómalos
    y_pred_adjusted = y_pred.copy()
    in_anomaly = False
    segment_detected = False
    segment_start = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            segment_start = i
            segment_detected = False
        if in_anomaly and y_pred[i] == 1:
            segment_detected = True
        if (y_true[i] == 0 or i == len(y_true) - 1) and in_anomaly:
            if segment_detected:
                y_pred_adjusted[segment_start:i] = 1
            in_anomaly = False

    return {
        "precision": float(precision_score(y_true, y_pred_adjusted, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_adjusted, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_adjusted, zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Índice Vectorial FAISS
# ─────────────────────────────────────────────────────────────────────────────

class TemporalVectorIndex:
    """
    Índice vectorial FAISS para recuperación de segmentos temporales similares.

    Envuelve un índice FAISS IVF-Flat (Inverted File con cuantización exacta)
    que permite búsqueda aproximada eficiente en el espacio de embeddings del
    encoder temporal. El índice almacena los embeddings de las ventanas
    temporales del conjunto de entrenamiento y recupera los k más cercanos
    durante la inferencia.

    Args:
        embed_dim:  Dimensión del espacio de embedding.
        n_lists:    Número de listas del índice IVF (sqrt(N) es heurística estándar).
        metric:     "cosine" o "l2".
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_lists: int = 100,
        metric: str = "cosine",
    ) -> None:
        self.embed_dim = embed_dim
        self.metric = metric

        if metric == "cosine":
            # Para similitud coseno se usa IndexFlatIP con vectores L2-normalizados
            quantizer = faiss.IndexFlatIP(embed_dim)
        else:
            quantizer = faiss.IndexFlatL2(embed_dim)

        self.index = faiss.IndexIVFFlat(
            quantizer,
            embed_dim,
            n_lists,
            faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2,
        )
        self._trained = False
        self._stored_windows: Optional[np.ndarray] = None

    def build(self, embeddings: np.ndarray, windows: np.ndarray) -> None:
        """
        Construye el índice a partir de los embeddings y ventanas temporales.

        Args:
            embeddings: Array (N, embed_dim) con los embeddings del encoder.
            windows:    Array (N, T, F) con las ventanas temporales originales.
        """
        if self.metric == "cosine":
            # L2-normalizar para similitud coseno exacta via producto interno
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            embeddings = embeddings / norms

        self.index.train(embeddings.astype(np.float32))
        self.index.add(embeddings.astype(np.float32))
        self._stored_windows = windows
        self._trained = True
        logger.info("Índice construido con %d vectores (dim=%d)", len(embeddings), self.embed_dim)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recupera los k segmentos más similares al embedding de consulta.

        Args:
            query_embedding: Array (embed_dim,) o (1, embed_dim).
            k:               Número de vecinos a recuperar.

        Returns:
            Tupla (scores, windows) donde scores es (k,) y windows es (k, T, F).
        """
        assert self._trained, "El índice debe construirse antes de recuperar."
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]

        if self.metric == "cosine":
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
            query_embedding = query_embedding / norm

        self.index.nprobe = min(10, self.index.nlist)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        retrieved_windows = self._stored_windows[indices[0]]
        return scores[0], retrieved_windows


# ─────────────────────────────────────────────────────────────────────────────
# Módulo de Fusión de Contexto Recuperado
# ─────────────────────────────────────────────────────────────────────────────

class ContextFusionModule(nn.Module):
    """
    Módulo de fusión entre la representación de la consulta y el contexto recuperado.

    Implementa un mecanismo de atención cruzada (cross-attention) donde la
    representación de la ventana de consulta actúa como query y los embeddings
    de los k segmentos recuperados actúan como keys y values. Produce una
    representación enriquecida contextualmente para forecasting o detección.

    Este diseño es análogo al decoder de un Transformer encoder-decoder
    (Vaswani et al., 2017) pero aplicado al espacio de embedding temporal.

    Args:
        d_model:   Dimensión del embedding.
        nhead:     Número de cabezas de atención cruzada.
        dropout:   Tasa de dropout.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        context_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fusión por atención cruzada con gate residual.

        Args:
            query_emb:    Embedding de consulta, forma (B, 1, d_model).
            context_embs: Embeddings recuperados, forma (B, k, d_model).

        Returns:
            Representación fusionada, forma (B, d_model).
        """
        attended, _ = self.cross_attention(
            query=query_emb,
            key=context_embs,
            value=context_embs,
        )                                                   # (B, 1, d_model)
        attended = attended.squeeze(1)                      # (B, d_model)
        query = query_emb.squeeze(1)                        # (B, d_model)

        # Gate residual: pondera cuánto contexto incorporar
        gate_weight = self.gate(torch.cat([query, attended], dim=-1))
        fused = gate_weight * attended + (1 - gate_weight) * query
        return self.norm(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Experimento de Ablación
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_experiment(
    encoder,
    index: TemporalVectorIndex,
    fusion_module: ContextFusionModule,
    predictor: nn.Module,
    test_windows: np.ndarray,
    test_labels: Optional[np.ndarray],
    test_targets: Optional[np.ndarray],
    k_values: List[int],
    device: torch.device,
    task: str = "forecasting",
    anomaly_threshold: float = 0.5,
    train_data: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Ejecuta el experimento de ablación completo sobre k_values.

    Para cada valor de k en k_values, ejecuta inferencia sobre el conjunto
    de test, calcula las métricas correspondientes y registra la latencia.

    Args:
        encoder:           TemporalEncoder entrenado.
        index:             TemporalVectorIndex poblado con embeddings de train.
        fusion_module:     ContextFusionModule para integrar contexto recuperado.
        predictor:         Cabezal de predicción/detección.
        test_windows:      Ventanas de test, forma (N_test, T, F).
        test_labels:       Etiquetas de anomalía binarias (solo para task="anomaly").
        test_targets:      Targets de forecasting (solo para task="forecasting").
        k_values:          Lista de valores k a evaluar, e.g. [0, 1, 5, 10].
        device:            Dispositivo de cómputo.
        task:              "forecasting" o "anomaly".
        anomaly_threshold: Umbral de decisión para detección de anomalías.
        train_data:        Serie de entrenamiento para calcular MASE.

    Returns:
        DataFrame con resultados del experimento de ablación.
    """
    results = []
    encoder.eval()
    fusion_module.eval()
    predictor.eval()

    for k in k_values:
        logger.info("Evaluando k=%d...", k)
        all_preds, all_targets, retrieval_times = [], [], []

        with torch.no_grad():
            for window in test_windows:
                window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

                # Encoding de la consulta
                query_emb = encoder.encode(window_t)           # (1, d_model)

                if k == 0:
                    # Baseline: sin recuperación
                    fused_emb = query_emb
                    retrieval_times.append(0.0)
                else:
                    # Recuperación de k vecinos
                    t0 = time.perf_counter()
                    _, retrieved_windows = index.retrieve(
                        query_emb.cpu().numpy().squeeze(), k=k
                    )
                    t1 = time.perf_counter()
                    retrieval_times.append((t1 - t0) * 1000)  # ms

                    # Encoding de los vecinos recuperados
                    retrieved_t = torch.tensor(
                        retrieved_windows, dtype=torch.float32
                    ).to(device)
                    context_embs = encoder.encode(retrieved_t)  # (k, d_model)

                    # Fusión: query + contexto
                    fused_emb = fusion_module(
                        query_emb.unsqueeze(1),                 # (1, 1, d_model)
                        context_embs.unsqueeze(0),              # (1, k, d_model)
                    )                                           # (1, d_model)

                # Predicción
                pred = predictor(fused_emb)
                all_preds.append(pred.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        avg_retrieval_ms = float(np.mean(retrieval_times))

        # ── Calcular métricas ─────────────────────────────────────────
        row = {
            "k": k,
            "avg_retrieval_ms": round(avg_retrieval_ms, 2),
        }

        if task == "forecasting" and test_targets is not None:
            row["MAE"] = round(mae(test_targets, all_preds), 4)
            row["MSE"] = round(mse(test_targets, all_preds), 4)
            if train_data is not None:
                row["MASE"] = round(mase(test_targets, all_preds, train_data), 4)
            # Mejora relativa respecto al baseline (k=0)
            if results:
                baseline_mae = results[0]["MAE"]
                row["delta_MAE_pct"] = round(
                    (baseline_mae - row["MAE"]) / baseline_mae * 100, 2
                )
            else:
                row["delta_MAE_pct"] = 0.0

        elif task == "anomaly" and test_labels is not None:
            metrics = point_adjust_f1(test_labels, all_preds.squeeze(), anomaly_threshold)
            row.update(metrics)
            try:
                row["AUROC"] = round(
                    roc_auc_score(test_labels, all_preds.squeeze()), 4
                )
            except ValueError:
                row["AUROC"] = float("nan")

        results.append(row)
        logger.info("k=%d resultados: %s", k, {k: v for k, v in row.items() if k != "k"})

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, output_dir: Path, task: str) -> None:
    """
    Guarda los resultados en CSV y genera tabla LaTeX para la memoria del TFM.

    Args:
        df:         DataFrame con resultados de la ablación.
        output_dir: Directorio de salida.
        task:       "forecasting" o "anomaly".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / f"ablation_{task}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Resultados CSV guardados en: %s", csv_path)

    # LaTeX
    latex_path = output_dir / f"ablation_{task}.tex"
    latex_str = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=f"Resultados del experimento de ablación sobre {task} (Temporal RAG).",
        label=f"tab:ablation_{task}",
        escape=False,
        bold_rows=False,
    )
    with open(latex_path, "w") as f:
        f.write(latex_str)
    logger.info("Tabla LaTeX guardada en: %s", latex_path)

    # Resumen en consola
    print("\n" + "=" * 60)
    print(f"RESULTADOS ABLACIÓN — {task.upper()}")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para el experimento de ablación."""
    parser = argparse.ArgumentParser(
        description="Experimento de Ablación Temporal RAG (k=0,1,5,10)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--encoder-path", type=str, required=True,
                        help="Ruta al checkpoint del encoder entrenado.")
    parser.add_argument("--index-path", type=str, required=True,
                        help="Ruta al índice FAISS serializado (.index).")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Ruta al .npy con datos de test (N, T, F).")
    parser.add_argument("--test-labels", type=str, default=None,
                        help="Ruta al .npy con etiquetas de anomalía (N,).")
    parser.add_argument("--test-targets", type=str, default=None,
                        help="Ruta al .npy con targets de forecasting (N, H).")
    parser.add_argument("--task", type=str, default="forecasting",
                        choices=["forecasting", "anomaly"])
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 5, 10])
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Iniciando experimento de ablación con k=%s", args.k_values)

    # Cargar datos de test
    test_windows = np.load(args.test_data)
    test_labels = np.load(args.test_labels) if args.test_labels else None
    test_targets = np.load(args.test_targets) if args.test_targets else None

    logger.info(
        "Datos cargados — test: %s | labels: %s | targets: %s",
        test_windows.shape,
        test_labels.shape if test_labels is not None else "N/A",
        test_targets.shape if test_targets is not None else "N/A",
    )

    logger.info(
        "Ejecutar con encoder_path=%s e index_path=%s para completar el experimento.",
        args.encoder_path,
        args.index_path,
    )
    logger.info("Ver función run_ablation_experiment() para la lógica completa.")
