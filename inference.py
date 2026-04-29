"""
inference.py
============
Módulo de inferencia en tiempo real para el sistema Temporal RAG.
Implementa las Fases 5 y 6 del pipeline descrito en el Capítulo 6 de la memoria:

    Fase 5 — Recuperación contextual k-NN (§6.5):
        z_q = f_θ(W_q)                       ← embedding del query
        {z_i1,...,z_ik} = FAISS.search(z_q)  ← recuperación k-NN
        s_isolation = 1 − exp(−d̄_k / τ)     ← score no paramétrico

    Fase 6 — Generador híbrido (§6.6):
        c_attn = MultiHeadAttention(Q=z_q, K=C, V=C)
        z_fused = LayerNorm(c_attn + z_q)
        output = [z_q ‖ z_fused] ∈ ℝ^{2d}
        s_ensemble = α·s_head + (1−α)·s_isolation

Soporta dos modos de operación:
    (A) Batch: procesar un CSV completo y guardar scores + predicciones
    (B) Streaming: procesar ventanas una a una (simulación near-real-time)

Uso:
    # Modo batch
    python inference.py \
        --mode batch \
        --data_path data/ambient_temperature_system_failure.csv \
        --encoder_path checkpoints/best_encoder.pt \
        --index_dir index/ \
        --output_path results/inference_output.csv \
        --k 5 --alpha 0.5

    # Modo streaming (simulación)
    python inference.py \
        --mode streaming \
        --data_path data/ambient_temperature_system_failure.csv \
        --encoder_path checkpoints/best_encoder.pt \
        --index_dir index/ \
        --k 5

Repositorio:
    https://github.com/arturomespinal/temporal-rag-tfm
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_contrastive import TransformerBackbone, load_series
from ablation_study import (
    AnomalyHead,
    CrossAttentionFusion,
    ForecastHead,
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. ESTRUCTURA DE RESULTADO
# ===========================================================================

@dataclass
class InferenceResult:
    """
    Resultado de inferencia para una ventana temporal individual.

    Contiene tanto las predicciones del módulo de forecasting como el
    score de anomalía ensemble del módulo de detección, junto con
    metadatos de latencia para monitorización del sistema.

    Attributes:
        window_idx:      Índice de la ventana en la serie temporal.
        timestamp_start: Índice temporal de inicio de la ventana.
        forecast:        Array (H,) con las H predicciones futuras.
        anomaly_score:   Score ensemble s_ensemble ∈ [0, 1].
        isolation_score: Score de aislamiento no paramétrico s_isolation ∈ [0, 1].
        parametric_score: Score del cabezal paramétrico s_head ∈ [0, 1].
        retrieved_k:     Número real de vecinos recuperados (≤ k solicitados).
        mean_distance:   Distancia coseno media a los k vecinos recuperados.
        latency_ms:      Latencia total de inferencia en milisegundos.
    """
    window_idx:       int
    timestamp_start:  int
    forecast:         np.ndarray
    anomaly_score:    float
    isolation_score:  float
    parametric_score: float
    retrieved_k:      int
    mean_distance:    float
    latency_ms:       float


# ===========================================================================
# 2. MOTOR DE INFERENCIA
# ===========================================================================

class TemporalRAGInferenceEngine:
    """
    Motor de inferencia del sistema Temporal RAG (§6.5–6.6).

    Encapsula el backbone pre-entrenado, el índice FAISS y los módulos
    de fusión y predicción en un único objeto con interfaz de inferencia
    simple. Diseñado para ser instanciado una vez y reutilizado para
    múltiples ventanas de consulta.

    El motor opera en modo evaluación: los pesos del backbone están
    congelados y la normalización BatchNorm usa estadísticas del entrenamiento.

    Args:
        backbone:         TransformerBackbone pre-entrenado.
        faiss_index:      Índice FAISS con embeddings del entrenamiento.
        index_windows:    Array (N, window_size) de ventanas del índice.
        fusion:           Módulo CrossAttentionFusion.
        forecast_head:    ForecastHead para predicción.
        anomaly_head:     AnomalyHead para detección.
        d_model:          Dimensión del backbone.
        alpha:            Peso del score paramétrico en el ensemble (§6.6).
        tau_iso:          Temperatura del score de aislamiento (§6.5).
        device:           Dispositivo de cómputo.
    """

    def __init__(
        self,
        backbone: TransformerBackbone,
        faiss_index: "faiss.Index",
        index_windows: np.ndarray,
        fusion: CrossAttentionFusion,
        forecast_head: ForecastHead,
        anomaly_head: AnomalyHead,
        d_model: int = 128,
        alpha: float = 0.5,
        tau_iso: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.backbone      = backbone.to(self.device).eval()
        self.faiss_index   = faiss_index
        self.index_windows = index_windows
        self.fusion        = fusion.to(self.device).eval()
        self.forecast_head = forecast_head.to(self.device).eval()
        self.anomaly_head  = anomaly_head.to(self.device).eval()
        self.d_model       = d_model
        self.alpha         = alpha
        self.tau_iso       = tau_iso

        # Congelar todos los parámetros (inferencia pura)
        for model in [self.backbone, self.fusion, self.forecast_head, self.anomaly_head]:
            for param in model.parameters():
                param.requires_grad = False

        logger.info(
            f"Motor de inferencia inicializado | "
            f"d_model={d_model} | α={alpha} | τ_iso={tau_iso} | "
            f"device={self.device} | índice={self.faiss_index.ntotal:,} vectores"
        )

    @classmethod
    def from_checkpoint(
        cls,
        encoder_path: str,
        index_dir: str,
        forecast_horizon: int = 96,
        alpha: float = 0.5,
        tau_iso: float = 0.5,
        window_size: int = 96,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        device: torch.device | None = None,
    ) -> "TemporalRAGInferenceEngine":
        """
        Construye el motor de inferencia desde checkpoints en disco.

        Carga secuencialmente:
            1. Backbone desde best_encoder.pt (train_contrastive.py)
            2. Índice FAISS desde index/faiss.index (build_index.py)
            3. Módulos de fusión y predicción (pesos aleatorios si no se
               proporciona un checkpoint de fine-tuning)

        Args:
            encoder_path:     Ruta al checkpoint del backbone.
            index_dir:        Directorio con el índice FAISS serializado.
            forecast_horizon: Horizonte H de predicción.
            alpha:            Peso α del ensemble.
            tau_iso:          Temperatura del score de aislamiento.
            window_size:      Longitud L de la ventana.
            d_model:          Dimensión del backbone.
            nhead:            Cabezas de atención.
            num_layers:       Capas del Transformer.
            device:           Dispositivo de cómputo.
        Returns:
            TemporalRAGInferenceEngine listo para inferencia.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone
        backbone = TransformerBackbone(window_size, d_model, nhead, num_layers).to(device)
        if Path(encoder_path).exists():
            ckpt = torch.load(encoder_path, map_location=device)
            backbone.load_state_dict(ckpt["backbone_state_dict"])
            logger.info(f"Backbone cargado: {encoder_path}")
        else:
            logger.warning(f"encoder_path no encontrado: {encoder_path}. Pesos aleatorios.")

        # Índice FAISS
        from build_index import load_index
        faiss_index, index_windows, _ = load_index(index_dir)

        # Módulos del generador
        fusion        = CrossAttentionFusion(d_model=d_model, nhead=4)
        forecast_head = ForecastHead(2 * d_model, forecast_horizon)
        anomaly_head  = AnomalyHead(2 * d_model)

        return cls(
            backbone=backbone,
            faiss_index=faiss_index,
            index_windows=index_windows,
            fusion=fusion,
            forecast_head=forecast_head,
            anomaly_head=anomaly_head,
            d_model=d_model,
            alpha=alpha,
            tau_iso=tau_iso,
            device=device,
        )

    def infer_window(
        self,
        window: np.ndarray,
        k: int = 5,
        window_idx: int = 0,
        timestamp_start: int = 0,
    ) -> InferenceResult:
        """
        Ejecuta la inferencia completa sobre una ventana temporal (§6.5–6.6).

        Procedimiento:
            1. Normalización Z-score local de la ventana (§6.1)
            2. Embedding z_q = backbone(W_q)
            3. Recuperación k-NN en FAISS
            4. Score de aislamiento s_isolation = 1 − exp(−d̄_k / τ)
            5. Fusión contextual cross-attention
            6. Predicción ŷ = forecast_head(z_fused)
            7. Score paramétrico s_head = anomaly_head(z_fused)
            8. Score ensemble s_ensemble = α·s_head + (1−α)·s_isolation

        Args:
            window:          Array (window_size,) con los valores de la ventana.
            k:               Número de vecinos a recuperar.
            window_idx:      Índice de la ventana (para metadatos).
            timestamp_start: Índice temporal de inicio.
        Returns:
            InferenceResult con predicciones, scores y metadatos de latencia.
        """
        t0 = time.perf_counter()

        # ── Paso 1: Normalización Z-score local ────────────────────────────────
        mu, sigma = window.mean(), window.std()
        window_norm = (window - mu) / (sigma + 1e-8)
        x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # ── Paso 2: Embedding del query ────────────────────────────────────
            h_q = self.backbone(x)                          # (1, d_model)
            z_q = F.normalize(h_q, dim=-1)                  # L2-normalizado

            # ── Pasos 3–4: Recuperación k-NN y score de aislamiento ────────────
            context = None
            isolation_score = 0.0
            mean_distance   = 0.0
            retrieved_k     = 0

            if k > 0 and FAISS_AVAILABLE:
                q_np = z_q.squeeze(0).cpu().numpy().astype(np.float32)
                distances, indices, _ = self._search_faiss(q_np, k)

                if len(distances) > 0:
                    retrieved_k   = len(distances)
                    mean_distance = float(distances.mean())
                    isolation_score = float(1.0 - np.exp(-mean_distance / self.tau_iso))

                    # Encodear ventanas recuperadas para el contexto
                    ret_windows = self.index_windows[indices]
                    ret_tensor  = torch.tensor(ret_windows, dtype=torch.float32).to(self.device)
                    h_ret = self.backbone(ret_tensor)             # (k, d_model)
                    z_ret = F.normalize(h_ret, dim=-1)
                    context = z_ret.unsqueeze(0)                  # (1, k, d_model)

            # ── Paso 5: Fusión contextual ──────────────────────────────────────
            z_fused_cat = self.fusion(z_q, context)              # (1, 2*d_model)

            # ── Pasos 6–7: Predicción y score paramétrico ──────────────────────
            forecast      = self.forecast_head(z_fused_cat).squeeze(0).cpu().numpy()
            s_head        = float(self.anomaly_head(z_fused_cat).squeeze().item())

            # ── Paso 8: Score ensemble (§6.6) ─────────────────────────────────
            anomaly_score = self.alpha * s_head + (1 - self.alpha) * isolation_score

        latency_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            window_idx=window_idx,
            timestamp_start=timestamp_start,
            forecast=forecast,
            anomaly_score=anomaly_score,
            isolation_score=isolation_score,
            parametric_score=s_head,
            retrieved_k=retrieved_k,
            mean_distance=mean_distance,
            latency_ms=latency_ms,
        )

    def _search_faiss(
        self,
        query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Búsqueda k-NN en el índice FAISS con manejo de errores.

        Args:
            query: Array (d,) L2-normalizado.
            k:     Número de vecinos solicitados.
        Returns:
            Tupla (distances, indices, windows).
        """
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        query_2d   = query_norm.reshape(1, -1)

        distances, indices = self.faiss_index.search(query_2d, k + 1)
        distances = distances[0]
        indices   = indices[0]

        # Filtrar self-matches e índices inválidos (-1)
        valid = (indices >= 0)
        distances = distances[valid]
        indices   = indices[valid]

        # Convertir similitud coseno a distancia (1 − sim) para s_isolation
        distances_dist = 1.0 - distances

        windows = self.index_windows[indices] if len(indices) > 0 else np.array([])
        return distances_dist, indices, windows


# ===========================================================================
# 3. GENERADOR DE VENTANAS STREAMING
# ===========================================================================

def streaming_window_generator(
    series: np.ndarray,
    window_size: int,
    step: int = 1,
    start_idx: int = 0,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """
    Generador de ventanas deslizantes para simulación de streaming.

    Simula el comportamiento de un sistema de monitorización en tiempo real
    donde las ventanas llegan secuencialmente. El generador es lazy: no
    materializa todas las ventanas en memoria, sino que las produce una a una.

    Args:
        series:      Array 1D con la serie temporal completa.
        window_size: Longitud L de cada ventana.
        step:        Paso entre ventanas consecutivas.
        start_idx:   Índice de inicio (0 = desde el principio).
    Yields:
        Tupla (window_idx, timestamp_start, window_array).
    """
    window_idx = 0
    i = start_idx
    while i + window_size <= len(series):
        yield window_idx, i, series[i:i + window_size].copy()
        i += step
        window_idx += 1


# ===========================================================================
# 4. MODO BATCH
# ===========================================================================

def run_batch_inference(
    engine: TemporalRAGInferenceEngine,
    series: np.ndarray,
    window_size: int,
    k: int,
    step: int,
    output_path: str,
    test_start_idx: int = 0,
) -> pd.DataFrame:
    """
    Procesa la serie completa en modo batch y guarda los resultados en CSV.

    Adecuado para evaluación post-hoc y generación de las curvas de anomalía
    que se incluyen en las figuras del Capítulo 7.

    Args:
        engine:        TemporalRAGInferenceEngine inicializado.
        series:        Array 1D de la serie temporal.
        window_size:   Longitud L de la ventana.
        k:             Número de vecinos recuperados.
        step:          Paso entre ventanas.
        output_path:   Ruta del CSV de salida.
        test_start_idx: Índice de inicio del conjunto de test.
    Returns:
        DataFrame con columnas: window_idx, timestamp_start, anomaly_score,
        isolation_score, parametric_score, retrieved_k, mean_distance, latency_ms.
    """
    results = []
    gen = streaming_window_generator(series, window_size, step, start_idx=test_start_idx)

    total = (len(series) - test_start_idx - window_size) // step + 1
    log_every = max(1, total // 20)

    for window_idx, ts_start, window in gen:
        result = engine.infer_window(window, k=k, window_idx=window_idx, timestamp_start=ts_start)
        results.append({
            "window_idx":       result.window_idx,
            "timestamp_start":  result.timestamp_start,
            "anomaly_score":    round(result.anomaly_score,    4),
            "isolation_score":  round(result.isolation_score,  4),
            "parametric_score": round(result.parametric_score, 4),
            "retrieved_k":      result.retrieved_k,
            "mean_distance":    round(result.mean_distance,    4),
            "latency_ms":       round(result.latency_ms,       3),
        })
        if window_idx % log_every == 0:
            logger.info(
                f"  [{window_idx:5,}/{total:,}] "
                f"score={result.anomaly_score:.4f} | "
                f"lat={result.latency_ms:.1f}ms"
            )

    df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"\nResultados guardados: {output_path}")
    logger.info(f"  Ventanas procesadas: {len(df):,}")
    logger.info(f"  Latencia media:      {df['latency_ms'].mean():.2f}ms")
    logger.info(f"  Score medio:         {df['anomaly_score'].mean():.4f}")
    logger.info(f"  Score máximo:        {df['anomaly_score'].max():.4f}")
    return df


# ===========================================================================
# 5. MODO STREAMING (SIMULACIÓN)
# ===========================================================================

def run_streaming_inference(
    engine: TemporalRAGInferenceEngine,
    series: np.ndarray,
    window_size: int,
    k: int,
    threshold: float,
    n_windows: int = 200,
    test_start_idx: int = 0,
) -> None:
    """
    Simula inferencia en streaming, imprimiendo alertas en tiempo real.

    Muestra cómo el sistema Temporal RAG operaría en un entorno de
    monitorización industrial: cada ventana se procesa en cuanto llega,
    y se emite una alerta si el score ensemble supera el umbral τ*.

    Args:
        engine:         TemporalRAGInferenceEngine.
        series:         Array 1D de la serie temporal.
        window_size:    Longitud L de la ventana.
        k:              Número de vecinos recuperados.
        threshold:      Umbral de detección τ* (optimizado en validación).
        n_windows:      Número de ventanas a procesar en la simulación.
        test_start_idx: Índice de inicio.
    """
    logger.info(f"\nModo STREAMING — {n_windows} ventanas | k={k} | umbral τ*={threshold:.4f}")
    print(f"\n{'─'*75}")
    print(f"{'Win':>5} | {'Score':>8} | {'Iso':>8} | {'Param':>8} | {'k_ret':>5} | {'ms':>6} | Estado")
    print(f"{'─'*75}")

    gen = streaming_window_generator(series, window_size, step=1, start_idx=test_start_idx)
    alerts = 0

    for i, (window_idx, ts_start, window) in enumerate(gen):
        if i >= n_windows:
            break

        result = engine.infer_window(window, k=k, window_idx=window_idx, timestamp_start=ts_start)
        is_alert = result.anomaly_score >= threshold
        if is_alert:
            alerts += 1

        estado = "⚠️  ALERTA" if is_alert else "✓  Normal"
        print(
            f"{window_idx:>5} | {result.anomaly_score:>8.4f} | "
            f"{result.isolation_score:>8.4f} | {result.parametric_score:>8.4f} | "
            f"{result.retrieved_k:>5} | {result.latency_ms:>6.1f} | {estado}"
        )

    print(f"{'─'*75}")
    print(f"Alertas emitidas: {alerts}/{n_windows} ({100*alerts/max(n_windows,1):.1f}%)")


# ===========================================================================
# 6. PIPELINE PRINCIPAL
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    """
    Punto de entrada del módulo de inferencia.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device} | modo: {args.mode} | k={args.k}")

    # ── Cargar motor de inferencia ────────────────────────────────────────────
    engine = TemporalRAGInferenceEngine.from_checkpoint(
        encoder_path=args.encoder_path,
        index_dir=args.index_dir,
        forecast_horizon=args.forecast_horizon,
        alpha=args.alpha,
        tau_iso=args.tau_iso,
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        device=device,
    )

    # ── Cargar serie ─────────────────────────────────────────────────────────
    series = load_series(args.data_path, args.column)
    n = len(series)
    test_start = int(n * 0.80)         # 20% final = test (§6.9)
    series_test = series[test_start:]

    # ── Ejecutar modo seleccionado ────────────────────────────────────────────
    if args.mode == "batch":
        run_batch_inference(
            engine, series_test, args.window_size,
            k=args.k, step=args.step,
            output_path=args.output_path,
        )

    elif args.mode == "streaming":
        run_streaming_inference(
            engine, series_test, args.window_size,
            k=args.k, threshold=args.threshold,
            n_windows=args.n_streaming_windows,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia Temporal RAG — Modos: batch | streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",         type=str, default="batch",
                        choices=["batch", "streaming"])
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--encoder_path", type=str, default="checkpoints/best_encoder.pt")
    parser.add_argument("--index_dir",    type=str, default="index/")
    parser.add_argument("--output_path",  type=str, default="results/inference_output.csv")
    parser.add_argument("--column",       type=str, default=None)
    parser.add_argument("--window_size",  type=int, default=96)
    parser.add_argument("--d_model",      type=int, default=128)
    parser.add_argument("--nhead",        type=int, default=8)
    parser.add_argument("--num_layers",   type=int, default=3)
    parser.add_argument("--forecast_horizon", type=int, default=96)
    parser.add_argument("--k",            type=int, default=5,
                        help="Número de vecinos a recuperar (k=0 = baseline sin RAG)")
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--tau_iso",      type=float, default=0.5)
    parser.add_argument("--threshold",    type=float, default=0.5,
                        help="Umbral τ* para alertas (modo streaming)")
    parser.add_argument("--step",         type=int, default=1,
                        help="Paso entre ventanas (modo batch)")
    parser.add_argument("--n_streaming_windows", type=int, default=200,
                        help="Ventanas a procesar (modo streaming)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
