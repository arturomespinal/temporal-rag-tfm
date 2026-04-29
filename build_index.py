"""
build_index.py
==============
Construye y serializa el índice vectorial FAISS a partir de los embeddings
del backbone entrenado con train_contrastive.py. Este script implementa
la Fase 4 del pipeline Temporal RAG (§6.4 de la memoria).

El índice resultante actúa como memoria histórica no paramétrica del sistema:
almacena los embeddings de todas las ventanas temporales del conjunto de
entrenamiento y permite recuperar los k vecinos más similares en tiempo
sublineal durante la inferencia.

Pipeline completo del índice (§6.4):
    1. Carga del backbone pre-entrenado
    2. Extracción de ventanas deslizantes con Z-score local (§6.1–6.2)
    3. Codificación en lotes → embeddings L2-normalizados (§6.3)
    4. Configuración del índice FAISS (IndexFlatL2 o IndexIVFFlat)
    5. Adición de vectores al índice
    6. Serialización en disco (.index) + metadatos (.npy)

Configuración de hiperparámetros FAISS (§6.4):
    n_list ≈ 4√N     (número de celdas de Voronoi para IVF)
    n_probe ≈ n_list / 8    (celdas inspeccionadas por consulta)

Garantiza recall > 95% con latencia ~1ms para N=10⁶ (Johnson et al., 2019).

Referencias:
    Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity
    search with GPUs. IEEE Transactions on Big Data, 7(3), 535–547.

    Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust
    approximate nearest neighbor search using HNSW. IEEE TPAMI, 42(4).

Uso:
    python build_index.py \
        --data_path data/ambient_temperature_system_failure.csv \
        --encoder_path checkpoints/best_encoder.pt \
        --index_dir index/ \
        --index_type ivf

Repositorio:
    https://github.com/arturomespinal/temporal-rag-tfm
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from train_contrastive import (
    SlidingWindowDataset,
    TransformerBackbone,
    load_series,
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS no disponible. Instalar: pip install faiss-cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. EXTRACCIÓN DE EMBEDDINGS
# ===========================================================================

def extract_embeddings(
    backbone: TransformerBackbone,
    series_train: np.ndarray,
    window_size: int,
    batch_size: int,
    device: torch.device,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Codifica todas las ventanas del conjunto de entrenamiento con el backbone.

    Protocolo de no-contaminación (§6.9): solo se indexan datos de entrenamiento.
    Ningún dato de validación o test puede contaminar el índice FAISS.

    Normalización L2 de los embeddings: garantiza que el producto interno
    en FAISS sea equivalente a la similitud coseno, alineado con la
    optimización NT-Xent del entrenamiento contrastivo (§7.4.1).

    Args:
        backbone:      TransformerBackbone pre-entrenado y congelado.
        series_train:  Array 1D con los datos de entrenamiento.
        window_size:   Longitud L de cada ventana.
        batch_size:    Tamaño de batch para la codificación.
        device:        Dispositivo de cómputo.
        step:          Paso entre ventanas (1 = máximo solapamiento, §6.2).
    Returns:
        Tupla (embeddings, windows):
            embeddings: Array (N, d_model) L2-normalizado, listo para FAISS.
            windows:    Array (N, window_size) con las ventanas originales
                        (almacenadas como metadatos para reconstrucción en §6.5).
    """
    backbone.eval()
    dataset = SlidingWindowDataset(series_train, window_size, step=step)
    N = len(dataset)
    logger.info(f"Codificando {N:,} ventanas (window_size={window_size}, step={step})...")

    all_embeddings: list[np.ndarray] = []
    all_windows:    list[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_views = [dataset[j][0] for j in range(i, end)]   # vista sin augmentación
            batch_tensor = torch.stack(batch_views).to(device)

            h = backbone(batch_tensor)                              # (B, d_model)
            z = F.normalize(h, dim=-1)                             # L2-normalización

            all_embeddings.append(z.cpu().numpy().astype(np.float32))
            all_windows.append(batch_tensor.cpu().numpy())

            if (i // batch_size) % 20 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(f"  [{i:6,}/{N:,}] — {elapsed:.1f}s transcurridos")

    embeddings = np.concatenate(all_embeddings, axis=0)
    windows    = np.concatenate(all_windows,    axis=0)

    elapsed_total = time.perf_counter() - t0
    throughput = N / elapsed_total
    logger.info(
        f"Codificación completada: {N:,} embeddings en {elapsed_total:.1f}s "
        f"({throughput:.0f} ventanas/s)"
    )
    return embeddings, windows


# ===========================================================================
# 2. CONSTRUCCIÓN DEL ÍNDICE FAISS
# ===========================================================================

def build_faiss_index(
    embeddings: np.ndarray,
    index_type: Literal["flat", "ivf", "hnsw"] = "ivf",
    n_list: int | None = None,
    n_probe: int | None = None,
    hnsw_m: int = 32,
) -> "faiss.Index":
    """
    Construye el índice FAISS con la configuración especificada (§6.4).

    Tipos de índice disponibles:
        'flat'  — IndexFlatIP: búsqueda exacta. Baseline para validación.
                  Complejidad O(N·d) por consulta.
        'ivf'   — IndexIVFFlat: k-means invertido. Recomendado para el TFM.
                  Complejidad ≈ O(n_probe · N/n_list · d).
                  Configuración óptima: n_list ≈ 4√N, n_probe ≈ n_list/8.
        'hnsw'  — IndexHNSW: grafo jerárquico. Óptimo para producción.
                  Complejidad O(log N). Mayor consumo de memoria.

    Referencia: Johnson et al. (2019). Billion-scale similarity search with GPUs.

    Args:
        embeddings:  Array (N, d) L2-normalizado.
        index_type:  Tipo de índice FAISS.
        n_list:      Celdas de Voronoi (IVF). None = 4√N automático.
        n_probe:     Celdas inspeccionadas por consulta. None = n_list/8 automático.
        hnsw_m:      Conexiones por nodo en HNSW (default: 32).
    Returns:
        Índice FAISS entrenado y poblado, listo para búsqueda.
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError(
            "FAISS no instalado. Ejecutar: pip install faiss-cpu  "
            "(o faiss-gpu si hay CUDA disponible)"
        )

    N, d = embeddings.shape
    logger.info(f"Construyendo índice FAISS tipo='{index_type}' | N={N:,} | d={d}")

    if index_type == "flat":
        # Producto interno exacto (equivalente a coseno con L2-normalización)
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        logger.info(f"IndexFlatIP construido: {index.ntotal:,} vectores")

    elif index_type == "ivf":
        # Configuración óptima (§6.4): n_list ≈ 4√N
        if n_list is None:
            n_list = max(10, int(4 * np.sqrt(N)))
        if n_probe is None:
            n_probe = max(1, n_list // 8)

        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_list, faiss.METRIC_INNER_PRODUCT)

        logger.info(f"Entrenando IndexIVFFlat: n_list={n_list}, n_probe={n_probe}...")
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = n_probe
        logger.info(
            f"IndexIVFFlat construido: {index.ntotal:,} vectores | "
            f"n_list={n_list} | n_probe={n_probe}"
        )

    elif index_type == "hnsw":
        # HNSW: latencia mínima para producción (Malkov & Yashunin, 2020)
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200    # calidad de construcción
        index.hnsw.efSearch = 64           # calidad de búsqueda
        index.add(embeddings)
        logger.info(f"IndexHNSWFlat construido: {index.ntotal:,} vectores | M={hnsw_m}")

    else:
        raise ValueError(f"index_type debe ser 'flat', 'ivf' o 'hnsw'. Recibido: {index_type}")

    return index


# ===========================================================================
# 3. BENCHMARK DE RECALL
# ===========================================================================

def benchmark_recall(
    index: "faiss.Index",
    embeddings: np.ndarray,
    k: int = 10,
    n_queries: int = 500,
    seed: int = 42,
) -> dict:
    """
    Evalúa el recall del índice comparando con búsqueda exacta bruta.

    Recall@k = fracción de los k vecinos exactos que el índice aproximado
    recupera correctamente. Para IndexIVFFlat con n_probe = n_list/8 se
    espera recall > 95% (Johnson et al., 2019).

    Args:
        index:      Índice FAISS a evaluar.
        embeddings: Embeddings del conjunto de entrenamiento.
        k:          Número de vecinos evaluados.
        n_queries:  Número de queries de evaluación (muestra aleatoria).
        seed:       Semilla para reproducibilidad.
    Returns:
        Dict con recall@k, tiempo medio de consulta y throughput.
    """
    if not FAISS_AVAILABLE:
        return {}

    rng = np.random.default_rng(seed)
    n_queries = min(n_queries, len(embeddings))
    query_idx = rng.choice(len(embeddings), size=n_queries, replace=False)
    queries   = embeddings[query_idx].astype(np.float32)

    # Búsqueda exacta (referencia)
    exact_index = faiss.IndexFlatIP(embeddings.shape[1])
    exact_index.add(embeddings.astype(np.float32))
    _, exact_I = exact_index.search(queries, k + 1)    # +1 para excluir el propio query
    exact_I = exact_I[:, 1:]                           # excluir self-match

    # Búsqueda aproximada (índice evaluado)
    t0 = time.perf_counter()
    _, approx_I = index.search(queries, k + 1)
    t1 = time.perf_counter()
    approx_I = approx_I[:, 1:]

    # Recall@k
    hits = 0
    for i in range(n_queries):
        exact_set  = set(exact_I[i].tolist())
        approx_set = set(approx_I[i].tolist())
        exact_set.discard(-1)
        approx_set.discard(-1)
        hits += len(exact_set & approx_set)

    recall = hits / (n_queries * k) if n_queries * k > 0 else 0.0
    mean_latency_ms = (t1 - t0) / n_queries * 1000
    throughput_qps  = n_queries / (t1 - t0)

    logger.info(f"Benchmark recall — Recall@{k}: {recall*100:.2f}% | "
                f"Latencia media: {mean_latency_ms:.3f}ms | "
                f"Throughput: {throughput_qps:.0f} queries/s")

    return {
        f"recall@{k}": recall,
        "mean_latency_ms": mean_latency_ms,
        "throughput_qps": throughput_qps,
        "n_queries": n_queries,
    }


# ===========================================================================
# 4. SERIALIZACIÓN
# ===========================================================================

def save_index(
    index: "faiss.Index",
    windows: np.ndarray,
    embeddings: np.ndarray,
    index_dir: str,
    config: dict,
) -> None:
    """
    Serializa el índice FAISS y los metadatos asociados.

    Estructura de archivos generados:
        index_dir/
            faiss.index        — índice FAISS serializado (binario)
            windows.npy        — ventanas originales (para reconstrucción en §6.5)
            embeddings.npy     — embeddings L2-normalizados (para debugging)
            config.json        — metadatos de configuración del índice

    Args:
        index:      Índice FAISS entrenado.
        windows:    Array (N, window_size) de ventanas originales.
        embeddings: Array (N, d) de embeddings L2-normalizados.
        index_dir:  Directorio de destino.
        config:     Diccionario de metadatos (parámetros de construcción).
    """
    import json

    os.makedirs(index_dir, exist_ok=True)

    # Serializar índice FAISS
    index_path = Path(index_dir) / "faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"Índice FAISS guardado: {index_path} ({index_path.stat().st_size / 1e6:.1f} MB)")

    # Guardar ventanas originales (metadatos para reconstrucción)
    windows_path = Path(index_dir) / "windows.npy"
    np.save(windows_path, windows)
    logger.info(f"Ventanas guardadas: {windows_path} ({windows.nbytes / 1e6:.1f} MB)")

    # Guardar embeddings (útil para debugging y análisis post-hoc)
    emb_path = Path(index_dir) / "embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"Embeddings guardados: {emb_path} ({embeddings.nbytes / 1e6:.1f} MB)")

    # Guardar configuración
    config["n_vectors"]    = int(index.ntotal)
    config["d_embedding"]  = int(embeddings.shape[1])
    config["windows_shape"] = list(windows.shape)
    config_path = Path(index_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuración guardada: {config_path}")


def load_index(index_dir: str) -> tuple["faiss.Index", np.ndarray, dict]:
    """
    Carga el índice FAISS y los metadatos desde disco.

    Args:
        index_dir: Directorio con los archivos generados por save_index().
    Returns:
        Tupla (index, windows, config).
    """
    import json

    index   = faiss.read_index(str(Path(index_dir) / "faiss.index"))
    windows = np.load(Path(index_dir) / "windows.npy")
    with open(Path(index_dir) / "config.json") as f:
        config = json.load(f)

    # Restaurar n_probe si es IVF
    if "n_probe" in config and hasattr(index, "nprobe"):
        index.nprobe = config["n_probe"]

    logger.info(f"Índice cargado: {index.ntotal:,} vectores | d={index.d} | tipo={type(index).__name__}")
    return index, windows, config


# ===========================================================================
# 5. PIPELINE PRINCIPAL
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    """
    Pipeline completo de construcción del índice FAISS (§6.4).

    Flujo:
        1. Carga de la serie y split temporal (solo train para el índice)
        2. Carga del backbone pre-entrenado (train_contrastive.py)
        3. Extracción de embeddings en lotes
        4. Construcción del índice FAISS con la configuración especificada
        5. Benchmark de recall (validación de la configuración)
        6. Serialización en disco
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    # ── 1. Datos (solo train — protocolo §6.9) ───────────────────────────────
    series = load_series(args.data_path, args.column)
    n = len(series)
    train_end = int(n * 0.70)
    series_train = series[:train_end]
    logger.info(f"Split: {train_end:,} puntos de entrenamiento para indexar")

    # ── 2. Backbone ──────────────────────────────────────────────────────────
    backbone = TransformerBackbone(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    if Path(args.encoder_path).exists():
        ckpt = torch.load(args.encoder_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        logger.info(f"Backbone cargado: {args.encoder_path}")
    else:
        logger.warning("Encoder no encontrado. Usando pesos aleatorios (solo para debugging).")

    for param in backbone.parameters():
        param.requires_grad = False

    # ── 3. Embeddings ─────────────────────────────────────────────────────────
    embeddings, windows = extract_embeddings(
        backbone, series_train, args.window_size,
        args.batch_size, device, step=1
    )

    # ── 4. Índice FAISS ──────────────────────────────────────────────────────
    if not FAISS_AVAILABLE:
        logger.error("FAISS no disponible. Instalar con: pip install faiss-cpu")
        return

    n_list = args.n_list if args.n_list > 0 else None
    n_probe = args.n_probe if args.n_probe > 0 else None

    index = build_faiss_index(
        embeddings,
        index_type=args.index_type,
        n_list=n_list,
        n_probe=n_probe,
        hnsw_m=args.hnsw_m,
    )

    # ── 5. Benchmark de recall ────────────────────────────────────────────────
    logger.info("Ejecutando benchmark de recall...")
    benchmark = benchmark_recall(index, embeddings, k=10, n_queries=500)

    # ── 6. Serialización ──────────────────────────────────────────────────────
    config = {
        "index_type":  args.index_type,
        "window_size": args.window_size,
        "d_model":     args.d_model,
        "n_list":      n_list,
        "n_probe":     n_probe,
        "hnsw_m":      args.hnsw_m,
        "encoder_path": args.encoder_path,
        "benchmark":   benchmark,
    }
    save_index(index, windows, embeddings, args.index_dir, config)

    logger.info(f"\nÍndice listo para inferencia. Cargar con:")
    logger.info(f"  from build_index import load_index")
    logger.info(f"  index, windows, cfg = load_index('{args.index_dir}')")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construcción del índice FAISS para Temporal RAG (§6.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--encoder_path", type=str, default="checkpoints/best_encoder.pt")
    parser.add_argument("--index_dir",    type=str, default="index/",
                        help="Directorio de salida para el índice serializado")
    parser.add_argument("--index_type",   type=str, default="ivf",
                        choices=["flat", "ivf", "hnsw"],
                        help="Tipo de índice FAISS (flat=exacto, ivf=approx, hnsw=prod)")
    parser.add_argument("--column",       type=str, default=None)
    parser.add_argument("--window_size",  type=int, default=96)
    parser.add_argument("--d_model",      type=int, default=128)
    parser.add_argument("--nhead",        type=int, default=8)
    parser.add_argument("--num_layers",   type=int, default=3)
    parser.add_argument("--batch_size",   type=int, default=512)
    parser.add_argument("--n_list",       type=int, default=0,
                        help="Celdas IVF (0 = automático: 4√N)")
    parser.add_argument("--n_probe",      type=int, default=0,
                        help="Celdas inspeccionadas por query (0 = automático: n_list/8)")
    parser.add_argument("--hnsw_m",       type=int, default=32,
                        help="Conexiones por nodo en HNSW")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
