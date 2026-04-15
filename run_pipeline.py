"""
run_pipeline.py
===============
End-to-end Temporal RAG pipeline runner.

Usage
-----
    python run_pipeline.py --config configs/default.yaml

This script ties together all pipeline stages:
    1. Load NAB series
    2. Segment into sliding windows
    3. Encode with selected encoder
    4. Build FAISS temporal memory
    5. Retrieve context for each query window
    6. Score with HybridGenerator
    7. Evaluate and report metrics

This is the reproducible experimental entry point described in §6 of the TFM.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.encoder.ts_encoder import AutoencoderEncoder, LSTMEncoder, TransformerEncoder
from src.generator.hybrid_generator import HybridGenerator
from src.index.faiss_index import TemporalFAISSIndex
from src.loader.nab_loader import NABLoader
from src.retrieval.context_retriever import ContextRetriever
from src.utils.metrics import anomaly_detection_report, forecasting_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "dataset": {
        "series_name": "ambient_temperature_system_failure",
        "window_size": 64,
        "step": 1,
    },
    "encoder": {
        "type": "lstm",          # "lstm" | "transformer" | "autoencoder"
        "latent_dim": 32,
        "hidden_size": 64,
    },
    "index": {
        "type": "flat",          # "flat" | "ivf"
        "nlist": 100,
        "nprobe": 10,
    },
    "retrieval": {"k": 5},
    "generator": {
        "fusion": "cross_attn",  # "concat" | "cross_attn"
        "task": "anomaly",       # "forecast" | "anomaly" | "both"
        "horizon": 1,
    },
    "split": {"train_ratio": 0.7},
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as fh:
            user_cfg = yaml.safe_load(fh)
        cfg = {**DEFAULT_CONFIG, **user_cfg}
        logger.info("Config loaded from %s", path)
    else:
        cfg = DEFAULT_CONFIG
        logger.info("Using default config.")
    return cfg


def build_encoder(cfg: dict):
    enc_type = cfg["encoder"]["type"]
    latent_dim = cfg["encoder"]["latent_dim"]
    window_size = cfg["dataset"]["window_size"]
    hidden_size = cfg["encoder"].get("hidden_size", 64)

    if enc_type == "lstm":
        return LSTMEncoder(
            input_size=window_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
        )
    elif enc_type == "transformer":
        return TransformerEncoder(
            input_size=window_size,
            d_model=hidden_size,
            latent_dim=latent_dim,
        )
    elif enc_type == "autoencoder":
        return AutoencoderEncoder(input_size=window_size, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")


def run(cfg: dict) -> None:
    # ---- 1. Load data ----
    logger.info("=== Stage 1: Loading data ===")
    loader = NABLoader(series_name=cfg["dataset"]["series_name"])
    windows, labels = loader.load_windows(
        window_size=cfg["dataset"]["window_size"],
        step=cfg["dataset"]["step"],
    )

    # ---- Train/test split (temporal: no shuffling) ----
    split_idx = int(len(windows) * cfg["split"]["train_ratio"])
    train_windows, train_labels = windows[:split_idx], labels[:split_idx]
    test_windows, test_labels = windows[split_idx:], labels[split_idx:]
    logger.info(
        "Split: train=%d, test=%d windows", len(train_windows), len(test_windows)
    )

    # ---- 2. Build encoder ----
    logger.info("=== Stage 2: Building encoder ===")
    encoder = build_encoder(cfg)
    logger.info("Encoder: %s", encoder.__class__.__name__)

    # NOTE: In the full experiment, the encoder is trained here via
    # reconstruction or contrastive loss. For this demonstration, we use
    # the randomly-initialised encoder to validate the pipeline end-to-end.
    # See notebooks/02_encoder_training.ipynb for the training procedure.

    # ---- 3. Encode training windows ----
    logger.info("=== Stage 3: Encoding training windows ===")
    train_embeddings = encoder.encode(train_windows)

    # ---- 4. Build FAISS index ----
    logger.info("=== Stage 4: Building FAISS index ===")
    index = TemporalFAISSIndex(
        dim=cfg["encoder"]["latent_dim"],
        index_type=cfg["index"]["type"],
        nlist=cfg["index"]["nlist"],
        nprobe=cfg["index"]["nprobe"],
    )
    metadata = [
        {"idx": i, "label": int(train_labels[i]), "window": train_windows[i]}
        for i in range(len(train_windows))
    ]
    index.build(train_embeddings, metadata=metadata)

    # ---- 5. Build retriever ----
    logger.info("=== Stage 5: Initialising retriever ===")
    retriever = ContextRetriever(
        index=index,
        encoder=encoder,
        k=cfg["retrieval"]["k"],
    )

    # ---- 6. Build hybrid generator ----
    logger.info("=== Stage 6: Building hybrid generator ===")
    generator = HybridGenerator(
        encoder=encoder,
        retriever=retriever,
        latent_dim=cfg["encoder"]["latent_dim"],
        fusion=cfg["generator"]["fusion"],
        task=cfg["generator"]["task"],
        horizon=cfg["generator"]["horizon"],
    )

    # ---- 7. Inference + Evaluation ----
    logger.info("=== Stage 7: Inference and evaluation ===")
    anomaly_scores = generator.anomaly_score(test_windows)

    report = anomaly_detection_report(
        y_true=test_labels.astype(int),
        y_scores=anomaly_scores,
        point_adjust_protocol=True,
    )

    logger.info("=== RESULTS (Temporal RAG) ===")
    for metric, value in report.items():
        logger.info("  %-15s %.4f", metric, value)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal RAG pipeline")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)
