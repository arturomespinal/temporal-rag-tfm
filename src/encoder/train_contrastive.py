"""
Temporal RAG — Contrastive Encoder Training
============================================
Script de entrenamiento del encoder de series temporales mediante pérdida
contrastiva NT-Xent con augmentaciones temporales. Produce embeddings que
forman la base del índice vectorial del sistema Temporal RAG.

Referencia teórica:
    - Chen et al. (2020). SimCLR. ICML 2020.
    - Yue et al. (2022). TS2Vec. AAAI 2022.
    - Eldele et al. (2021). TNC. IJCAI 2021.

Autor: Arturo Miguel Espinal Reyes
TFM: RAG Temporal — Máster en Ciencia de Datos, 2026
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("temporal_rag.encoder")


# ─────────────────────────────────────────────────────────────────────────────
# Augmentaciones temporales
# ─────────────────────────────────────────────────────────────────────────────

def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """
    Augmentación por ruido gaussiano aditivo.

    Añade ruido blanco gaussiano N(0, sigma²) a cada punto de la serie,
    preservando la estructura temporal de largo plazo mientras introduce
    variabilidad local. Equivalente a modelar errores de medición del sensor.

    Args:
        x:     Tensor de forma (batch, time, features).
        sigma: Desviación estándar del ruido gaussiano.

    Returns:
        Tensor augmentado de igual forma que x.
    """
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Augmentación por escalado de amplitud.

    Multiplica la serie por un factor aleatorio por canal, extraído de
    N(1, sigma²), lo que simula variaciones de ganancia del sensor o
    cambios de unidad sin alterar la forma del patrón temporal.

    Args:
        x:     Tensor de forma (batch, time, features).
        sigma: Desviación estándar del factor de escala.

    Returns:
        Tensor augmentado de igual forma que x.
    """
    factor = torch.randn(x.size(0), 1, x.size(2), device=x.device) * sigma + 1.0
    return x * factor


def permutation(
    x: torch.Tensor,
    max_segments: int = 5,
    seg_mode: str = "random",
) -> torch.Tensor:
    """
    Augmentación por permutación de subsegmentos temporales.

    Divide la serie en max_segments subsegmentos y los reordena aleatoriamente.
    Preserva la distribución estadística marginal pero destruye dependencias
    temporales de corto plazo, forzando al encoder a capturar patrones de
    largo plazo para el aprendizaje contrastivo.

    Args:
        x:            Tensor de forma (batch, time, features).
        max_segments: Número máximo de segmentos en los que dividir la serie.
        seg_mode:     Modo de segmentación. "random" usa cortes aleatorios;
                      "equal" usa segmentos de igual longitud.

    Returns:
        Tensor augmentado de igual forma que x.
    """
    orig_steps = x.size(1)
    num_segs = np.random.randint(1, max_segments + 1)

    if seg_mode == "random":
        split_points = sorted(random.sample(range(1, orig_steps), min(num_segs - 1, orig_steps - 1)))
    else:
        split_points = list(range(orig_steps // num_segs, orig_steps, orig_steps // num_segs))

    # Construir segmentos y permutarlos
    splits = [0] + split_points + [orig_steps]
    segments = [x[:, splits[i]:splits[i + 1], :] for i in range(len(splits) - 1)]
    random.shuffle(segments)
    return torch.cat(segments, dim=1)


def apply_augmentation(x: torch.Tensor) -> torch.Tensor:
    """
    Aplica una pipeline de augmentaciones temporales aleatoriamente compuesta.

    La estrategia de augmentación compuesta se selecciona de forma estocástica
    en cada llamada, lo que maximiza la diversidad del espacio de vistas
    positivas durante el entrenamiento contrastivo.

    Args:
        x: Tensor de forma (batch, time, features).

    Returns:
        Vista augmentada del tensor de entrada.
    """
    aug_fns = [
        lambda t: jitter(t, sigma=random.uniform(0.01, 0.05)),
        lambda t: scaling(t, sigma=random.uniform(0.05, 0.15)),
        lambda t: permutation(t, max_segments=random.randint(3, 7)),
    ]
    # Aplicar 1 o 2 augmentaciones en secuencia aleatoria
    n_augs = random.randint(1, 2)
    selected = random.sample(aug_fns, n_augs)
    for fn in selected:
        x = fn(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesWindowDataset(Dataset):
    """
    Dataset de ventanas temporales deslizantes para entrenamiento contrastivo.

    Genera pares (x_i, x_i_plus) donde ambas son vistas augmentadas de la
    misma ventana temporal de la serie original. La ventana deslizante con
    stride configurable permite balancear la densidad de muestreo con el
    solapamiento entre muestras.

    Args:
        data:        Array NumPy de forma (T, F) — T pasos temporales, F features.
        window_size: Longitud de cada ventana temporal (número de timesteps).
        stride:      Paso entre ventanas consecutivas.
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 96,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.stride = stride

        # Índices de inicio de cada ventana
        self.indices = list(range(0, len(data) - window_size + 1, stride))
        logger.info(
            "Dataset creado: %d ventanas de longitud %d (stride=%d)",
            len(self.indices), window_size, stride,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        window = self.data[start : start + self.window_size]  # (T, F)

        # Generar dos vistas positivas del mismo segmento
        x_i = apply_augmentation(window.unsqueeze(0)).squeeze(0)
        x_j = apply_augmentation(window.unsqueeze(0)).squeeze(0)
        return x_i, x_j


# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura del Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Encoder Transformer para series temporales con proyección contrastiva.

    Arquitectura de dos etapas:
    1. Backbone Transformer: extrae representaciones contextuales de la secuencia
       de entrada mediante atención multi-cabeza. La salida es el token [CLS]
       (media global sobre el eje temporal), representando la ventana completa.
    2. Projection Head: red MLP no-lineal de dos capas que mapea la representación
       del backbone a un espacio de embeddings normalizado donde se aplica la
       pérdida contrastiva NT-Xent. Siguiendo Chen et al. (2020), el projection
       head se descarta tras el entrenamiento; solo el backbone se usa para indexar.

    Args:
        input_dim:      Número de features de la serie temporal de entrada.
        d_model:        Dimensión del espacio interno del Transformer.
        nhead:          Número de cabezas de atención multi-head.
        num_layers:     Número de capas Transformer encoder.
        dim_feedforward: Dimensión de la capa feedforward interna.
        dropout:        Tasa de dropout.
        embed_dim:      Dimensión del espacio de embedding contrastivo (projection head).
        max_seq_len:    Longitud máxima de secuencia para el encoding posicional.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 64,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        # Proyección de entrada: (T, F) → (T, d_model)
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding sinusoidal fijo (Vaswani et al., 2017)
        self.pos_encoding = self._build_positional_encoding(max_seq_len, d_model)

        # Backbone Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection Head: MLP de 2 capas con BatchNorm (siguiendo SimCLR)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

        self._init_weights()

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> nn.Parameter:
        """Construye el tensor de positional encoding sinusoidal."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, T, d_model)

    def _init_weights(self) -> None:
        """Inicialización de pesos con Xavier uniforme para capas lineales."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica una ventana temporal en su embedding del backbone.

        Este método se usa durante la inferencia para construir el índice
        vectorial. Solo utiliza el backbone (sin projection head).

        Args:
            x: Tensor de forma (batch, time, features).

        Returns:
            Embedding de forma (batch, d_model) — representación de la ventana.
        """
        T = x.size(1)
        x = self.input_projection(x)                    # (B, T, d_model)
        x = x + self.pos_encoding[:, :T, :]            # add positional encoding
        x = self.transformer(x)                         # (B, T, d_model)
        return x.mean(dim=1)                            # global average pooling → (B, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward completo: backbone + projection head (usado solo en entrenamiento).

        Args:
            x: Tensor de forma (batch, time, features).

        Returns:
            Embedding L2-normalizado de forma (batch, embed_dim).
        """
        h = self.encode(x)                              # (B, d_model)
        z = self.projection_head(h)                     # (B, embed_dim)
        return F.normalize(z, dim=-1)                   # L2 norm → cosine space


# ─────────────────────────────────────────────────────────────────────────────
# Pérdida NT-Xent (Contrastiva)
# ─────────────────────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Pérdida NT-Xent (Normalized Temperature-scaled Cross Entropy).

    Implementación de la función de pérdida contrastiva de SimCLR (Chen et al.,
    2020) con soporte para vectores ya L2-normalizados de entrada. Para un batch
    de N pares, construye la matriz de similaridad 2N × 2N y optimiza que cada
    muestra tenga mayor similitud con su par positivo que con cualquier negativo.

    La pérdida NT-Xent para el par (i, j) es:
        ℓ(i, j) = -log [ exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) ]

    donde τ es el parámetro de temperatura. Temperaturas bajas (τ → 0) crean
    gradientes más enfocados en los negativos más difíciles; temperaturas altas
    (τ → ∞) suavizan la distribución y pueden generar embeddings más generales.

    Args:
        temperature: Parámetro de temperatura τ. Valor recomendado: 0.07–0.5.
        device:      Dispositivo de cómputo.
    """

    def __init__(self, temperature: float = 0.2, device: str = "cpu") -> None:
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida NT-Xent sobre un batch de pares positivos.

        Args:
            z_i: Embeddings de la primera vista, forma (N, embed_dim), L2-normalizados.
            z_j: Embeddings de la segunda vista, forma (N, embed_dim), L2-normalizados.

        Returns:
            Escalar con la pérdida media sobre el batch.
        """
        N = z_i.size(0)

        # Concatenar las 2N representaciones: [z_i; z_j]
        z = torch.cat([z_i, z_j], dim=0)               # (2N, embed_dim)

        # Matriz de similitud coseno 2N × 2N (z ya está L2-normalizado)
        sim = torch.mm(z, z.T) / self.temperature       # (2N, 2N)

        # Máscara para excluir diagonal (self-similarity)
        mask = torch.eye(2 * N, device=self.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Labels: para z_i[k], el positivo está en z_j[k] → posición k+N
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N),
        ]).to(self.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Bucle de Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    encoder: TemporalEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NTXentLoss,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> float:
    """
    Entrena el encoder durante una época completa.

    Args:
        encoder:   Modelo TemporalEncoder a entrenar.
        loader:    DataLoader que produce pares (x_i, x_j) augmentados.
        optimizer: Optimizador (AdamW recomendado).
        criterion: Función de pérdida NTXentLoss.
        device:    Dispositivo de cómputo (cpu/cuda/mps).
        scheduler: LR scheduler opcional (se llama después de cada step).

    Returns:
        Pérdida media de la época.
    """
    encoder.train()
    total_loss = 0.0

    for batch_idx, (x_i, x_j) in enumerate(loader):
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        z_i = encoder(x_i)
        z_j = encoder(x_j)

        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            logger.debug("  Batch %d / %d — Loss: %.4f", batch_idx, len(loader), loss.item())

    return total_loss / len(loader)


def train(
    data: np.ndarray,
    output_dir: str = "./checkpoints",
    window_size: int = 96,
    stride: int = 12,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.2,
    d_model: int = 128,
    embed_dim: int = 64,
    nhead: int = 8,
    num_layers: int = 3,
    seed: int = 42,
    device_str: str = "auto",
) -> TemporalEncoder:
    """
    Pipeline completo de entrenamiento del encoder contrastivo.

    Configura el dataset, el modelo, el optimizador con cosine annealing,
    y ejecuta el bucle de entrenamiento guardando checkpoints periódicos.

    Args:
        data:        Array (T, F) con la serie temporal de entrenamiento.
        output_dir:  Directorio donde se guardan checkpoints y el modelo final.
        window_size: Longitud de ventana temporal.
        stride:      Stride del dataset deslizante.
        batch_size:  Tamaño de batch.
        epochs:      Número de épocas de entrenamiento.
        lr:          Learning rate inicial para AdamW.
        weight_decay: Regularización L2.
        temperature: Temperatura τ para NT-Xent.
        d_model:     Dimensión interna del Transformer.
        embed_dim:   Dimensión del espacio de embedding contrastivo.
        nhead:       Número de cabezas de atención.
        num_layers:  Número de capas Transformer.
        seed:        Semilla para reproducibilidad.
        device_str:  "auto" | "cpu" | "cuda" | "mps".

    Returns:
        Encoder entrenado (TemporalEncoder) listo para indexación vectorial.
    """
    # ── Reproducibilidad ──────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Dispositivo ───────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Dispositivo de entrenamiento: %s", device)

    # ── Dataset y DataLoader ──────────────────────────────────────────────
    dataset = TimeSeriesWindowDataset(data, window_size=window_size, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ── Modelo ────────────────────────────────────────────────────────────
    input_dim = data.shape[1] if data.ndim == 2 else 1
    encoder = TemporalEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        embed_dim=embed_dim,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info("Parámetros entrenables: %s", f"{n_params:,}")

    # ── Optimizador y Scheduler ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01
    )
    criterion = NTXentLoss(temperature=temperature, device=str(device))

    # ── Bucle de Entrenamiento ────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(encoder, loader, optimizer, criterion, device, scheduler)

        logger.info(
            "Época %3d / %d — Loss: %.4f — LR: %.2e",
            epoch, epochs, avg_loss, scheduler.get_last_lr()[0],
        )

        # Guardar checkpoint cada 10 épocas y el mejor modelo
        if epoch % 10 == 0:
            ckpt_path = output_path / f"encoder_epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "state_dict": encoder.state_dict()}, ckpt_path)
            logger.info("Checkpoint guardado: %s", ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_path / "encoder_best.pt"
            torch.save({"epoch": epoch, "loss": best_loss, "state_dict": encoder.state_dict()}, best_path)

    logger.info("Entrenamiento completado. Mejor loss: %.4f", best_loss)
    logger.info("Modelo guardado en: %s", output_path / "encoder_best.pt")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento del Encoder Contrastivo para Temporal RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Ruta al fichero .npy con la serie temporal (T, F).")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directorio de salida para checkpoints.")
    parser.add_argument("--window-size", type=int, default=96,
                        help="Longitud de la ventana temporal.")
    parser.add_argument("--stride", type=int, default=12,
                        help="Stride del dataset deslizante.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate inicial.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperatura τ para NT-Xent.")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Dimensión interna del Transformer.")
    parser.add_argument("--embed-dim", type=int, default=64,
                        help="Dimensión del espacio de embedding contrastivo.")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Número de cabezas de atención.")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Número de capas Transformer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Cargar datos
    data = np.load(args.data_path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    logger.info("Datos cargados: shape=%s, dtype=%s", data.shape, data.dtype)

    # Normalización Z-score global
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    logger.info("Datos normalizados (Z-score).")

    # Entrenamiento
    train(
        data=data,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        d_model=args.d_model,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        seed=args.seed,
        device_str=args.device,
    )
