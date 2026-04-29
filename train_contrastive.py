"""
train_contrastive.py
====================
Entrenamiento del encoder temporal mediante aprendizaje contrastivo auto-supervisado
con pérdida NT-Xent (Normalized Temperature-scaled Cross Entropy).

Arquitectura:
    TemporalEncoder = TransformerBackbone + ProjectionHead (descartado post-entrenamiento)

Augmentaciones implementadas (§7.4.2 de la memoria):
    1. Jitter gaussiano          — simula ruido de sensor
    2. Scaling de amplitud       — variaciones de ganancia sin alterar forma
    3. Permutación de subsegmentos — destruye dependencias de corto plazo

Referencia fundacional:
    Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
    A simple framework for contrastive learning of visual representations.
    ICML 2020, 1597–1607.

    Yue, Z., et al. (2022). TS2Vec: Towards universal representation of time series.
    AAAI 2022, 36(8), 8980–8987.

    Eldele, E., et al. (2021). Time-series representation learning via temporal
    and contextual contrasting. IJCAI 2021, 2352–2359.

Uso:
    python train_contrastive.py --data_path data/ambient_temperature_system_failure.csv \
                                 --epochs 50 --batch_size 256 --window_size 96 \
                                 --embed_dim 64 --d_model 128 --tau 0.2

Repositorio:
    https://github.com/arturomespinal/temporal-rag-tfm
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. AUGMENTACIONES TEMPORALES
# ===========================================================================

def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """
    Augmentación por jitter gaussiano (§7.4.2).

    Añade ruido gaussiano aditivo i.i.d. a cada punto de la ventana temporal.
    Simula errores de medición del sensor sin alterar la dinámica global.

    Args:
        x:     Tensor de forma (batch, window_size) o (window_size,).
        sigma: Desviación estándar del ruido. Rango efectivo: [0.01, 0.05].

    Returns:
        Tensor perturbado con la misma forma que x.
    """
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor,
            scale_min: float = 0.85,
            scale_max: float = 1.15) -> torch.Tensor:
    """
    Augmentación por scaling de amplitud (§7.4.2).

    Multiplica la ventana por un escalar aleatorio uniforme en [scale_min, scale_max].
    Modela variaciones de ganancia del sensor preservando la forma del patrón temporal.

    Args:
        x:         Tensor de forma (batch, window_size).
        scale_min: Límite inferior del factor de escala.
        scale_max: Límite superior del factor de escala.

    Returns:
        Tensor escalado con la misma forma que x.
    """
    factor = torch.empty(x.shape[0], 1, device=x.device).uniform_(scale_min, scale_max)
    return x * factor


def permutation(x: torch.Tensor,
                max_segments: int = 5,
                min_segments: int = 3) -> torch.Tensor:
    """
    Augmentación por permutación de subsegmentos (§7.4.2).

    Divide la ventana en n segmentos aleatorios y los reordena aleatoriamente.
    Destruye dependencias de corto plazo, forzando al encoder a capturar
    patrones de largo plazo robustos.

    Args:
        x:            Tensor de forma (batch, window_size).
        max_segments: Número máximo de cortes.
        min_segments: Número mínimo de cortes.

    Returns:
        Tensor permutado con la misma forma que x.
    """
    batch_size, window_size = x.shape
    n_segs = random.randint(min_segments, max_segments)
    # Puntos de corte aleatorios (excluyendo extremos)
    cut_points = sorted(random.sample(range(1, window_size), min(n_segs - 1, window_size - 1)))
    cut_points = [0] + cut_points + [window_size]

    segments = []
    for i in range(len(cut_points) - 1):
        segments.append(x[:, cut_points[i]:cut_points[i + 1]])

    random.shuffle(segments)
    return torch.cat(segments, dim=1)


def apply_augmentation(x: torch.Tensor) -> torch.Tensor:
    """
    Pipeline de augmentación completo para generar una vista positiva.

    Aplica secuencialmente jitter → scaling → permutación con probabilidades
    independientes, generando variabilidad suficiente para el aprendizaje
    contrastivo sin destruir la semántica dinámica de la ventana.

    Args:
        x: Tensor de forma (batch, window_size).

    Returns:
        Vista augmentada con la misma forma que x.
    """
    x = jitter(x, sigma=random.uniform(0.01, 0.05))
    x = scaling(x)
    if random.random() > 0.5:          # permutación aplicada al 50% de los batches
        x = permutation(x)
    return x


# ===========================================================================
# 2. ARQUITECTURA DEL ENCODER TEMPORAL
# ===========================================================================

class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal (Vaswani et al., 2017).

    Inyecta información de posición absoluta en los tokens temporales antes
    del mecanismo de self-attention, preservando el orden secuencial de la
    serie temporal dentro de la ventana.

    Args:
        d_model:  Dimensión del espacio de representación del Transformer.
        max_len:  Longitud máxima de secuencia soportada.
        dropout:  Tasa de dropout aplicada tras la suma posicional.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)                      # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch, seq_len, d_model).
        Returns:
            Tensor con codificación posicional sumada, misma forma.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBackbone(nn.Module):
    """
    Backbone Transformer para codificación de ventanas temporales (§7.4.3 — Etapa 1).

    Proyecta cada punto escalar de la ventana a d_model dimensiones mediante
    una capa lineal de entrada, aplica codificación posicional, y procesa
    la secuencia completa con num_layers capas de TransformerEncoder.

    La representación global de la ventana se obtiene mediante global average
    pooling sobre el eje temporal (equivalente al token CLS en BERT), produciendo
    un vector de forma (batch, d_model) que captura la dinámica agregada.

    Args:
        window_size: Longitud L de la ventana temporal (default: 96, §7.4.4).
        d_model:     Dimensión interna del Transformer (default: 128).
        nhead:       Número de cabezas de atención (default: 8).
        num_layers:  Profundidad del encoder (default: 3).
        dropout:     Tasa de dropout (default: 0.1).
    """

    def __init__(
        self,
        window_size: int = 96,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN: mejor estabilidad del gradiente
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch, window_size) — serie univariada normalizada.
        Returns:
            Representación global de forma (batch, d_model).
        """
        x = x.unsqueeze(-1)                        # (batch, window_size, 1)
        x = self.input_projection(x)               # (batch, window_size, d_model)
        x = self.pos_encoding(x)                   # (batch, window_size, d_model)
        x = self.transformer(x)                    # (batch, window_size, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)                          # Global Average Pooling → (batch, d_model)
        return x


class ProjectionHead(nn.Module):
    """
    Cabezal de proyección MLP para el espacio contrastivo (§7.4.3 — Etapa 2).

    Mapea la representación del backbone (d_model) al espacio contrastivo
    (embed_dim) mediante dos capas lineales con BatchNorm y activación GELU.

    Siguiendo Chen et al. (2020), este cabezal se descarta tras el entrenamiento.
    Solo el TransformerBackbone se usa para indexar el espacio vectorial FAISS.

    Args:
        d_model:   Dimensión de entrada (salida del backbone).
        embed_dim: Dimensión del espacio contrastivo (default: 64, §7.4.4).
        hidden_dim: Dimensión de la capa oculta MLP (default: 256).
    """

    def __init__(
        self,
        d_model: int = 128,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch, d_model).
        Returns:
            Embedding L2-normalizado de forma (batch, embed_dim).
        """
        z = self.net(x)
        return F.normalize(z, dim=-1)              # Normalización L2 en la esfera unitaria


class TemporalEncoder(nn.Module):
    """
    Encoder temporal completo: backbone + projection head.

    Combina TransformerBackbone y ProjectionHead en un único módulo
    que puede entrenarse end-to-end con la pérdida NT-Xent.

    En inferencia (modo evaluación), se usa únicamente el backbone
    para obtener representaciones de dimensión d_model destinadas
    al índice FAISS (§6.4 de la memoria).

    Args:
        window_size: Longitud de la ventana temporal.
        d_model:     Dimensión interna del Transformer.
        embed_dim:   Dimensión del espacio contrastivo.
        nhead:       Cabezas de atención.
        num_layers:  Capas del encoder.
        dropout:     Tasa de dropout.
    """

    def __init__(
        self,
        window_size: int = 96,
        d_model: int = 128,
        embed_dim: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(window_size, d_model, nhead, num_layers, dropout)
        self.projection = ProjectionHead(d_model, embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor de forma (batch, window_size).
        Returns:
            Tupla (h, z):
                h: representación backbone (batch, d_model) — para FAISS
                z: embedding contrastivo L2-normalizado (batch, embed_dim) — para NT-Xent
        """
        h = self.backbone(x)
        z = self.projection(h)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Modo inferencia: devuelve solo la representación backbone L2-normalizada.
        Este vector es el que se almacena en el índice FAISS (§6.4).

        Args:
            x: Tensor de forma (batch, window_size).
        Returns:
            Representación L2-normalizada de forma (batch, d_model).
        """
        with torch.no_grad():
            h = self.backbone(x)
            return F.normalize(h, dim=-1)


# ===========================================================================
# 3. PÉRDIDA NT-XENT
# ===========================================================================

class NTXentLoss(nn.Module):
    """
    Pérdida NT-Xent (Normalized Temperature-scaled Cross Entropy) (§7.4.1).

    Formulación matemática (Chen et al., 2020; Yue et al., 2022):

        L = -(1/2N) * Σ_i [
            log( exp(sim(z_i, z_i⁺) / τ) / Σ_{j≠i} exp(sim(z_i, z_j) / τ) )
          + log( exp(sim(z_i⁺, z_i) / τ) / Σ_{j≠i} exp(sim(z_i⁺, z_j) / τ) )
        ]

    donde:
        z_i, z_i⁺ : par positivo (dos vistas augmentadas de la misma ventana)
        τ          : temperatura de calibración (§7.4.4: τ = 0.2)
        sim(·,·)   : similitud coseno (los embeddings ya están L2-normalizados)

    La implementación vectorizada computa la matriz de similitudes completa
    (2N × 2N) en una sola operación de producto matricial, con enmascaramiento
    de la diagonal para excluir el par (z_i, z_i) del denominador.

    Args:
        temperature: Parámetro τ. Valores bajos focalizan gradientes en los
                     negativos más difíciles; valores altos suavizan la distribución.
        device:      Dispositivo de cómputo ('cuda' o 'cpu').
    """

    def __init__(self, temperature: float = 0.2, device: str = "cpu") -> None:
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Computa la pérdida NT-Xent para un batch de pares positivos.

        Args:
            z_i: Embeddings de la vista 1, forma (N, embed_dim), L2-normalizados.
            z_j: Embeddings de la vista 2, forma (N, embed_dim), L2-normalizados.
        Returns:
            Escalar con la pérdida media del batch.
        """
        N = z_i.shape[0]
        # Concatenar ambas vistas → (2N, embed_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Matriz de similitudes coseno (2N × 2N)
        # Dado que z está L2-normalizado, sim(u,v) = u·vᵀ
        sim_matrix = torch.mm(z, z.t()) / self.temperature    # (2N, 2N)

        # Máscara de diagonal: excluir sim(z_i, z_i) del denominador
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Pares positivos: (i, i+N) y (i+N, i)
        labels = torch.arange(N, device=self.device)
        labels = torch.cat([labels + N, labels])               # (2N,)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# ===========================================================================
# 4. DATASET DE VENTANAS TEMPORALES
# ===========================================================================

class SlidingWindowDataset(Dataset):
    """
    Dataset de ventanas deslizantes con normalización Z-score local (§6.1 y §6.2).

    Cada ítem del dataset devuelve dos vistas augmentadas de la misma ventana,
    formando el par positivo requerido por NT-Xent.

    Normalización local por ventana (§6.1):
        x̃(t) = [x(t) − μ_W] / (σ_W + ε)

    donde μ_W y σ_W son la media y desviación estándar de la ventana W,
    y ε = 1×10⁻⁸ es la constante de estabilidad numérica.

    Args:
        series:      Array numpy 1D con la serie temporal completa.
        window_size: Longitud L de cada ventana (default: 96, §6.2).
        step:        Paso entre ventanas consecutivas (default: 1 para máximo solapamiento).
    """

    def __init__(
        self,
        series: np.ndarray,
        window_size: int = 96,
        step: int = 1,
    ) -> None:
        self.windows = self._extract_windows(series, window_size, step)
        logger.info(f"Dataset creado: {len(self.windows)} ventanas de longitud {window_size}")

    @staticmethod
    def _extract_windows(
        series: np.ndarray,
        window_size: int,
        step: int,
    ) -> np.ndarray:
        """
        Extrae ventanas deslizantes y aplica normalización Z-score local.

        Cardinalidad del índice (§6.2):
            N_index = ⌊(T − L) / s⌋ + 1

        Args:
            series:      Array 1D de longitud T.
            window_size: Longitud L de cada ventana.
            step:        Paso s entre ventanas.
        Returns:
            Array de forma (N_index, window_size) con ventanas normalizadas.
        """
        T = len(series)
        indices = range(0, T - window_size + 1, step)
        windows = []
        for i in indices:
            w = series[i:i + window_size].astype(np.float32)
            mu, sigma = w.mean(), w.std()
            w = (w - mu) / (sigma + 1e-8)             # Z-score local (§6.1)
            windows.append(w)
        return np.stack(windows, axis=0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve dos vistas augmentadas de la ventana idx.

        Returns:
            Tupla (view_1, view_2): par positivo para NT-Xent.
        """
        w = torch.tensor(self.windows[idx])
        view_1 = apply_augmentation(w.unsqueeze(0)).squeeze(0)
        view_2 = apply_augmentation(w.unsqueeze(0)).squeeze(0)
        return view_1, view_2


# ===========================================================================
# 5. BUCLE DE ENTRENAMIENTO
# ===========================================================================

def train_one_epoch(
    model: TemporalEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NTXentLoss,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """
    Ejecuta una época completa de entrenamiento contrastivo.

    Args:
        model:      TemporalEncoder en modo entrenamiento.
        loader:     DataLoader con pares (view_1, view_2).
        optimizer:  Optimizador AdamW (§7.4.4).
        criterion:  Pérdida NT-Xent.
        device:     Dispositivo de cómputo.
        scheduler:  Scheduler de tasa de aprendizaje (Cosine Annealing, §7.4.4).
    Returns:
        Pérdida media de la época.
    """
    model.train()
    total_loss = 0.0

    for view_1, view_2 in loader:
        view_1 = view_1.to(device)
        view_2 = view_2.to(device)

        optimizer.zero_grad()

        _, z_i = model(view_1)
        _, z_j = model(view_2)

        loss = criterion(z_i, z_j)
        loss.backward()

        # Gradient clipping para estabilidad (especialmente en Transformers)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(
    model: TemporalEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_dir: str,
) -> None:
    """
    Guarda checkpoint del modelo con metadatos de entrenamiento.

    Estructura del checkpoint:
        {
            'epoch': int,
            'model_state_dict': OrderedDict,
            'backbone_state_dict': OrderedDict,   ← para carga directa en FAISS pipeline
            'optimizer_state_dict': OrderedDict,
            'loss': float,
        }

    Args:
        model:     TemporalEncoder entrenado.
        optimizer: Estado del optimizador.
        epoch:     Época actual.
        loss:      Pérdida de la época.
        save_dir:  Directorio de destino.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = Path(save_dir) / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    logger.info(f"Checkpoint guardado: {path}")


# ===========================================================================
# 6. PIPELINE PRINCIPAL
# ===========================================================================

def load_series(data_path: str, column: str | None = None) -> np.ndarray:
    """
    Carga la serie temporal desde CSV y devuelve un array 1D.

    Compatible con el formato del dataset NAB (ambient_temperature_system_failure).
    Si el CSV tiene múltiples columnas, selecciona 'value' o la columna indicada.

    Args:
        data_path: Ruta al archivo CSV.
        column:    Nombre de la columna de valores (None = auto-detección).
    Returns:
        Array numpy 1D con los valores de la serie.
    """
    df = pd.read_csv(data_path, parse_dates=True)
    if column is not None:
        series = df[column].values
    elif "value" in df.columns:
        series = df["value"].values
    else:
        # Seleccionar la primera columna numérica disponible
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No se encontraron columnas numéricas en {data_path}")
        series = df[numeric_cols[0]].values
        logger.warning(f"Columna no especificada; usando '{numeric_cols[0]}'")

    # Imputación de NaN por interpolación lineal (§6.1)
    series = pd.Series(series).interpolate(method="linear").ffill().bfill().values
    logger.info(f"Serie cargada: {len(series)} puntos temporales")
    return series.astype(np.float32)


def main(args: argparse.Namespace) -> None:
    """
    Punto de entrada principal del script de entrenamiento contrastivo.

    Flujo:
        1. Configuración del dispositivo y semilla aleatoria
        2. Carga y preparación de la serie temporal
        3. Construcción del dataset y DataLoader
        4. Inicialización del modelo, optimizador y scheduler
        5. Bucle de entrenamiento con NT-Xent
        6. Guardado del modelo final

    Args:
        args: Namespace de argparse con todos los hiperparámetros (§7.4.4).
    """
    # ── Reproducibilidad ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de cómputo: {device}")

    # ── Datos ────────────────────────────────────────────────────────────────
    series = load_series(args.data_path, args.column)

    # Split temporal 70/10/20 — el índice FAISS solo se construye con train (§6.9)
    n = len(series)
    train_end = int(n * 0.70)
    series_train = series[:train_end]

    dataset = SlidingWindowDataset(series_train, args.window_size, step=1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,               # NT-Xent requiere batches completos para diversidad de negativos
    )

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = TemporalEncoder(
        window_size=args.window_size,
        d_model=args.d_model,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros entrenables: {n_params:,}")

    # ── Optimizador y Scheduler (§7.4.4) ─────────────────────────────────────
    # AdamW con weight decay desacoplado (Loshchilov & Hutter, 2019)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    # Cosine Annealing: decaimiento suave hasta lr_min en T_max pasos
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )

    # ── Pérdida ──────────────────────────────────────────────────────────────
    criterion = NTXentLoss(temperature=args.tau, device=str(device))

    # ── Entrenamiento ────────────────────────────────────────────────────────
    best_loss = float("inf")
    logger.info(f"Iniciando entrenamiento: {args.epochs} épocas")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device, scheduler)
        logger.info(f"Época {epoch:03d}/{args.epochs} | Loss NT-Xent: {loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Guardar checkpoint cada 10 épocas y si es el mejor
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, loss, args.checkpoint_dir)

        if loss < best_loss:
            best_loss = loss
            best_path = Path(args.checkpoint_dir) / "best_encoder.pt"
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "model_config": {
                        "window_size": args.window_size,
                        "d_model": args.d_model,
                        "embed_dim": args.embed_dim,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                    },
                    "loss": loss,
                },
                best_path,
            )
            logger.info(f"  → Mejor modelo actualizado (loss={loss:.6f})")

    logger.info(f"Entrenamiento completado. Mejor loss: {best_loss:.6f}")
    logger.info(f"Modelo guardado en: {Path(args.checkpoint_dir) / 'best_encoder.pt'}")
    logger.info("El backbone está listo para indexación FAISS (ablation_study.py)")


# ===========================================================================
# 7. ARGPARSE
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos.

    Los valores por defecto corresponden a la configuración de entrenamiento
    reportada en la Tabla 7.4.4 de la memoria del TFM.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento contrastivo del Temporal Encoder (NT-Xent) — Temporal RAG TFM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Datos ────────────────────────────────────────────────────────────────
    parser.add_argument("--data_path", type=str, required=True,
                        help="Ruta al CSV de la serie temporal (NAB format)")
    parser.add_argument("--column", type=str, default=None,
                        help="Nombre de la columna de valores (default: auto)")

    # ── Arquitectura (§7.4.3 y §7.4.4) ──────────────────────────────────────
    parser.add_argument("--window_size", type=int, default=96,
                        help="Longitud L de la ventana temporal")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Dimensión interna del Transformer")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Dimensión del espacio contrastivo (FAISS)")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Número de cabezas de atención")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Número de capas del TransformerEncoder")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Tasa de dropout")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=50,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Tamaño del batch (maximiza diversidad de negativos en NT-Xent)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Tasa de aprendizaje inicial para AdamW")
    parser.add_argument("--tau", type=float, default=0.2,
                        help="Temperatura τ de la pérdida NT-Xent")

    # ── Infraestructura ───────────────────────────────────────────────────────
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directorio para guardar checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
