"""
encoder/ts_encoder.py
=====================
Stage 3 of the Temporal RAG pipeline: generating dynamic embeddings.

This module provides three encoder architectures evaluated in §6 (Phase 3):

1. ``AutoencoderEncoder``   — non-linear reconstruction baseline.
2. ``LSTMEncoder``          — sequential dependency extraction.
3. ``TransformerEncoder``   — attention-based complex pattern capture.

All encoders share a common ``BaseEncoder`` interface, making them
interchangeable downstream in the FAISS indexing and retrieval stages.

Theoretical context (TFM §6 Phase 3)
--------------------------------------
The central requirement of a temporal embedding is that **geometric proximity
in the latent space reflects dynamic similarity** between windows — i.e., two
windows with similar shapes, trends and volatility regimes should produce
nearby vectors, regardless of their absolute position in time.

Formally, given a sliding window W_i ∈ ℝ^L, the encoder f_θ: ℝ^L → ℝ^d
must satisfy:

    d(f_θ(W_i), f_θ(W_j)) ≈ 0  ⟺  W_i and W_j share the same dynamic regime

where d(·,·) is Euclidean distance or cosine distance. This property is not
guaranteed by autoencoders alone; it requires training signals that enforce
the latent geometry (contrastive, reconstruction, or prediction objectives).

References
----------
Franceschi, J.-Y., Dieuleveut, A., & Jaggi, M. (2019). Unsupervised scalable
    representation learning for multivariate time series. *NeurIPS 32*.
Eldele, E., et al. (2021). Time-series representation learning via temporal
    and contextual contrasting. *IJCAI*, 2352–2359.
Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 30*.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseEncoder(ABC, nn.Module):
    """
    Abstract interface for all Temporal RAG encoders.

    Subclasses must implement ``forward`` (PyTorch forward pass) and
    ``encode`` (convenience batch-inference wrapper returning numpy arrays).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, L) or (B, L, C)
            Batch of B windows, each of length L (C channels for multivariate).

        Returns
        -------
        z : torch.Tensor of shape (B, d)
            Latent embedding vectors.
        """

    def encode(
        self,
        windows: np.ndarray,
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convenience method: encode a numpy array of windows in mini-batches.

        Parameters
        ----------
        windows : np.ndarray of shape (N, L)
            Pre-normalised sliding windows from NABLoader.
        batch_size : int
            Mini-batch size for GPU inference.
        device : str, optional
            ``"cuda"`` or ``"cpu"``. Auto-detected if None.

        Returns
        -------
        embeddings : np.ndarray of shape (N, d)
            L2-normalised latent vectors (unit sphere ⟹ cosine similarity
            equals dot product, enabling efficient FAISS IndexFlatIP).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.eval()
        self.to(device)

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = torch.tensor(
                    windows[i : i + batch_size], dtype=torch.float32
                ).to(device)
                z = self.forward(batch)  # (B, d)
                # L2-normalise for cosine similarity compatibility
                z = nn.functional.normalize(z, p=2, dim=1)
                all_embeddings.append(z.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(
            "Encoded %d windows → embedding matrix shape %s",
            len(windows),
            embeddings.shape,
        )
        return embeddings


# ---------------------------------------------------------------------------
# 1. LSTM Encoder
# ---------------------------------------------------------------------------


class LSTMEncoder(BaseEncoder):
    """
    Bidirectional LSTM encoder that maps a univariate window of length L
    to a dense latent vector of dimension ``latent_dim``.

    Architecture
    ------------
    Input (B, L) → unsqueeze → (B, L, 1)
        → BiLSTM (num_layers=2)
        → last hidden state concatenation [h_fwd; h_bwd] → (B, 2·hidden_size)
        → Linear projection → (B, latent_dim)
        → Tanh activation

    The bidirectional design ensures that the embedding captures both
    forward (causal) and backward (anti-causal) temporal dependencies,
    which is particularly valuable for detecting the onset and offset of
    anomalous regimes.

    Parameters
    ----------
    input_size : int
        Length of each input window (L).
    hidden_size : int
        Number of LSTM units per direction. Total LSTM output = 2*hidden_size.
    latent_dim : int
        Dimensionality of the output embedding vector.
    num_layers : int
        Depth of the stacked BiLSTM.
    dropout : float
        Dropout applied between LSTM layers (0 to disable).

    References
    ----------
    Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
        *Neural Computation, 9*(8), 1735–1780.
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(
            input_size=1,  # univariate — one feature per time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project (2 * hidden_size) → latent_dim
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_size, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, L)
        """
        # (B, L) → (B, L, 1)
        x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * 2, B, hidden_size)
        # Extract last layer's forward and backward hidden states
        h_fwd = h_n[-2]  # (B, hidden_size)
        h_bwd = h_n[-1]  # (B, hidden_size)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden_size)
        return self.projection(h_cat)  # (B, latent_dim)


# ---------------------------------------------------------------------------
# 2. Transformer Encoder
# ---------------------------------------------------------------------------


class TransformerEncoder(BaseEncoder):
    """
    Transformer encoder for time series windows.

    Architecture
    ------------
    Input (B, L) → Linear patch embedding → (B, L, d_model)
        → Positional encoding (sinusoidal)
        → N × TransformerEncoderLayer (multi-head self-attention + FFN)
        → Global mean pooling over time → (B, d_model)
        → Linear projection → (B, latent_dim)

    Self-attention allows the model to discover arbitrary-range dependencies
    within the window, capturing complex seasonal and trend interactions that
    exceed the modelling capacity of recurrent architectures (Vaswani et al.,
    2017; Zhou et al., 2021).

    Parameters
    ----------
    input_size : int
        Length of each input window (L).
    d_model : int
        Internal Transformer dimensionality.
    nhead : int
        Number of attention heads (must divide d_model).
    num_encoder_layers : int
        Number of stacked self-attention layers.
    latent_dim : int
        Output embedding dimension.
    dropout : float
        Dropout in attention and FFN layers.

    References
    ----------
    Zhou, H., et al. (2021). Informer. *AAAI 35*(12), 11106–11115.
    Wu, H., et al. (2021). Autoformer. *NeurIPS 34*, 22419–22430.
    """

    def __init__(
        self,
        input_size: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size

        # Scalar-to-d_model linear projection (patch embedding)
        self.input_projection = nn.Linear(1, d_model)

        # Sinusoidal positional encoding
        self.register_buffer(
            "pos_encoding",
            self._build_pos_encoding(input_size, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.projection = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Tanh(),
        )

    @staticmethod
    def _build_pos_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        """Classic sinusoidal positional encoding (Vaswani et al., 2017)."""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe  # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, L)
        """
        x = x.unsqueeze(-1)  # (B, L, 1)
        x = self.input_projection(x)  # (B, L, d_model)
        x = x + self.pos_encoding  # add positional encoding
        x = self.transformer(x)  # (B, L, d_model)
        x = x.mean(dim=1)  # global average pooling → (B, d_model)
        return self.projection(x)  # (B, latent_dim)


# ---------------------------------------------------------------------------
# 3. Autoencoder Encoder (reconstruction baseline)
# ---------------------------------------------------------------------------


class AutoencoderEncoder(BaseEncoder):
    """
    Convolutional Autoencoder — encoder component only.

    This serves as the **reconstruction baseline** in the ablation study.
    Its bottleneck representation is used as the temporal embedding.
    The decoder (``AutoencoderDecoder``) is trained alongside but discarded
    at indexing time.

    Architecture
    ------------
    (B, L) → Conv1D stack (progressively halving temporal resolution)
           → Flatten → Linear → (B, latent_dim)

    Parameters
    ----------
    input_size : int
        Length of each input window.
    latent_dim : int
        Bottleneck / embedding dimensionality.

    References
    ----------
    Xu, H., et al. (2018). Unsupervised anomaly detection via Variational
        Auto-Encoders. *WWW*, 187–196.
    """

    def __init__(self, input_size: int = 64, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.conv_stack = nn.Sequential(
            # Layer 1: (B, 1, L) → (B, 16, L/2)
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 2: (B, 16, L/2) → (B, 32, L/4)
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: (B, 32, L/4) → (B, 64, L/8)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        conv_out_size = 64 * (input_size // 8)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, L)
        """
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.conv_stack(x)  # (B, 64, L/8)
        x = x.flatten(start_dim=1)  # (B, 64*L/8)
        return self.fc(x)  # (B, latent_dim)
