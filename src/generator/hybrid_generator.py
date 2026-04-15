"""
generator/hybrid_generator.py
==============================
Stage 6 of the Temporal RAG pipeline: the hybrid generator.

This is the component that unifies parametric memory (the neural network
weights) with non-parametric memory (the retrieved historical context)
to produce the final forecast or anomaly score.

Theoretical context (TFM §6 Phase 6)
--------------------------------------
In standard RAG for NLP, the generator is a language model that conditions
on the concatenation of the query and retrieved passages. The temporal
analogue is a neural network that receives both:

    (a) the latent representation of the current window z_q ∈ ℝ^d
    (b) a context representation C ∈ ℝ^{k×d} of retrieved embeddings

Two integration strategies are implemented:

**Concatenation (concat)**
    The k retrieved embeddings are aggregated (mean-pooled) into a single
    context vector c̄ ∈ ℝ^d and concatenated with z_q:

        input = [z_q ‖ c̄] ∈ ℝ^{2d}

    Simple and effective; loses inter-context relationships.

**Cross-Attention (cross_attn)**
    z_q acts as the *query*, the k retrieved embeddings form the *keys*
    and *values* in a single multi-head attention layer:

        c_attn = MultiHeadAttention(Q=z_q, K=C, V=C) ∈ ℝ^d
        input  = [z_q ‖ c_attn] ∈ ℝ^{2d}

    Allows the model to *selectively* weight the retrieved neighbours,
    giving full attention mass to the most informative episodes
    (Li et al., 2025; Zhang et al., 2025).

Both variants feed into a shared prediction head that produces either:
- A multi-step forecast (regression head)
- A scalar anomaly score in [0, 1] (binary classification head)

References
----------
Lewis, P., et al. (2020). Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks. *NeurIPS 33*, 9459–9474.
Li, J., et al. (2025). TimeRAG: Boosting LLM Time Series Forecasting
    via Retrieval-Augmented Generation. *AAAI 2025*.
Zhang, H., et al. (2025). TS-RAG: Retrieval-Augmented Generation based
    Time Series Foundation Models. *ICLR 2025*.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.encoder.ts_encoder import BaseEncoder
from src.retrieval.context_retriever import ContextRetriever, RetrievedContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-Attention context integrator
# ---------------------------------------------------------------------------


class CrossAttentionFusion(nn.Module):
    """
    Single-layer multi-head cross-attention for fusing query and context.

    The query window embedding attends over the k retrieved embeddings,
    producing a dynamically weighted context representation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of query and context embeddings.
    nhead : int
        Number of attention heads.
    dropout : float
        Attention dropout probability.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : torch.Tensor of shape (B, d)
            Current window embedding.
        context : torch.Tensor of shape (B, k, d)
            k retrieved neighbour embeddings.

        Returns
        -------
        fused : torch.Tensor of shape (B, d)
            Context-enriched representation.
        """
        # MultiheadAttention expects (B, seq_len, d)
        query_seq = query.unsqueeze(1)  # (B, 1, d)
        attended, _ = self.attn(
            query=query_seq,
            key=context,
            value=context,
        )  # (B, 1, d)
        attended = attended.squeeze(1)  # (B, d)
        return self.norm(attended + query)  # residual connection


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------


class ForecastHead(nn.Module):
    """
    Multi-layer perceptron regression head for multi-step forecasting.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (2*d for concat fusion).
    horizon : int
        Number of future time steps to forecast.
    """

    def __init__(self, input_dim: int, horizon: int = 1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, horizon)


class AnomalyHead(nn.Module):
    """
    Binary classification head for anomaly scoring.

    Outputs a probability score in [0, 1]; values above a tuned threshold
    τ are classified as anomalous (§7.2).

    Parameters
    ----------
    input_dim : int
        Input feature dimension (2*d for concat fusion).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# HybridGenerator
# ---------------------------------------------------------------------------


class HybridGenerator(nn.Module):
    """
    The core Temporal RAG model: parametric encoder + non-parametric
    retrieval + learned fusion and prediction head.

    Parameters
    ----------
    encoder : BaseEncoder
        Pre-trained (or jointly trained) temporal encoder f_θ.
    retriever : ContextRetriever
        Populated retriever wrapping the FAISS memory.
    latent_dim : int
        Embedding dimensionality d.
    fusion : {"concat", "cross_attn"}
        Context integration strategy (see module docstring).
    task : {"forecast", "anomaly", "both"}
        Which prediction head(s) to activate.
    horizon : int
        Forecast horizon (only used when task ∈ {"forecast", "both"}).
    nhead : int
        Attention heads (only used when fusion="cross_attn").

    Attributes
    ----------
    forecast_head : ForecastHead or None
    anomaly_head : AnomalyHead or None
    fusion_layer : CrossAttentionFusion or None
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        retriever: ContextRetriever,
        latent_dim: int = 32,
        fusion: Literal["concat", "cross_attn"] = "cross_attn",
        task: Literal["forecast", "anomaly", "both"] = "both",
        horizon: int = 1,
        nhead: int = 4,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.retriever = retriever
        self.latent_dim = latent_dim
        self.fusion = fusion
        self.task = task

        # 2*d because query embedding + fused context are concatenated
        head_input_dim = 2 * latent_dim

        if fusion == "cross_attn":
            self.fusion_layer = CrossAttentionFusion(latent_dim, nhead=nhead)
        else:
            self.fusion_layer = None  # type: ignore

        self.forecast_head: Optional[ForecastHead] = (
            ForecastHead(head_input_dim, horizon=horizon)
            if task in ("forecast", "both")
            else None
        )
        self.anomaly_head: Optional[AnomalyHead] = (
            AnomalyHead(head_input_dim)
            if task in ("anomaly", "both")
            else None
        )

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def _fuse(
        self,
        query_emb: torch.Tensor,
        contexts: List[RetrievedContext],
        device: str,
    ) -> torch.Tensor:
        """
        Produce the fused representation [z_q ‖ c] for each item in the batch.

        Parameters
        ----------
        query_emb : torch.Tensor of shape (B, d)
        contexts : list of RetrievedContext (length B)
        device : str

        Returns
        -------
        fused : torch.Tensor of shape (B, 2*d)
        """
        k_list = [
            torch.tensor(ctx.neighbour_embeddings, dtype=torch.float32).to(device)
            if len(ctx.neighbour_embeddings) > 0
            else torch.zeros(1, self.latent_dim, device=device)
            for ctx in contexts
        ]

        # Pad to equal k for batched attention
        max_k = max(t.shape[0] for t in k_list)
        context_padded = torch.stack(
            [
                torch.nn.functional.pad(t, (0, 0, 0, max_k - t.shape[0]))
                for t in k_list
            ]
        )  # (B, max_k, d)

        if self.fusion == "cross_attn" and self.fusion_layer is not None:
            context_repr = self.fusion_layer(query_emb, context_padded)  # (B, d)
        else:
            # Simple mean-pooling over retrieved neighbours
            context_repr = context_padded.mean(dim=1)  # (B, d)

        return torch.cat([query_emb, context_repr], dim=-1)  # (B, 2d)

    def forward(
        self,
        windows: np.ndarray,
        k: Optional[int] = None,
        device: Optional[str] = None,
    ) -> dict:
        """
        Full Temporal RAG forward pass.

        Parameters
        ----------
        windows : np.ndarray of shape (B, L)
            Normalised query windows.
        k : int, optional
            Override the retriever's default k.
        device : str, optional

        Returns
        -------
        outputs : dict with keys:
            - ``"forecast"``       np.ndarray (B, horizon) or None
            - ``"anomaly_score"``  np.ndarray (B,)         or None
            - ``"contexts"``       list[RetrievedContext]
        """
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()

        # 1. Encode query windows
        query_emb_np = self.encoder.encode(windows, device=dev)
        query_emb = torch.tensor(query_emb_np, dtype=torch.float32).to(dev)

        # 2. Retrieve context
        contexts = self.retriever.retrieve(windows, k=k)

        # 3. Fuse
        with torch.no_grad():
            fused = self._fuse(query_emb, contexts, dev)  # (B, 2d)

        # 4. Predict
        outputs: dict = {"contexts": contexts}

        with torch.no_grad():
            if self.forecast_head is not None:
                outputs["forecast"] = (
                    self.forecast_head(fused).cpu().numpy()
                )
            else:
                outputs["forecast"] = None

            if self.anomaly_head is not None:
                outputs["anomaly_score"] = (
                    self.anomaly_head(fused).cpu().numpy()
                )
            else:
                outputs["anomaly_score"] = None

        return outputs

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def anomaly_score(
        self,
        windows: np.ndarray,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute anomaly scores for a batch of windows.

        Combines the **learned anomaly head score** (parametric) with the
        **isolation score** derived from retrieval distances (non-parametric).
        The ensemble score is:

            s_ensemble = α · s_head + (1 - α) · s_isolation

        where α is a learnable or heuristic mixing weight (default 0.5).

        Parameters
        ----------
        windows : np.ndarray of shape (N, L)
        k : int, optional

        Returns
        -------
        scores : np.ndarray of shape (N,)
            Ensemble anomaly scores in [0, 1].
        """
        alpha = 0.5
        out = self.forward(windows, k=k)

        isolation_scores = np.array(
            [ctx.anomaly_isolation_score for ctx in out["contexts"]]
        )

        if out["anomaly_score"] is not None:
            return alpha * out["anomaly_score"] + (1 - alpha) * isolation_scores
        else:
            return isolation_scores
