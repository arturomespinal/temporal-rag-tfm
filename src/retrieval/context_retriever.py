"""
retrieval/context_retriever.py
==============================
Stage 5 of the Temporal RAG pipeline: contextual retrieval.
 
This module encodes a query window and retrieves the k most dynamically
similar historical windows from the FAISS memory, forming the **episodic
context** that is injected into the hybrid generator.
 
Theoretical context (TFM §6 Phase 5)
--------------------------------------
At inference time, the system receives a new query window W_q ∈ ℝ^L.
The retrieval procedure is:
 
    1. Encode: z_q = f_θ(W_q) ∈ ℝ^d
    2. Normalise: z_q ← z_q / ‖z_q‖₂  (for cosine distance consistency)
    3. k-NN search: {(z_i, m_i)}_{i=1}^k = FAISS.search(z_q, k)
    4. Return: the raw windows corresponding to the k retrieved embeddings
 
The intuition is that the k retrieved windows represent historical episodes
where the system exhibited *similar* dynamic behaviour. Under the RAG
paradigm, these constitute non-parametric "evidence" that the hybrid
generator can use to condition its forecast or anomaly score — analogous
to how a language model conditions on retrieved document passages
(Lewis et al., 2020).
 
The sensitivity analysis over k (§7.2) investigates the trade-off:
- Small k  → low context diversity, fast retrieval
- Large k  → richer context, risk of noisy/irrelevant neighbours
 
References
----------
Lewis, P., et al. (2020). Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks. *NeurIPS 33*, 9459–9474.
Zhang, H., et al. (2025). TS-RAG: Retrieval-Augmented Generation based
    Time Series Foundation Models. *ICLR 2025 submissions*.
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
from typing import List, Optional
 
import numpy as np
import torch
 
from src.encoder.ts_encoder import BaseEncoder
from src.index.faiss_index import TemporalFAISSIndex
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------
 
 
@dataclass
class RetrievedContext:
    """
    Container for the results of a single retrieval call.
 
    Attributes
    ----------
    query_window : np.ndarray of shape (L,)
        The normalised query window used to produce the query embedding.
    query_embedding : np.ndarray of shape (d,)
        Latent representation of the query window.
    neighbour_windows : np.ndarray of shape (k, L)
        Raw (normalised) windows of the k retrieved historical neighbours.
    neighbour_embeddings : np.ndarray of shape (k, d)
        Latent representations of the retrieved neighbours.
    distances : np.ndarray of shape (k,)
        L2 distances from the query to each neighbour.
    metadata : list[dict]
        Metadata associated with each retrieved neighbour.
    """
 
    query_window: np.ndarray
    query_embedding: np.ndarray
    neighbour_windows: np.ndarray
    neighbour_embeddings: np.ndarray
    distances: np.ndarray
    metadata: List[dict] = field(default_factory=list)
 
    @property
    def mean_distance(self) -> float:
        """Mean L2 distance to retrieved neighbours (isolation proxy)."""
        return float(self.distances.mean())
 
    @property
    def anomaly_isolation_score(self) -> float:
        """
        Normalised isolation score ∈ [0, 1].
 
        A high score indicates that the query window is far from all
        retrieved neighbours in the latent space, which is a proxy for
        anomalousness. This is the RAG-based anomaly signal (§7.2).
 
            score = 1 - exp(-mean_distance / τ)
 
        where τ is a temperature parameter (default = 1.0). Values close
        to 1 suggest the query resides in a low-density region of the
        latent space — a characteristic of contextual anomalies.
        """
        tau = 1.0
        return float(1.0 - np.exp(-self.mean_distance / tau))
 
 
# ---------------------------------------------------------------------------
# ContextRetriever
# ---------------------------------------------------------------------------
 
 
class ContextRetriever:
    """
    Encodes query windows and retrieves their k nearest historical neighbours
    from the FAISS temporal memory.
 
    Parameters
    ----------
    index : TemporalFAISSIndex
        The populated vector index (non-parametric historical memory).
    encoder : BaseEncoder
        The same encoder used to build the index. Consistency is critical:
        using a different encoder would map queries to an incompatible space.
    k : int
        Default number of neighbours to retrieve. Can be overridden per call.
    device : str, optional
        ``"cuda"`` or ``"cpu"``. Auto-detected if None.
 
    Notes
    -----
    The ``encoder`` is used in eval mode with gradients disabled. Callers
    should ensure the encoder has been trained before instantiating this
    class.
    """
 
    def __init__(
        self,
        index: TemporalFAISSIndex,
        encoder: BaseEncoder,
        k: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.k = k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
 
        self.encoder.eval()
        self.encoder.to(self.device)
 
    def retrieve(
        self,
        query_windows: np.ndarray,
        k: Optional[int] = None,
    ) -> List[RetrievedContext]:
        """
        Retrieve context for a batch of query windows.
 
        Parameters
        ----------
        query_windows : np.ndarray of shape (Q, L)
            Batch of normalised query windows.
        k : int, optional
            Override the default k for this call.
 
        Returns
        -------
        contexts : list[RetrievedContext]
            One RetrievedContext per query window.
        """
        k_eff = k or self.k
 
        # Encode query batch
        query_embeddings = self.encoder.encode(
            query_windows, device=self.device
        )  # (Q, d)
 
        # k-NN retrieval
        distances, indices, metadata_lists = self.index.search(
            query_embeddings, k=k_eff
        )
 
        contexts = []
        for q in range(len(query_windows)):
            valid_mask = indices[q] >= 0
 
            # Reconstruct neighbour windows from metadata
            neighbour_windows = np.stack(
                [
                    meta["window"]
                    for meta, valid in zip(metadata_lists[q], valid_mask)
                    if valid and "window" in meta
                ],
                axis=0,
            ) if any(valid_mask) and "window" in metadata_lists[q][0] else np.empty((0, query_windows.shape[1]))
 
            neighbour_emb = (
                query_embeddings[indices[q][valid_mask]]
                if valid_mask.any()
                else np.empty((0, query_embeddings.shape[1]))
            )
 
            ctx = RetrievedContext(
                query_window=query_windows[q],
                query_embedding=query_embeddings[q],
                neighbour_windows=neighbour_windows,
                neighbour_embeddings=neighbour_emb,
                distances=distances[q][valid_mask],
                metadata=[
                    m
                    for m, v in zip(metadata_lists[q], valid_mask)
                    if v
                ],
            )
            contexts.append(ctx)
 
        logger.debug(
            "Retrieved k=%d neighbours for %d queries. "
            "Avg isolation score: %.4f",
            k_eff,
            len(query_windows),
            np.mean([c.anomaly_isolation_score for c in contexts]),
        )
        return contexts
 
    def retrieve_one(
        self, window: np.ndarray, k: Optional[int] = None
    ) -> RetrievedContext:
        """
        Convenience wrapper to retrieve context for a single window.
 
        Parameters
        ----------
        window : np.ndarray of shape (L,)
        k : int, optional
 
        Returns
        -------
        RetrievedContext
        """
        return self.retrieve(window[np.newaxis, :], k=k)[0]
