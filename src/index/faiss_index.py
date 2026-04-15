"""
index/faiss_index.py
====================
Stage 4 of the Temporal RAG pipeline: vector indexing.

This module wraps Facebook AI Similarity Search (FAISS) to build and query
the **non-parametric temporal memory** of the Temporal RAG system.

Theoretical context (TFM §6 Phase 4)
--------------------------------------
Brute-force k-nearest-neighbour search has O(N·d) complexity per query,
where N is the total number of indexed windows and d is the embedding
dimensionality. For a deployment scenario with months of sensor data at
1-minute resolution (N ≈ 500 000) and d = 32–128, this becomes infeasible
for real-time inference.

FAISS addresses this via **approximate nearest-neighbour (ANN) search**.
Two index types are provided here:

* ``IndexFlatL2``   — exact Euclidean search; used for correctness tests
                      on small datasets and baseline comparison.
* ``IndexIVFFlat``  — Inverted File index; partitions the embedding space
                      into ``nlist`` Voronoi cells. At query time, only
                      ``nprobe`` cells are searched (nprobe ≪ nlist), giving
                      sub-linear latency at a small recall cost.

The choice between these index types follows the trade-off studied in
Johnson et al. (2019): for N < 10⁵ exact search is sufficient; for
N ≥ 10⁵, IVFFlat with nlist = 4√N and nprobe ≈ 8 provides >95% recall
at O(log N) latency.

References
----------
Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search
    with GPUs. *IEEE Transactions on Big Data, 7*(3), 535–547.
Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate
    nearest-neighbour search. *IEEE TPAMI, 42*(4), 824–836.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "FAISS is required: pip install faiss-cpu  (or faiss-gpu for GPU support)"
    ) from exc

logger = logging.getLogger(__name__)


class TemporalFAISSIndex:
    """
    Non-parametric temporal memory backed by a FAISS index.

    At build time, embeddings of all historical windows are added to the
    index together with their metadata (timestamps, original values, labels).
    At query time, the index returns the k most similar historical windows
    for any new embedding vector.

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding vectors (must match encoder output).
    index_type : {"flat", "ivf"}
        ``"flat"``  — exact search (IndexFlatL2), safe for small N.
        ``"ivf"``   — approximate search (IndexIVFFlat), required for large N.
    nlist : int
        Number of Voronoi cells for IVF (used only when index_type="ivf").
        Rule of thumb: nlist ≈ 4 * sqrt(N).
    nprobe : int
        Number of cells to visit at query time (higher ⟹ better recall,
        slower query). Typically nprobe = nlist / 8.
    metric : {"l2", "cosine"}
        Distance metric. Cosine requires unit-norm embeddings (enforced by
        ``BaseEncoder.encode``).

    Attributes
    ----------
    index : faiss.Index
        The underlying FAISS index object.
    metadata : list[dict]
        Per-vector metadata stored in insertion order.
    is_trained : bool
        Whether the IVF quantiser has been trained (always True for flat).
    """

    def __init__(
        self,
        dim: int = 32,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        metric: str = "l2",
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric

        self.metadata: list[dict] = []
        self._faiss_metric = (
            faiss.METRIC_L2
            if metric == "l2"
            else faiss.METRIC_INNER_PRODUCT
        )
        self.index: faiss.Index = self._init_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_index(self) -> faiss.Index:
        """Instantiate (but do not train) the FAISS index."""
        if self.index_type == "flat":
            idx = faiss.IndexFlatL2(self.dim)
            logger.info("Initialized IndexFlatL2(dim=%d)", self.dim)
        elif self.index_type == "ivf":
            quantiser = faiss.IndexFlatL2(self.dim)
            idx = faiss.IndexIVFFlat(
                quantiser, self.dim, self.nlist, self._faiss_metric
            )
            idx.nprobe = self.nprobe
            logger.info(
                "Initialized IndexIVFFlat(dim=%d, nlist=%d, nprobe=%d)",
                self.dim,
                self.nlist,
                self.nprobe,
            )
        else:
            raise ValueError(
                f"Unknown index_type '{self.index_type}'. Choose 'flat' or 'ivf'."
            )
        return idx

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        """FAISS requires contiguous float32 arrays."""
        return np.ascontiguousarray(arr, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self.index.is_trained

    @property
    def n_vectors(self) -> int:
        return self.index.ntotal

    def build(
        self,
        embeddings: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Train (IVF only) and populate the index.

        Parameters
        ----------
        embeddings : np.ndarray of shape (N, d)
            L2-normalised embedding vectors from ``BaseEncoder.encode``.
        metadata : list[dict], optional
            Per-vector metadata dictionaries (e.g., timestamps, labels,
            raw window values). If None, indices are used as metadata.

        Raises
        ------
        ValueError
            If embedding dimensionality does not match ``self.dim``.
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} ≠ index dim {self.dim}."
            )

        vecs = self._to_float32(embeddings)

        if self.index_type == "ivf":
            logger.info("Training IVF quantiser on %d vectors …", len(vecs))
            self.index.train(vecs)

        self.index.add(vecs)

        if metadata is not None:
            assert len(metadata) == len(embeddings), (
                "metadata length must equal number of embeddings"
            )
            self.metadata = metadata
        else:
            self.metadata = [{"idx": i} for i in range(len(embeddings))]

        logger.info("Index built. Total vectors: %d", self.n_vectors)

    def add(
        self,
        embeddings: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Incrementally add new vectors to an existing index.

        This enables **online/streaming** operation: as new windows arrive,
        their embeddings can be appended without rebuilding the full index.
        Note: IVF indices must be trained before calling ``add``.

        Parameters
        ----------
        embeddings : np.ndarray of shape (M, d)
        metadata : list[dict], optional
        """
        if not self.is_trained:
            raise RuntimeError(
                "IVF index must be trained before adding vectors. "
                "Call build() with initial data first."
            )
        vecs = self._to_float32(embeddings)
        self.index.add(vecs)

        if metadata:
            self.metadata.extend(metadata)
        else:
            start = self.n_vectors - len(embeddings)
            self.metadata.extend([{"idx": start + i} for i in range(len(embeddings))])

        logger.debug("Added %d vectors. Total: %d", len(embeddings), self.n_vectors)

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, list[list[dict]]]:
        """
        Retrieve the k nearest historical windows for each query embedding.

        Parameters
        ----------
        query_embeddings : np.ndarray of shape (Q, d)
            Embedding vectors of the query windows (current time windows).
        k : int
            Number of nearest neighbours to retrieve per query.

        Returns
        -------
        distances : np.ndarray of shape (Q, k)
            L2 distances (or inner product scores if metric="cosine") to
            the retrieved neighbours.
        indices : np.ndarray of shape (Q, k)
            Integer indices of the retrieved vectors in the index.
        retrieved_metadata : list of lists of dict
            ``retrieved_metadata[q][i]`` is the metadata dict for the i-th
            nearest neighbour of query q.
        """
        vecs = self._to_float32(query_embeddings)
        distances, indices = self.index.search(vecs, k)

        retrieved_metadata = []
        for row in indices:
            row_meta = []
            for idx in row:
                if idx >= 0 and idx < len(self.metadata):
                    row_meta.append(self.metadata[idx])
                else:
                    row_meta.append({"idx": -1})  # FAISS returns -1 for padding
            retrieved_metadata.append(row_meta)

        return distances, indices, retrieved_metadata

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Parameters
        ----------
        path : str or Path
            Directory where ``faiss.index`` and ``metadata.pkl`` will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "metadata.pkl", "wb") as fh:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "dim": self.dim,
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                    "metric": self.metric,
                },
                fh,
            )
        logger.info("Index saved to %s (%d vectors)", path, self.n_vectors)

    @classmethod
    def load(cls, path: str | Path) -> "TemporalFAISSIndex":
        """
        Restore a previously saved index from disk.

        Parameters
        ----------
        path : str or Path
            Directory containing ``faiss.index`` and ``metadata.pkl``.

        Returns
        -------
        TemporalFAISSIndex
        """
        path = Path(path)

        with open(path / "metadata.pkl", "rb") as fh:
            state = pickle.load(fh)

        instance = cls(
            dim=state["dim"],
            index_type=state["index_type"],
            nlist=state["nlist"],
            nprobe=state["nprobe"],
            metric=state["metric"],
        )
        instance.index = faiss.read_index(str(path / "faiss.index"))
        instance.metadata = state["metadata"]

        logger.info(
            "Index loaded from %s (%d vectors)", path, instance.n_vectors
        )
        return instance
