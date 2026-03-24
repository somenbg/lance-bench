"""Brute-force k-NN (L2) on float32 embeddings — shared by both backends."""

from __future__ import annotations

import numpy as np


def knn_l2(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    queries: (Q, d) float32
    corpus: (N, d) float32
    Returns (indices (Q, k), distances (Q, k)), sorted by distance ascending per row.
    """
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32, copy=False)
    if corpus.dtype != np.float32:
        corpus = corpus.astype(np.float32, copy=False)
    # (Q, N) squared L2 via expansion: ||q-c||^2 = ||q||^2 + ||c||^2 - 2 q·c
    q2 = np.sum(queries * queries, axis=1, keepdims=True)  # (Q, 1)
    c2 = np.sum(corpus * corpus, axis=1, keepdims=True).T  # (1, N)
    d2 = q2 + c2 - 2.0 * (queries @ corpus.T)
    # k smallest per row
    if k >= corpus.shape[0]:
        idx = np.argsort(d2, axis=1)
    else:
        idx = np.argpartition(d2, kth=k - 1, axis=1)
        idx = idx[:, :k]
        row = np.arange(queries.shape[0])[:, None]
        order = np.argsort(d2[row, idx], axis=1)
        idx = idx[row, order]
    dist = np.sqrt(np.maximum(d2[np.arange(queries.shape[0])[:, None], idx], 0.0))
    return idx.astype(np.int64), dist.astype(np.float32)
