"""Synthetic corpus + queries for repeatable micro-benchmarks."""

from __future__ import annotations

import numpy as np


def make_random_embeddings(
    n_rows: int,
    dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (ids (n_rows,), embeddings (n_rows, dim)) float32.
    """
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_rows, dim), dtype=np.float64).astype(np.float32)
    ids = np.arange(n_rows, dtype=np.int64)
    return ids, emb


def make_queries(
    n_queries: int,
    dim: int,
    seed: int = 43,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_queries, dim), dtype=np.float64).astype(np.float32)
