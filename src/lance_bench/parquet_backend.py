"""Read embeddings from Parquet into a dense float32 matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def embedding_table(ids: np.ndarray, embeddings: np.ndarray) -> pa.Table:
    """Shared Arrow schema for Parquet and Lance corpora."""
    _, d = embeddings.shape
    flat = np.asarray(embeddings, dtype=np.float32).reshape(-1)
    list_array = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float32()),
        list_size=d,
    )
    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "embedding": list_array,
        }
    )


def write_corpus_parquet(path: Path, ids: np.ndarray, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = embedding_table(ids, embeddings)
    pq.write_table(table, path, compression="zstd")


def as_fixed_size_list(col: pa.Array | pa.ChunkedArray) -> pa.FixedSizeListArray:
    if isinstance(col, pa.ChunkedArray):
        col = pa.concat_arrays(col.chunks)
    if not pa.types.is_fixed_size_list(col.type):
        raise TypeError(f"expected fixed_size_list embedding column, got {col.type}")
    return col


def load_corpus_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["id", "embedding"])
    ids = table["id"].to_numpy(zero_copy_only=False)
    emb_col = as_fixed_size_list(table["embedding"])
    values = emb_col.values.to_numpy(zero_copy_only=False)
    d = emb_col.type.list_size
    n = len(emb_col)
    emb = values.reshape(n, d).astype(np.float32, copy=False)
    return ids.astype(np.int64, copy=False), emb
