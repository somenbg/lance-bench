"""Write / read Lance table with the same logical schema as Parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lance_bench.parquet_backend import as_fixed_size_list, embedding_table


def write_corpus_lance(uri: Path, table_name: str, ids: np.ndarray, embeddings: np.ndarray) -> None:
    uri.mkdir(parents=True, exist_ok=True)
    import lancedb

    table = embedding_table(ids, embeddings)
    db = lancedb.connect(str(uri))
    db.create_table(table_name, data=table, mode="overwrite")


def load_corpus_matrix(uri: Path, table_name: str) -> tuple[np.ndarray, np.ndarray]:
    import lancedb

    db = lancedb.connect(str(uri))
    t = db.open_table(table_name)
    arrow_table = t.to_arrow()
    ids = arrow_table["id"].to_numpy(zero_copy_only=False)
    emb_col = as_fixed_size_list(arrow_table["embedding"])
    values = emb_col.values.to_numpy(zero_copy_only=False)
    d = emb_col.type.list_size
    n = len(emb_col)
    emb = values.reshape(n, d).astype(np.float32, copy=False)
    return ids.astype(np.int64, copy=False), emb
