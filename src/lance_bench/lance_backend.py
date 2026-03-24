"""Write / read Lance columnar dataset (pylance from https://github.com/lance-format/lance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lance_bench.parquet_backend import as_fixed_size_list, embedding_table


def write_corpus_lance(uri: Path, _table_name: str, ids: np.ndarray, embeddings: np.ndarray) -> None:
    """Write corpus as a Lance dataset under ``uri`` (directory).

    ``_table_name`` is ignored; it remains in the signature so call sites match the old
    LanceDB-oriented API (one named table per DB directory).
    """
    from lance.dataset import write_dataset

    uri.mkdir(parents=True, exist_ok=True)
    table = embedding_table(ids, embeddings)
    write_dataset(table, str(uri), mode="overwrite", schema=table.schema)


def load_corpus_matrix(uri: Path, _table_name: str) -> tuple[np.ndarray, np.ndarray]:
    from lance import LanceDataset

    ds = LanceDataset(str(uri))
    arrow_table = ds.to_table(columns=["id", "embedding"])
    ids = arrow_table["id"].to_numpy(zero_copy_only=False)
    emb_col = as_fixed_size_list(arrow_table["embedding"])
    values = emb_col.values.to_numpy(zero_copy_only=False)
    d = emb_col.type.list_size
    n = len(emb_col)
    emb = values.reshape(n, d).astype(np.float32, copy=False)
    return ids.astype(np.int64, copy=False), emb
