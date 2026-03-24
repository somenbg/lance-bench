"""Time load + shared brute-force kNN for Parquet vs Lance."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lance_bench.bruteforce import knn_l2
from lance_bench.lance_backend import load_corpus_matrix as load_lance
from lance_bench.parquet_backend import load_corpus_matrix as load_parquet
from lance_bench.resource_profile import dir_tree_size_bytes, file_size_bytes


@dataclass
class BenchResult:
    backend: str
    n_rows: int
    dim: int
    n_queries: int
    k: int
    load_seconds: float
    search_seconds: float
    total_seconds: float
    parquet_path: str | None = None
    lance_uri: str | None = None
    scenario_id: str = ""
    scenario_group: str = ""
    trial: int = 0
    storage_bytes: int | None = None
    rss_peak_load_bytes: int | None = None
    rss_peak_search_bytes: int | None = None
    repeat_index: int = 0
    sample_count: int = 1
    load_stdev: float | None = None
    search_stdev: float | None = None
    total_stdev: float | None = None
    rss_peak_load_stdev: float | None = None
    rss_peak_search_stdev: float | None = None


def _time_load_parquet(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    ids, emb = load_parquet(path)
    t1 = time.perf_counter()
    return ids, emb, t1 - t0


def _time_load_lance(uri: Path, table: str) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    ids, emb = load_lance(uri, table)
    t1 = time.perf_counter()
    return ids, emb, t1 - t0


def run_pair(
    parquet_path: Path,
    lance_uri: Path,
    lance_table: str,
    queries: np.ndarray,
    k: int,
    *,
    profile: bool = False,
) -> tuple[BenchResult, BenchResult]:
    pq_disk = file_size_bytes(parquet_path)
    lance_disk = dir_tree_size_bytes(lance_uri)

    if not profile:
        _, emb_pq, load_pq = _time_load_parquet(parquet_path)
        t0 = time.perf_counter()
        knn_l2(queries, emb_pq, k)
        search_pq = time.perf_counter() - t0
        r_pq = BenchResult(
            backend="parquet",
            n_rows=emb_pq.shape[0],
            dim=emb_pq.shape[1],
            n_queries=queries.shape[0],
            k=k,
            load_seconds=load_pq,
            search_seconds=search_pq,
            total_seconds=load_pq + search_pq,
            parquet_path=str(parquet_path),
            storage_bytes=pq_disk,
        )
        del emb_pq
        gc.collect()

        _, emb_lance, lance_load_s = _time_load_lance(lance_uri, lance_table)
        t0 = time.perf_counter()
        knn_l2(queries, emb_lance, k)
        search_lance = time.perf_counter() - t0
        r_lance = BenchResult(
            backend="lance",
            n_rows=emb_lance.shape[0],
            dim=emb_lance.shape[1],
            n_queries=queries.shape[0],
            k=k,
            load_seconds=lance_load_s,
            search_seconds=search_lance,
            total_seconds=lance_load_s + search_lance,
            lance_uri=str(lance_uri),
            storage_bytes=lance_disk,
        )
        return r_pq, r_lance

    from lance_bench.resource_profile import peak_rss_during

    t0 = time.perf_counter()
    (_ids_pq, emb_pq), peak_load_pq = peak_rss_during(lambda: load_parquet(parquet_path))
    load_pq = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, peak_search_pq = peak_rss_during(lambda: knn_l2(queries, emb_pq, k))
    search_pq = time.perf_counter() - t0
    r_pq = BenchResult(
        backend="parquet",
        n_rows=emb_pq.shape[0],
        dim=emb_pq.shape[1],
        n_queries=queries.shape[0],
        k=k,
        load_seconds=load_pq,
        search_seconds=search_pq,
        total_seconds=load_pq + search_pq,
        parquet_path=str(parquet_path),
        storage_bytes=pq_disk,
        rss_peak_load_bytes=peak_load_pq,
        rss_peak_search_bytes=peak_search_pq,
    )
    del emb_pq
    gc.collect()

    t0 = time.perf_counter()
    (_ids_l, emb_lance), peak_load_l = peak_rss_during(lambda: load_lance(lance_uri, lance_table))
    lance_load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, peak_search_l = peak_rss_during(lambda: knn_l2(queries, emb_lance, k))
    search_lance = time.perf_counter() - t0
    r_lance = BenchResult(
        backend="lance",
        n_rows=emb_lance.shape[0],
        dim=emb_lance.shape[1],
        n_queries=queries.shape[0],
        k=k,
        load_seconds=lance_load_s,
        search_seconds=search_lance,
        total_seconds=lance_load_s + search_lance,
        lance_uri=str(lance_uri),
        storage_bytes=lance_disk,
        rss_peak_load_bytes=peak_load_l,
        rss_peak_search_bytes=peak_search_l,
    )
    return r_pq, r_lance


def write_results_json(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
