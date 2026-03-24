# First benchmark: Parquet vs Lance

This document describes the **initial harness** added to compare reading the same logical corpus from **Parquet** versus **Lance** (via LanceDB), then running the **same** brute-force k-nearest-neighbor search on the materialized embedding matrix.

It is intentionally narrow: one schema, one distance metric (L2), one search implementation, explicit timings for **load** vs **search** so results stay interpretable.

---

## What was implemented

### Packaging and layout

- **`pyproject.toml`** — Python 3.10+, dependencies (`lancedb`, `pyarrow`, `numpy`, `typer`), Hatchling wheel over `src/lance_bench`, console entry point **`lance-bench`**.
- **`src/lance_bench/`** — importable package with small, single-purpose modules (see below).
- **`.gitignore`** — ignores `.venv/`, `data/`, `results/`, build artifacts so generated corpora and timing JSON do not clutter version control.

### Shared data model

Both backends use the same Apache Arrow shape before persistence:

| Column      | Type                         |
| ----------- | ---------------------------- |
| `id`        | `int64`                      |
| `embedding` | `fixed_size_list<float32>`   |

`parquet_backend.embedding_table()` builds this `pa.Table` once; Parquet writes it with Zstd compression, Lance ingests the same table through `lancedb.create_table`.

### Modules

| Module               | Responsibility |
| -------------------- | -------------- |
| `dataset.py`         | Seeded synthetic corpus and query vectors (`float32`). |
| `parquet_backend.py` | `embedding_table`, Parquet write/read, `as_fixed_size_list()` to normalize Arrow list columns (including **chunked** columns after Parquet read). |
| `lance_backend.py`   | Lance write from the same `pa.Table`, read via `to_arrow()` and the same coalescing helper. |
| `bruteforce.py`      | Batched L2 kNN: builds an `(Q, N)` distance matrix with one matrix multiply and returns top-`k` indices per query. |
| `benchmark.py`       | Times **load** (path → dense `(N, d)` matrix) and **search** (kNN on that matrix); writes JSON rows (`BenchResult`). |
| `cli.py`             | Typer CLI: `prepare` and `run`. |

### Command-line workflow

1. **`lance-bench prepare`** — Generates a synthetic corpus and writes:
   - `data/corpus.parquet`
   - `data/lance/` with LanceDB table `corpus`
2. **`lance-bench run`** — Loads each corpus into memory, runs the shared kNN on the same query batch, prints and writes `results/bench.json` (path configurable with `--results-json`).

Example:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/lance-bench prepare --n-rows 50000 --dim 128
.venv/bin/lance-bench run --n-queries 256 --k 10
```

---

## What is being measured

For each backend:

- **`load_seconds`** — Time to read the full table and produce a contiguous `float32` matrix of shape `(N, d)`.
- **`search_seconds`** — Time to run `knn_l2(queries, corpus, k)` after load; **identical code** for both backends, so differences here are mostly from memory layout / allocator effects at small `N`, and should converge for large batches where compute dominates.
- **`total_seconds`** — Sum of the two (one full pass each; no shared cache between backends in a single `run`).

This is **not** yet a Lance **indexed** ANN benchmark; enabling IVF/PQ (or similar) would be a separate, explicitly labeled mode so Parquet is not compared unfairly to an approximate index.

---

## Fixes and edge cases handled

- **Chunked Arrow columns** — `pq.read_table` can return a `ChunkedArray` for `embedding`; `.values` is not available on chunked columns. `as_fixed_size_list()` concatenates chunks into one `FixedSizeListArray` before flattening to NumPy.

---

## Limitations (by design for v1)

- **Synthetic data only** — Replace `dataset.make_random_embeddings` with a real loader when moving to production-like evaluation.
- **Full materialization** — Entire corpus must fit in RAM for this path; larger runs need chunked scan + batched kNN or on-disk engines.
- **No cold-cache protocol** — OS page cache is not explicitly dropped between runs; repeat runs and document hardware if you need “cold” numbers.
- **No percentile latency** — Single timing sample per phase; extend with repeated trials for p50/p95 if needed.

---

## Relation to PLAN.md

This harness is the **smallest vertical slice** toward the broader plan: ingestion, dual storage, shared evaluation, JSON results. Later phases can add Ray load generation, S3 paths, ground truth / recall, and comparison baselines (FAISS, Zvec) as described in [PLAN.md](../PLAN.md).

To inspect produced files and schemas, see [Reading benchmark data](reading-data.md).

For multi-scenario scaling runs and the generated Markdown report, see [Sweep and report](sweep-and-report.md).

For on-disk size and RSS-style memory sampling, see [Disk and memory profiling](disk-and-memory.md).
