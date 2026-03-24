# Reading benchmark data

This guide describes **what files the first benchmark produces**, **how they relate**, and **how to inspect them** (schema, samples, JSON results).

Paths below use the repository root; adjust if you run from elsewhere.

---

## 0. Sweep outputs (`lance-bench sweep`)

After a sweep you also get:

| File | Role |
| ---- | ---- |
| `results/sweep.json` | All trials × backends × scenarios (`scenario_id`, `scenario_group`, `trial`, timings) |
| `results/sweep_meta.json` | `{ "trials", "quick" }` for `lance-bench report` |
| `results/benchmark-report.md` | Markdown tables (median timings) |

Corpora live under `data/sweep/n{N}_d{d}_seed{seed}/`. Details: [Sweep and report](sweep-and-report.md).

---

## 1. `results/bench.json`

**What it is:** Plain JSON — one object per backend (`parquet`, `lance`) with timing fields and run parameters.

**Fields (typical):**

| Field | Meaning |
| ----- | ------- |
| `backend` | `"parquet"` or `"lance"` |
| `n_rows` | Corpus size after load |
| `dim` | Embedding dimension |
| `n_queries` | Number of query vectors |
| `k` | k for kNN |
| `load_seconds` | Time to read storage into a dense `(N, d)` matrix |
| `search_seconds` | Time for shared brute-force kNN after load |
| `total_seconds` | `load_seconds` + `search_seconds` |
| `parquet_path` / `lance_uri` | Which artifact was used (the other is `null`) |
| `storage_bytes` | On-disk footprint for that backend’s artifact (Parquet file or Lance directory tree) |
| `rss_peak_load_bytes` / `rss_peak_search_bytes` | Peak RSS during load / search when profiling is on (`null` with `--no-profile`) |

**Read from the shell:**

```bash
cat results/bench.json
python3 -m json.tool results/bench.json
```

**Read in Python:**

```python
import json
from pathlib import Path

rows = json.loads(Path("results/bench.json").read_text(encoding="utf-8"))
for r in rows:
    print(r["backend"], r["total_seconds"])
```

---

## 2. `data/corpus.parquet`

**What it is:** The **corpus** written by `lance-bench prepare` — the same logical rows the benchmark loads for the Parquet path.

**Schema:**

| Column | Type (Arrow) | Meaning |
| ------ | ------------ | ------- |
| `id` | `int64` | Row identifier (for synthetic data: `0 … N-1`) |
| `embedding` | `fixed_size_list<float>[d]` | One vector per row, length `d` (e.g. 128) |

**Inspect with PyArrow** (project dependency):

```python
import pyarrow.parquet as pq

path = "data/corpus.parquet"
table = pq.read_table(path)
print(table.schema)
print("rows:", table.num_rows)

# First rows: id + embedding as Python lists
for i in range(min(2, table.num_rows)):
    row_id = table["id"][i].as_py()
    vec = table["embedding"][i].as_py()
    print(row_id, len(vec), vec[:3])
```

**Metadata only (no full column read):**

```python
import pyarrow.parquet as pq

schema = pq.read_schema("data/corpus.parquet")
dim = schema.field("embedding").type.list_size
print(dim)
```

---

## 3. `data/lance/` (Lance format, pylance)

**What it is:** A Lance columnar dataset directory written by [pylance](https://github.com/lance-format/lance) (`lance.dataset.write_dataset`). Same logical content as `corpus.parquet` when both are produced by the same `prepare` run. The CLI still passes a legacy table name (`corpus`) to shared helpers; native Lance stores one dataset per directory, not a named table inside a database.

**Inspect with pylance:**

```python
from lance import LanceDataset

ds = LanceDataset("data/lance")
arrow = ds.to_table(columns=["id", "embedding"])
print(arrow.schema)
print("rows:", arrow.num_rows)

for i in range(min(2, arrow.num_rows)):
    row_id = arrow["id"][i].as_py()
    vec = arrow["embedding"][i].as_py()
    print(row_id, len(vec), vec[:3])
```

**Note:** Fragments under `data/lance/` (e.g. `*.lance`) are Lance’s internal layout. Prefer **pylance** (`LanceDataset`, scanners) for reading rather than editing files manually.

---

## 4. Queries (in-memory only)

For the current CLI, **query vectors are not written to disk**. `lance-bench run` generates them in memory (seeded random, same `dim` as the corpus) to match the synthetic benchmark.

Conceptually they form a matrix of shape `(n_queries, dim)` with `float32` values — same family as the corpus for synthetic runs.

To persist queries later, extend `prepare` or `run` to write e.g. `data/queries.parquet` and document the schema alongside the corpus.

---

## Sanity check: Parquet vs Lance match

After `prepare`, the first rows of `corpus.parquet` and Lance table `corpus` should show **identical** `id` and `embedding` values. If they diverge, the two backends are not pointing at the same logical dataset.

---

## See also

- [First benchmark: Parquet vs Lance](first-benchmark.md) — what the harness measures and how it is structured
- [PLAN.md](../PLAN.md) — broader benchmarking plan
