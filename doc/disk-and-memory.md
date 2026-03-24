# Disk and memory profiling

The harness can record **on-disk footprint** and **approximate RAM (RSS) peaks** alongside wall-clock time so you can compare Parquet vs Lance under the same logical corpus.

---

## What is measured

### On-disk (`storage_bytes` in JSON)

| Backend | What we count |
| ------- | ------------- |
| **Parquet** | Size of `corpus.parquet` (single file; Zstd-compressed columnar data). |
| **Lance** | Sum of **all file sizes** under the Lance dataset directory (e.g. `*.lance` fragments, manifests, sidecars). |

This answers: **“How much disk does each format use for this table?”** It does not include temp files elsewhere on the volume.

### Memory (`rss_peak_load_bytes`, `rss_peak_search_bytes`)

While **load** (read → dense `N × d` matrix) or **search** (batched brute-force kNN) runs, a **background thread** samples the process **RSS** about every **5 ms** via **psutil** and keeps the **maximum** seen during that phase.

- **RSS** is what the OS attributes to your process (heap, some mapped pages, etc.). It is **not** the same as “bytes allocated by `malloc`” and it can include noise from other activity in the process.
- Peaks are **per phase** (load vs search), not a single number for the whole run.
- Medians across **trials** appear in the Markdown report when profiling is enabled.

Use this to answer: **“Roughly how much RAM pressure does each phase add?”** — good for spotting blow-ups or comparing backends directionally, not for sub-percent accounting.

---

## How to enable

```bash
# Single run (default: profile on)
.venv/bin/lance-bench run
.venv/bin/lance-bench run --no-profile   # timings only; skips RSS sampling thread

# Sweep
.venv/bin/lance-bench sweep
.venv/bin/lance-bench sweep --no-profile
```

**Disk sizes** are still written to JSON for each backend whenever `run_pair` runs (cheap `stat` / tree walk). **RSS** fields are `null` when `--no-profile` is used.

After `prepare`, print sizes without running a benchmark:

```bash
.venv/bin/lance-bench prepare --show-sizes
```

---

## Where results appear

- **`results/bench.json`** / **`results/sweep.json`** — `storage_bytes`, `rss_peak_load_bytes`, `rss_peak_search_bytes` per row (when applicable).
- **`results/benchmark-report.md`** — Extra columns: disk (MiB), disk ratio L/P, and RSS peaks (MiB) when data is present.
- **`lance-bench plot`** — PNGs: `scale_disk_mib.png`, `dimension_disk_mib.png`, `scale_rss_mib.png`, `dimension_rss_mib.png`, `resources_dashboard.png` (see [Benchmark visualizations](visualizations.md)).

---

## Limitations (read before deciding)

- **RSS** depends on **allocator behavior**, **memory mapping**, **Python GC**, and **OS caching**; repeat runs and stable environments matter.
- **Subprocess isolation** would give cleaner per-backend RSS, but this harness runs both backends **in one process** sequentially, so the **second** backend’s RSS may reflect retained memory from the first unless the process is fresh.
- For **strict** memory work, add **tracemalloc**, **jemalloc profiling**, or **separate processes** per backend.

---

## See also

- [First benchmark: Parquet vs Lance](first-benchmark.md)
- [Sweep and report](sweep-and-report.md)
