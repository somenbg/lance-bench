# lance-bench

Benchmark harness that compares **Apache Parquet** vs **Lance** using the **[pylance](https://github.com/lance-format/lance)** Python SDK for the same synthetic or prepared corpora: time to **load** embeddings into memory, run identical **L2 brute-force kNN**, and optionally record **on-disk size** and **RSS peaks**. Supports **multi-scenario sweeps** (scale, dimension, query batch, **k**), **Markdown reports**, **matplotlib plots**, and **repeated sweeps** with mean ± sample stdev.

Longer product/research context: [PLAN.md](PLAN.md).

---

## Requirements

- **Python 3.10+**
- Dependencies are declared in [pyproject.toml](pyproject.toml) (`pylance`, `pyarrow`, `numpy`, `typer`, `matplotlib`, `psutil`).

---

## Install

```bash
cd lance_bench   # this repository
python3 -m venv .venv
.venv/bin/pip install -e .
```

The CLI entry point is **`lance-bench`**.

---

## How to run

### One corpus + single comparison

```bash
# Write the same data as Parquet + Lance under ./data/
.venv/bin/lance-bench prepare --n-rows 50000 --dim 128

# Time load + shared kNN (default: disk sizes + RSS profiling)
.venv/bin/lance-bench run --n-queries 256 --k 10

# Timings only (no RSS sampling thread)
.venv/bin/lance-bench run --no-profile
```

Outputs: `data/corpus.parquet`, `data/lance/`, and `results/bench.json`.

### Full scenario sweep (recommended for comparisons)

```bash
# Full grid (15 scenarios; scale up to N=1M), 2 inner trials, report + default profiling
.venv/bin/lance-bench sweep

# Same, plus PNG plots
.venv/bin/lance-bench sweep --plot-to results/plots

# Smaller grid (smoke test)
.venv/bin/lance-bench sweep --quick
```

Outputs: `results/sweep.json`, `results/sweep_meta.json`, `results/benchmark-report.md`, and optional `results/sweep_raw.json` / `results/plots/*.png`.

**Repeats** (outer loop, mean ± stdev per cell):

```bash
.venv/bin/lance-bench sweep --quick --repeats 100 --trials 1 --no-profile --no-keep-raw
```

Corpora are created under **`data/sweep/`** (can grow large); **`results/`** is gitignored by default.

### Regenerate report or plots from existing JSON

```bash
.venv/bin/lance-bench report --from results/sweep.json
.venv/bin/lance-bench plot --from results/sweep.json --out-dir results/plots
```

### CLI help

```bash
.venv/bin/lance-bench --help
.venv/bin/lance-bench sweep --help
```

---

## Documentation

| Doc | Topic |
| --- | ----- |
| [doc/first-benchmark.md](doc/first-benchmark.md) | Harness layout and metrics |
| [doc/sweep-and-report.md](doc/sweep-and-report.md) | Sweeps, repeats, reports |
| [doc/visualizations.md](doc/visualizations.md) | Plot outputs (`lance-bench plot`) |
| [doc/disk-and-memory.md](doc/disk-and-memory.md) | Disk bytes, RSS profiling |
| [doc/reading-data.md](doc/reading-data.md) | JSON schemas, inspecting Parquet/Lance |
| [doc/reports/benchmark-report.md](doc/reports/benchmark-report.md) | Checked-in snapshot of a full sweep |

The `doc/` folder holds these Markdown sources; this README is the entry point.

---

## Layout

- `src/lance_bench/` — package (CLI, backends, sweep, plots, profiling)
- `doc/` — documentation
- `PLAN.md` — broader benchmarking plan
