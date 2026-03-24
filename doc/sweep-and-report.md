# Sweep scenarios and reports

The **`lance-bench sweep`** command runs a **grid of scenarios** to compare Parquet vs Lance across **scale**, **embedding dimension**, **query batch size**, and **k**, then writes machine-readable results and a Markdown report.

---

## Commands

```bash
# Editable install (once)
python3 -m venv .venv
.venv/bin/pip install -e .

# Full sweep (default: 15 scenarios, 2 trials each → median in report; scale includes N up to 1M)
.venv/bin/lance-bench sweep

# Smaller grid for CI or smoke tests
.venv/bin/lance-bench sweep --quick --trials 1

# Regenerate only the Markdown report from existing JSON
.venv/bin/lance-bench report --from results/sweep.json --meta results/sweep_meta.json
```

### Outputs

| Path | Contents |
| ---- | -------- |
| `data/sweep/n{N}_d{d}_seed{seed}/` | One directory per unique corpus (`corpus.parquet` + `lance/`) |
| `results/sweep.json` | **`--repeats 1`:** one row per backend per inner trial. **`--repeats N>1`:** one row per scenario × backend with **mean** timings and **sample stdev** (`sample_count` = N×trials). |
| `results/sweep_raw.json` | Optional: all rows from every repeat (default **on** when `repeats>1`; use **`--no-keep-raw`** to skip). |
| `results/sweep_meta.json` | Includes `trials`, `quick`, `profile`, **`repeats`**, **`samples_per_cell`**, `aggregated`, `keep_raw`. |
| `results/benchmark-report.md` | Tables; with `repeats>1`, time columns use **mean ± stdev** when present. |

**Repeats:** `--repeats 100 --trials 1` runs the full scenario list 100 times and **averages** (mean ± sample stdev) per cell. Example: `lance-bench sweep --quick --repeats 100 --trials 1 --no-profile --no-keep-raw`. Full grid × 100 repeats can take a long time.

By default **`results/`** is gitignored; a **checked-in snapshot** of the last full sweep lives in [reports/benchmark-report.md](reports/benchmark-report.md). Copy fresh output there after major runs if you want history in git.

**Charts:** `lance-bench plot --from results/sweep.json --out-dir results/plots`, or `lance-bench sweep --plot-to results/plots`. See [Benchmark visualizations](visualizations.md).

**Disk / RSS:** `lance-bench sweep` records **on-disk** sizes per backend; with **`--profile`** (default), it also samples **RSS peaks** during load and search. See [Disk and memory profiling](disk-and-memory.md).

---

## Scenario groups

Defined in `lance_bench/sweep.py` (`build_scenarios`).

| Group | Varies | Fixed (full mode) |
| ----- | ------ | ----------------- |
| **scale** | N ∈ {5k, 20k, 50k, 100k, 200k, 500k, **1M**} | d=128, Q=256, k=10 |
| **dimension** | d ∈ {128, 384, 768} | N=50k, Q=256, k=10 |
| **queries** | Q ∈ {64, 256, 1024} | N=50k, d=128, k=10 |
| **k** | k ∈ {10, 50} | N=50k, d=128, Q=256 |

**Quick mode** uses a subset (7 scenarios) with smaller N and fewer dimensions.

**Large N:** At **N=1M** and **d=128**, the dense corpus matrix alone is about **512 MiB** `float32` per load, plus queries and temporaries — ensure **RAM** and **`data/sweep/`** disk space before running a full sweep.

Corpora are **deterministic** for a given `(N, d, corpus_seed)` (default seed `42`). Query vectors differ per **trial** using a stable CRC-based offset from `scenario_id` so repeats are comparable without identical queries across scenarios.

---

## How to read the report

- **Pq load / L load** — Time to read storage into a dense `float32` matrix.
- **Pq search / L search** — Shared batched L2 brute-force kNN on that matrix.
- **Load L/P** — Lance load ÷ Parquet load (**below 1** means Lance faster on load for that row).
- **Total L/P** — Ratio of end-to-end median times.

Numbers depend on **hardware**, **OS page cache**, and **library versions**; treat cross-machine comparisons as directional unless you pin the environment.

---

## See also

- [First benchmark: Parquet vs Lance](first-benchmark.md)
- [Reading benchmark data](reading-data.md)
- [PLAN.md](../PLAN.md)
