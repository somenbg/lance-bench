# Parquet vs Lance — sweep benchmark

> **Snapshot:** Generated from `lance-bench sweep` and copied here for version control. Raw rows live in `results/sweep.json` (gitignored by default). Re-run: `lance-bench sweep` then `lance-bench report`, or see [Sweep and report](../sweep-and-report.md).

- **Generated:** 2026-03-23 23:57 UTC
- **Platform:** `Darwin 24.6.0` (arm64)
- **Python:** `3.10.15`
- **Trials per scenario:** 2
- **Sweep mode:** full

## What was compared

Each scenario loads the full corpus from **Parquet** vs **Lance** into a dense `float32` matrix, then runs the **same** batched L2 brute-force kNN. Metrics are wall time: **load**, **search**, **total** (median over trials).

**Interpretation:** Differences in **search** time should be small (same NumPy kernel). Differences in **load** reflect PyArrow Parquet decode vs Lance scan/materialization in this harness (local disk, warm-ish cache after first touch in a trial sequence).

## Results (median over trials)

### Scale

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scale_n5000_d128_nq256_k10` | 5000 | 128 | 256 | 10 | 0.0055 | 0.0025 | 0.0085 | 0.0084 | 0.0140 | 0.0110 | 0.46× | 0.78× |
| `scale_n20000_d128_nq256_k10` | 20000 | 128 | 256 | 10 | 0.0215 | 0.0057 | 0.0484 | 0.0470 | 0.0699 | 0.0526 | 0.26× | 0.75× |
| `scale_n50000_d128_nq256_k10` | 50000 | 128 | 256 | 10 | 0.0559 | 0.0120 | 0.1089 | 0.1041 | 0.1648 | 0.1161 | 0.21× | 0.70× |
| `scale_n100000_d128_nq256_k10` | 100000 | 128 | 256 | 10 | 0.1151 | 0.0222 | 0.2351 | 0.2438 | 0.3501 | 0.2660 | 0.19× | 0.76× |

### Dimension

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dim_d128_n50k_nq256_k10` | 50000 | 128 | 256 | 10 | 0.0557 | 0.0102 | 0.1186 | 0.1074 | 0.1743 | 0.1176 | 0.18× | 0.67× |
| `dim_d384_n50k_nq256_k10` | 50000 | 384 | 256 | 10 | 0.1848 | 0.0343 | 0.1413 | 0.1479 | 0.3261 | 0.1823 | 0.19× | 0.56× |
| `dim_d768_n50k_nq256_k10` | 50000 | 768 | 256 | 10 | 0.4272 | 0.0821 | 0.1686 | 0.1844 | 0.5957 | 0.2664 | 0.19× | 0.45× |

### Queries

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `queries_n50k_nq64_k10` | 50000 | 128 | 64 | 10 | 0.0580 | 0.0164 | 0.0281 | 0.0285 | 0.0861 | 0.0448 | 0.28× | 0.52× |
| `queries_n50k_nq256_k10` | 50000 | 128 | 256 | 10 | 0.0483 | 0.0098 | 0.1022 | 0.0993 | 0.1505 | 0.1091 | 0.20× | 0.72× |
| `queries_n50k_nq1024_k10` | 50000 | 128 | 1024 | 10 | 0.0486 | 0.0109 | 0.5180 | 0.4935 | 0.5666 | 0.5044 | 0.22× | 0.89× |

### K

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k_n50k_nq256_k10` | 50000 | 128 | 256 | 10 | 0.0501 | 0.0099 | 0.1015 | 0.1000 | 0.1516 | 0.1099 | 0.20× | 0.72× |
| `k_n50k_nq256_k50` | 50000 | 128 | 256 | 50 | 0.0469 | 0.0097 | 0.1000 | 0.0987 | 0.1469 | 0.1084 | 0.21× | 0.74× |

## Observations (this run)

- **Load (Lance / Parquet)** across **scale** scenarios was **0.19×–0.46×** (always below 1 → Lance **faster** to materialize the matrix in this run, after local caching effects).
- **Search** times should track query count and N×d multiply cost; small Lance vs Parquet spread is expected after both corpora are in RAM.
- This harness measures **full materialization** + in-memory kNN, not Lance IVF/PQ ANN or streaming Parquet scans.

## Raw data

See `results/sweep.json` (all trials) alongside this report.
