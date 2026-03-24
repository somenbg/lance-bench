# Parquet vs Lance — sweep benchmark

- **Generated:** 2026-03-24 01:49 UTC
- **Platform:** `Darwin 24.6.0` (arm64)
- **Python:** `3.10.15`
- **Trials per scenario (inner):** 1
- **Sweep repeats (outer):** 1
- **Per-cell timing:** **median** over **1** inner trial(s) in a single sweep pass.
- **Sweep mode:** full

## What was compared

Each scenario loads the full corpus from **Parquet** vs **Lance** into a dense `float32` matrix, then runs the **same** batched L2 brute-force kNN. Wall-time columns are **load**, **search**, **total** (see header above for median vs mean±stdev).

**Interpretation:** Differences in **search** time should be small (same NumPy kernel). Differences in **load** reflect PyArrow Parquet decode vs Lance scan/materialization in this harness (local disk, warm-ish cache after first touch in a trial sequence).

**On-disk size:** Parquet = `corpus.parquet` file size; Lance = sum of file sizes under the Lance dataset directory (fragments + sidecars). Comparable logical row counts, different physical layout and compression.

## Results

### Scale

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P | Pq disk (MiB) | L disk (MiB) | Disk L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scale_n5000_d128_nq256_k10` | 5000 | 128 | 256 | 10 | 0.0240 | 0.0595 | 0.0187 | 0.0111 | 0.0427 | 0.0705 | 2.48× | 1.65× | 2.9 | 9.9 | 3.47× |
| `scale_n20000_d128_nq256_k10` | 20000 | 128 | 256 | 10 | 0.0241 | 0.0060 | 0.0558 | 0.0503 | 0.0799 | 0.0564 | 0.25× | 0.71× | 9.7 | 39.7 | 4.10× |
| `scale_n50000_d128_nq256_k10` | 50000 | 128 | 256 | 10 | 0.0569 | 0.0182 | 0.1199 | 0.1095 | 0.1768 | 0.1277 | 0.32× | 0.72× | 23.3 | 99.2 | 4.26× |
| `scale_n100000_d128_nq256_k10` | 100000 | 128 | 256 | 10 | 0.1229 | 0.0261 | 0.2522 | 0.2212 | 0.3751 | 0.2473 | 0.21× | 0.66× | 46.0 | 148.8 | 3.23× |
| `scale_n200000_d128_nq256_k10` | 200000 | 128 | 256 | 10 | 0.2968 | 0.1000 | 0.6024 | 0.5746 | 0.8993 | 0.6746 | 0.34× | 0.75× | 91.3 | 297.6 | 3.26× |
| `scale_n500000_d128_nq256_k10` | 500000 | 128 | 256 | 10 | 0.9705 | 0.2008 | 1.7097 | 2.0810 | 2.6802 | 2.2818 | 0.21× | 0.85× | 227.0 | 743.9 | 3.28× |
| `scale_n1000000_d128_nq256_k10` | 1000000 | 128 | 256 | 10 | 2.1559 | 0.5003 | 4.3154 | 3.8542 | 6.4713 | 4.3545 | 0.23× | 0.67× | 453.1 | 1487.8 | 3.28× |

### Dimension

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P | Pq disk (MiB) | L disk (MiB) | Disk L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dim_d128_n50k_nq256_k10` | 50000 | 128 | 256 | 10 | 0.1075 | 0.0242 | 0.1110 | 0.1045 | 0.2185 | 0.1286 | 0.22× | 0.59× | 23.3 | 99.2 | 4.26× |
| `dim_d384_n50k_nq256_k10` | 50000 | 384 | 256 | 10 | 0.1722 | 0.0254 | 0.1423 | 0.1428 | 0.3145 | 0.1683 | 0.15× | 0.54× | 68.4 | 147.3 | 2.15× |
| `dim_d768_n50k_nq256_k10` | 50000 | 768 | 256 | 10 | 0.3762 | 0.0501 | 0.1748 | 0.1720 | 0.5510 | 0.2221 | 0.13× | 0.40× | 136.1 | 293.7 | 2.16× |

### Queries

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P | Pq disk (MiB) | L disk (MiB) | Disk L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `queries_n50k_nq64_k10` | 50000 | 128 | 64 | 10 | 0.0512 | 0.0110 | 0.0315 | 0.0292 | 0.0827 | 0.0402 | 0.22× | 0.49× | 23.3 | 99.2 | 4.26× |
| `queries_n50k_nq256_k10` | 50000 | 128 | 256 | 10 | 0.0537 | 0.0101 | 0.1470 | 0.1102 | 0.2006 | 0.1203 | 0.19× | 0.60× | 23.3 | 99.2 | 4.26× |
| `queries_n50k_nq1024_k10` | 50000 | 128 | 1024 | 10 | 0.0546 | 0.0109 | 0.5656 | 0.4494 | 0.6202 | 0.4603 | 0.20× | 0.74× | 23.3 | 99.2 | 4.26× |

### K

| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | Load L/P | Total L/P | Pq disk (MiB) | L disk (MiB) | Disk L/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k_n50k_nq256_k10` | 50000 | 128 | 256 | 10 | 0.0494 | 0.0113 | 0.1064 | 0.1070 | 0.1558 | 0.1183 | 0.23× | 0.76× | 23.3 | 99.2 | 4.26× |
| `k_n50k_nq256_k50` | 50000 | 128 | 256 | 50 | 0.0500 | 0.0098 | 0.1018 | 0.0982 | 0.1518 | 0.1081 | 0.20× | 0.71× | 23.3 | 99.2 | 4.26× |

## Observations (this run)

- **Load (Lance / Parquet)** across **scale** scenarios spanned **0.21×–2.48×** (mixed; see table).
- **Search** times should track query count and N×d multiply cost; small Lance vs Parquet spread is expected after both corpora are in RAM.
- This harness measures **full materialization** + in-memory kNN, not Lance IVF/PQ ANN or streaming Parquet scans.

## Raw data

See `results/sweep.json` alongside this report.
