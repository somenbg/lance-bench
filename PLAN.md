# LanceDB Benchmarking & Utility Framework

## 0. Objectives

### Primary Goals

* Benchmark **LanceDB** across:

  * Recall (quality)
  * Latency (P50/P95/P99)
  * Throughput (QPS under concurrency)
  * Cost (compute + storage)
* Build reusable utilities for:

  * Dataset ingestion
  * Index tuning
  * Evaluation pipelines
  * Reporting

### Secondary Goals

* Compare against:

  * **FAISS**
  * **Zvec**
* Validate fit for:

  * embedding evaluation
  * retrieval workloads
  * hybrid filtering

---

## 1. High-Level Architecture

```
                +----------------------+
                |  Public S3 Datasets  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Ingestion Layer     |
                | (Ray Jobs)           |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  LanceDB Storage     |
                | (Local / S3-backed)  |
                +----------+-----------+
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
+---------------------+           +----------------------+
| Benchmark Engine    |           | Utility Layer        |
| (Recall/Latency)    |           | (Wrappers/APIs)      |
+----------+----------+           +----------+-----------+
           |                                 |
           v                                 v
   +----------------+              +----------------------+
   | Results Store  |              | Reports / Dashboards |
   | (Parquet/S3)   |              | (Markdown/Plots)     |
   +----------------+              +----------------------+
```

---

## 2. Dataset Strategy (Public S3)

### Requirements

* Large-scale (≥ 1M vectors)
* Realistic embedding distributions
* Metadata for filtering

### Candidate Sources

* AWS Open Data Registry (public S3 buckets)
* Common options:

  * LAION embeddings (image/text)
  * OpenSearch benchmark datasets
  * Synthetic embeddings (fallback)

### Utility: Dataset Loader

```python
def load_s3_dataset(s3_path: str) -> pd.DataFrame:
    # supports parquet / csv / json
    pass
```

---

## 3. Data Preparation

### Standard Schema

```
id: int
embedding: float32[d]
metadata:
    - category
    - timestamp
    - numeric features
```

### Transformations

* Normalize embeddings (if cosine similarity)
* Ensure float32 (avoid float64 overhead)
* Optional:

  * PCA (dim reduction experiments)

---

## 4. LanceDB Setup

### Table Creation

```python
db = lancedb.connect("s3://bucket/lancedb")
table = db.create_table("benchmark", data=df)
```

---

### Index Configurations (Experiment Matrix)

| Config ID | num_partitions | num_sub_vectors | Notes       |
| --------- | -------------- | --------------- | ----------- |
| A         | 64             | 32              | Fast build  |
| B         | 256            | 64              | Balanced    |
| C         | 512            | 96              | High recall |

---

### Utility: Index Builder

```python
def build_index(table, config):
    table.create_index(
        num_partitions=config["num_partitions"],
        num_sub_vectors=config["num_sub_vectors"]
    )
```

---

## 5. Ground Truth (Critical)

### Exact Search Baseline

```python
def brute_force_search(query, embeddings, k):
    # numpy / torch implementation
    pass
```

Store:

* top-k neighbors
* distances

---

## 6. Benchmark Engine

### Metrics

#### 1. Recall@K

```
recall = |ANN ∩ Exact| / K
```

#### 2. Latency

* P50 / P95 / P99
* Cold vs warm

#### 3. Throughput

* Queries/sec under concurrency

#### 4. Index Build Time

#### 5. Storage Size

---

### Utility: Benchmark Runner

```python
def run_benchmark(table, queries, k=10):
    # returns latency + recall metrics
    pass
```

---

## 7. Distributed Benchmarking (Ray)

Use **Ray** to simulate load.

### Pattern

```python
@ray.remote
def query_task(q):
    return table.search(q).limit(10).to_pandas()
```

### Experiments

* concurrency: [1, 10, 100, 500]
* dataset sizes: [1M, 10M, 100M]

---

## 8. Benchmark Scenarios

### A. Local Disk (Baseline)

* isolate LanceDB performance

---

### B. S3-backed

* measure:

  * cold read latency
  * caching effects

---

### C. Hybrid Filtering

```python
table.search(q).where("category == 'x'")
```

Measure:

* latency impact
* recall impact

---

### D. Incremental Updates

* append new embeddings
* measure:

  * reindex cost
  * query degradation

---

## 9. Comparison Layer

### FAISS Baseline

* in-memory ANN
* best-case latency

---

### Zvec Baseline

* embedded engine
* low-latency comparison

---

## 10. Results Storage

Store all results in:

* Parquet (S3)

Schema:

```
experiment_id
dataset_size
index_config
recall@k
p50_latency
p95_latency
throughput
cost_estimate
```

---

## 11. Reporting

### Outputs

* Markdown reports
* Plots:

  * Recall vs Latency
  * Throughput vs Concurrency

---

### Utility: Report Generator

```python
def generate_report(results_df):
    # produce markdown + charts
    pass
```

---

## 12. Utility Layer (Reusable)

Build a small internal SDK:

```
lancedb_utils/
    ingestion.py
    indexing.py
    query.py
    benchmark.py
    reporting.py
```

---

## 13. Integration with Your ML System

### Use Cases

#### A. Embedding Evaluation

* store embeddings per model version
* compare recall across versions

#### B. Drift Detection

* distribution shift analysis

#### C. Retrieval Validation

* “does embedding retrieve similar trips?”

---

## 14. Experiment Automation

### Pipeline

* Trigger via:

  * CLI
  * Airflow / Argo
  * Ray Jobs

### Config-driven runs

```yaml
dataset: s3://...
index_configs:
  - partitions: 256
    subvectors: 64
queries: 10000
concurrency: [1, 10, 100]
```

---

## 15. Deliverables

### Phase 1 (Week 1)

* Dataset ingestion
* LanceDB setup
* Basic benchmark (latency only)

### Phase 2 (Week 2)

* Recall measurement
* Index tuning experiments

### Phase 3 (Week 3)

* Ray distributed benchmarking
* S3 experiments

### Phase 4 (Week 4)

* Comparison vs FAISS/Zvec
* Reporting dashboard

---

## 16. Success Criteria

* ≥ 90% recall@10 for target workload
* Stable P95 latency under concurrency
* Predictable scaling behavior
* Clear cost vs performance tradeoffs

---

## Final Insight

You are not just benchmarking a database.

You are building:

> a **repeatable evaluation system for embedding quality + retrieval performance**

This becomes a **core asset** for:

* model validation
* infra decisions
* cost optimization

---
