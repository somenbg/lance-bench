"""Parameter sweeps: prepare corpora once per (n_rows, dim, seed), run timed pairs."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import zlib
from pathlib import Path

import pyarrow.parquet as pq

from lance_bench.benchmark import BenchResult, run_pair, write_results_json
from lance_bench.dataset import make_queries, make_random_embeddings
from lance_bench.lance_backend import write_corpus_lance
from lance_bench.parquet_backend import write_corpus_parquet


@dataclass(frozen=True)
class Scenario:
    """One benchmark scenario (same corpus path reused when n_rows/dim/seed match)."""

    group: str
    scenario_id: str
    n_rows: int
    dim: int
    n_queries: int
    k: int
    corpus_seed: int = 42


def corpus_dir(base: Path, n_rows: int, dim: int, corpus_seed: int) -> Path:
    return base / f"n{n_rows}_d{dim}_seed{corpus_seed}"


def prepare_corpus(data_root: Path, n_rows: int, dim: int, corpus_seed: int) -> Path:
    out = corpus_dir(data_root, n_rows, dim, corpus_seed)
    out.mkdir(parents=True, exist_ok=True)
    ids, emb = make_random_embeddings(n_rows, dim, seed=corpus_seed)
    pq_path = out / "corpus.parquet"
    lance_dir = out / "lance"
    write_corpus_parquet(pq_path, ids, emb)
    write_corpus_lance(lance_dir, "corpus", ids, emb)
    return out


def build_scenarios(quick: bool) -> list[Scenario]:
    """Full grid (quick=False) or reduced grid for fast smoke runs."""
    out: list[Scenario] = []

    def add(group: str, sid: str, n: int, d: int, nq: int, k: int) -> None:
        out.append(Scenario(group=group, scenario_id=sid, n_rows=n, dim=d, n_queries=nq, k=k))

    if quick:
        for n in (5_000, 20_000):
            add("scale", f"scale_n{n}_d128_nq256_k10", n, 128, 256, 10)
        add("dimension", "dim_d128_n50k_nq256_k10", 50_000, 128, 256, 10)
        add("dimension", "dim_d384_n50k_nq256_k10", 50_000, 384, 256, 10)
        add("queries", "queries_n50k_nq64_k10", 50_000, 128, 64, 10)
        add("queries", "queries_n50k_nq1024_k10", 50_000, 128, 1024, 10)
        add("k", "k_n50k_nq256_k50", 50_000, 128, 256, 50)
        return out

    for n in (5_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000):
        add("scale", f"scale_n{n}_d128_nq256_k10", n, 128, 256, 10)
    for d in (128, 384, 768):
        add("dimension", f"dim_d{d}_n50k_nq256_k10", 50_000, d, 256, 10)
    for nq in (64, 256, 1024):
        add("queries", f"queries_n50k_nq{nq}_k10", 50_000, 128, nq, 10)
    for k in (10, 50):
        add("k", f"k_n50k_nq256_k{k}", 50_000, 128, 256, k)
    return out


def run_scenario(
    data_dir: Path,
    scenario: Scenario,
    query_seed: int,
    trial: int,
    *,
    profile: bool,
) -> tuple[BenchResult, BenchResult]:
    pq_path = data_dir / "corpus.parquet"
    lance_uri = data_dir / "lance"
    schema = pq.read_schema(pq_path)
    dim = schema.field("embedding").type.list_size
    queries = make_queries(scenario.n_queries, dim, seed=query_seed)
    r_pq, r_lance = run_pair(
        pq_path,
        lance_uri,
        "corpus",
        queries,
        scenario.k,
        profile=profile,
    )
    meta = dict(
        scenario_id=scenario.scenario_id,
        scenario_group=scenario.group,
        trial=trial,
    )
    return (
        replace(r_pq, **meta),
        replace(r_lance, **meta),
    )


def run_sweep(
    data_root: Path,
    scenarios: list[Scenario],
    trials: int,
    query_seed_base: int,
    *,
    profile: bool = True,
    prepared: set[tuple[int, int, int]] | None = None,
) -> list[BenchResult]:
    """If ``prepared`` is passed (e.g. from ``run_sweep_repeated``), corpora are not rebuilt when already done."""
    if prepared is None:
        prepared = set()
    rows: list[BenchResult] = []
    for sc in scenarios:
        key = (sc.n_rows, sc.dim, sc.corpus_seed)
        if key not in prepared:
            prepare_corpus(data_root, sc.n_rows, sc.dim, sc.corpus_seed)
            prepared.add(key)
        ddir = corpus_dir(data_root, sc.n_rows, sc.dim, sc.corpus_seed)
        for t in range(trials):
            seed = query_seed_base + t * 997 + (zlib.crc32(sc.scenario_id.encode()) % 10_000)
            rp, rl = run_scenario(ddir, sc, query_seed=seed, trial=t, profile=profile)
            rows.extend([rp, rl])
    return rows


def aggregate_mean_stdev_by_scenario_backend(rows: list[BenchResult]) -> list[BenchResult]:
    """
    Collapse raw rows into one BenchResult per (scenario_id, backend).
    Timings: mean and sample stdev across all input rows in each group.
    RSS: mean (rounded int) and sample stdev when multiple samples.
    """
    groups: dict[tuple[str, str], list[BenchResult]] = defaultdict(list)
    for r in rows:
        if not r.scenario_id:
            continue
        groups[(r.scenario_id, r.backend)].append(r)

    out: list[BenchResult] = []
    for sid, be in sorted(groups.keys()):
        xs = groups[(sid, be)]
        n = len(xs)
        x0 = xs[0]

        def mean_stdev(getter) -> tuple[float, float | None]:
            vals = [getter(x) for x in xs]
            m = float(statistics.mean(vals))
            if n < 2:
                return m, None
            return m, float(statistics.stdev(vals))

        lm, ls = mean_stdev(lambda x: x.load_seconds)
        sm, ss = mean_stdev(lambda x: x.search_seconds)
        tm, ts = mean_stdev(lambda x: x.total_seconds)

        rss_l = [x.rss_peak_load_bytes for x in xs if x.rss_peak_load_bytes is not None]
        rss_s = [x.rss_peak_search_bytes for x in xs if x.rss_peak_search_bytes is not None]
        rlm = int(round(statistics.mean(rss_l))) if rss_l else None
        rls = float(statistics.stdev(rss_l)) if len(rss_l) > 1 else None
        rsm = int(round(statistics.mean(rss_s))) if rss_s else None
        rss_s_stdev = float(statistics.stdev(rss_s)) if len(rss_s) > 1 else None

        out.append(
            replace(
                x0,
                backend=be,
                scenario_id=sid,
                load_seconds=lm,
                search_seconds=sm,
                total_seconds=tm,
                load_stdev=ls,
                search_stdev=ss,
                total_stdev=ts,
                trial=0,
                repeat_index=0,
                sample_count=n,
                storage_bytes=x0.storage_bytes,
                rss_peak_load_bytes=rlm,
                rss_peak_search_bytes=rsm,
                rss_peak_load_stdev=rls,
                rss_peak_search_stdev=rss_s_stdev,
            )
        )
    return out


def run_sweep_repeated(
    data_root: Path,
    scenarios: list[Scenario],
    trials: int,
    query_seed_base: int,
    *,
    profile: bool = True,
    repeats: int = 1,
) -> tuple[list[BenchResult], list[BenchResult]]:
    """
    Run run_sweep `repeats` times with different query RNG offsets.
    Returns (all_raw_rows_with_repeat_index, aggregated_rows_if_repeats_gt_1_else_raw).
    """
    raw: list[BenchResult] = []
    step = 100_003
    shared_prepared: set[tuple[int, int, int]] = set()
    for rep in range(repeats):
        base = query_seed_base + rep * step
        chunk = run_sweep(
            data_root,
            scenarios,
            trials,
            base,
            profile=profile,
            prepared=shared_prepared,
        )
        for r in chunk:
            raw.append(replace(r, repeat_index=rep))
    if repeats > 1:
        return raw, aggregate_mean_stdev_by_scenario_backend(raw)
    return raw, raw


def median_by_scenario(rows: list[BenchResult]) -> dict[str, dict[str, dict[str, float]]]:
    """scenario_id -> backend -> metric -> median."""
    by_sid: dict[str, list[BenchResult]] = {}
    for r in rows:
        by_sid.setdefault(r.scenario_id, []).append(r)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for sid, rs in by_sid.items():
        by_be: dict[str, list[BenchResult]] = {}
        for r in rs:
            by_be.setdefault(r.backend, []).append(r)
        out[sid] = {}
        for be, xs in by_be.items():
            d: dict[str, float] = {
                "load_seconds": float(statistics.median(x.load_seconds for x in xs)),
                "search_seconds": float(statistics.median(x.search_seconds for x in xs)),
                "total_seconds": float(statistics.median(x.total_seconds for x in xs)),
            }
            stor = [x.storage_bytes for x in xs if x.storage_bytes is not None]
            if stor:
                d["storage_bytes"] = float(stor[0])
            rpl = [x.rss_peak_load_bytes for x in xs if x.rss_peak_load_bytes is not None]
            if rpl:
                d["rss_peak_load_bytes"] = float(statistics.median(rpl))
            rps = [x.rss_peak_search_bytes for x in xs if x.rss_peak_search_bytes is not None]
            if rps:
                d["rss_peak_search_bytes"] = float(statistics.median(rps))
            for key_src, key_dst in (
                ("load_stdev", "load_stdev"),
                ("search_stdev", "search_stdev"),
                ("total_stdev", "total_stdev"),
                ("rss_peak_load_stdev", "rss_peak_load_stdev"),
                ("rss_peak_search_stdev", "rss_peak_search_stdev"),
            ):
                vals = [getattr(x, key_src) for x in xs if getattr(x, key_src) is not None]
                if vals:
                    d[key_dst] = float(statistics.median(vals))
            out[sid][be] = d
    return out


def write_sweep_json(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in rows], indent=2),
        encoding="utf-8",
    )


def load_sweep_json(path: Path) -> list[BenchResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[BenchResult] = []
    for r in raw:
        r.setdefault("scenario_id", "")
        r.setdefault("scenario_group", "")
        r.setdefault("trial", 0)
        r.setdefault("storage_bytes", None)
        r.setdefault("rss_peak_load_bytes", None)
        r.setdefault("rss_peak_search_bytes", None)
        r.setdefault("repeat_index", 0)
        r.setdefault("sample_count", 1)
        r.setdefault("load_stdev", None)
        r.setdefault("search_stdev", None)
        r.setdefault("total_stdev", None)
        r.setdefault("rss_peak_load_stdev", None)
        r.setdefault("rss_peak_search_stdev", None)
        out.append(BenchResult(**r))
    return out
