"""Generate a Markdown report from sweep JSON."""

from __future__ import annotations

import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone

from lance_bench.benchmark import BenchResult
from lance_bench.sweep import median_by_scenario


def _scenario_meta(rows: list[BenchResult]) -> dict[str, tuple[str, int, int, int, int]]:
    """scenario_id -> (group, n_rows, dim, n_queries, k) from first row."""
    m: dict[str, tuple[str, int, int, int, int]] = {}
    for r in rows:
        if r.scenario_id and r.scenario_id not in m:
            m[r.scenario_id] = (r.scenario_group, r.n_rows, r.dim, r.n_queries, r.k)
    return m


def _ratio(lance: float, parquet: float) -> float:
    if parquet == 0:
        return float("inf")
    return lance / parquet


def _mib(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x / (1024**2):.1f}"


def _fmt_sec(mean: float, stdev: float | None) -> str:
    if stdev is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {stdev:.4f}"


def _mib_pm(mean_b: float | None, std_b: float | None) -> str:
    if mean_b is None:
        return "—"
    if std_b is None:
        return _mib(mean_b)
    return f"{_mib(mean_b)} ± {std_b / (1024**2):.1f}"


def generate_report_md(
    rows: list[BenchResult],
    *,
    title: str,
    trials: int,
    quick: bool,
    repeats: int = 1,
) -> str:
    med = median_by_scenario(rows)
    meta = _scenario_meta(rows)
    groups_order = ["scale", "dimension", "queries", "k"]
    by_group: dict[str, list[str]] = defaultdict(list)
    for sid, (g, _, _, _, _) in meta.items():
        by_group[g].append(sid)
    def _sort_key(group: str, sid: str) -> tuple:
        _, n, d, nq, k = meta[sid]
        if group == "scale":
            return (n, sid)
        if group == "dimension":
            return (d, sid)
        if group == "queries":
            return (nq, sid)
        if group == "k":
            return (k, sid)
        return (sid,)

    for g in by_group:
        by_group[g].sort(key=lambda sid: _sort_key(g, sid))

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Platform:** `{platform.system()} {platform.release()}` ({platform.machine()})")
    lines.append(f"- **Python:** `{sys.version.split()[0]}`")
    lines.append(f"- **Trials per scenario (inner):** {trials}")
    lines.append(f"- **Sweep repeats (outer):** {repeats}")
    if repeats > 1:
        lines.append(
            f"- **Samples per (scenario, backend) cell:** {repeats * trials} — timings in the table are "
            "**mean ± sample stdev** over those samples when ± is shown."
        )
    else:
        lines.append(
            f"- **Per-cell timing:** **median** over **{trials}** inner trial(s) in a single sweep pass."
        )
    lines.append(f"- **Sweep mode:** {'quick (reduced grid)' if quick else 'full'}")
    lines.append("")
    lines.append("## What was compared")
    lines.append("")
    lines.append(
        "Each scenario loads the full corpus from **Parquet** vs **Lance** into a dense "
        "`float32` matrix, then runs the **same** batched L2 brute-force kNN. "
        "Wall-time columns are **load**, **search**, **total** (see header above for median vs mean±stdev)."
    )
    lines.append("")
    lines.append(
        "**Interpretation:** Differences in **search** time should be small (same NumPy kernel). "
        "Differences in **load** reflect PyArrow Parquet decode vs Lance scan/materialization "
        "in this harness (local disk, warm-ish cache after first touch in a trial sequence)."
    )
    lines.append("")
    show_disk = any(r.storage_bytes is not None for r in rows)
    show_rss = any(r.rss_peak_load_bytes is not None for r in rows)
    if show_disk:
        lines.append(
            "**On-disk size:** Parquet = `corpus.parquet` file size; Lance = sum of file sizes under "
            "the LanceDB directory (fragments + sidecars). Comparable logical row counts, different "
            "physical layout and compression."
        )
        lines.append("")
    if show_rss:
        lines.append(
            "**RSS peaks:** Sampled process resident set size during **load** and **search** "
            "(background thread, ~5 ms interval via psutil). This approximates **heap + mapped pages** "
            "visible to the OS; it is not allocator-level precision and can include unrelated churn."
        )
        lines.append("")

    lines.append("## Results")
    lines.append("")
    for group in groups_order:
        sids = by_group.get(group)
        if not sids:
            continue
        lines.append(f"### {group.title()}")
        lines.append("")
        head = (
            "| Scenario | N | d | Q | k | Pq load | L load | Pq search | L search | Pq total | L total | "
            "Load L/P | Total L/P |"
        )
        sep = (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        if show_disk:
            head += " Pq disk (MiB) | L disk (MiB) | Disk L/P |"
            sep += " ---: | ---: | ---: |"
        if show_rss:
            head += " Pq RSS load (MiB) | L RSS load (MiB) | Pq RSS search (MiB) | L RSS search (MiB) |"
            sep += " ---: | ---: | ---: | ---: |"
        lines.append(head)
        lines.append(sep)
        for sid in sids:
            _, n, d, nq, k = meta[sid]
            pq_m = med[sid]["parquet"]
            la_m = med[sid]["lance"]
            rl = _ratio(la_m["load_seconds"], pq_m["load_seconds"])
            rt = _ratio(la_m["total_seconds"], pq_m["total_seconds"])
            row = (
                f"| `{sid}` | {n} | {d} | {nq} | {k} | "
                f"{_fmt_sec(pq_m['load_seconds'], pq_m.get('load_stdev'))} | "
                f"{_fmt_sec(la_m['load_seconds'], la_m.get('load_stdev'))} | "
                f"{_fmt_sec(pq_m['search_seconds'], pq_m.get('search_stdev'))} | "
                f"{_fmt_sec(la_m['search_seconds'], la_m.get('search_stdev'))} | "
                f"{_fmt_sec(pq_m['total_seconds'], pq_m.get('total_stdev'))} | "
                f"{_fmt_sec(la_m['total_seconds'], la_m.get('total_stdev'))} | "
                f"{rl:.2f}× | {rt:.2f}× |"
            )
            if show_disk:
                pq_d = pq_m.get("storage_bytes")
                la_d = la_m.get("storage_bytes")
                dr = _ratio(la_d, pq_d) if pq_d and la_d else float("nan")
                dr_s = f"{dr:.2f}×" if dr == dr else "—"
                row += f" {_mib(pq_d)} | {_mib(la_d)} | {dr_s} |"
            if show_rss:
                row += (
                    f" {_mib_pm(pq_m.get('rss_peak_load_bytes'), pq_m.get('rss_peak_load_stdev'))} | "
                    f"{_mib_pm(la_m.get('rss_peak_load_bytes'), la_m.get('rss_peak_load_stdev'))} | "
                    f"{_mib_pm(pq_m.get('rss_peak_search_bytes'), pq_m.get('rss_peak_search_stdev'))} | "
                    f"{_mib_pm(la_m.get('rss_peak_search_bytes'), la_m.get('rss_peak_search_stdev'))} |"
                )
            lines.append(row)
        lines.append("")

    lines.append("## Observations (this run)")
    lines.append("")
    scale_sids = by_group.get("scale", [])
    if scale_sids:
        ratios: list[float] = []
        for sid in scale_sids:
            pq_m = med[sid]["parquet"]
            la_m = med[sid]["lance"]
            ratios.append(_ratio(la_m["load_seconds"], pq_m["load_seconds"]))
        lo, hi = min(ratios), max(ratios)
        if hi < 1.0:
            lines.append(
                f"- **Load (Lance / Parquet)** across **scale** scenarios was **{lo:.2f}×–{hi:.2f}×** "
                f"(always below 1 → Lance **faster** to materialize the matrix in this run, after local caching effects)."
            )
        elif lo > 1.0:
            lines.append(
                f"- **Load (Lance / Parquet)** across **scale** scenarios was **{lo:.2f}×–{hi:.2f}×** "
                f"(always above 1 → Parquet **faster** on load in this run)."
            )
        else:
            lines.append(
                f"- **Load (Lance / Parquet)** across **scale** scenarios spanned **{lo:.2f}×–{hi:.2f}×** (mixed; see table)."
            )
    lines.append(
        "- **Search** times should track query count and N×d multiply cost; small Lance vs Parquet "
        "spread is expected after both corpora are in RAM."
    )
    lines.append(
        "- This harness measures **full materialization** + in-memory kNN, not Lance IVF/PQ ANN or "
        "streaming Parquet scans."
    )
    lines.append("")
    lines.append("## Raw data")
    lines.append("")
    if repeats > 1:
        lines.append(
            "See `results/sweep.json` (**mean-aggregated** per scenario × backend). "
            "All per-repeat rows are in `results/sweep_raw.json` when the sweep was run with `--keep-raw` (default on for repeats > 1)."
        )
    else:
        lines.append("See `results/sweep.json` alongside this report.")
    lines.append("")
    return "\n".join(lines)
