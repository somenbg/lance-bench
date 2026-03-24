"""Matplotlib figures from sweep JSON (median per scenario × backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lance_bench.benchmark import BenchResult
from lance_bench.sweep import median_by_scenario


def _scenario_meta(rows: list[BenchResult]) -> dict[str, tuple[str, int, int, int, int]]:
    m: dict[str, tuple[str, int, int, int, int]] = {}
    for r in rows:
        if r.scenario_id and r.scenario_id not in m:
            m[r.scenario_id] = (r.scenario_group, r.n_rows, r.dim, r.n_queries, r.k)
    return m


def _sids_for_group(meta: dict[str, tuple], group: str) -> list[str]:
    sids = [sid for sid, (g, *_rest) in meta.items() if g == group]
    return sids


def _bytes_to_mib(b: float | None) -> float | None:
    if b is None:
        return None
    return float(b) / (1024.0**2)


def _med_has_disk(med: dict, sid: str) -> bool:
    return med[sid]["parquet"].get("storage_bytes") is not None


def _med_has_rss(med: dict, sid: str) -> bool:
    return med[sid]["parquet"].get("rss_peak_load_bytes") is not None


def write_sweep_figures(rows: list[BenchResult], out_dir: Path) -> list[Path]:
    """Write PNGs: dashboard (2×2) plus optional singles. Returns paths created."""
    out_dir.mkdir(parents=True, exist_ok=True)
    med = median_by_scenario(rows)
    meta = _scenario_meta(rows)
    if not meta:
        raise ValueError("no scenario_id on rows; run a sweep first")

    pq_color = "#4472C4"
    la_color = "#ED7D31"

    paths: list[Path] = []

    def fig_scale_total() -> None:
        sids = sorted(_sids_for_group(meta, "scale"), key=lambda sid: meta[sid][1])
        if len(sids) < 2:
            return
        xs = [meta[s][1] for s in sids]
        y_pq = [med[s]["parquet"]["total_seconds"] for s in sids]
        y_la = [med[s]["lance"]["total_seconds"] for s in sids]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet", linewidth=2, markersize=6)
        ax.plot(xs, y_la, "s-", color=la_color, label="Lance", linewidth=2, markersize=6)
        ax.set_xlabel("N — corpus rows")
        ax.set_ylabel("Total time (s), median")
        ax.set_title("Scale: end-to-end time vs N\n(d=128, Q=256, k=10)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "scale_total.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_scale_load() -> None:
        sids = sorted(_sids_for_group(meta, "scale"), key=lambda sid: meta[sid][1])
        if len(sids) < 2:
            return
        xs = [meta[s][1] for s in sids]
        y_pq = [med[s]["parquet"]["load_seconds"] for s in sids]
        y_la = [med[s]["lance"]["load_seconds"] for s in sids]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet load", linewidth=2, markersize=6)
        ax.plot(xs, y_la, "s-", color=la_color, label="Lance load", linewidth=2, markersize=6)
        ax.set_xlabel("N — corpus rows")
        ax.set_ylabel("Load time (s), median")
        ax.set_title("Scale: load-only vs N")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "scale_load.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_dimension_total() -> None:
        sids = sorted(_sids_for_group(meta, "dimension"), key=lambda sid: meta[sid][2])
        if not sids:
            return
        xs = [meta[s][2] for s in sids]
        y_pq = [med[s]["parquet"]["total_seconds"] for s in sids]
        y_la = [med[s]["lance"]["total_seconds"] for s in sids]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet", linewidth=2, markersize=6)
        ax.plot(xs, y_la, "s-", color=la_color, label="Lance", linewidth=2, markersize=6)
        ax.set_xlabel("d — embedding dimension")
        ax.set_ylabel("Total time (s), median")
        ax.set_title("Dimension: total vs d\n(N=50k, Q=256, k=10)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "dimension_total.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_queries_search() -> None:
        sids = sorted(_sids_for_group(meta, "queries"), key=lambda sid: meta[sid][3])
        if not sids:
            return
        xs = [meta[s][3] for s in sids]
        y_pq = [med[s]["parquet"]["search_seconds"] for s in sids]
        y_la = [med[s]["lance"]["search_seconds"] for s in sids]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet", linewidth=2, markersize=6)
        ax.plot(xs, y_la, "s-", color=la_color, label="Lance", linewidth=2, markersize=6)
        ax.set_xlabel("Q — number of queries")
        ax.set_ylabel("Search time (s), median")
        ax.set_title("Query batch: kNN time vs Q\n(N=50k, d=128, k=10)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "queries_search.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_k_bars() -> None:
        sids = sorted(_sids_for_group(meta, "k"), key=lambda sid: meta[sid][4])
        if not sids:
            return
        labels = [f"k={meta[s][4]}" for s in sids]
        pq_t = [med[s]["parquet"]["total_seconds"] for s in sids]
        la_t = [med[s]["lance"]["total_seconds"] for s in sids]
        x = np.arange(len(sids))
        w = 0.35
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(x - w / 2, pq_t, w, label="Parquet", color=pq_color)
        ax.bar(x + w / 2, la_t, w, label="Lance", color=la_color)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Total time (s), median")
        ax.set_title("k sweep\n(N=50k, d=128, Q=256)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        p = out_dir / "k_total.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_scale_disk() -> None:
        sids = sorted(_sids_for_group(meta, "scale"), key=lambda sid: meta[sid][1])
        if len(sids) < 2 or not _med_has_disk(med, sids[0]):
            return
        xs = [meta[s][1] for s in sids]
        y_pq = [_bytes_to_mib(med[s]["parquet"]["storage_bytes"]) for s in sids]
        y_la = [_bytes_to_mib(med[s]["lance"]["storage_bytes"]) for s in sids]
        if any(v is None for v in y_pq + y_la):
            return
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet file", linewidth=2, markersize=6)
        ax.plot(xs, y_la, "s-", color=la_color, label="Lance dir (sum files)", linewidth=2, markersize=6)
        ax.set_xlabel("N — corpus rows")
        ax.set_ylabel("On-disk size (MiB)")
        ax.set_title("Scale: storage footprint vs N\n(same logical rows, different layout)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "scale_disk_mib.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_dimension_disk() -> None:
        sids = sorted(_sids_for_group(meta, "dimension"), key=lambda sid: meta[sid][2])
        if not sids or not _med_has_disk(med, sids[0]):
            return
        xs = [meta[s][2] for s in sids]
        y_pq = [_bytes_to_mib(med[s]["parquet"]["storage_bytes"]) for s in sids]
        y_la = [_bytes_to_mib(med[s]["lance"]["storage_bytes"]) for s in sids]
        if any(v is None for v in y_pq + y_la):
            return
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet file", linewidth=2, markersize=6)
        ax.plot(xs, y_la, "s-", color=la_color, label="Lance dir (sum files)", linewidth=2, markersize=6)
        ax.set_xlabel("d — embedding dimension")
        ax.set_ylabel("On-disk size (MiB)")
        ax.set_title("Dimension: storage footprint vs d\n(N=50k)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "dimension_disk_mib.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_scale_rss() -> None:
        sids = sorted(_sids_for_group(meta, "scale"), key=lambda sid: meta[sid][1])
        if len(sids) < 2 or not _med_has_rss(med, sids[0]):
            return
        xs = [meta[s][1] for s in sids]
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
        for ax, key, title in (
            (axes[0], "rss_peak_load_bytes", "Peak RSS during load"),
            (axes[1], "rss_peak_search_bytes", "Peak RSS during search"),
        ):
            y_pq = [_bytes_to_mib(med[s]["parquet"].get(key)) for s in sids]
            y_la = [_bytes_to_mib(med[s]["lance"].get(key)) for s in sids]
            if any(v is None for v in y_pq + y_la):
                ax.text(0.5, 0.5, f"no {key} in medians", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet", linewidth=2, markersize=6)
            ax.plot(xs, y_la, "s-", color=la_color, label="Lance", linewidth=2, markersize=6)
            ax.set_ylabel("RSS (MiB)")
            ax.set_title(title + " vs N (median trials)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[1].set_xlabel("N — corpus rows")
        fig.suptitle("Scale: memory peaks (sampled RSS)", fontsize=11, y=1.02)
        fig.tight_layout()
        p = out_dir / "scale_rss_mib.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_dimension_rss() -> None:
        sids = sorted(_sids_for_group(meta, "dimension"), key=lambda sid: meta[sid][2])
        if not sids or not _med_has_rss(med, sids[0]):
            return
        xs = [meta[s][2] for s in sids]
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
        for ax, key, title in (
            (axes[0], "rss_peak_load_bytes", "Peak RSS during load"),
            (axes[1], "rss_peak_search_bytes", "Peak RSS during search"),
        ):
            y_pq = [_bytes_to_mib(med[s]["parquet"].get(key)) for s in sids]
            y_la = [_bytes_to_mib(med[s]["lance"].get(key)) for s in sids]
            if any(v is None for v in y_pq + y_la):
                ax.text(0.5, 0.5, f"no {key}", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.plot(xs, y_pq, "o-", color=pq_color, label="Parquet", linewidth=2, markersize=6)
            ax.plot(xs, y_la, "s-", color=la_color, label="Lance", linewidth=2, markersize=6)
            ax.set_ylabel("RSS (MiB)")
            ax.set_title(title + " vs d (median trials)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[1].set_xlabel("d — embedding dimension")
        fig.suptitle("Dimension: memory peaks (sampled RSS)", fontsize=11, y=1.02)
        fig.tight_layout()
        p = out_dir / "dimension_rss_mib.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def fig_resources_dashboard() -> None:
        scale_sids = sorted(_sids_for_group(meta, "scale"), key=lambda sid: meta[sid][1])
        dim_sids = sorted(_sids_for_group(meta, "dimension"), key=lambda sid: meta[sid][2])
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        ax = axes[0, 0]
        if len(scale_sids) >= 2 and _med_has_disk(med, scale_sids[0]):
            xs = [meta[s][1] for s in scale_sids]
            pq = [_bytes_to_mib(med[s]["parquet"]["storage_bytes"]) for s in scale_sids]
            la = [_bytes_to_mib(med[s]["lance"]["storage_bytes"]) for s in scale_sids]
            if not any(v is None for v in pq + la):
                ax.plot(xs, pq, "o-", color=pq_color, label="Parquet", markersize=5)
                ax.plot(xs, la, "s-", color=la_color, label="Lance", markersize=5)
                ax.set_xlabel("N")
                ax.set_ylabel("Disk (MiB)")
                ax.set_title("On-disk vs N")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "incomplete disk", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "no scale disk", ha="center", va="center", transform=ax.transAxes)

        ax = axes[0, 1]
        if dim_sids and _med_has_disk(med, dim_sids[0]):
            xs = [meta[s][2] for s in dim_sids]
            pq = [_bytes_to_mib(med[s]["parquet"]["storage_bytes"]) for s in dim_sids]
            la = [_bytes_to_mib(med[s]["lance"]["storage_bytes"]) for s in dim_sids]
            if not any(v is None for v in pq + la):
                ax.plot(xs, pq, "o-", color=pq_color, label="Parquet", markersize=5)
                ax.plot(xs, la, "s-", color=la_color, label="Lance", markersize=5)
                ax.set_xlabel("d")
                ax.set_ylabel("Disk (MiB)")
                ax.set_title("On-disk vs d")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "incomplete disk", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "no dim disk", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 0]
        if len(scale_sids) >= 2 and _med_has_rss(med, scale_sids[0]):
            xs = [meta[s][1] for s in scale_sids]
            pq = [_bytes_to_mib(med[s]["parquet"].get("rss_peak_load_bytes")) for s in scale_sids]
            la = [_bytes_to_mib(med[s]["lance"].get("rss_peak_load_bytes")) for s in scale_sids]
            if not any(v is None for v in pq + la):
                ax.plot(xs, pq, "o-", color=pq_color, label="Parquet", markersize=5)
                ax.plot(xs, la, "s-", color=la_color, label="Lance", markersize=5)
                ax.set_xlabel("N")
                ax.set_ylabel("RSS load peak (MiB)")
                ax.set_title("RSS load peak vs N")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "incomplete RSS", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "no scale RSS\n(use --profile)", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 1]
        if dim_sids and _med_has_rss(med, dim_sids[0]):
            xs = [meta[s][2] for s in dim_sids]
            pq = [_bytes_to_mib(med[s]["parquet"].get("rss_peak_load_bytes")) for s in dim_sids]
            la = [_bytes_to_mib(med[s]["lance"].get("rss_peak_load_bytes")) for s in dim_sids]
            if not any(v is None for v in pq + la):
                ax.plot(xs, pq, "o-", color=pq_color, label="Parquet", markersize=5)
                ax.plot(xs, la, "s-", color=la_color, label="Lance", markersize=5)
                ax.set_xlabel("d")
                ax.set_ylabel("RSS load peak (MiB)")
                ax.set_title("RSS load peak vs d")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "incomplete RSS", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "no dim RSS\n(use --profile)", ha="center", va="center", transform=ax.transAxes)

        fig.suptitle("Parquet vs Lance — disk & RSS load (medians)", fontsize=13, y=1.02)
        fig.tight_layout()
        p = out_dir / "resources_dashboard.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    fig_scale_total()
    fig_scale_load()
    fig_dimension_total()
    fig_queries_search()
    fig_k_bars()
    fig_scale_disk()
    fig_dimension_disk()
    fig_scale_rss()
    fig_dimension_rss()
    fig_resources_dashboard()

    # 2×2 dashboard (reuse logic inline)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    def ax_scale_total(ax: Any) -> None:
        sids = sorted(_sids_for_group(meta, "scale"), key=lambda sid: meta[sid][1])
        if len(sids) < 2:
            ax.text(0.5, 0.5, "no scale scenarios", ha="center", va="center", transform=ax.transAxes)
            return
        xs = [meta[s][1] for s in sids]
        ax.plot(xs, [med[s]["parquet"]["total_seconds"] for s in sids], "o-", color=pq_color, label="Parquet")
        ax.plot(xs, [med[s]["lance"]["total_seconds"] for s in sids], "s-", color=la_color, label="Lance")
        ax.set_xlabel("N")
        ax.set_ylabel("Total (s)")
        ax.set_title("Total vs corpus size N")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def ax_dim(ax: Any) -> None:
        sids = sorted(_sids_for_group(meta, "dimension"), key=lambda sid: meta[sid][2])
        if not sids:
            ax.text(0.5, 0.5, "no dim scenarios", ha="center", va="center", transform=ax.transAxes)
            return
        xs = [meta[s][2] for s in sids]
        ax.plot(xs, [med[s]["parquet"]["total_seconds"] for s in sids], "o-", color=pq_color, label="Parquet")
        ax.plot(xs, [med[s]["lance"]["total_seconds"] for s in sids], "s-", color=la_color, label="Lance")
        ax.set_xlabel("d")
        ax.set_ylabel("Total (s)")
        ax.set_title("Total vs dimension d")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def ax_q(ax: Any) -> None:
        sids = sorted(_sids_for_group(meta, "queries"), key=lambda sid: meta[sid][3])
        if not sids:
            ax.text(0.5, 0.5, "no query scenarios", ha="center", va="center", transform=ax.transAxes)
            return
        xs = [meta[s][3] for s in sids]
        ax.plot(xs, [med[s]["parquet"]["search_seconds"] for s in sids], "o-", color=pq_color, label="Parquet")
        ax.plot(xs, [med[s]["lance"]["search_seconds"] for s in sids], "s-", color=la_color, label="Lance")
        ax.set_xlabel("Q")
        ax.set_ylabel("Search (s)")
        ax.set_title("kNN time vs query count Q")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def ax_k(ax: Any) -> None:
        sids = sorted(_sids_for_group(meta, "k"), key=lambda sid: meta[sid][4])
        if not sids:
            ax.text(0.5, 0.5, "no k scenarios", ha="center", va="center", transform=ax.transAxes)
            return
        labels = [f"k={meta[s][4]}" for s in sids]
        x = np.arange(len(sids))
        w = 0.35
        ax.bar(x - w / 2, [med[s]["parquet"]["total_seconds"] for s in sids], w, color=pq_color, label="Parquet")
        ax.bar(x + w / 2, [med[s]["lance"]["total_seconds"] for s in sids], w, color=la_color, label="Lance")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Total (s)")
        ax.set_title("Total vs k")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    ax_scale_total(axes[0, 0])
    ax_dim(axes[0, 1])
    ax_q(axes[1, 0])
    ax_k(axes[1, 1])
    fig.suptitle("Parquet vs Lance — sweep (median timings)", fontsize=13, y=1.02)
    fig.tight_layout()
    dash = out_dir / "sweep_dashboard.png"
    fig.savefig(dash, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(dash)

    return paths
