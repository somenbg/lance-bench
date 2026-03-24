"""CLI: prepare data and run Parquet vs Lance (load + brute-force kNN)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import typer

from lance_bench.dataset import make_queries, make_random_embeddings
from lance_bench.lance_backend import write_corpus_lance
from lance_bench.parquet_backend import write_corpus_parquet
from lance_bench.benchmark import run_pair, write_results_json
from lance_bench.plot_sweep import write_sweep_figures
from lance_bench.report_md import generate_report_md
from lance_bench.sweep import (
    build_scenarios,
    load_sweep_json,
    run_sweep_repeated,
    write_sweep_json,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def prepare(
    out_dir: Path = typer.Option(Path("data"), help="Output directory"),
    n_rows: int = typer.Option(50_000, help="Corpus size"),
    dim: int = typer.Option(128, help="Embedding dimension"),
    seed: int = typer.Option(42, help="RNG seed for corpus"),
    show_sizes: bool = typer.Option(
        False,
        "--show-sizes",
        help="Print on-disk bytes for Parquet file vs Lance directory",
    ),
) -> None:
    """Write the same synthetic corpus to Parquet and Lance."""
    from lance_bench.resource_profile import dir_tree_size_bytes, file_size_bytes

    ids, emb = make_random_embeddings(n_rows, dim, seed=seed)
    pq_path = out_dir / "corpus.parquet"
    lance_dir = out_dir / "lance"
    write_corpus_parquet(pq_path, ids, emb)
    write_corpus_lance(lance_dir, "corpus", ids, emb)
    typer.echo(f"Parquet: {pq_path}")
    typer.echo(f"Lance:   {lance_dir} (table corpus)")
    if show_sizes:
        pq_b = file_size_bytes(pq_path)
        la_b = dir_tree_size_bytes(lance_dir)
        typer.echo(f"Parquet bytes (file): {pq_b}")
        typer.echo(f"Lance bytes (tree):     {la_b}")


@app.command()
def run(
    data_dir: Path = typer.Option(Path("data"), help="Directory from prepare"),
    n_queries: int = typer.Option(256, help="Number of query vectors"),
    k: int = typer.Option(10, help="k for kNN"),
    query_seed: int = typer.Option(43, help="RNG seed for queries"),
    results_json: Path = typer.Option(Path("results/bench.json"), help="Write timings here"),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Sample RSS peaks during load/search (psutil); disk sizes always recorded",
    ),
) -> None:
    """Load corpus from each backend and time load + brute-force search."""
    pq_path = data_dir / "corpus.parquet"
    lance_uri = data_dir / "lance"
    if not pq_path.is_file():
        raise typer.BadParameter(f"Missing {pq_path}; run prepare first.")
    if not lance_uri.is_dir():
        raise typer.BadParameter(f"Missing {lance_uri}; run prepare first.")

    schema = pq.read_schema(pq_path)
    dim = schema.field("embedding").type.list_size
    queries = make_queries(n_queries, dim, seed=query_seed)

    r_pq, r_lance = run_pair(pq_path, lance_uri, "corpus", queries, k, profile=profile)
    write_results_json(results_json, [r_pq, r_lance])

    typer.echo(results_json.read_text(encoding="utf-8"))


@app.command()
def sweep(
    data_root: Path = typer.Option(Path("data/sweep"), help="Per-corpus directories created here"),
    out_json: Path = typer.Option(Path("results/sweep.json"), help="All trial rows (JSON)"),
    report_path: Path = typer.Option(Path("results/benchmark-report.md"), help="Markdown report"),
    quick: bool = typer.Option(False, "--quick", help="Smaller scenario grid for smoke tests"),
    trials: int = typer.Option(2, min=1, help="Repeated trials per scenario (median in report)"),
    query_seed_base: int = typer.Option(10_000, help="Base seed; trials offset deterministically"),
    plot_to: Optional[Path] = typer.Option(
        None,
        "--plot-to",
        help="If set, write matplotlib PNGs to this directory",
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Sample RSS peaks during load/search (psutil); disk sizes always recorded",
    ),
    repeats: int = typer.Option(
        1,
        min=1,
        max=500,
        help="Run the full sweep this many times; timings in sweep.json are mean±stdev over repeats×trials",
    ),
    keep_raw: bool = typer.Option(
        True,
        "--keep-raw/--no-keep-raw",
        help="When repeats>1, also write sweep_raw.json with every pass",
    ),
) -> None:
    """Run multiple scenarios (scale, dim, query batch, k) and write JSON + Markdown report."""
    scenarios = build_scenarios(quick=quick)
    typer.echo(
        f"Scenarios: {len(scenarios)} | trials: {trials} | repeats: {repeats} | "
        f"samples/cell: {repeats * trials} | quick={quick} | profile={profile}"
    )
    raw_all, rows = run_sweep_repeated(
        data_root,
        scenarios,
        trials=trials,
        query_seed_base=query_seed_base,
        profile=profile,
        repeats=repeats,
    )
    if repeats > 1 and keep_raw:
        raw_path = out_json.with_name(out_json.stem + "_raw.json")
        write_sweep_json(raw_path, raw_all)
        typer.echo(f"Wrote {raw_path} ({len(raw_all)} rows)")
    write_sweep_json(out_json, rows)
    meta_path = out_json.with_name(out_json.stem + "_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "trials": trials,
                "quick": quick,
                "profile": profile,
                "repeats": repeats,
                "samples_per_cell": repeats * trials,
                "aggregated": repeats > 1,
                "keep_raw": bool(repeats > 1 and keep_raw),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    title = "Parquet vs Lance — sweep benchmark"
    md = generate_report_md(rows, title=title, trials=trials, quick=quick, repeats=repeats)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    typer.echo(f"Wrote {out_json} ({len(rows)} rows)")
    typer.echo(f"Wrote {meta_path}")
    typer.echo(f"Wrote {report_path}")
    if plot_to is not None:
        for p in write_sweep_figures(rows, plot_to):
            typer.echo(f"Plot: {p}")


@app.command()
def plot(
    from_json: Path = typer.Option(Path("results/sweep.json"), "--from", help="Sweep JSON"),
    out_dir: Path = typer.Option(Path("results/plots"), help="PNG output directory"),
) -> None:
    """Plot median Parquet vs Lance metrics from a sweep (matplotlib PNGs)."""
    rows = load_sweep_json(from_json)
    paths = write_sweep_figures(rows, out_dir)
    for p in paths:
        typer.echo(f"Wrote {p}")


@app.command("report")
def report_cmd(
    from_json: Path = typer.Option(Path("results/sweep.json"), "--from", help="Sweep JSON"),
    meta_json: Optional[Path] = typer.Option(
        None,
        "--meta",
        help="sweep_meta.json from sweep (defaults for trials/quick)",
    ),
    out: Path = typer.Option(Path("results/benchmark-report.md"), help="Markdown output"),
) -> None:
    """Regenerate Markdown report from an existing sweep JSON."""
    rows = load_sweep_json(from_json)
    trials, quick, repeats = 2, False, 1
    meta_path = meta_json
    if meta_path is None:
        cand = from_json.with_name(from_json.stem + "_meta.json")
        if cand.is_file():
            meta_path = cand
    if meta_path is not None and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trials = int(meta.get("trials", trials))
        quick = bool(meta.get("quick", quick))
        repeats = int(meta.get("repeats", repeats))
    title = "Parquet vs Lance — sweep benchmark"
    md = generate_report_md(rows, title=title, trials=trials, quick=quick, repeats=repeats)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    typer.echo(f"Wrote {out}")
