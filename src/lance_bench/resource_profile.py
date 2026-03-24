"""On-disk size and sampled process RSS peaks during a code section."""

from __future__ import annotations

import gc
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def file_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return 0


def dir_tree_size_bytes(root: Path) -> int:
    """Sum of file sizes under root (recursive)."""
    if not root.exists():
        return 0
    total = 0
    if root.is_file():
        return int(root.stat().st_size)
    for p in root.rglob("*"):
        if p.is_file():
            total += int(p.stat().st_size)
    return total


def peak_rss_during(fn: Callable[[], T], *, interval_s: float = 0.005) -> tuple[T, int]:
    """
    Run fn while a background thread samples this process's RSS.
    Returns (fn_result, peak_rss_bytes). Requires psutil.
    """
    import psutil

    gc.collect()
    proc = psutil.Process(os.getpid())
    peak: list[int] = [int(proc.memory_info().rss)]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(interval_s):
            peak[0] = max(peak[0], int(proc.memory_info().rss))

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        out = fn()
    finally:
        stop.set()
        th.join(timeout=10.0)
    return out, peak[0]
