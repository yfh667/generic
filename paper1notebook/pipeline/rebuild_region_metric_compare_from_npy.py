from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


DEFAULT_RUN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\Starlink_72_22_1_550"
    r"\motif_w_le4_h_le3_region_internal_plus_grid"
    r"\region_internal_grid_metrics_t0_86160_stride60"
)
DEFAULT_PAIRS = ("china_america", "china_europe", "china_africa")
MOTIF_RE = re.compile(r"^(.*?)(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild paper1 region-metric compare/summary CSVs from completed topology npy files."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS))
    parser.add_argument("--metric", choices=("delay", "hops", "both"), default="both")
    parser.add_argument("--write-meta", action="store_true")
    return parser.parse_args()


def finite_or_none(value: float) -> float | None:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def topology_sort_key(name: str) -> tuple[int, str, int]:
    if name == "full_link":
        return (1, name, 0)
    if name == "gridplus":
        return (2, name, 0)
    match = MOTIF_RE.match(name)
    if match:
        return (0, match.group(1), int(match.group(2)))
    return (0, name, -1)


def discover_topologies(pair_dir: Path, metric_file: str, expected_steps: int) -> list[str]:
    names: list[str] = []
    for topology_dir in (pair_dir / "topologies").iterdir():
        if not topology_dir.is_dir():
            continue
        path = topology_dir / metric_file
        time_path = topology_dir / "time_indices.npy"
        if not path.exists() or not time_path.exists():
            continue
        try:
            values = np.load(path, mmap_mode="r")
            saved_steps = np.load(time_path, mmap_mode="r")
            if values.shape == (expected_steps,) and saved_steps.shape == (expected_steps,):
                names.append(topology_dir.name)
        except Exception:
            continue
    return sorted(names, key=topology_sort_key)


def write_compare_csv(
    path: Path,
    *,
    steps: np.ndarray,
    values_by_topology: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(values_by_topology.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *names])
        writer.writeheader()
        for idx, step in enumerate(steps):
            row = {"step": int(step)}
            for name in names:
                row[name] = finite_or_none(float(values_by_topology[name][idx]))
            writer.writerow(row)


def write_summary_csv(path: Path, values_by_topology: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["topology", "mean", "min", "max", "finite_steps"])
        writer.writeheader()
        for name, values_raw in values_by_topology.items():
            values = np.asarray(values_raw, dtype=np.float64)
            finite = values[np.isfinite(values)]
            writer.writerow(
                {
                    "topology": name,
                    "mean": float(np.mean(finite)) if finite.size else None,
                    "min": float(np.min(finite)) if finite.size else None,
                    "max": float(np.max(finite)) if finite.size else None,
                    "finite_steps": int(finite.size),
                }
            )


def metric_specs(selected: str) -> Iterable[tuple[str, str, str]]:
    if selected in ("delay", "both"):
        yield "mean_shortest_delay_ms.npy", "compare_mean_shortest_delay_ms.csv", "summary_mean_shortest_delay_ms.csv"
    if selected in ("hops", "both"):
        yield "mean_shortest_hops.npy", "compare_mean_shortest_hops.csv", "summary_mean_shortest_hops.csv"


def rebuild_pair(pair_dir: Path, steps: np.ndarray, selected_metric: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for npy_name, compare_name, summary_name in metric_specs(selected_metric):
        names = discover_topologies(pair_dir, npy_name, int(steps.size))
        values = {
            name: np.asarray(np.load(pair_dir / "topologies" / name / npy_name), dtype=np.float32)
            for name in names
        }
        write_compare_csv(pair_dir / compare_name, steps=steps, values_by_topology=values)
        write_summary_csv(pair_dir / summary_name, values)
        result[npy_name] = len(names)
    return result


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    steps = np.asarray(np.load(run_dir / "time_indices.npy"), dtype=np.int64)
    meta: dict[str, object] = {
        "run_dir": str(run_dir),
        "num_steps": int(steps.size),
        "pairs": {},
    }
    for pair in args.pairs:
        pair_dir = run_dir / str(pair)
        counts = rebuild_pair(pair_dir, steps, str(args.metric))
        meta["pairs"][str(pair)] = counts
        print(f"[rebuild-region-compare] pair={pair} counts={counts}", flush=True)
    if args.write_meta:
        (run_dir / "rebuilt_compare_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
