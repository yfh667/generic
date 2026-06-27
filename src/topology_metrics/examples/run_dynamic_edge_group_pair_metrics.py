from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_metrics.module import (  # noqa: E402
    build_group_pair_node_arrays,
    compute_group_pair_shortest_timeseries,
    plot_group_pair_shortest_timeseries,
    read_edge_table_csv,
    result_summary,
    write_group_pair_shortest_timeseries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute group-pair shortest hops and optional delay for a dynamic edge-mask series."
    )
    parser.add_argument("--edges-csv", type=Path, required=True)
    parser.add_argument("--steps-npy", type=Path, required=True)
    parser.add_argument("--active-mask-npy", type=Path, required=True)
    parser.add_argument("--group-data-json", type=Path, required=True)
    parser.add_argument("--source-group-id", type=int, required=True)
    parser.add_argument("--target-group-id", type=int, required=True)
    parser.add_argument("--total-nodes", type=int, required=True)
    parser.add_argument("--weights-npy", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--label", type=str, default="dynamic_edge_series")
    return parser.parse_args()


def load_group_data_json(path: Path) -> dict[int, dict]:
    """Load either a region group cache JSON or the already-normalized mapping."""

    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("groups_by_step"), dict):
        groups_by_step = raw["groups_by_step"]
        out: dict[int, dict] = {}
        for step_text, groups in groups_by_step.items():
            out[int(step_text)] = {
                "groups": {
                    int(group_id): set(int(node) for node in nodes)
                    for group_id, nodes in dict(groups).items()
                }
            }
        return out
    out = {}
    for step_text, payload in dict(raw).items():
        groups = dict(payload.get("groups", {})) if isinstance(payload, dict) else {}
        out[int(step_text)] = {
            "groups": {
                int(group_id): set(int(node) for node in nodes)
                for group_id, nodes in groups.items()
            }
        }
    return out


def main() -> int:
    args = parse_args()
    steps_all = np.asarray(np.load(args.steps_npy, mmap_mode="r"), dtype=np.int64)
    start = int(steps_all[0]) if args.start is None else int(args.start)
    end = int(steps_all[-1]) if args.end is None else int(args.end)
    rows = np.flatnonzero(
        (steps_all >= start)
        & (steps_all <= end)
        & (((steps_all - start) % max(1, int(args.stride))) == 0)
    )
    if rows.size == 0:
        raise ValueError(f"no steps selected for start={start}, end={end}, stride={args.stride}")
    steps = np.asarray(steps_all[rows], dtype=np.int64)
    edge_table = read_edge_table_csv(args.edges_csv, total_nodes=int(args.total_nodes))
    active_all = np.load(args.active_mask_npy, mmap_mode="r")
    active_mask = active_all if active_all.ndim == 1 else np.asarray(active_all[rows], dtype=bool)
    weights = None
    if args.weights_npy is not None:
        weights_all = np.load(args.weights_npy, mmap_mode="r")
        weights = weights_all if weights_all.ndim == 1 else np.asarray(weights_all[rows], dtype=np.float32)
    group_data = load_group_data_json(args.group_data_json)
    group_nodes = build_group_pair_node_arrays(
        group_data=group_data,
        steps=steps,
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
    )
    result = compute_group_pair_shortest_timeseries(
        edge_table=edge_table,
        total_nodes=int(args.total_nodes),
        steps=steps,
        edge_active_mask=active_mask,
        edge_weights_ms=weights,
        group_nodes=group_nodes,
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        progress_every_chunks=10,
    )
    files = write_group_pair_shortest_timeseries(
        result,
        args.out_dir,
        meta={
            "label": str(args.label),
            "edges_csv": str(args.edges_csv),
            "steps_npy": str(args.steps_npy),
            "active_mask_npy": str(args.active_mask_npy),
            "weights_npy": None if args.weights_npy is None else str(args.weights_npy),
            "group_data_json": str(args.group_data_json),
            "source_group_id": int(args.source_group_id),
            "target_group_id": int(args.target_group_id),
            "total_nodes": int(args.total_nodes),
            "start": int(start),
            "end": int(end),
            "stride": int(args.stride),
        },
    )
    plot_group_pair_shortest_timeseries(
        {str(args.label): result},
        args.out_dir,
        metric="mean_shortest_hops",
        ylabel="mean shortest hops",
        title=f"{args.label}: mean shortest hops",
        filename="mean_shortest_hops.png",
    )
    if result.has_delay:
        plot_group_pair_shortest_timeseries(
            {str(args.label): result},
            args.out_dir,
            metric="mean_shortest_delay_ms",
            ylabel="mean shortest delay (ms)",
            title=f"{args.label}: mean shortest delay",
            filename="mean_shortest_delay_ms.png",
        )

    summary = result_summary(result)
    print(
        f"label={args.label} mean_hops={summary['mean_hops']:.6f} "
        f"mean_delay_ms={summary['mean_shortest_delay_ms']:.6f}",
        flush=True,
    )
    print(f"out_dir={args.out_dir}", flush=True)
    for key, value in files.items():
        print(f"{key}={value}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
