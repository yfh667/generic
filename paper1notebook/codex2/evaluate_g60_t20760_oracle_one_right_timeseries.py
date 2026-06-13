from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data

from build_g60_motif_gridplus_shortest_delay_timeseries_parallel import (
    DEFAULT_XML,
    dijkstra_targets,
    group_name,
    group_nodes_for_step,
    load_steps_and_rows,
    make_worker_spec,
)


DEFAULT_EDGE_CSV = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "g60_oracle_one_right_topology_t20760_china_europe"
    / "oracle_one_right_with_intra_edges.csv"
)
DEFAULT_DELAY_STORE = PROJECT_ROOT / "data" / "linshi" / "G60_full_options_plus_intra_t0_86164_stride1"
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "data" / "linshi" / "g60_oracle_one_right_t20760_china_europe_timeseries_t0_86160_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed 20760s one-right/one-left oracle topology over a time interval."
    )
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--source-group", type=int, default=2)
    parser.add_argument("--target-group", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def read_edge_table(path: Path) -> EdgeTable:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"edge csv is empty: {path}")

    src = np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32)
    dst = np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32)
    option = np.asarray([int(row["option"]) for row in rows], dtype=np.int16)
    sat_ids = [str(i + 1) for i in range(int(G60_CONFIG.total_sats))]
    for row in rows:
        s = int(row["src_node"])
        d = int(row["dst_node"])
        if 0 <= s < len(sat_ids):
            sat_ids[s] = str(row.get("src_sat_id", sat_ids[s]))
        if 0 <= d < len(sat_ids):
            sat_ids[d] = str(row.get("dst_sat_id", sat_ids[d]))

    n_count = int(G60_CONFIG.N)
    return EdgeTable(
        src=src,
        dst=dst,
        option=option,
        src_plane=(src // n_count).astype(np.int16),
        src_y=(src % n_count).astype(np.int16),
        dst_plane=(dst // n_count).astype(np.int16),
        dst_y=(dst % n_count).astype(np.int16),
        sat_ids=sat_ids,
    )


def evaluate_step(
    *,
    spec,
    weights: np.ndarray,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> dict:
    values: list[float] = []
    for source in sources:
        dist, _ = dijkstra_targets(
            spec=spec,
            weights=weights,
            source=int(source),
            targets=targets,
            want_prev=False,
        )
        for target in targets:
            value = float(dist[int(target)])
            if math.isfinite(value):
                values.append(value)

    expected = int(len(sources) * len(targets))
    if values:
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
    else:
        mean = math.nan
        min_value = math.nan
        max_value = math.nan
    return {
        "source_nodes": int(len(sources)),
        "target_nodes": int(len(targets)),
        "reachable_pairs": int(len(values)),
        "expected_pairs": int(expected),
        "mean_delay_ms": mean,
        "min_delay_ms": min_value,
        "max_delay_ms": max_value,
    }


def write_step_summary(path: Path, steps: list[int], rows: list[dict]) -> None:
    fields = [
        "step",
        "source_nodes",
        "target_nodes",
        "reachable_pairs",
        "expected_pairs",
        "mean_delay_ms",
        "min_delay_ms",
        "max_delay_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step, row in zip(steps, rows):
            item = {"step": int(step)}
            item.update(row)
            writer.writerow(item)


def write_plot(out_dir: Path, steps: list[int], values: np.ndarray) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[oracle-timeseries] plot skipped: {exc}", flush=True)
        return None

    plot_path = out_dir / "oracle_t20760_mean_shortest_delay.png"
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(steps, values, color="#8b1a1a", linewidth=1.4, label="fixed oracle designed at 20760s")
    ax.set_xlabel("time step (s)")
    ax.set_ylabel("China-Europe mean shortest delay (ms)")
    ax.set_title("G60 fixed one-right/one-left oracle topology designed at 20760s")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main() -> int:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_table = read_edge_table(Path(args.edge_csv))
    write_edges_csv(edge_table, out_dir / "edges.csv")

    delay_store, steps, delay_rows = load_steps_and_rows(
        Path(args.delay_store_dir),
        int(args.start),
        int(args.end),
        int(args.stride),
    )
    spec = make_worker_spec(edge_table, delay_store, "oracle_one_right_t20760_fixed")

    group_data = load_or_build_group_data(
        xml_file=Path(args.xml_file),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    summaries: list[dict] = []
    means = np.full(len(steps), np.nan, dtype=np.float32)
    for row_idx, (step, delay_row) in enumerate(zip(steps, delay_rows)):
        sources = group_nodes_for_step(group_data, int(step), int(args.source_group))
        targets = group_nodes_for_step(group_data, int(step), int(args.target_group))
        weights = np.asarray(delay_store.delay_ms_array[int(delay_row), spec.store_edge_indices], dtype=np.float32)
        summary = evaluate_step(spec=spec, weights=weights, sources=sources, targets=targets)
        summaries.append(summary)
        means[row_idx] = float(summary["mean_delay_ms"])
        if int(args.progress_every) > 0 and ((row_idx + 1) % int(args.progress_every) == 0 or row_idx + 1 == len(steps)):
            print(
                f"[oracle-timeseries] {row_idx + 1}/{len(steps)} "
                f"step={step} mean={float(means[row_idx]):.4f} "
                f"reachable={summary['reachable_pairs']}/{summary['expected_pairs']}",
                flush=True,
            )

    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "mean_shortest_delay_ms.npy", means.astype(np.float32))
    write_step_summary(out_dir / "step_summary.csv", steps, summaries)
    plot_path = write_plot(out_dir, steps, means)

    expected = np.asarray([row["expected_pairs"] for row in summaries], dtype=np.int64)
    reachable = np.asarray([row["reachable_pairs"] for row in summaries], dtype=np.int64)
    finite = means[np.isfinite(means)]
    meta = {
        "topology": "fixed_one_right_one_left_oracle_designed_at_20760s",
        "edge_csv": str(Path(args.edge_csv)),
        "num_edges": int(edge_table.num_edges),
        "start": int(args.start),
        "end": int(args.end),
        "actual_start_step": int(steps[0]) if steps else None,
        "actual_end_step": int(steps[-1]) if steps else None,
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "source_group": int(args.source_group),
        "source_group_name": group_name(int(args.source_group)),
        "target_group": int(args.target_group),
        "target_group_name": group_name(int(args.target_group)),
        "full_reachable_steps": int(np.sum(reachable == expected)),
        "min_reachable_pairs": int(np.min(reachable)) if reachable.size else 0,
        "mean_delay_ms": float(np.mean(finite)) if finite.size else math.nan,
        "min_delay_ms": float(np.min(finite)) if finite.size else math.nan,
        "max_delay_ms": float(np.max(finite)) if finite.size else math.nan,
        "elapsed_s": float(time.time() - started_at),
        "plot": str(plot_path) if plot_path else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
