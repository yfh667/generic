from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.delay_store import build_or_load_artifacts, write_edge_index_matrix, write_query_meta
from src.link_delay.module.position_cache import PositionCacheStore
from src.link_delay.module.query import FullLinkDelayStore
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.satellite_topology_viewer.module.topology_edges import build_full_option_plus_intra_edges
from src.topology_workflow.module.batch_shortest_hops import all_pairs_hop_dist
from src.topology_workflow.module.shortest_delay import (
    build_bidirectional_sparse_parts,
    build_weight_lookup,
    edge_weights_for_step,
)


STATION_GROUPS = {
    0: {"name": "America", "stations": list(range(0, 5))},
    1: {"name": "Africa", "stations": list(range(5, 11))},
    2: {"name": "China", "stations": list(range(11, 23))},
    3: {"name": "Europe", "stations": list(range(23, 31))},
}
GROUP_COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FFA500"]
PAIR_SPECS = {
    "china_europe": (2, 3, "China-Europe"),
    "china_america": (2, 0, "China-America"),
    "china_africa": (2, 1, "China-Africa"),
}


DEFAULT_POSITION_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "Starlink_72_22_1_550"
    / "satellitesposition"
    / "_position_cache"
    / "cache_0_86160_1s"
)
DEFAULT_XML_FILE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "Starlink_72_22_1_550"
    / "satellitesposition"
    / "station_visible_satellites_baseRaan_1.xml"
)
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "cache" / "group_data" / "Starlink_72_22_1_550"
DEFAULT_DELAY_OUTPUT_BASE = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "cache"
    / "link_delay"
    / "Starlink_72_22_1_550"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "Starlink_72_22_1_550"
    / "plus_grid_region_pairs_metrics"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Starlink plus-grid China-region-pair shortest delay and hop metrics."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE_DIR)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML_FILE)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--delay-store-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-steps", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--force-delay-store", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    return parser.parse_args(argv)


def finite_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "min": None, "max": None, "finite_steps": 0}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "finite_steps": int(finite.size),
    }


def group_nodes(group_data: dict[int, dict], step: int, group_id: int) -> list[int]:
    return sorted(int(x) for x in group_data.get(int(step), {}).get("groups", {}).get(int(group_id), set()))


def summarize_values(values: np.ndarray, required_pairs: int) -> tuple[float, float, float, int]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return math.nan, math.nan, math.nan, 0
    return float(np.mean(finite)), float(np.min(finite)), float(np.max(finite)), int(finite.size)


def ensure_option0_delay_store(
    *,
    config: ViewerConfig,
    position_cache_dir: Path,
    delay_store_dir: Path,
    start: int,
    end: int,
    stride: int,
    chunk_steps: int,
    force: bool,
) -> None:
    artifacts = build_or_load_artifacts(
        cache_dir=position_cache_dir,
        out_dir=delay_store_dir,
        config=config,
        start=int(start),
        end=int(end),
        stride=int(stride),
        force=bool(force),
        chunk_steps=int(chunk_steps),
        allow_incomplete_cache=False,
        options=(0,),
        wrap_planes=True,
    )
    edge_index_path = write_edge_index_matrix(artifacts, total_sats=config.total_sats, directed_storage=False)
    write_query_meta(artifacts, edge_index_path, options=(0,), directed_storage=False, wrap_planes=True)


def metrics_complete(out_dir: Path, steps: list[int]) -> bool:
    expected = np.asarray(steps, dtype=np.int64)
    for pair_key in PAIR_SPECS:
        csv_path = out_dir / f"{pair_key}_metrics.csv"
        npy_delay = out_dir / f"{pair_key}_mean_shortest_delay_ms.npy"
        npy_hops = out_dir / f"{pair_key}_mean_shortest_hops.npy"
        if not (csv_path.exists() and npy_delay.exists() and npy_hops.exists()):
            return False
        try:
            saved_steps = np.load(out_dir / "time_indices.npy", mmap_mode="r")
            if not np.array_equal(np.asarray(saved_steps, dtype=np.int64), expected):
                return False
            if np.load(npy_delay, mmap_mode="r").shape != (len(steps),):
                return False
            if np.load(npy_hops, mmap_mode="r").shape != (len(steps),):
                return False
        except Exception:
            return False
    return True


def write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "hour",
        "source_nodes",
        "target_nodes",
        "required_pairs",
        "delay_reachable_pairs",
        "mean_shortest_delay_ms",
        "min_shortest_delay_ms",
        "max_shortest_delay_ms",
        "hop_reachable_pairs",
        "mean_shortest_hops",
        "min_shortest_hops",
        "max_shortest_hops",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(out_dir: Path, steps: np.ndarray, pair_arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_hours = np.asarray(steps, dtype=np.float64) / 3600.0
    colors = {
        "china_europe": "#2563eb",
        "china_america": "#dc2626",
        "china_africa": "#16a34a",
    }

    def line_plot(metric: str, ylabel: str, title: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(15, 6.6), dpi=180)
        for key, (_src, _dst, label) in PAIR_SPECS.items():
            ax.plot(x_hours, pair_arrays[key][metric], linewidth=1.05, color=colors[key], label=label)
        ax.set_xlabel("time (hour)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return path

    delay_plot = line_plot(
        "delay",
        "mean shortest delay (ms)",
        "Starlink 72x22 plus-grid mean shortest delay",
        "starlink_plus_grid_mean_shortest_delay_ms_three_pairs.png",
    )
    hops_plot = line_plot(
        "hops",
        "mean shortest path (hops)",
        "Starlink 72x22 plus-grid mean shortest hops",
        "starlink_plus_grid_mean_shortest_hops_three_pairs.png",
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 11.5), dpi=180, sharex=True)
    for row, (key, (_src, _dst, label)) in enumerate(PAIR_SPECS.items()):
        axes[row, 0].plot(x_hours, pair_arrays[key]["delay"], color=colors[key], linewidth=1.0)
        axes[row, 0].set_ylabel("delay (ms)")
        axes[row, 0].set_title(f"{label} shortest delay")
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 1].plot(x_hours, pair_arrays[key]["hops"], color=colors[key], linewidth=1.0)
        axes[row, 1].set_ylabel("hops")
        axes[row, 1].set_title(f"{label} shortest hops")
        axes[row, 1].grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("time (hour)")
    axes[-1, 1].set_xlabel("time (hour)")
    fig.suptitle("Starlink 72x22 plus-grid region-pair shortest metrics", y=0.995)
    fig.tight_layout()
    panel_plot = out_dir / "starlink_plus_grid_delay_hops_three_pairs_panel.png"
    fig.savefig(panel_plot)
    plt.close(fig)
    return {"delay_plot": str(delay_plot), "hops_plot": str(hops_plot), "panel_plot": str(panel_plot)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.end < args.start:
        raise ValueError("--end must be >= --start")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    config = ViewerConfig(
        name="Starlink_72_22",
        P=72,
        N=22,
        station_groups=STATION_GROUPS,
        group_colors=GROUP_COLORS,
    )
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    delay_store_dir = Path(args.delay_store_dir) if args.delay_store_dir is not None else (
        DEFAULT_DELAY_OUTPUT_BASE
        / f"Starlink_72_22_option0_wrap_t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    if metrics_complete(out_dir, steps) and not bool(args.force_metrics):
        print(f"[starlink-plus-grid-metrics] Reusing existing metrics in {out_dir}", flush=True)
        pair_arrays = {
            key: {
                "delay": np.asarray(np.load(out_dir / f"{key}_mean_shortest_delay_ms.npy"), dtype=np.float32),
                "hops": np.asarray(np.load(out_dir / f"{key}_mean_shortest_hops.npy"), dtype=np.float32),
            }
            for key in PAIR_SPECS
        }
        plots = plot_outputs(out_dir, np.asarray(steps, dtype=np.int64), pair_arrays)
        print(json.dumps({"out_dir": str(out_dir), **plots}, ensure_ascii=False, indent=2), flush=True)
        return 0

    ensure_option0_delay_store(
        config=config,
        position_cache_dir=Path(args.position_cache_dir),
        delay_store_dir=delay_store_dir,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
        chunk_steps=int(args.chunk_steps),
        force=bool(args.force_delay_store),
    )
    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )

    edge_table = build_full_option_plus_intra_edges(
        config,
        inter_options=(0,),
        include_intra=True,
        wrap_planes=True,
    )
    delay_store = FullLinkDelayStore(delay_store_dir)
    position_store = PositionCacheStore(args.position_cache_dir)
    delay_rows = delay_store.rows_for_interval(int(args.start), int(args.end), int(args.stride))
    position_rows = position_store.rows_for_interval(int(args.start), int(args.end), int(args.stride))
    lookup = build_weight_lookup(edge_table, delay_store, config=config, allow_intra_fallback=True)
    row_index, col_index = build_bidirectional_sparse_parts(edge_table)
    hop_dist = all_pairs_hop_dist(edge_table, int(config.total_sats))

    pair_arrays = {
        key: {
            "delay": np.full(len(steps), np.nan, dtype=np.float32),
            "hops": np.full(len(steps), np.nan, dtype=np.float32),
            "rows": [],
        }
        for key in PAIR_SPECS
    }

    print(
        f"[starlink-plus-grid-metrics] steps={len(steps)} edges={edge_table.num_edges} "
        f"delay_store_edges={lookup.topology_edge_indices.size} "
        f"intra_position_fallback={lookup.fallback_intra_edge_indices.size}",
        flush=True,
    )
    started = time.time()
    for idx, step in enumerate(steps):
        china_nodes = group_nodes(group_data, int(step), 2)
        target_nodes_by_pair = {
            key: group_nodes(group_data, int(step), target_group)
            for key, (_source_group, target_group, _label) in PAIR_SPECS.items()
        }

        weights = None
        dist_matrix = None
        if china_nodes:
            nonempty_targets = sorted({node for nodes in target_nodes_by_pair.values() for node in nodes})
            if nonempty_targets:
                weights = edge_weights_for_step(
                    edge_table=edge_table,
                    lookup=lookup,
                    delay_store=delay_store,
                    position_store=position_store,
                    delay_row=int(delay_rows[idx]),
                    position_row=int(position_rows[idx]),
                )
                graph = csr_matrix(
                    (np.concatenate([weights, weights]), (row_index, col_index)),
                    shape=(int(config.total_sats), int(config.total_sats)),
                )
                dist_matrix = np.atleast_2d(
                    np.asarray(
                        dijkstra(
                            graph,
                            directed=True,
                            indices=np.asarray(china_nodes, dtype=np.int32),
                            return_predecessors=False,
                        ),
                        dtype=np.float64,
                    )
                )

        source_idx = np.asarray(china_nodes, dtype=np.int32)
        for key, (_source_group, _target_group, _label) in PAIR_SPECS.items():
            targets = target_nodes_by_pair[key]
            target_idx = np.asarray(targets, dtype=np.int32)
            required_pairs = int(len(china_nodes) * len(targets))

            if dist_matrix is not None and target_idx.size:
                delay_values = dist_matrix[:, target_idx]
                delay_mean, delay_min, delay_max, delay_reachable = summarize_values(delay_values, required_pairs)
            else:
                delay_mean, delay_min, delay_max, delay_reachable = math.nan, math.nan, math.nan, 0

            if source_idx.size and target_idx.size:
                hop_values = hop_dist[source_idx[:, None], target_idx].astype(np.float64, copy=False)
                hop_mean, hop_min, hop_max, hop_reachable = summarize_values(hop_values, required_pairs)
            else:
                hop_mean, hop_min, hop_max, hop_reachable = math.nan, math.nan, math.nan, 0

            pair_arrays[key]["delay"][idx] = delay_mean
            pair_arrays[key]["hops"][idx] = hop_mean
            pair_arrays[key]["rows"].append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "source_nodes": int(len(china_nodes)),
                    "target_nodes": int(len(targets)),
                    "required_pairs": int(required_pairs),
                    "delay_reachable_pairs": int(delay_reachable),
                    "mean_shortest_delay_ms": None if not math.isfinite(delay_mean) else float(delay_mean),
                    "min_shortest_delay_ms": None if not math.isfinite(delay_min) else float(delay_min),
                    "max_shortest_delay_ms": None if not math.isfinite(delay_max) else float(delay_max),
                    "hop_reachable_pairs": int(hop_reachable),
                    "mean_shortest_hops": None if not math.isfinite(hop_mean) else float(hop_mean),
                    "min_shortest_hops": None if not math.isfinite(hop_min) else float(hop_min),
                    "max_shortest_hops": None if not math.isfinite(hop_max) else float(hop_max),
                }
            )

        if int(args.progress_every) > 0 and ((idx + 1) % int(args.progress_every) == 0 or idx + 1 == len(steps)):
            elapsed = time.time() - started
            eta = elapsed / max(1, idx + 1) * (len(steps) - idx - 1)
            print(
                f"[starlink-plus-grid-metrics] {idx + 1}/{len(steps)} step={step} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    for key in PAIR_SPECS:
        np.save(out_dir / f"{key}_mean_shortest_delay_ms.npy", pair_arrays[key]["delay"])
        np.save(out_dir / f"{key}_mean_shortest_hops.npy", pair_arrays[key]["hops"])
        write_pair_csv(out_dir / f"{key}_metrics.csv", pair_arrays[key]["rows"])

    compare_path = out_dir / "compare_region_pairs_metrics.csv"
    with compare_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["step", "hour"]
        for key in PAIR_SPECS:
            fieldnames.extend([f"{key}_delay_ms", f"{key}_hops"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            row: dict[str, Any] = {"step": int(step), "hour": float(step) / 3600.0}
            for key in PAIR_SPECS:
                row[f"{key}_delay_ms"] = float(pair_arrays[key]["delay"][idx])
                row[f"{key}_hops"] = float(pair_arrays[key]["hops"][idx])
            writer.writerow(row)

    plots = plot_outputs(out_dir, np.asarray(steps, dtype=np.int64), pair_arrays)
    summary = {
        "constellation": config.name,
        "topology": "plus_grid_option0_plus_intra_wrap",
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "num_steps": len(steps),
        "position_cache_dir": str(Path(args.position_cache_dir)),
        "delay_store_dir": str(delay_store_dir),
        "group_cache_dir": str(Path(args.group_cache_dir)),
        "pair_summary": {
            key: {
                "label": PAIR_SPECS[key][2],
                "delay_ms": finite_summary(pair_arrays[key]["delay"]),
                "hops": finite_summary(pair_arrays[key]["hops"]),
            }
            for key in PAIR_SPECS
        },
        "compare_csv": str(compare_path),
        **plots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
