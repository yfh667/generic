from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_m56_local_patch_hybrid_topology import DEFAULT_CONFIG, path_from, region_pairs_from_workflow  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.config import load_workflow_yaml, viewer_config_from_workflow  # noqa: E402


DEFAULT_DYNAMIC_DIR = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3\shortest_hops_t0_86160_stride60")
    / "dynamic_splice_topologies"
    / "dyn_m056_m040_ca_b6-12-18_c_t0_86100_s1_sel-transition-dp_pen0p001"
)
DEFAULT_LST_DIR = DEFAULT_DYNAMIC_DIR / "lst30s_backward"
DEFAULT_DELAY_STORE = Path(r"E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LST-constrained dynamic-splice topology metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--lst-dir", type=Path, default=DEFAULT_LST_DIR)
    parser.add_argument("--delay-store", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--delay-sample-stride", type=int, default=60)
    parser.add_argument("--skip-delay", action="store_true")
    parser.add_argument("--pairs", nargs="+", default=None)
    return parser.parse_args()


def finite_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def read_union_edges(path: Path) -> tuple[np.ndarray, np.ndarray]:
    src: list[int] = []
    dst: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src.append(int(row["src_node"]))
            dst.append(int(row["dst_node"]))
    return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32)


def read_step_stats(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: int(value) for key, value in row.items()})
    return rows


def read_dynamic_schedule(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = dict(row)
            for key in (
                "step",
                "candidate_idx",
                "band_len",
                "added_edges",
                "removed_edges",
                "changed_edges",
                "new_edges_from_prev",
            ):
                if key in out and out[key] != "":
                    out[key] = int(out[key])
            for key, value in list(out.items()):
                if key.endswith("_mean_hops") or key.endswith("_delta_vs_base") or key.endswith("_improvement_vs_base"):
                    out[key] = float(value)
            rows.append(out)
    return rows


def mean_between_groups(dist: np.ndarray, sources: tuple[int, ...], targets: tuple[int, ...]) -> tuple[float, int, int]:
    if not sources or not targets:
        return math.nan, 0, int(len(sources) * len(targets))
    block = dist[np.ix_(np.asarray(sources, dtype=np.int32), np.asarray(targets, dtype=np.int32))]
    finite = np.isfinite(block) & (block > 0)
    reachable = int(np.count_nonzero(finite))
    total = int(block.size)
    if reachable == 0:
        return math.nan, 0, total
    return float(np.mean(block[finite])), reachable, total


def build_sparse_graph(
    *,
    total_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    active_cols: np.ndarray,
    weights: np.ndarray | None = None,
) -> csr_matrix:
    u = src[active_cols]
    v = dst[active_cols]
    if weights is None:
        data = np.ones(u.size * 2, dtype=np.float32)
    else:
        data = np.asarray(np.concatenate([weights, weights]), dtype=np.float32)
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    return csr_matrix((data, (rows, cols)), shape=(int(total_nodes), int(total_nodes)))


def load_group_data(*, workflow: dict[str, Any], config: Any, steps: list[int], stride: int) -> dict:
    paths_raw = workflow.get("paths", {})
    return load_or_build_group_data(
        xml_file=path_from(paths_raw, "group_xml"),
        group_cache_dir=path_from(paths_raw, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(stride),
        enabled=True,
        force=False,
    )


def compute_hop_metrics(
    *,
    out_csv: Path,
    steps: np.ndarray,
    active_mask: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    config: Any,
    group_data: dict,
    pairs: dict[str, Any],
) -> dict[str, Any]:
    packed = np.packbits(active_mask, axis=1)
    unique_rows, inverse = np.unique(packed, axis=0, return_inverse=True)
    dist_by_state: dict[int, np.ndarray] = {}
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pair_values: dict[str, list[float]] = defaultdict(list)
    pair_reachable: dict[str, list[int]] = defaultdict(list)
    pair_total: dict[str, list[int]] = defaultdict(list)

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["step", "pair", "source_nodes", "target_nodes", "reachable_pairs", "total_pairs", "mean_hops"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, step_raw in enumerate(steps.tolist()):
            state = int(inverse[row_idx])
            if state not in dist_by_state:
                unpacked = np.unpackbits(unique_rows[state], count=active_mask.shape[1]).astype(bool)
                cols = np.flatnonzero(unpacked)
                graph = build_sparse_graph(total_nodes=config.total_sats, src=src, dst=dst, active_cols=cols)
                dist_by_state[state] = np.asarray(shortest_path(graph, directed=False, unweighted=True), dtype=np.float32)

            dist = dist_by_state[state]
            step = int(step_raw)
            for pair_key, pair in pairs.items():
                sources = group_nodes_for_step(group_data, step, int(pair.source_group_id))
                targets = group_nodes_for_step(group_data, step, int(pair.target_group_id))
                mean_hops, reachable, total = mean_between_groups(dist, sources, targets)
                pair_values[pair_key].append(mean_hops)
                pair_reachable[pair_key].append(reachable)
                pair_total[pair_key].append(total)
                writer.writerow(
                    {
                        "step": step,
                        "pair": pair_key,
                        "source_nodes": len(sources),
                        "target_nodes": len(targets),
                        "reachable_pairs": reachable,
                        "total_pairs": total,
                        "mean_hops": mean_hops,
                    }
                )

    return {
        "num_unique_active_topology_states": int(unique_rows.shape[0]),
        "pairs": {
            pair_key: {
                **finite_stats(values),
                "reachable_pairs_mean": float(np.mean(pair_reachable[pair_key])),
                "total_pairs_mean": float(np.mean(pair_total[pair_key])),
            }
            for pair_key, values in pair_values.items()
        },
    }


def compute_delay_metrics(
    *,
    out_csv: Path,
    steps: np.ndarray,
    active_mask: np.ndarray,
    step_stats: list[dict[str, int]],
    src: np.ndarray,
    dst: np.ndarray,
    config: Any,
    group_data: dict,
    pairs: dict[str, Any],
    delay_store: Path,
    sample_stride: int,
) -> dict[str, Any]:
    delay_path = delay_store / "edge_delay_ms.npy"
    edge_index_path = delay_store / "edge_index_matrix.npy"
    if not delay_path.exists() or not edge_index_path.exists():
        raise FileNotFoundError(f"missing delay store files in {delay_store}")

    delay_ms = np.load(delay_path, mmap_mode="r")
    edge_index = np.load(edge_index_path, mmap_mode="r")
    delay_cols = np.asarray(edge_index[src, dst], dtype=np.int32)
    if np.any(delay_cols < 0):
        bad = int(np.flatnonzero(delay_cols < 0)[0])
        raise ValueError(f"union edge has no delay-store entry: col={bad}, edge=({src[bad]}, {dst[bad]})")

    building = np.asarray([row["building_edges"] for row in step_stats], dtype=np.int32)
    sample_mask = np.zeros(steps.size, dtype=bool)
    sample_mask[building > 0] = True
    if int(sample_stride) > 0:
        sample_mask[(steps % int(sample_stride)) == 0] = True
    sample_mask[0] = True
    sample_mask[-1] = True
    sample_rows = np.flatnonzero(sample_mask)

    pair_values: dict[str, list[float]] = defaultdict(list)
    pair_values_building: dict[str, list[float]] = defaultdict(list)
    pair_values_regular: dict[str, list[float]] = defaultdict(list)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "step",
            "sample_reason",
            "pair",
            "source_nodes",
            "target_nodes",
            "reachable_pairs",
            "total_pairs",
            "mean_delay_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pos, row_idx in enumerate(sample_rows.tolist(), 1):
            step = int(steps[row_idx])
            active_cols = np.flatnonzero(active_mask[row_idx])
            weights = np.asarray(delay_ms[step, delay_cols[active_cols]], dtype=np.float32)
            graph = build_sparse_graph(
                total_nodes=config.total_sats,
                src=src,
                dst=dst,
                active_cols=active_cols,
                weights=weights,
            )
            source_group_id = int(next(iter(pairs.values())).source_group_id)
            common_sources = group_nodes_for_step(group_data, step, source_group_id)
            if common_sources:
                dist_from_sources = np.asarray(
                    shortest_path(
                        graph,
                        directed=False,
                        unweighted=False,
                        indices=np.asarray(common_sources, dtype=np.int32),
                    ),
                    dtype=np.float64,
                )
            else:
                dist_from_sources = np.empty((0, int(config.total_sats)), dtype=np.float64)

            reason = "building" if int(building[row_idx]) > 0 else "regular"
            for pair_key, pair in pairs.items():
                sources = group_nodes_for_step(group_data, step, int(pair.source_group_id))
                targets = group_nodes_for_step(group_data, step, int(pair.target_group_id))
                if tuple(sources) != tuple(common_sources):
                    dist = np.asarray(shortest_path(graph, directed=False, unweighted=False), dtype=np.float64)
                    mean_delay, reachable, total = mean_between_groups(dist, sources, targets)
                elif not targets or dist_from_sources.size == 0:
                    mean_delay, reachable, total = math.nan, 0, int(len(sources) * len(targets))
                else:
                    block = dist_from_sources[:, np.asarray(targets, dtype=np.int32)]
                    finite = np.isfinite(block) & (block > 0)
                    reachable = int(np.count_nonzero(finite))
                    total = int(block.size)
                    mean_delay = float(np.mean(block[finite])) if reachable else math.nan
                pair_values[pair_key].append(mean_delay)
                if reason == "building":
                    pair_values_building[pair_key].append(mean_delay)
                else:
                    pair_values_regular[pair_key].append(mean_delay)
                writer.writerow(
                    {
                        "step": step,
                        "sample_reason": reason,
                        "pair": pair_key,
                        "source_nodes": len(sources),
                        "target_nodes": len(targets),
                        "reachable_pairs": reachable,
                        "total_pairs": total,
                        "mean_delay_ms": mean_delay,
                    }
                )

            if pos % 500 == 0 or pos == len(sample_rows):
                print(f"[lst-summary] delay samples {pos}/{len(sample_rows)} step={step}", flush=True)

    return {
        "sample_rows": int(sample_rows.size),
        "sample_stride_seconds": int(sample_stride),
        "building_sample_rows": int(np.count_nonzero(building[sample_rows] > 0)),
        "regular_sample_rows": int(np.count_nonzero(building[sample_rows] == 0)),
        "pairs": {
            pair_key: {
                "all_sampled": finite_stats(pair_values[pair_key]),
                "building_rows": finite_stats(pair_values_building[pair_key]),
                "regular_rows": finite_stats(pair_values_regular[pair_key]),
            }
            for pair_key in pair_values
        },
    }


def main() -> int:
    args = parse_args()
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.lst_dir) / "metric_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = np.load(Path(args.lst_dir) / "steps.npy").astype(np.int64)
    active_mask = np.load(Path(args.lst_dir) / "edge_active_mask.npy", mmap_mode="r")
    src, dst = read_union_edges(Path(args.lst_dir) / "union_edges.csv")
    step_stats = read_step_stats(Path(args.lst_dir) / "lst_step_stats.csv")
    dynamic_rows = read_dynamic_schedule(Path(args.dynamic_dir) / "dynamic_schedule.csv")

    group_data = load_group_data(workflow=workflow, config=config, steps=steps.astype(int).tolist(), stride=1)
    pair_by_key = region_pairs_from_workflow(workflow)
    pair_keys = tuple(str(x) for x in (args.pairs if args.pairs else ("china_europe", "china_america", "china_africa")))
    missing_pairs = [key for key in pair_keys if key not in pair_by_key]
    if missing_pairs:
        raise ValueError(f"unknown pair(s) {missing_pairs}; available={sorted(pair_by_key)}")
    pairs = {key: pair_by_key[key] for key in pair_keys}

    link_stats: dict[str, Any] = {}
    for key in ("target_active_edges", "active_edges", "building_edges", "active_dropped_by_building", "building_dropped_by_conflict"):
        values = [float(row[key]) for row in step_stats]
        link_stats[key] = finite_stats(values)
        link_stats[key]["nonzero_rows"] = int(sum(1 for v in values if v > 0))
        link_stats[key]["sum_edge_seconds"] = float(sum(values))

    if dynamic_rows and "expanded_from_step" in dynamic_rows[0]:
        transition_rows = [
            row
            for row in dynamic_rows
            if str(row.get("expanded_from_step", "")) == str(row.get("step", ""))
        ]
    else:
        transition_rows = dynamic_rows
    new_edges_values = [float(row.get("new_edges_from_prev", 0)) for row in transition_rows]
    link_stats["dynamic_schedule_new_edges_from_prev"] = finite_stats(new_edges_values)
    link_stats["dynamic_schedule_new_edges_from_prev"]["nonzero_rows"] = int(sum(1 for v in new_edges_values if v > 0))
    link_stats["dynamic_schedule_new_edges_from_prev"]["counted_transition_rows"] = int(len(transition_rows))

    coarse_hops: dict[str, dict[str, Any]] = {}
    for pair_key in pair_keys:
        dyn_key = f"{pair_key}_mean_hops"
        base_key = f"{pair_key}_base_mean_hops"
        if not dynamic_rows or dyn_key not in dynamic_rows[0] or base_key not in dynamic_rows[0]:
            coarse_hops[pair_key] = {
                "dynamic_target_topology": None,
                "base_000056": None,
                "mean_improvement_vs_base": None,
                "note": f"dynamic_schedule.csv has no {dyn_key}/{base_key} columns",
            }
            continue
        dyn_values = [float(row[dyn_key]) for row in dynamic_rows]
        base_values = [float(row[base_key]) for row in dynamic_rows]
        coarse_hops[pair_key] = {
            "dynamic_target_topology": finite_stats(dyn_values),
            "base_000056": finite_stats(base_values),
            "mean_improvement_vs_base": float(np.mean(np.asarray(base_values) - np.asarray(dyn_values))),
        }

    hop_summary = compute_hop_metrics(
        out_csv=out_dir / "lst_active_hops_timeseries.csv",
        steps=steps,
        active_mask=active_mask,
        src=src,
        dst=dst,
        config=config,
        group_data=group_data,
        pairs=pairs,
    )

    delay_summary = None
    if not args.skip_delay:
        delay_summary = compute_delay_metrics(
            out_csv=out_dir / "lst_active_delay_sample_timeseries.csv",
            steps=steps,
            active_mask=active_mask,
            step_stats=step_stats,
            src=src,
            dst=dst,
            config=config,
            group_data=group_data,
            pairs=pairs,
            delay_store=Path(args.delay_store),
            sample_stride=int(args.delay_sample_stride),
        )

    summary = {
        "inputs": {
            "dynamic_dir": str(args.dynamic_dir),
            "lst_dir": str(args.lst_dir),
            "delay_store": None if args.skip_delay else str(args.delay_store),
            "delay_sample_stride": int(args.delay_sample_stride),
        },
        "time_axis": {
            "start": int(steps[0]),
            "end": int(steps[-1]),
            "num_steps": int(steps.size),
            "stride_seconds": int(steps[1] - steps[0]) if steps.size > 1 else None,
        },
        "link_setup": link_stats,
        "coarse_60s_target_hops_from_dynamic_schedule": coarse_hops,
        "lst_active_hops_full_1s": hop_summary,
        "lst_active_delay_ms_sampled": delay_summary,
        "outputs": {
            "summary_json": str(out_dir / "lst_dynamic_metric_summary.json"),
            "hop_timeseries_csv": str(out_dir / "lst_active_hops_timeseries.csv"),
            "delay_sample_csv": None if args.skip_delay else str(out_dir / "lst_active_delay_sample_timeseries.csv"),
        },
    }

    (out_dir / "lst_dynamic_metric_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
