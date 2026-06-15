from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from apply_lst_to_dynamic_splice import subset_edge_table  # noqa: E402
from run_m56_m40_dynamic_splice_86100 import read_edge_table_csv  # noqa: E402

from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import LazyEdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.multi_viewer import Topology2DPanel, UnifiedControlTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.edge_betweenness import (  # noqa: E402
    build_undirected_adjacency,
    edge_betweenness_between_node_sets,
)
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import (  # noqa: E402
    TopologySpec,
    topology_specs_from_motif_csv,
)
from src.topology_workflow.module.config import (  # noqa: E402
    load_workflow_yaml,
    time_axis_from_config,
    viewer_config_from_workflow,
)


DEFAULT_CONFIG = (
    GENERIC_ROOT
    / "paper1notebook"
    / "pipeline"
    / "configs"
    / "g60_w_le4_h_le3_shortest_hops.yaml"
)
DEFAULT_DYNAMIC_LST_DIR = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3")
    / "shortest_hops_t0_86160_stride60"
    / "dynamic_splice_topologies"
    / "dyn_m056_m040_ca_b6-12-18_c_t0_86100_s1_sel-transition-dp_pen0p001"
    / "lst30s_backward"
)
DEFAULT_SCREENSHOT = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\lst30_dp_c_pen001_stats")
    / "viewer_motif56_vs_dynamic_china_europe"
    / "motif56_vs_dynamic_lst30_china_europe_t0.png"
)


@dataclass(frozen=True)
class RegionPair:
    key: str
    label: str
    source_group_id: int
    target_group_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronized 2D viewer: fixed motif000056 vs dynamic LST30, "
            "with China-Europe shortest-path edge betweenness overlay."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dynamic-lst-dir", type=Path, default=DEFAULT_DYNAMIC_LST_DIR)
    parser.add_argument("--motif", default="56")
    parser.add_argument("--pair", default="china_europe")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--show-panel-controls", action="store_true")
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--initial-value-max", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def path_from(raw: dict[str, Any], key: str) -> Path:
    value = raw.get(key)
    if value in (None, ""):
        raise ValueError(f"missing paths.{key} in workflow config")
    return Path(str(value))


def normalize_motif_name(raw: str, *, name_prefix: str) -> str:
    token = str(raw).strip()
    if not token:
        raise ValueError("empty motif token")
    if token.isdigit():
        return f"{name_prefix}_motif_{int(token):06d}"
    if token.startswith("motif_") and token.removeprefix("motif_").isdigit():
        return f"{name_prefix}_{token}"
    if token.startswith(f"{name_prefix}_motif_"):
        return token
    if token.startswith("combined_motif_"):
        return token
    raise ValueError(f"unsupported motif token {raw!r}")


def region_pairs_from_workflow(raw: dict[str, Any]) -> dict[str, RegionPair]:
    out: dict[str, RegionPair] = {}
    for item in raw.get("region_pairs", []) or []:
        pair = RegionPair(
            key=str(item["key"]),
            label=str(item.get("label", item["key"])),
            source_group_id=int(item["source_group_id"]),
            target_group_id=int(item["target_group_id"]),
        )
        out[pair.key] = pair
    return out


def select_dynamic_rows(dynamic_steps: np.ndarray, *, start: int, end: int, stride: int) -> np.ndarray:
    if int(stride) <= 0:
        raise ValueError("--stride must be positive")
    steps = np.asarray(dynamic_steps, dtype=np.int64)
    mask = (steps >= int(start)) & (steps <= int(end)) & (((steps - int(start)) % int(stride)) == 0)
    rows = np.flatnonzero(mask).astype(np.int64)
    if rows.size == 0:
        raise ValueError(f"no dynamic steps selected by start={start}, end={end}, stride={stride}")
    return rows


def load_static_motif_spec(workflow: dict[str, Any], config, motif: str) -> TopologySpec:
    paths_raw = workflow.get("paths", {})
    motif_raw = workflow.get("motif_library", {})
    name_prefix = str(motif_raw.get("name_prefix", "combined"))
    motif_csv = path_from(paths_raw, "motif_library_dir") / str(motif_raw.get("csv_name"))
    wanted_name = normalize_motif_name(str(motif), name_prefix=name_prefix)
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library=name_prefix,
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=False,
    )
    by_name = {spec.name: spec for spec in specs}
    if wanted_name not in by_name:
        raise ValueError(f"motif {wanted_name!r} not found in {motif_csv}")
    return by_name[wanted_name]


def make_static_provider(*, spec: TopologySpec, config, pair: RegionPair, group_data: dict):
    adjacency = build_undirected_adjacency(spec.edge_table, int(config.total_sats))
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[np.ndarray, str]] = {}

    def provider(_row: int, step: int) -> tuple[np.ndarray, str]:
        source_nodes = group_nodes_for_step(group_data, int(step), int(pair.source_group_id))
        target_nodes = group_nodes_for_step(group_data, int(step), int(pair.target_group_id))
        key = (source_nodes, target_nodes)
        cached = cache.get(key)
        if cached is None:
            values, summary, _samples = edge_betweenness_between_node_sets(
                spec.edge_table,
                total_nodes=int(config.total_sats),
                source_nodes=source_nodes,
                target_nodes=target_nodes,
                adjacency=adjacency,
                sample_path_limit=0,
            )
            status = (
                f"static motif000056 | cached states {len(cache) + 1} | "
                f"src {summary.source_nodes} tgt {summary.target_nodes} | "
                f"pairs {summary.reachable_pairs} | mean hops {summary.mean_shortest_distance_hops:.3f} | "
                f"max edge {summary.max_edge_betweenness:.3f}"
            )
            cached = (values, status)
            cache[key] = cached
        return cached

    return provider


def make_dynamic_provider(*, edge_table, active_mask: np.ndarray, config, pair: RegionPair, group_data: dict):
    cache: dict[int, tuple[np.ndarray, str]] = {}

    def provider(row: int, step: int) -> tuple[np.ndarray, str]:
        row = int(row)
        cached = cache.get(row)
        if cached is not None:
            return cached
        active_cols = np.flatnonzero(active_mask[row])
        active_table = subset_edge_table(edge_table, active_cols)
        adjacency = build_undirected_adjacency(active_table, int(config.total_sats))
        source_nodes = group_nodes_for_step(group_data, int(step), int(pair.source_group_id))
        target_nodes = group_nodes_for_step(group_data, int(step), int(pair.target_group_id))
        local_values, summary, _samples = edge_betweenness_between_node_sets(
            active_table,
            total_nodes=int(config.total_sats),
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            adjacency=adjacency,
            sample_path_limit=0,
        )
        values = np.zeros(int(edge_table.num_edges), dtype=np.float32)
        values[active_cols] = local_values
        status = (
            f"dynamic LST30 active | cached rows {len(cache) + 1} | active {active_cols.size} | "
            f"src {summary.source_nodes} tgt {summary.target_nodes} | pairs {summary.reachable_pairs} | "
            f"mean hops {summary.mean_shortest_distance_hops:.3f} | max edge {summary.max_edge_betweenness:.3f}"
        )
        payload = (values, status)
        cache[row] = payload
        return payload

    return provider


def main() -> int:
    args = parse_args()
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    _cfg_start, _cfg_end, cfg_stride = time_axis_from_config(workflow)
    stride = int(args.stride if args.stride is not None else cfg_stride)
    if stride <= 0:
        raise ValueError("--stride must be positive")

    paths_raw = workflow.get("paths", {})
    pair_by_key = region_pairs_from_workflow(workflow)
    if args.pair not in pair_by_key:
        raise ValueError(f"unknown pair {args.pair!r}; available={sorted(pair_by_key)}")
    pair = pair_by_key[str(args.pair)]

    motif_spec = load_static_motif_spec(workflow, config, str(args.motif))
    dynamic_dir = Path(args.dynamic_lst_dir)
    dynamic_steps_all = np.load(dynamic_dir / "steps.npy")
    row_indices = select_dynamic_rows(dynamic_steps_all, start=int(args.start), end=int(args.end), stride=stride)
    steps = [int(dynamic_steps_all[idx]) for idx in row_indices]
    full_range = row_indices.size == dynamic_steps_all.size and int(row_indices[0]) == 0 and int(row_indices[-1]) == dynamic_steps_all.size - 1

    dynamic_edge_table = read_edge_table_csv(dynamic_dir / "union_edges.csv", total_sats=int(config.total_sats))
    active_all = np.load(dynamic_dir / "edge_active_mask.npy", mmap_mode="r")
    building_all = np.load(dynamic_dir / "edge_building_mask.npy", mmap_mode="r")
    if full_range:
        dynamic_active_mask = active_all
        dynamic_building_mask = building_all
    else:
        dynamic_active_mask = np.asarray(active_all[row_indices], dtype=bool)
        dynamic_building_mask = np.asarray(building_all[row_indices], dtype=bool)

    group_data = load_or_build_group_data(
        xml_file=path_from(paths_raw, "group_xml"),
        group_cache_dir=path_from(paths_raw, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=True,
        force=bool(args.force_group_cache),
    )

    print(
        "[motif56-vs-dynamic-viewer] "
        f"steps={len(steps)} range={steps[0]}..{steps[-1]} stride={stride} | "
        f"motif_edges={motif_spec.edge_table.num_edges} dynamic_union_edges={dynamic_edge_table.num_edges} | "
        f"pair={pair.label}",
        flush=True,
    )

    if args.check_only:
        return 0

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = DEFAULT_SCREENSHOT
    if screenshot is not None:
        screenshot = Path(screenshot)
        screenshot.parent.mkdir(parents=True, exist_ok=True)

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    motif_viewer = LazyEdgeUsageTopology2DViewer(
        config,
        steps=steps,
        edge_table=motif_spec.edge_table,
        edge_usage_provider=make_static_provider(
            spec=motif_spec,
            config=config,
            pair=pair,
            group_data=group_data,
        ),
        window_title=f"{pair.label} motif000056 edge betweenness",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        initial_value_max=float(args.initial_value_max),
    )
    dynamic_viewer = LazyEdgeUsageTopology2DViewer(
        config,
        steps=steps,
        edge_table=dynamic_edge_table,
        edge_usage_provider=make_dynamic_provider(
            edge_table=dynamic_edge_table,
            active_mask=dynamic_active_mask,
            config=config,
            pair=pair,
            group_data=group_data,
        ),
        window_title=f"{pair.label} dynamic LST30 edge betweenness",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        initial_value_max=float(args.initial_value_max),
        edge_active_mask=dynamic_active_mask,
        edge_building_mask=dynamic_building_mask,
    )
    window = UnifiedControlTopology2DViewer(
        title=f"{config.name} motif000056 vs dynamic LST30 | {pair.label} edge betweenness",
        panels=[
            Topology2DPanel("static motif000056", motif_viewer),
            Topology2DPanel("dynamic: 000056 + 000040(C patch), DP, LST=30s", dynamic_viewer),
        ],
        shared_value_scale=True,
    )
    return run_viewer_widget(
        window,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
