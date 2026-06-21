from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.multi_viewer import (  # noqa: E402
    Topology2DPanel,
    UnifiedControlTopology2DViewer,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION  # noqa: E402


DEFAULT_EXPERIMENT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\full_link_guided_one_right_stride1"
    r"\t0_7200_stride1_online_fast"
)
DEFAULT_FULL_LINK_CACHE = Path(r"E:\paper11\data\linshi\g60_full_link_multi_region_weighted_betweenness_t0_86164_stride1")
DEFAULT_GROUP_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show ideal dynamic one-right topology and LST-constrained active/building links "
            "in the shared 2D topology viewer."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--full-link-cache", type=Path, default=DEFAULT_FULL_LINK_CACHE)
    parser.add_argument("--lst", type=int, default=30)
    parser.add_argument("--step", type=int, default=None, help="Initial step. Defaults to max dropped-active step.")
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty edge CSV: {path}")
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=[str(i + 1) for i in range(int(total_sats))],
    )


def port_keys_by_edge(edge_table: EdgeTable) -> list[tuple[tuple[int, str], ...]]:
    out: list[tuple[tuple[int, str], ...]] = []
    for idx in range(int(edge_table.num_edges)):
        if int(edge_table.option[idx]) == INTRA_OPTION:
            out.append(tuple())
        else:
            out.append(((int(edge_table.src[idx]), "right"), (int(edge_table.dst[idx]), "left")))
    return out


def build_target_port_owner_by_row(
    *,
    target_mask: np.ndarray,
    ports_by_edge: list[tuple[tuple[int, str], ...]],
) -> list[dict[tuple[int, str], int]]:
    owners: list[dict[tuple[int, str], int]] = []
    for row in range(int(target_mask.shape[0])):
        owner: dict[tuple[int, str], int] = {}
        for edge in np.flatnonzero(target_mask[row]):
            for port in ports_by_edge[int(edge)]:
                owner[port] = int(edge)
        owners.append(owner)
    return owners


def build_setup_events(
    *,
    target_mask: np.ndarray,
    steps: np.ndarray,
    lst_s: int,
) -> dict[int, list[tuple[int, int]]]:
    events: dict[int, list[tuple[int, int]]] = {}
    if int(lst_s) <= 0:
        return events
    step_to_row = {int(step): idx for idx, step in enumerate(np.asarray(steps, dtype=np.int64).tolist())}
    for row in range(1, int(target_mask.shape[0])):
        activation_time = int(steps[row])
        added = np.flatnonzero(target_mask[row] & ~target_mask[row - 1])
        if added.size == 0:
            continue
        start_time = max(int(steps[0]), activation_time - int(lst_s))
        # This viewer is designed for 1s steps. Snap to the nearest available row if needed.
        if start_time not in step_to_row:
            start_time = int(steps[max(0, np.searchsorted(steps, start_time, side="left"))])
        events.setdefault(int(start_time), []).extend((activation_time, int(edge)) for edge in added.tolist())
    return events


def compute_lst_masks(
    *,
    target_mask: np.ndarray,
    steps: np.ndarray,
    ports_by_edge: list[tuple[tuple[int, str], ...]],
    lst_s: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    steps = np.asarray(steps, dtype=np.int64)
    target_owner = build_target_port_owner_by_row(target_mask=target_mask, ports_by_edge=ports_by_edge)
    start_events = build_setup_events(target_mask=target_mask, steps=steps, lst_s=int(lst_s))
    active_mask = np.asarray(target_mask, dtype=bool).copy()
    building_mask = np.zeros_like(active_mask, dtype=bool)
    current_events: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []

    for row, step in enumerate(steps.tolist()):
        current_events.extend(start_events.get(int(step), []))
        if current_events:
            current_events = [(deadline, edge) for deadline, edge in current_events if int(deadline) > int(step)]

        target_row = target_mask[row]
        building_port_owner: dict[tuple[int, str], int] = {}
        accepted_building_edges: set[int] = set()
        building_dropped = 0
        candidates = sorted(
            (
                (int(deadline), int(edge))
                for deadline, edge in current_events
                if not bool(target_row[int(edge)])
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _deadline, edge in candidates:
            ports = ports_by_edge[int(edge)]
            if not ports:
                continue
            if any(port in building_port_owner for port in ports):
                building_dropped += 1
                continue
            for port in ports:
                building_port_owner[port] = int(edge)
            accepted_building_edges.add(int(edge))

        dropped_target_edges: set[int] = set()
        for port, building_edge in building_port_owner.items():
            target_edge = target_owner[row].get(port)
            if target_edge is not None and int(target_edge) != int(building_edge):
                dropped_target_edges.add(int(target_edge))

        if accepted_building_edges:
            building_mask[row, np.asarray(sorted(accepted_building_edges), dtype=np.int32)] = True
        if dropped_target_edges:
            active_mask[row, np.asarray(sorted(dropped_target_edges), dtype=np.int32)] = False

        rows.append(
            {
                "step": int(step),
                "target_active_edges": int(np.count_nonzero(target_row)),
                "active_edges": int(np.count_nonzero(active_mask[row])),
                "building_edges": int(len(accepted_building_edges)),
                "active_dropped_by_building": int(len(dropped_target_edges)),
                "building_dropped_by_conflict": int(building_dropped),
            }
        )

    return active_mask, building_mask, pd.DataFrame(rows)


def load_or_build_lst_viewer_cache(
    *,
    experiment_dir: Path,
    edge_table: EdgeTable,
    steps: np.ndarray,
    target_mask: np.ndarray,
    lst_s: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Path]:
    cache_dir = experiment_dir / f"lst{int(lst_s):03d}_viewer_data"
    active_path = cache_dir / "edge_active_mask.npy"
    building_path = cache_dir / "edge_building_mask.npy"
    stats_path = cache_dir / "lst_step_stats.csv"
    meta_path = cache_dir / "meta.json"
    expected_shape = (int(steps.shape[0]), int(edge_table.num_edges))
    if not force and active_path.exists() and building_path.exists() and stats_path.exists():
        active = np.load(active_path, mmap_mode="r")
        building = np.load(building_path, mmap_mode="r")
        if active.shape == expected_shape and building.shape == expected_shape:
            return active, building, pd.read_csv(stats_path), cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    active, building, stats = compute_lst_masks(
        target_mask=target_mask,
        steps=steps,
        ports_by_edge=port_keys_by_edge(edge_table),
        lst_s=int(lst_s),
    )
    np.save(active_path, active)
    np.save(building_path, building)
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    meta = {
        "experiment_dir": str(experiment_dir),
        "lst_s": int(lst_s),
        "shape": list(expected_shape),
        "rule": (
            "Building edges are future target additions in [activation_time-LST, activation_time). "
            "Accepted building edges reserve right/left ports; conflicting current target edges are removed from active."
        ),
        "outputs": {
            "edge_active_mask": str(active_path),
            "edge_building_mask": str(building_path),
            "lst_step_stats": str(stats_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return active, building, stats, cache_dir


def make_panel(
    *,
    title: str,
    steps: list[int],
    edge_table: EdgeTable,
    active_mask: np.ndarray,
    building_mask: np.ndarray | None,
    group_data: dict,
) -> Topology2DPanel:
    values = np.zeros((1, int(edge_table.num_edges)), dtype=np.float32)
    viewer = EdgeUsageTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_usage_values=values,
        value_max=1.0,
        edge_active_mask=active_mask,
        edge_building_mask=building_mask,
        window_title=title,
        group_data=group_data,
        show_groups=True,
        show_topology_under_edge_values=True,
        zero_value_edges_visible=False,
        topology_edge_alpha=150,
        topology_edge_width=0.014,
        building_edge_color="#2563EB",
        building_edge_alpha=210,
        building_edge_width=0.035,
        show_grid_lines=False,
    )
    return Topology2DPanel(title=title, viewer=viewer, stretch=1)


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    edge_table = read_edge_table_csv(Path(args.full_link_cache) / "edges.csv", total_sats=int(G60_CONFIG.total_sats))
    steps = np.asarray(np.load(experiment_dir / "dynamic_one_right_time_indices.npy", mmap_mode="r"), dtype=np.int64)
    target_mask = np.load(experiment_dir / "dynamic_one_right_active_full_edge_mask.npy", mmap_mode="r")
    if target_mask.shape != (int(steps.shape[0]), int(edge_table.num_edges)):
        raise ValueError(f"target mask shape {target_mask.shape} != ({steps.shape[0]}, {edge_table.num_edges})")

    lst_active, lst_building, lst_stats, cache_dir = load_or_build_lst_viewer_cache(
        experiment_dir=experiment_dir,
        edge_table=edge_table,
        steps=steps,
        target_mask=np.asarray(target_mask, dtype=bool),
        lst_s=int(args.lst),
        force=bool(args.force),
    )
    initial_row = int(lst_stats["active_dropped_by_building"].to_numpy().argmax())
    if args.step is not None:
        matches = np.flatnonzero(steps == int(args.step))
        if matches.size == 0:
            raise ValueError(f"step {args.step} not found in {steps[0]}..{steps[-1]}")
        initial_row = int(matches[0])

    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=[int(x) for x in steps.tolist()],
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(steps[1] - steps[0]) if int(steps.shape[0]) > 1 else 1,
        enabled=True,
        force=False,
    )

    max_row = lst_stats.iloc[int(initial_row)]
    print(
        f"[dynamic-one-right-lst-viewer] experiment={experiment_dir} lst={int(args.lst)} "
        f"steps={steps[0]}..{steps[-1]} count={steps.shape[0]} cache={cache_dir}",
        flush=True,
    )
    print(
        f"[dynamic-one-right-lst-viewer] initial_step={int(steps[initial_row])} "
        f"building={int(max_row['building_edges'])} dropped={int(max_row['active_dropped_by_building'])} "
        f"active={int(max_row['active_edges'])}/{int(max_row['target_active_edges'])}",
        flush=True,
    )
    if bool(args.check_only):
        return 0

    if bool(args.offscreen):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    panels = [
        make_panel(
            title="ideal dynamic one-right target",
            steps=[int(x) for x in steps.tolist()],
            edge_table=edge_table,
            active_mask=target_mask,
            building_mask=np.zeros((1, int(edge_table.num_edges)), dtype=bool),
            group_data=group_data,
        ),
        make_panel(
            title=f"LST={int(args.lst)}s active black + building blue dashed",
            steps=[int(x) for x in steps.tolist()],
            edge_table=edge_table,
            active_mask=lst_active,
            building_mask=lst_building,
            group_data=group_data,
        ),
    ]
    combined = UnifiedControlTopology2DViewer(
        title=f"G60 China-Europe dynamic one-right with LST={int(args.lst)}s",
        panels=panels,
        shared_value_scale=True,
        orientation=QtCore.Qt.Horizontal,
    )
    combined.master.update_step(int(initial_row))
    return run_viewer_widget(
        combined,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
