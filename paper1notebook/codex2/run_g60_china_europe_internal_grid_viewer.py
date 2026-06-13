from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data

from topology_edges import INTRA_OPTION, build_full_option_plus_intra_edges


DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "linshi" / "g60_region_internal_grid_topology"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a G60 full-link topology where selected region-internal inter-plane "
            "links are constrained to +grid option 0."
        )
    )
    parser.add_argument("--step", type=int, default=0, help="Time step used to read region groups.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--constrained-groups", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--grid-option", type=int, default=0)
    parser.add_argument(
        "--constraint-mode",
        choices=["same_region"],
        default="same_region",
        help=(
            "Current experiment rule: constrain only edges whose endpoints are in the same selected group. "
            "For each selected-region node side, boundary edges remain full-link only if that side has no "
            "internal grid edge."
        ),
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--no-groups", action="store_true")
    return parser.parse_args(argv)


def group_label(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def make_edge_table_subset(edge_table: EdgeTable, mask: np.ndarray) -> EdgeTable:
    mask = np.asarray(mask, dtype=bool)
    return EdgeTable(
        src=np.asarray(edge_table.src[mask], dtype=np.int32),
        dst=np.asarray(edge_table.dst[mask], dtype=np.int32),
        option=np.asarray(edge_table.option[mask], dtype=np.int16),
        src_plane=np.asarray(edge_table.src_plane[mask], dtype=np.int16),
        src_y=np.asarray(edge_table.src_y[mask], dtype=np.int16),
        dst_plane=np.asarray(edge_table.dst_plane[mask], dtype=np.int16),
        dst_y=np.asarray(edge_table.dst_y[mask], dtype=np.int16),
        sat_ids=list(edge_table.sat_ids),
    )


def selected_group_for_node(
    node: int,
    *,
    group_nodes: dict[int, set[int]],
    constrained_groups: list[int],
) -> int | None:
    for gid in constrained_groups:
        if int(node) in group_nodes[gid]:
            return int(gid)
    return None


def edge_region_relation(
    *,
    src: int,
    dst: int,
    group_nodes: dict[int, set[int]],
    constrained_groups: list[int],
) -> str:
    """Classify an edge for the region-internal +grid rule.

    same_selected_group is the only relation that is constrained to grid_option.
    Boundary edges, including selected-region nodes connected to outside nodes,
    remain full-link.
    """

    src_gid = selected_group_for_node(src, group_nodes=group_nodes, constrained_groups=constrained_groups)
    dst_gid = selected_group_for_node(dst, group_nodes=group_nodes, constrained_groups=constrained_groups)
    if src_gid is not None and src_gid == dst_gid:
        return "same_selected_group"
    if src_gid is not None or dst_gid is not None:
        return "boundary_or_cross_selected_group"
    return "outside_selected_groups"


def edge_side_for_node(edge_table: EdgeTable, edge_idx: int, node: int) -> str | None:
    src = int(edge_table.src[edge_idx])
    dst = int(edge_table.dst[edge_idx])
    src_plane = int(edge_table.src_plane[edge_idx])
    dst_plane = int(edge_table.dst_plane[edge_idx])
    if int(node) == src:
        return "right" if dst_plane > src_plane else "left"
    if int(node) == dst:
        return "left" if src_plane < dst_plane else "right"
    return None


def node_is_in_group(node: int, group_nodes: dict[int, set[int]], group_id: int) -> bool:
    return int(node) in group_nodes.get(int(group_id), set())


def edge_is_inside_one_selected_group(
    *,
    src: int,
    dst: int,
    group_nodes: dict[int, set[int]],
    constrained_groups: list[int],
) -> bool:
    return any(
        node_is_in_group(src, group_nodes, gid) and node_is_in_group(dst, group_nodes, gid)
        for gid in constrained_groups
    )


def edge_is_constrained(
    *,
    src: int,
    dst: int,
    group_nodes: dict[int, set[int]],
    constrained_groups: list[int],
    mode: str,
) -> bool:
    if mode == "same_region":
        return any(int(src) in group_nodes[gid] and int(dst) in group_nodes[gid] for gid in constrained_groups)

    union_nodes: set[int] = set()
    for gid in constrained_groups:
        union_nodes.update(group_nodes[gid])

    if mode == "union":
        return int(src) in union_nodes and int(dst) in union_nodes
    if mode == "touching":
        return int(src) in union_nodes or int(dst) in union_nodes
    raise ValueError(f"Unknown constraint mode: {mode}")


def filter_region_grid_edges(
    edge_table: EdgeTable,
    *,
    group_nodes: dict[int, set[int]],
    constrained_groups: list[int],
    grid_option: int = 0,
    constraint_mode: str = "same_region",
) -> tuple[EdgeTable, dict]:
    keep = np.ones(int(edge_table.num_edges), dtype=bool)
    drop_by_option: dict[str, int] = {}
    drop_by_reason: dict[str, int] = {}
    constrained_by_option: dict[str, int] = {}
    kept_boundary_by_option: dict[str, int] = {}
    kept_outside_by_option: dict[str, int] = {}
    kept_constrained_grid = 0
    occupied_side_to_allowed_neighbors: dict[tuple[int, str], set[int]] = {}

    for idx in range(int(edge_table.num_edges)):
        option = int(edge_table.option[idx])
        if option == INTRA_OPTION or option != int(grid_option):
            continue
        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        if not edge_is_inside_one_selected_group(
            src=src,
            dst=dst,
            group_nodes=group_nodes,
            constrained_groups=constrained_groups,
        ):
            continue
        for node in (src, dst):
            side = edge_side_for_node(edge_table, idx, node)
            if side is not None:
                neighbor = dst if int(node) == src else src
                occupied_side_to_allowed_neighbors.setdefault((int(node), side), set()).add(int(neighbor))

    for idx in range(int(edge_table.num_edges)):
        option = int(edge_table.option[idx])
        if option == INTRA_OPTION:
            continue

        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        inside_selected_group = edge_is_inside_one_selected_group(
            src=src,
            dst=dst,
            group_nodes=group_nodes,
            constrained_groups=constrained_groups,
        )
        if inside_selected_group:
            constrained_by_option[str(option)] = constrained_by_option.get(str(option), 0) + 1
        relation = edge_region_relation(
            src=src,
            dst=dst,
            group_nodes=group_nodes,
            constrained_groups=constrained_groups,
        )
        constrained = edge_is_constrained(
            src=src,
            dst=dst,
            group_nodes=group_nodes,
            constrained_groups=constrained_groups,
            mode=constraint_mode,
        )

        side_conflict_nodes: list[tuple[int, str]] = []
        for node in (src, dst):
            if selected_group_for_node(node, group_nodes=group_nodes, constrained_groups=constrained_groups) is None:
                continue
            side = edge_side_for_node(edge_table, idx, node)
            if side is None:
                continue
            neighbor = dst if int(node) == src else src
            allowed_neighbors = occupied_side_to_allowed_neighbors.get((int(node), side))
            if allowed_neighbors and int(neighbor) not in allowed_neighbors:
                side_conflict_nodes.append((int(node), side))

        if side_conflict_nodes:
            keep[idx] = False
            drop_by_option[str(option)] = drop_by_option.get(str(option), 0) + 1
            drop_by_reason["selected_node_side_has_internal_grid"] = (
                drop_by_reason.get("selected_node_side_has_internal_grid", 0) + 1
            )
            continue

        if inside_selected_group and option != int(grid_option):
            keep[idx] = False
            drop_by_option[str(option)] = drop_by_option.get(str(option), 0) + 1
            drop_by_reason["same_region_non_grid"] = drop_by_reason.get("same_region_non_grid", 0) + 1
            continue

        if not constrained:
            if relation == "boundary_or_cross_selected_group":
                kept_boundary_by_option[str(option)] = kept_boundary_by_option.get(str(option), 0) + 1
            elif relation == "outside_selected_groups":
                kept_outside_by_option[str(option)] = kept_outside_by_option.get(str(option), 0) + 1
            continue

        if option == int(grid_option):
            kept_constrained_grid += 1
        else:
            keep[idx] = False
            drop_by_option[str(option)] = drop_by_option.get(str(option), 0) + 1
            drop_by_reason["constrained_non_grid"] = drop_by_reason.get("constrained_non_grid", 0) + 1

    filtered = make_edge_table_subset(edge_table, keep)
    same_region_non_grid_remaining = 0
    selected_node_side_conflict_remaining = 0
    for idx in range(int(filtered.num_edges)):
        option = int(filtered.option[idx])
        if option == INTRA_OPTION:
            continue
        relation = edge_region_relation(
            src=int(filtered.src[idx]),
            dst=int(filtered.dst[idx]),
            group_nodes=group_nodes,
            constrained_groups=constrained_groups,
        )
        if relation == "same_selected_group" and option != int(grid_option):
            same_region_non_grid_remaining += 1
        for node in (int(filtered.src[idx]), int(filtered.dst[idx])):
            if selected_group_for_node(node, group_nodes=group_nodes, constrained_groups=constrained_groups) is None:
                continue
            side = edge_side_for_node(filtered, idx, node)
            if side is None:
                continue
            neighbor = int(filtered.dst[idx]) if int(node) == int(filtered.src[idx]) else int(filtered.src[idx])
            allowed_neighbors = occupied_side_to_allowed_neighbors.get((int(node), side))
            if allowed_neighbors and int(neighbor) not in allowed_neighbors:
                selected_node_side_conflict_remaining += 1

    option_counts_full = {
        str(option): int(np.count_nonzero(edge_table.option == int(option)))
        for option in sorted(set(int(x) for x in edge_table.option.tolist()))
    }
    option_counts_kept = {
        str(option): int(np.count_nonzero(filtered.option == int(option)))
        for option in sorted(set(int(x) for x in filtered.option.tolist()))
    }

    stats = {
        "full_edges": int(edge_table.num_edges),
        "kept_edges": int(filtered.num_edges),
        "dropped_edges": int(edge_table.num_edges - filtered.num_edges),
        "grid_option": int(grid_option),
        "constraint_mode": str(constraint_mode),
        "constrained_groups": [
            {"id": int(gid), "name": group_label(gid), "nodes": len(group_nodes.get(int(gid), set()))}
            for gid in constrained_groups
        ],
        "option_counts_full": option_counts_full,
        "option_counts_kept": option_counts_kept,
        "constrained_inter_edges_by_option_before_filter": constrained_by_option,
        "kept_constrained_grid_edges": int(kept_constrained_grid),
        "selected_region_node_sides_with_internal_grid": int(len(occupied_side_to_allowed_neighbors)),
        "dropped_edges_by_option": drop_by_option,
        "dropped_constrained_non_grid_edges_by_option": drop_by_option,
        "dropped_edges_by_reason": drop_by_reason,
        "kept_boundary_or_cross_selected_group_full_edges_by_option": kept_boundary_by_option,
        "kept_outside_selected_groups_full_edges_by_option": kept_outside_by_option,
        "validation": {
            "same_selected_group_non_grid_edges_remaining": int(same_region_non_grid_remaining),
            "selected_node_same_side_extra_edges_remaining": int(selected_node_side_conflict_remaining),
            "boundary_rule": (
                "Boundary edges are kept as full-link only on a selected-region node side that does not "
                "already have an internal grid edge."
            ),
        },
    }
    return filtered, stats


def write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def output_dir_for(args: argparse.Namespace) -> Path:
    groups = "_".join(str(int(x)) for x in args.constrained_groups)
    return Path(args.out_root) / f"step{int(args.step)}_groups{groups}_{args.constraint_mode}_grid{int(args.grid_option)}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    step = int(args.step)
    constrained_groups = [int(x) for x in args.constrained_groups]
    out_dir = output_dir_for(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_table_full = build_full_option_plus_intra_edges(G60_CONFIG)
    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=[step],
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=1,
        enabled=True,
        force=False,
    )
    if step not in group_data:
        raise ValueError(f"Missing group data for step={step}; available={sorted(group_data)[:5]}")

    groups_obj = group_data[step].get("groups", {})
    group_nodes = {gid: set(int(x) for x in groups_obj.get(gid, set())) for gid in constrained_groups}
    filtered_edge_table, stats = filter_region_grid_edges(
        edge_table_full,
        group_nodes=group_nodes,
        constrained_groups=constrained_groups,
        grid_option=int(args.grid_option),
        constraint_mode=str(args.constraint_mode),
    )

    write_edges_csv(filtered_edge_table, out_dir / "edges.csv")
    meta = {
        "constellation": str(G60_CONFIG.name),
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "step": step,
        "xml_file": str(Path(args.xml_file)),
        "rule": (
            "Start from full option 0/1/2/4 inter-plane links plus intra y-ring links. "
            "For each selected-region node, left and right sides are checked independently. If a side has "
            "an internal grid_option link to a node in the same selected region, that side keeps only this "
            "internal grid link and drops all other inter-plane links. A boundary side with no internal "
            "grid link remains full-link."
        ),
        **stats,
        "outputs": {
            "edges_csv": str(out_dir / "edges.csv"),
            "meta_json": str(out_dir / "meta.json"),
        },
    }
    write_meta(out_dir / "meta.json", meta)

    print(
        f"[region-grid-viewer] step={step} mode={args.constraint_mode} "
        f"groups={[f'{gid}:{group_label(gid)}' for gid in constrained_groups]} "
        f"edges={stats['kept_edges']}/{stats['full_edges']} dropped={stats['dropped_edges']} "
        f"out={out_dir}",
        flush=True,
    )
    print(
        f"[region-grid-viewer] dropped_by_option={stats['dropped_constrained_non_grid_edges_by_option']} "
        f"kept_option_counts={stats['option_counts_kept']}",
        flush=True,
    )
    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=[step],
        edge_table=filtered_edge_table,
        edge_values=None,
        window_title=(
            f"G60 full-link with {args.constraint_mode} region +grid "
            f"groups={','.join(str(x) for x in constrained_groups)} step={step}"
        ),
        group_data={} if args.no_groups else group_data,
        show_groups=not bool(args.no_groups),
        topology_edge_color="#000000",
        topology_edge_alpha=165,
        topology_edge_width=0.018,
    )
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
