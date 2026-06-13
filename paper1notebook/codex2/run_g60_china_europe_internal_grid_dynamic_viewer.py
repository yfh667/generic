from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "linshi"


LEFT_SIDE = 0
RIGHT_SIDE = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build/open a dynamic G60 topology sequence. China/Europe internal inter-plane "
            "links use side-aware +grid constraints; unconstrained sides keep full-link."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164, help="Inclusive end step.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--constrained-groups", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--grid-option", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--hide-groups", action="store_true")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def default_out_dir(start: int, end: int, stride: int, groups: list[int]) -> Path:
    group_text = "_".join(str(int(x)) for x in groups)
    return (
        DEFAULT_OUT_ROOT
        / f"g60_region_internal_grid_dynamic_topology_groups{group_text}_t{int(start)}_{int(end)}_stride{int(stride)}"
    )


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def group_signature(group_data: dict, step: int, constrained_groups: list[int]) -> tuple[tuple[int, ...], ...]:
    current = group_data.get(int(step), {}) if group_data else {}
    groups = current.get("groups", {}) if isinstance(current, dict) else {}
    return tuple(tuple(sorted(int(x) for x in groups.get(int(gid), set()) or set())) for gid in constrained_groups)


def node_group_bits_from_signature(
    signature: tuple[tuple[int, ...], ...],
    *,
    total_nodes: int,
) -> np.ndarray:
    bits = np.zeros(int(total_nodes), dtype=np.uint16)
    for offset, nodes in enumerate(signature):
        bit = np.uint16(1 << int(offset))
        for node in nodes:
            if 0 <= int(node) < int(total_nodes):
                bits[int(node)] |= bit
    return bits


def edge_side_arrays(edge_table: EdgeTable) -> tuple[np.ndarray, np.ndarray]:
    src_side = np.full(int(edge_table.num_edges), -1, dtype=np.int8)
    dst_side = np.full(int(edge_table.num_edges), -1, dtype=np.int8)
    for idx in range(int(edge_table.num_edges)):
        if int(edge_table.option[idx]) == INTRA_OPTION:
            continue
        src_plane = int(edge_table.src_plane[idx])
        dst_plane = int(edge_table.dst_plane[idx])
        if dst_plane > src_plane:
            src_side[idx] = RIGHT_SIDE
            dst_side[idx] = LEFT_SIDE
        elif dst_plane < src_plane:
            src_side[idx] = LEFT_SIDE
            dst_side[idx] = RIGHT_SIDE
    return src_side, dst_side


def active_mask_for_signature(
    edge_table: EdgeTable,
    signature: tuple[tuple[int, ...], ...],
    *,
    src_side: np.ndarray,
    dst_side: np.ndarray,
    grid_option: int,
) -> tuple[np.ndarray, dict]:
    total_nodes = int(G60_CONFIG.total_sats)
    node_bits = node_group_bits_from_signature(signature, total_nodes=total_nodes)
    allowed_neighbors = [[set(), set()] for _ in range(total_nodes)]

    src_arr = edge_table.src
    dst_arr = edge_table.dst
    opt_arr = edge_table.option

    for idx in range(int(edge_table.num_edges)):
        option = int(opt_arr[idx])
        if option == INTRA_OPTION or option != int(grid_option):
            continue
        src = int(src_arr[idx])
        dst = int(dst_arr[idx])
        if int(node_bits[src] & node_bits[dst]) == 0:
            continue
        if int(src_side[idx]) >= 0:
            allowed_neighbors[src][int(src_side[idx])].add(dst)
        if int(dst_side[idx]) >= 0:
            allowed_neighbors[dst][int(dst_side[idx])].add(src)

    keep = np.ones(int(edge_table.num_edges), dtype=bool)
    dropped_by_option: dict[str, int] = {}
    dropped_by_reason = {
        "same_region_non_grid": 0,
        "selected_node_side_has_internal_grid": 0,
    }

    for idx in range(int(edge_table.num_edges)):
        option = int(opt_arr[idx])
        if option == INTRA_OPTION:
            continue
        src = int(src_arr[idx])
        dst = int(dst_arr[idx])
        same_selected_region = int(node_bits[src] & node_bits[dst]) != 0
        if same_selected_region and option != int(grid_option):
            keep[idx] = False
            dropped_by_option[str(option)] = dropped_by_option.get(str(option), 0) + 1
            dropped_by_reason["same_region_non_grid"] += 1
            continue

        conflict = False
        if int(node_bits[src]) != 0 and int(src_side[idx]) >= 0:
            allowed = allowed_neighbors[src][int(src_side[idx])]
            conflict = bool(allowed and dst not in allowed)
        if not conflict and int(node_bits[dst]) != 0 and int(dst_side[idx]) >= 0:
            allowed = allowed_neighbors[dst][int(dst_side[idx])]
            conflict = bool(allowed and src not in allowed)
        if conflict:
            keep[idx] = False
            dropped_by_option[str(option)] = dropped_by_option.get(str(option), 0) + 1
            dropped_by_reason["selected_node_side_has_internal_grid"] += 1

    validation_same_region_non_grid_remaining = 0
    validation_side_conflict_remaining = 0
    for idx in range(int(edge_table.num_edges)):
        if not bool(keep[idx]) or int(opt_arr[idx]) == INTRA_OPTION:
            continue
        option = int(opt_arr[idx])
        src = int(src_arr[idx])
        dst = int(dst_arr[idx])
        if int(node_bits[src] & node_bits[dst]) != 0 and option != int(grid_option):
            validation_same_region_non_grid_remaining += 1
        if int(node_bits[src]) != 0 and int(src_side[idx]) >= 0:
            allowed = allowed_neighbors[src][int(src_side[idx])]
            if allowed and dst not in allowed:
                validation_side_conflict_remaining += 1
        if int(node_bits[dst]) != 0 and int(dst_side[idx]) >= 0:
            allowed = allowed_neighbors[dst][int(dst_side[idx])]
            if allowed and src not in allowed:
                validation_side_conflict_remaining += 1

    stats = {
        "active_edges": int(np.count_nonzero(keep)),
        "dropped_edges": int(edge_table.num_edges - np.count_nonzero(keep)),
        "dropped_edges_by_option": dropped_by_option,
        "dropped_edges_by_reason": dropped_by_reason,
        "selected_region_node_sides_with_internal_grid": int(
            sum(1 for node in range(total_nodes) for side in (LEFT_SIDE, RIGHT_SIDE) if allowed_neighbors[node][side])
        ),
        "validation_same_region_non_grid_remaining": int(validation_same_region_non_grid_remaining),
        "validation_selected_node_same_side_extra_edges_remaining": int(validation_side_conflict_remaining),
    }
    return keep, stats


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def artifacts_valid(out_dir: Path, expected_shape: tuple[int, int]) -> bool:
    active_path = out_dir / "edge_active_mask.npy"
    steps_path = out_dir / "time_indices.npy"
    edges_path = out_dir / "edges.csv"
    meta_path = out_dir / "meta.json"
    if not (active_path.exists() and steps_path.exists() and edges_path.exists() and meta_path.exists()):
        return False
    try:
        active = np.load(active_path, mmap_mode="r")
        return tuple(active.shape) == tuple(expected_shape)
    except Exception:
        return False


def build_or_load_sequence(
    *,
    out_dir: Path,
    steps: list[int],
    edge_table: EdgeTable,
    group_data: dict,
    constrained_groups: list[int],
    grid_option: int,
    force: bool,
) -> tuple[np.ndarray, dict]:
    expected_shape = (len(steps), int(edge_table.num_edges))
    if not force and artifacts_valid(out_dir, expected_shape):
        active = np.load(out_dir / "edge_active_mask.npy", mmap_mode="r")
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        print(f"[region-grid-dynamic] Reusing active mask: {out_dir / 'edge_active_mask.npy'}", flush=True)
        return active, meta

    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    active_path = out_dir / "edge_active_mask.npy"
    active = np.lib.format.open_memmap(
        active_path,
        mode="w+",
        dtype=np.bool_,
        shape=expected_shape,
    )

    src_side, dst_side = edge_side_arrays(edge_table)
    signature_to_state: dict[tuple[tuple[int, ...], ...], int] = {}
    state_masks: list[np.ndarray] = []
    state_stats: list[dict] = []
    state_ids = np.empty(len(steps), dtype=np.int32)
    step_rows: list[dict] = []
    started = time.perf_counter()

    for row, step in enumerate(steps):
        signature = group_signature(group_data, int(step), constrained_groups)
        state_id = signature_to_state.get(signature)
        if state_id is None:
            mask, stats = active_mask_for_signature(
                edge_table,
                signature,
                src_side=src_side,
                dst_side=dst_side,
                grid_option=int(grid_option),
            )
            state_id = len(state_masks)
            signature_to_state[signature] = state_id
            state_masks.append(mask)
            state_stats.append(
                {
                    "state_id": int(state_id),
                    "group_node_counts": ";".join(str(len(nodes)) for nodes in signature),
                    **stats,
                }
            )
        active[row, :] = state_masks[int(state_id)]
        state_ids[row] = int(state_id)
        if row < 20 or row == len(steps) - 1:
            stats = state_stats[int(state_id)]
            step_rows.append(
                {
                    "row": int(row),
                    "step": int(step),
                    "state_id": int(state_id),
                    "active_edges": int(stats["active_edges"]),
                    "dropped_edges": int(stats["dropped_edges"]),
                    "validation_same_region_non_grid_remaining": int(
                        stats["validation_same_region_non_grid_remaining"]
                    ),
                    "validation_selected_node_same_side_extra_edges_remaining": int(
                        stats["validation_selected_node_same_side_extra_edges_remaining"]
                    ),
                }
            )
        if (row + 1) % 5000 == 0 or row + 1 == len(steps):
            elapsed = time.perf_counter() - started
            print(
                f"[region-grid-dynamic] rows {row + 1}/{len(steps)} "
                f"unique_states={len(state_masks)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    active.flush()
    np.save(out_dir / "state_ids.npy", state_ids)
    write_csv_rows(out_dir / "state_summary.csv", state_stats)
    write_csv_rows(out_dir / "sample_step_summary.csv", step_rows)

    active_counts = np.asarray(active.sum(axis=1), dtype=np.int32)
    validation_same = max(int(x["validation_same_region_non_grid_remaining"]) for x in state_stats)
    validation_side = max(int(x["validation_selected_node_same_side_extra_edges_remaining"]) for x in state_stats)
    meta = {
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "stride": int(steps[1] - steps[0]) if len(steps) > 1 else 1,
        "num_steps": int(len(steps)),
        "num_edges_full": int(edge_table.num_edges),
        "constrained_groups": [
            {"id": int(gid), "name": group_name(int(gid))}
            for gid in constrained_groups
        ],
        "grid_option": int(grid_option),
        "rule": (
            "For each selected-region node, left and right sides are checked independently. "
            "If a side has an internal grid_option link to a node in the same selected region, "
            "that side keeps only this internal grid link and drops all other inter-plane links. "
            "A boundary side with no internal grid link remains full-link. Intra y-ring links are kept."
        ),
        "active_mask_shape": [int(x) for x in active.shape],
        "unique_group_states": int(len(state_masks)),
        "active_edges_min": int(active_counts.min()),
        "active_edges_max": int(active_counts.max()),
        "active_edges_mean": float(active_counts.mean()),
        "validation_same_region_non_grid_remaining_max": int(validation_same),
        "validation_selected_node_same_side_extra_edges_remaining_max": int(validation_side),
        "outputs": {
            "edge_active_mask": str(active_path),
            "time_indices": str(out_dir / "time_indices.npy"),
            "edges_csv": str(out_dir / "edges.csv"),
            "state_ids": str(out_dir / "state_ids.npy"),
            "state_summary": str(out_dir / "state_summary.csv"),
            "sample_step_summary": str(out_dir / "sample_step_summary.csv"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return np.load(active_path, mmap_mode="r"), meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("stride must be positive")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")
    constrained_groups = [int(x) for x in args.constrained_groups]
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(
        int(args.start),
        int(args.end),
        int(args.stride),
        constrained_groups,
    )

    edge_table = build_full_option_plus_intra_edges(G60_CONFIG)
    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    active_mask, meta = build_or_load_sequence(
        out_dir=out_dir,
        steps=steps,
        edge_table=edge_table,
        group_data=group_data,
        constrained_groups=constrained_groups,
        grid_option=int(args.grid_option),
        force=bool(args.force),
    )
    print(
        f"[region-grid-dynamic] ready steps={len(steps)} edges={edge_table.num_edges} "
        f"active={meta['active_edges_min']}..{meta['active_edges_max']} "
        f"unique_states={meta['unique_group_states']} out={out_dir}",
        flush=True,
    )

    if args.build_only or args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        edge_values=None,
        window_title=(
            f"G60 side-aware region internal +grid topology "
            f"{steps[0]}..{steps[-1]} stride {int(args.stride)}"
        ),
        group_data={} if args.hide_groups else group_data,
        show_groups=not bool(args.hide_groups),
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
