from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402


DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\static_z_chain_056_061_056_china_europe\t0_86160_stride1"
    r"\seq0056_0061_0056_tau36000_54000_lst060_latest"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast full-timeseries inter-port conflict check for a plan-dir.")
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--occupancy-mode",
        choices=("active_building", "active", "building"),
        default="active_building",
        help="Which edge masks count as terminal occupancy. Default counts active OR building.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def read_union_edges(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def selected_rows(steps: np.ndarray, *, start: int, end: int, stride: int) -> np.ndarray:
    mask = (steps >= int(start)) & (steps <= int(end)) & (((steps - int(start)) % int(stride)) == 0)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError(f"no rows for start={start}, end={end}, stride={stride}")
    return rows.astype(np.int64)


def edge_key(a: int, b: int) -> str:
    a, b = int(a), int(b)
    return f"{a}-{b}" if a <= b else f"{b}-{a}"


def build_inter_owner_arrays(edge_rows: list[dict], *, p: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    total_nodes = int(p) * int(n)
    num_edges = len(edge_rows)
    left_owner_by_edge = np.full(num_edges, -1, dtype=np.int32)
    right_owner_by_edge = np.full(num_edges, -1, dtype=np.int32)
    inter_mask = np.zeros(num_edges, dtype=bool)
    inter_meta: list[dict] = []
    for idx, row in enumerate(edge_rows):
        src = int(row["src_node"])
        dst = int(row["dst_node"])
        src_plane = int(row["src_plane"])
        dst_plane = int(row["dst_plane"])
        option = int(row["option"])
        if src_plane == dst_plane or option == -1:
            continue
        if src_plane < dst_plane:
            right_owner = src
            left_owner = dst
        else:
            right_owner = dst
            left_owner = src
        right_owner_by_edge[idx] = int(right_owner)
        left_owner_by_edge[idx] = int(left_owner)
        inter_mask[idx] = True
        inter_meta.append(
            {
                "edge_idx": int(idx),
                "edge_key": edge_key(src, dst),
                "src": src,
                "dst": dst,
                "right_owner": int(right_owner),
                "left_owner": int(left_owner),
                "option": option,
            }
        )
    return left_owner_by_edge, right_owner_by_edge, inter_mask, inter_meta


def violation_details_for_row(
    *,
    occupied_row: np.ndarray,
    active_row: np.ndarray | None,
    building_row: np.ndarray | None,
    edge_rows: list[dict],
    left_count: np.ndarray,
    right_count: np.ndarray,
    p: int,
    n: int,
) -> list[dict]:
    left_neighbors: list[list[int]] = [[] for _ in range(int(p) * int(n))]
    right_neighbors: list[list[int]] = [[] for _ in range(int(p) * int(n))]
    left_edge_indices: list[list[int]] = [[] for _ in range(int(p) * int(n))]
    right_edge_indices: list[list[int]] = [[] for _ in range(int(p) * int(n))]

    for idx in np.flatnonzero(occupied_row):
        row = edge_rows[int(idx)]
        src = int(row["src_node"])
        dst = int(row["dst_node"])
        src_plane = int(row["src_plane"])
        dst_plane = int(row["dst_plane"])
        option = int(row["option"])
        if src_plane == dst_plane or option == -1:
            continue
        if src_plane < dst_plane:
            right_owner = src
            left_owner = dst
        else:
            right_owner = dst
            left_owner = src
        state = []
        if active_row is not None and bool(active_row[int(idx)]):
            state.append("active")
        if building_row is not None and bool(building_row[int(idx)]):
            state.append("building")
        state_text = "+".join(state) if state else "occupied"
        right_neighbors[right_owner].append(left_owner)
        right_edge_indices[right_owner].append(f"{int(idx)}:{state_text}")
        left_neighbors[left_owner].append(right_owner)
        left_edge_indices[left_owner].append(f"{int(idx)}:{state_text}")

    details: list[dict] = []
    total_nodes = int(p) * int(n)
    for node in range(total_nodes):
        reasons: list[str] = []
        if int(left_count[node]) > 1:
            reasons.append("left_count_gt1")
        if int(right_count[node]) > 1:
            reasons.append("right_count_gt1")
        plane, y = divmod(int(node), int(n))
        if plane == 0 and int(left_count[node]) > 0:
            reasons.append("first_plane_left_count_gt0")
        if plane == int(p) - 1 and int(right_count[node]) > 0:
            reasons.append("last_plane_right_count_gt0")
        if not reasons:
            continue
        details.append(
            {
                "node": int(node),
                "plane": int(plane),
                "y": int(y),
                "left_count": int(left_count[node]),
                "right_count": int(right_count[node]),
                "left_neighbors": " ".join(str(x) for x in sorted(left_neighbors[node])),
                "right_neighbors": " ".join(str(x) for x in sorted(right_neighbors[node])),
                "left_edge_indices": " ".join(str(x) for x in left_edge_indices[node]),
                "right_edge_indices": " ".join(str(x) for x in right_edge_indices[node]),
                "reasons": " ".join(reasons),
            }
        )
    return details


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    plan_dir = Path(args.plan_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else plan_dir / "fast_inter_port_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(np.load(plan_dir / "steps.npy", mmap_mode="r"), dtype=np.int64)
    selected = selected_rows(steps, start=int(args.start), end=int(args.end), stride=int(args.stride))
    active = np.load(plan_dir / "edge_active_mask.npy", mmap_mode="r")
    building = np.load(plan_dir / "edge_building_mask.npy", mmap_mode="r")
    edge_rows = read_union_edges(plan_dir / "union_edges.csv")
    if int(active.shape[1]) != len(edge_rows):
        raise ValueError(f"active edge columns {active.shape[1]} != union_edges rows {len(edge_rows)}")
    if tuple(building.shape) != tuple(active.shape):
        raise ValueError(f"building mask shape {building.shape} != active mask shape {active.shape}")

    left_owner_by_edge, right_owner_by_edge, inter_mask, inter_meta = build_inter_owner_arrays(
        edge_rows,
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
    )
    total_nodes = int(G60_CONFIG.P) * int(G60_CONFIG.N)
    first_bad: dict | None = None
    max_bad: dict | None = None
    step_summary: list[dict] = []
    first_bad_counts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    max_bad_counts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    chunk_size = max(1, int(args.chunk_size))
    for chunk_start in range(0, int(selected.size), chunk_size):
        rows = selected[chunk_start : chunk_start + chunk_size]
        active_chunk = np.asarray(active[rows, :], dtype=bool)
        building_chunk = np.asarray(building[rows, :], dtype=bool)
        if args.occupancy_mode == "active_building":
            occupied_chunk = np.logical_or(active_chunk, building_chunk)
        elif args.occupancy_mode == "active":
            occupied_chunk = active_chunk
        else:
            occupied_chunk = building_chunk
        for local_idx, row_idx in enumerate(rows):
            occupied_inter = np.flatnonzero(occupied_chunk[local_idx] & inter_mask)
            if occupied_inter.size:
                left_counts = np.bincount(left_owner_by_edge[occupied_inter], minlength=total_nodes).astype(np.int16)
                right_counts = np.bincount(right_owner_by_edge[occupied_inter], minlength=total_nodes).astype(np.int16)
            else:
                left_counts = np.zeros(total_nodes, dtype=np.int16)
                right_counts = np.zeros(total_nodes, dtype=np.int16)
            violation_mask = (left_counts > 1) | (right_counts > 1)
            payload = {
                "step": int(steps[int(row_idx)]),
                "violation_nodes": int(np.count_nonzero(violation_mask)),
                "left_gt1_nodes": int(np.count_nonzero(left_counts > 1)),
                "right_gt1_nodes": int(np.count_nonzero(right_counts > 1)),
                "max_left_count": int(np.max(left_counts)) if left_counts.size else 0,
                "max_right_count": int(np.max(right_counts)) if right_counts.size else 0,
                "active_edges": int(np.count_nonzero(active_chunk[local_idx])),
                "building_edges": int(np.count_nonzero(building_chunk[local_idx])),
                "occupied_edges": int(np.count_nonzero(occupied_chunk[local_idx])),
            }
            step_summary.append(payload)
            if payload["violation_nodes"] > 0 and first_bad is None:
                first_bad = dict(payload)
                first_bad_counts = (
                    np.asarray(occupied_chunk[local_idx], dtype=bool),
                    np.asarray(active_chunk[local_idx], dtype=bool),
                    np.asarray(building_chunk[local_idx], dtype=bool),
                    np.asarray(left_counts, dtype=np.int16),
                    np.asarray(right_counts, dtype=np.int16),
                )
            if max_bad is None or payload["violation_nodes"] > int(max_bad["violation_nodes"]):
                max_bad = dict(payload)
                max_bad_counts = (
                    np.asarray(occupied_chunk[local_idx], dtype=bool),
                    np.asarray(active_chunk[local_idx], dtype=bool),
                    np.asarray(building_chunk[local_idx], dtype=bool),
                    np.asarray(left_counts, dtype=np.int16),
                    np.asarray(right_counts, dtype=np.int16),
                )

    write_csv(out_dir / "step_summary.csv", step_summary)
    if first_bad and first_bad_counts is not None:
        occupied_row, active_row, building_row, left_count, right_count = first_bad_counts
        write_csv(
            out_dir / f"violations_first_bad_t{int(first_bad['step'])}.csv",
            violation_details_for_row(
                occupied_row=occupied_row,
                active_row=active_row,
                building_row=building_row,
                edge_rows=edge_rows,
                left_count=left_count,
                right_count=right_count,
                p=int(G60_CONFIG.P),
                n=int(G60_CONFIG.N),
            ),
        )
    if max_bad and max_bad_counts is not None:
        occupied_row, active_row, building_row, left_count, right_count = max_bad_counts
        write_csv(
            out_dir / f"violations_max_bad_t{int(max_bad['step'])}.csv",
            violation_details_for_row(
                occupied_row=occupied_row,
                active_row=active_row,
                building_row=building_row,
                edge_rows=edge_rows,
                left_count=left_count,
                right_count=right_count,
                p=int(G60_CONFIG.P),
                n=int(G60_CONFIG.N),
            ),
        )

    aggregate = {
        "ok": first_bad is None,
        "plan_dir": str(plan_dir),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "occupancy_mode": str(args.occupancy_mode),
        "checked_steps": int(len(step_summary)),
        "inter_edges_in_union": int(len(inter_meta)),
        "first_bad": first_bad,
        "max_bad": max_bad,
        "steps_with_conflict": int(sum(1 for row in step_summary if int(row["violation_nodes"]) > 0)),
    }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)
    print(f"out_dir={out_dir}", flush=True)
    return 0 if first_bad is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
