from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv  # noqa: E402
from src.link_setup_time.module import (  # noqa: E402
    SwitchEvent,
    TopologyProfile,
    apply_switch_chain_static_z,
    chain_summary,
    materialize_chain_static_z,
    plan_row_to_dict,
    transition_to_dict,
)
from src.topology_workflow.module import INTRA_OPTION, build_motif_text_edge_table, make_edge_table_from_records  # noqa: E402
from src.topology_workflow.module.region_constraints import edge_records_from_table  # noqa: E402


MOTIF_LIBRARY_CSV = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "libraries"
    / "motif"
    / "exact_box"
    / "w_le_4_h_le_3"
    / "combined_w_le4_h_le3_808.csv"
)
DEFAULT_USAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "switch_setup"
    / "usage_driven_switch_056_061_056_china_europe_1s_delay"
    / "t0_86160_stride1"
    / "switch36000_54000"
    / "setup600_delay"
    / "tracked_usage"
)
DEFAULT_OUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "switch_setup"
    / "static_z_chain_056_061_056_china_europe"
)

OPTION_SYMBOL = {-1: "I", 0: "A", 1: "B", 2: "D", 4: "C"}


@dataclass(frozen=True)
class RightLink:
    owner: int
    right: int
    edge_key: tuple[int, int]
    edge_idx: int
    option: int
    symbol: str


@dataclass(frozen=True)
class StaticTopology:
    name: str
    motif_id: int
    motif_text: str
    edge_table: EdgeTable
    right_by_owner: dict[int, RightLink]
    z_by_owner: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static-z chain switch sequence for G60 motif000056 -> 000061 -> 000056."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lst", type=int, default=60)
    parser.add_argument("--delta", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--strategy", choices=("latest", "earliest", "near_tau"), default="latest")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--tau", type=int, nargs="+", default=[36000, 54000])
    parser.add_argument("--motif-sequence", type=int, nargs="+", default=[56, 61, 56])
    parser.add_argument("--usage-root", type=Path, default=DEFAULT_USAGE_ROOT)
    parser.add_argument("--motif-library-csv", type=Path, default=MOTIF_LIBRARY_CSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def edge_key(src: int, dst: int) -> tuple[int, int]:
    a, b = int(src), int(dst)
    return (a, b) if a <= b else (b, a)


def read_motif_rows(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["motif_id"])] = dict(row)
    return rows


def build_edge_key_index(edge_table: EdgeTable) -> dict[tuple[int, int], int]:
    return {
        edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(int(edge_table.num_edges))
    }


def build_right_links(edge_table: EdgeTable) -> dict[int, RightLink]:
    right_by_owner: dict[int, RightLink] = {}
    left_by_right: dict[int, RightLink] = {}
    for idx in range(int(edge_table.num_edges)):
        option = int(edge_table.option[idx])
        if option == int(INTRA_OPTION):
            continue
        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        src_p = int(edge_table.src_plane[idx])
        dst_p = int(edge_table.dst_plane[idx])
        if src_p == dst_p:
            continue
        owner, right = (src, dst) if src_p < dst_p else (dst, src)
        link = RightLink(
            owner=int(owner),
            right=int(right),
            edge_key=edge_key(src, dst),
            edge_idx=int(idx),
            option=option,
            symbol=OPTION_SYMBOL.get(option, str(option)),
        )
        if int(owner) in right_by_owner:
            raise ValueError(f"right port has multiple links: owner={owner}")
        if int(right) in left_by_right:
            raise ValueError(f"left port has multiple incoming links: right={right}")
        right_by_owner[int(owner)] = link
        left_by_right[int(right)] = link
    return right_by_owner


def read_tracked_cols(path: Path) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            left, right = str(row["edge_key"]).split("-", 1)
            out[edge_key(int(left), int(right))] = int(row["tracked_col"])
    return out


def selected_rows(all_steps: np.ndarray, *, start: int, end: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    wanted = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
    rows = np.searchsorted(all_steps, wanted)
    bad = (rows >= all_steps.size) | (np.asarray(all_steps[rows], dtype=np.int64) != wanted)
    if bool(np.any(bad)):
        examples = wanted[np.flatnonzero(bad)[:10]]
        raise KeyError(f"usage store does not contain requested steps, examples={examples.tolist()}")
    return wanted, rows.astype(np.int64)


def load_z_by_owner(
    *,
    usage_root: Path,
    topology_name: str,
    right_by_owner: Mapping[int, RightLink],
    steps: np.ndarray,
    row_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    usage_dir = Path(usage_root) / topology_name
    time_path = usage_dir / "time_indices.npy"
    edges_path = usage_dir / "tracked_edges.csv"
    counts_path = usage_dir / "tracked_edge_usage_counts.npy"
    missing = [path for path in (time_path, edges_path, counts_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing usage files: " + ", ".join(str(path) for path in missing))

    tracked_cols = read_tracked_cols(edges_path)
    counts = np.load(counts_path, mmap_mode="r")
    zero = np.zeros(int(steps.size), dtype=np.float32)
    z_by_owner: dict[int, np.ndarray] = {}
    for owner, link in right_by_owner.items():
        col = tracked_cols.get(link.edge_key)
        if col is None:
            z_by_owner[int(owner)] = zero.copy()
        else:
            z_by_owner[int(owner)] = np.asarray(counts[row_indices, int(col)], dtype=np.float32)
    return z_by_owner


def load_static_topology(
    *,
    motif_id: int,
    motif_rows: Mapping[int, dict[str, str]],
    usage_root: Path,
    steps: np.ndarray,
    row_indices: np.ndarray,
) -> StaticTopology:
    row = motif_rows[int(motif_id)]
    motif_text = str(row["motif"]).strip()
    edge_table = build_motif_text_edge_table(
        motif_text=motif_text,
        config=G60_CONFIG,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
        wrap_planes=False,
        add_intra_ring=True,
    )
    right_by_owner = build_right_links(edge_table)
    name = f"combined_motif_{int(motif_id):06d}"
    z_by_owner = load_z_by_owner(
        usage_root=Path(usage_root),
        topology_name=name,
        right_by_owner=right_by_owner,
        steps=steps,
        row_indices=row_indices,
    )
    return StaticTopology(
        name=name,
        motif_id=int(motif_id),
        motif_text=motif_text,
        edge_table=edge_table,
        right_by_owner=right_by_owner,
        z_by_owner=z_by_owner,
    )


def union_edge_table(topologies: Sequence[StaticTopology]) -> EdgeTable:
    records: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for topology in topologies:
        for idx, record in enumerate(edge_records_from_table(topology.edge_table)):
            key = edge_key(int(topology.edge_table.src[idx]), int(topology.edge_table.dst[idx]))
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return make_edge_table_from_records(p=int(G60_CONFIG.P), n=int(G60_CONFIG.N), records=records)


def topology_profile(topology: StaticTopology) -> TopologyProfile:
    return TopologyProfile(
        name=str(topology.motif_id),
        right={int(owner): int(link.right) for owner, link in topology.right_by_owner.items()},
        z={int(owner): np.asarray(values, dtype=np.float32) for owner, values in topology.z_by_owner.items()},
    )


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_masks_and_values(
    *,
    steps: np.ndarray,
    edge_table: EdgeTable,
    topologies: Mapping[str, StaticTopology],
    initial_name: str,
    transitions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_to_idx = build_edge_key_index(edge_table)
    total_rows = int(steps.size)
    total_edges = int(edge_table.num_edges)
    active = np.zeros((total_rows, total_edges), dtype=bool)
    building = np.zeros((total_rows, total_edges), dtype=bool)
    values = np.zeros((total_rows, total_edges), dtype=np.float32)

    for idx in range(total_edges):
        if int(edge_table.option[idx]) == int(INTRA_OPTION):
            active[:, idx] = True

    initial = topologies[str(initial_name)]
    for owner, link in initial.right_by_owner.items():
        col = key_to_idx.get(link.edge_key)
        if col is None:
            continue
        active[:, int(col)] = True
        values[:, int(col)] = np.asarray(initial.z_by_owner.get(int(owner), np.zeros(total_rows)), dtype=np.float32)

    for transition in sorted(
        transitions,
        key=lambda item: (
            10**30 if item.start is None else int(item.start),
            int(item.switch_index),
            int(item.owner),
        ),
    ):
        if transition.start is None or transition.ready is None:
            continue
        start_row = int(np.searchsorted(steps, int(transition.start), side="left"))
        ready_row = int(np.searchsorted(steps, int(transition.ready), side="left"))
        start_row = max(0, min(total_rows, start_row))
        ready_row = max(0, min(total_rows, ready_row))

        if transition.old_right is not None:
            old_col = key_to_idx.get(edge_key(int(transition.owner), int(transition.old_right)))
            if old_col is not None:
                active[start_row:, int(old_col)] = False
                values[start_row:, int(old_col)] = 0.0

        if transition.new_right is None:
            continue

        new_col = key_to_idx.get(edge_key(int(transition.owner), int(transition.new_right)))
        if new_col is None:
            continue
        if ready_row > start_row:
            building[start_row:ready_row, int(new_col)] = True
        active[ready_row:, int(new_col)] = True
        to_topology = topologies[str(transition.to_name)]
        z = to_topology.z_by_owner.get(int(transition.owner))
        if z is not None:
            values[ready_row:, int(new_col)] = np.asarray(z[ready_row:], dtype=np.float32)
    return active, building, values


def main() -> int:
    args = parse_args()
    motif_sequence = [int(x) for x in args.motif_sequence]
    taus = [int(x) for x in args.tau]
    if len(motif_sequence) != len(taus) + 1:
        raise ValueError("--motif-sequence length must equal len(--tau)+1")

    usage_steps_all = np.asarray(
        np.load(Path(args.usage_root) / f"combined_motif_{motif_sequence[0]:06d}" / "time_indices.npy", mmap_mode="r"),
        dtype=np.int64,
    )
    steps, row_indices = selected_rows(
        usage_steps_all,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    motif_rows = read_motif_rows(Path(args.motif_library_csv))

    unique_motifs = sorted(set(motif_sequence))
    topologies = {
        str(motif_id): load_static_topology(
            motif_id=int(motif_id),
            motif_rows=motif_rows,
            usage_root=Path(args.usage_root),
            steps=steps,
            row_indices=row_indices,
        )
        for motif_id in unique_motifs
    }
    profiles = {name: topology_profile(topology) for name, topology in topologies.items()}
    events = [SwitchEvent(tau=int(tau), target=str(target)) for tau, target in zip(taus, motif_sequence[1:])]
    chain = apply_switch_chain_static_z(
        steps=steps,
        profiles=profiles,
        initial=str(motif_sequence[0]),
        events=events,
        lst=int(args.lst),
        delta=int(args.delta),
        threshold=float(args.threshold),
        strategy=str(args.strategy),
        enforce_previous_ready=not bool(args.allow_overlap),
    )
    _active_right, _building_right = materialize_chain_static_z(
        steps=steps,
        initial_right=profiles[str(motif_sequence[0])].right,
        transitions=chain.transitions,
        nodes=range(int(G60_CONFIG.total_sats)),
    )

    edge_table = union_edge_table(list(topologies.values()))
    active, building, values = build_masks_and_values(
        steps=steps,
        edge_table=edge_table,
        topologies=topologies,
        initial_name=str(motif_sequence[0]),
        transitions=chain.transitions,
    )

    if args.out_dir is None:
        tau_text = "_".join(str(x) for x in taus)
        seq_text = "_".join(f"{x:04d}" for x in motif_sequence)
        out_dir = (
            DEFAULT_OUT_ROOT
            / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
            / f"seq{seq_text}_tau{tau_text}_lst{int(args.lst):03d}_{args.strategy}"
        )
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_edges_csv(edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "steps.npy", steps.astype(np.int64))
    np.save(out_dir / "edge_active_mask.npy", active.astype(bool))
    np.save(out_dir / "edge_building_mask.npy", building.astype(bool))
    np.save(out_dir / "edge_usage_values.npy", values.astype(np.float32))

    for switch in chain.switches:
        rows = [plan_row_to_dict(row) for row in switch.rows]
        for row in rows:
            row["switch_index"] = int(switch.index)
            row["from"] = switch.from_name
            row["to"] = switch.to_name
        write_csv(out_dir / f"switch_{int(switch.index):03d}_{switch.from_name}_to_{switch.to_name}_plan.csv", rows)
    write_csv(out_dir / "transitions.csv", [transition_to_dict(item) for item in chain.transitions])

    building_counts = np.sum(building, axis=1).astype(np.int32)
    with (out_dir / "building_concurrency_by_step.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hour", "building_count"])
        for step, count in zip(steps, building_counts):
            writer.writerow([int(step), f"{int(step) / 3600.0:.6f}", int(count)])
    np.save(out_dir / "building_concurrency.npy", building_counts)

    summary = {
        "method": "static_z_chain",
        "initial": chain.initial,
        "final": chain.final,
        "motif_sequence": motif_sequence,
        "taus": taus,
        "lst": int(args.lst),
        "delta": int(args.delta),
        "threshold": float(args.threshold),
        "strategy": str(args.strategy),
        "enforce_previous_ready": not bool(args.allow_overlap),
        "steps": {"start": int(steps[0]), "end": int(steps[-1]), "stride": int(args.stride), "count": int(steps.size)},
        "topologies": {
            name: {"motif_id": topology.motif_id, "motif": topology.motif_text, "right_links": len(topology.right_by_owner)}
            for name, topology in topologies.items()
        },
        "switches": chain_summary(chain),
        "warnings": list(chain.warnings),
        "union_edges": int(edge_table.num_edges),
        "max_building_concurrency": int(building_counts.max()) if building_counts.size else 0,
        "first_building_step": None if not np.any(building_counts > 0) else int(steps[int(np.flatnonzero(building_counts > 0)[0])]),
        "last_building_step": None if not np.any(building_counts > 0) else int(steps[int(np.flatnonzero(building_counts > 0)[-1])]),
        "files": {
            "steps": "steps.npy",
            "union_edges": "union_edges.csv",
            "edge_active_mask": "edge_active_mask.npy",
            "edge_building_mask": "edge_building_mask.npy",
            "edge_usage_values": "edge_usage_values.npy",
            "transitions": "transitions.csv",
            "building_concurrency": "building_concurrency_by_step.csv",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}", flush=True)
    for item in summary["switches"]:
        print(
            f"switch{item['switch_index']}: {item['from']}->{item['to']} tau={item['tau']} "
            f"changed={item['changed_nodes']} lossless={item['lossless']} conflict={item['conflict']} "
            f"remove_only={item['remove_only']} max_lag={item['max_lag']}",
            flush=True,
        )
    print(
        f"max_building={summary['max_building_concurrency']} "
        f"first_building={summary['first_building_step']} last_building={summary['last_building_step']}",
        flush=True,
    )
    if chain.warnings:
        print(f"warnings={len(chain.warnings)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
