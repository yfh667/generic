from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.motif_generator.module.config_io import load_yaml_dict
from src.motif_generator.module.support import motif_support_from_dict, motif_support_to_edge_records
from src.motif_generator.module.tiling import tile_edge_records_on_grid
from src.motif_generator.module.viewer_adapter import option_from_delta
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = THIS_DIR / "cache" / "group_data_cache"
DEFAULT_MOTIF_CONFIG = GENERIC_ROOT / "src" / "motif_generator" / "examples" / "configs" / "dad_cxx_support.yaml"
DEFAULT_GRIDPLUS_CONFIG = PROJECT_ROOT / "data" / "topology_design" / "grid+" / "config" / "motif.json"
DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "g60_motif_gridplus_shortest_path_t0_86164"

INTRA_OPTION = -1
LEGACY_OPTION_DELTAS = {
    0: (1, 0),
    1: (1, -1),
    2: (2, 0),
    4: (1, 1),
    5: (2, -1),
    6: (1, 2),
}


@dataclass(frozen=True)
class StateSummary:
    state_id: int
    source_nodes: int
    target_nodes: int
    reachable_pairs: int
    total_shortest_distance_hops: float
    mean_shortest_distance_hops: float
    min_shortest_distance_hops: float
    max_shortest_distance_hops: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute G60 China-Europe shortest-path time series for a support motif and grid+."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164, help="Inclusive end step.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--source-group", type=int, default=2, help="Default G60 group 2 is China.")
    parser.add_argument("--target-group", type=int, default=3, help="Default G60 group 3 is Europe.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--motif-config", type=Path, default=DEFAULT_MOTIF_CONFIG)
    parser.add_argument("--gridplus-config", type=Path, default=DEFAULT_GRIDPLUS_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-steps", type=int, default=3)
    parser.add_argument("--sample-pairs-per-step", type=int, default=80)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def make_edge_table_from_records(
    *,
    p: int,
    n: int,
    records: list[tuple[int, int, int, int, int]],
    sat_ids: list[str] | None = None,
) -> EdgeTable:
    unique: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    for src_plane, src_y, dst_plane, dst_y, option in records:
        src = int(src_plane) * int(n) + int(src_y)
        dst = int(dst_plane) * int(n) + int(dst_y)
        if src == dst:
            continue
        a, b = (src, dst) if src < dst else (dst, src)
        unique.setdefault((a, b), (int(src_plane), int(src_y), int(dst_plane), int(dst_y), int(option)))

    src_values: list[int] = []
    dst_values: list[int] = []
    opt_values: list[int] = []
    src_plane_values: list[int] = []
    src_y_values: list[int] = []
    dst_plane_values: list[int] = []
    dst_y_values: list[int] = []

    for src, dst in sorted(unique):
        src_plane = src // int(n)
        src_y = src % int(n)
        dst_plane = dst // int(n)
        dst_y = dst % int(n)
        _orig_src_plane, _orig_src_y, _orig_dst_plane, _orig_dst_y, option = unique[(src, dst)]
        src_values.append(src)
        dst_values.append(dst)
        opt_values.append(int(option))
        src_plane_values.append(src_plane)
        src_y_values.append(src_y)
        dst_plane_values.append(dst_plane)
        dst_y_values.append(dst_y)

    total_sats = int(p) * int(n)
    if sat_ids is None:
        sat_ids = [str(i + 1) for i in range(total_sats)]
    if len(sat_ids) != total_sats:
        raise ValueError(f"sat_ids length {len(sat_ids)} != p*n={total_sats}")

    return EdgeTable(
        src=np.asarray(src_values, dtype=np.int32),
        dst=np.asarray(dst_values, dtype=np.int32),
        option=np.asarray(opt_values, dtype=np.int16),
        src_plane=np.asarray(src_plane_values, dtype=np.int16),
        src_y=np.asarray(src_y_values, dtype=np.int16),
        dst_plane=np.asarray(dst_plane_values, dtype=np.int16),
        dst_y=np.asarray(dst_y_values, dtype=np.int16),
        sat_ids=sat_ids,
    )


def add_intra_ring_records(records: list[tuple[int, int, int, int, int]], *, p: int, n: int) -> None:
    for plane in range(int(p)):
        for y in range(int(n)):
            records.append((plane, y, plane, (y + 1) % int(n), INTRA_OPTION))


def build_support_motif_edge_table(*, motif_config: Path, p: int, n: int) -> EdgeTable:
    raw = load_yaml_dict(motif_config)
    motif_support = motif_support_from_dict(raw)
    tiling_raw = raw.get("tiling", {})
    if not isinstance(tiling_raw, dict):
        raise ValueError("motif YAML tiling section must be a mapping when present")

    result = tile_edge_records_on_grid(
        p=int(p),
        n=int(n),
        motif_width=motif_support.w,
        motif_height=motif_support.h,
        local_edges=motif_support_to_edge_records(motif_support),
        horizontal_step=tiling_raw.get("horizontal_step"),
        allow_vertical_overlap=bool(tiling_raw.get("allow_vertical_overlap", True)),
        allow_clipped_right=bool(tiling_raw.get("allow_clipped_right", True)),
    )

    records: list[tuple[int, int, int, int, int]] = []
    for edge in result.placed_edges:
        dx = int(edge.dst_col) - int(edge.src_col)
        dy = int(edge.dst_row) - int(edge.src_row)
        option = option_from_delta(dx, dy)
        records.append((int(edge.src_col), int(edge.src_row), int(edge.dst_col), int(edge.dst_row), int(option)))
    add_intra_ring_records(records, p=p, n=n)
    return make_edge_table_from_records(p=p, n=n, records=records)


def legacy_target(x: int, y: int, *, n: int, option: int) -> tuple[int, int]:
    if int(option) not in LEGACY_OPTION_DELTAS:
        raise ValueError(f"unsupported legacy option: {option}")
    dx, dy = LEGACY_OPTION_DELTAS[int(option)]
    return int(x) + int(dx), (int(y) + int(dy) + int(n)) % int(n)


def build_legacy_gridplus_edge_table(*, motif_json: Path) -> EdgeTable:
    raw = json.loads(Path(motif_json).read_text(encoding="utf-8"))
    p = int(raw["P"])
    n = int(raw["N"])
    outgoing: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    incoming: dict[tuple[int, int], tuple[int, int]] = {}

    def set_edge(src: tuple[int, int], dst: tuple[int, int], option: int) -> None:
        old = outgoing.get(src)
        if old is not None:
            old_dst, _old_option = old
            if incoming.get(old_dst) == src:
                incoming.pop(old_dst, None)
        old_src = incoming.get(dst)
        if old_src is not None:
            old_out = outgoing.get(old_src)
            if old_out is not None and old_out[0] == dst:
                outgoing.pop(old_src, None)
        outgoing[src] = (dst, int(option))
        incoming[dst] = src

    for motif in raw.get("motifs", []):
        p_start = int(motif["p_start"])
        p_end = int(motif["p_end"])
        y_start = int(motif["y_start"])
        y_end = int(motif["y_end"])
        option = int(motif["option"])

        for x in range(p_start, p_end + 1):
            for y in range(y_start, y_end + 1):
                target_x, target_y = legacy_target(x, y, n=n, option=option)
                if target_x < p_start or target_x > p_end:
                    continue
                if target_y < y_start or target_y > y_end:
                    continue
                set_edge((x, y), (target_x, target_y), option)

    records: list[tuple[int, int, int, int, int]] = []
    for (x, y), ((target_x, target_y), option) in outgoing.items():
        # Match the old transform_nodes_2_adjacent behavior: right-neighbor
        # inter-plane sources are emitted only for planes 0..P-2.
        if not (0 <= x < p - 1 and 0 <= y < n and 0 <= target_x < p and 0 <= target_y < n):
            continue
        records.append((x, y, target_x, target_y, option))
    add_intra_ring_records(records, p=p, n=n)
    return make_edge_table_from_records(p=p, n=n, records=records)


def build_adjacency(edge_table: EdgeTable, total_nodes: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(int(total_nodes))]
    for idx in range(int(edge_table.num_edges)):
        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def precompute_hop_and_next(edge_table: EdgeTable, total_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    adjacency = build_adjacency(edge_table, total_nodes)
    n = int(total_nodes)
    dist = np.full((n, n), -1, dtype=np.int16)
    next_hop = np.full((n, n), -1, dtype=np.int32)

    for source in range(n):
        dist[source, source] = 0
        next_hop[source, source] = source
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            next_dist = int(dist[source, node]) + 1
            for neighbor in adjacency[node]:
                if dist[source, neighbor] >= 0:
                    continue
                dist[source, neighbor] = next_dist
                next_hop[source, neighbor] = neighbor if node == source else int(next_hop[source, node])
                queue.append(neighbor)
    return dist, next_hop


def reconstruct_path(next_hop: np.ndarray, source: int, target: int) -> list[int]:
    source = int(source)
    target = int(target)
    if source < 0 or target < 0 or source >= next_hop.shape[0] or target >= next_hop.shape[1]:
        return []
    if int(next_hop[source, target]) < 0:
        return []
    path = [source]
    current = source
    guard = 0
    while current != target:
        current = int(next_hop[current, target])
        if current < 0:
            return []
        path.append(current)
        guard += 1
        if guard > next_hop.shape[0]:
            return []
    return path


def group_nodes_for_step(group_data: dict, step: int, group_id: int) -> tuple[int, ...]:
    current = group_data.get(int(step), {}) if group_data else {}
    groups = current.get("groups", {}) if isinstance(current, dict) else {}
    return tuple(sorted(int(x) for x in groups.get(int(group_id), set()) or set()))


def build_state_index(
    *,
    group_data: dict,
    steps: list[int],
    source_group: int,
    target_group: int,
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], np.ndarray]:
    state_to_id: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    unique_states: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    state_ids = np.empty(len(steps), dtype=np.int32)
    for row, step in enumerate(steps):
        key = (
            group_nodes_for_step(group_data, step, int(source_group)),
            group_nodes_for_step(group_data, step, int(target_group)),
        )
        state_id = state_to_id.get(key)
        if state_id is None:
            state_id = len(unique_states)
            state_to_id[key] = state_id
            unique_states.append(key)
        state_ids[row] = int(state_id)
    return unique_states, state_ids


def summarize_state(dist: np.ndarray, source_nodes: tuple[int, ...], target_nodes: tuple[int, ...], state_id: int) -> StateSummary:
    source = np.asarray(source_nodes, dtype=np.int32)
    target = np.asarray(target_nodes, dtype=np.int32)
    if source.size == 0 or target.size == 0:
        return StateSummary(int(state_id), int(source.size), int(target.size), 0, 0.0, float("nan"), float("nan"), float("nan"))

    values = dist[np.ix_(source, target)]
    reachable = values >= 0
    if not bool(np.any(reachable)):
        return StateSummary(int(state_id), int(source.size), int(target.size), 0, 0.0, float("nan"), float("nan"), float("nan"))

    reachable_values = values[reachable].astype(np.float64)
    total = float(np.sum(reachable_values))
    count = int(reachable_values.size)
    return StateSummary(
        state_id=int(state_id),
        source_nodes=int(source.size),
        target_nodes=int(target.size),
        reachable_pairs=count,
        total_shortest_distance_hops=total,
        mean_shortest_distance_hops=float(total / count),
        min_shortest_distance_hops=float(np.min(reachable_values)),
        max_shortest_distance_hops=float(np.max(reachable_values)),
    )


def write_state_definitions(path: Path, unique_states: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> None:
    payload = [
        {
            "state_id": int(idx),
            "source_nodes": [int(x) for x in source_nodes],
            "target_nodes": [int(x) for x in target_nodes],
        }
        for idx, (source_nodes, target_nodes) in enumerate(unique_states)
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_state_summary(path: Path, summaries: list[StateSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(StateSummary.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(item.__dict__)


def write_step_summary(
    *,
    path: Path,
    steps: list[int],
    state_ids: np.ndarray,
    state_summaries: list[StateSummary],
    source_group: int,
    target_group: int,
) -> None:
    by_state = {int(item.state_id): item for item in state_summaries}
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "source_group_id",
            "target_group_id",
            "source_nodes",
            "target_nodes",
            "reachable_pairs",
            "total_shortest_distance_hops",
            "mean_shortest_distance_hops",
            "min_shortest_distance_hops",
            "max_shortest_distance_hops",
            "state_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, step in enumerate(steps):
            item = by_state[int(state_ids[row])]
            writer.writerow(
                {
                    "step": int(step),
                    "source_group_id": int(source_group),
                    "target_group_id": int(target_group),
                    "source_nodes": int(item.source_nodes),
                    "target_nodes": int(item.target_nodes),
                    "reachable_pairs": int(item.reachable_pairs),
                    "total_shortest_distance_hops": float(item.total_shortest_distance_hops),
                    "mean_shortest_distance_hops": float(item.mean_shortest_distance_hops),
                    "min_shortest_distance_hops": float(item.min_shortest_distance_hops),
                    "max_shortest_distance_hops": float(item.max_shortest_distance_hops),
                    "state_id": int(item.state_id),
                }
            )


def write_path_samples(
    *,
    path: Path,
    steps: list[int],
    group_data: dict,
    dist: np.ndarray,
    next_hop: np.ndarray,
    source_group: int,
    target_group: int,
    sample_steps: int,
    sample_pairs_per_step: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step", "source", "target", "distance_hops", "path", "path_indexed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps[: max(0, int(sample_steps))]:
            sources = group_nodes_for_step(group_data, step, int(source_group))
            targets = group_nodes_for_step(group_data, step, int(target_group))
            written = 0
            for source in sources:
                for target in targets:
                    distance = int(dist[int(source), int(target)])
                    if distance < 0:
                        continue
                    nodes = reconstruct_path(next_hop, int(source), int(target))
                    writer.writerow(
                        {
                            "step": int(step),
                            "source": int(source),
                            "target": int(target),
                            "distance_hops": distance,
                            "path": "->".join(str(x) for x in nodes),
                            "path_indexed": ",".join(f"{idx + 1}:{node}" for idx, node in enumerate(nodes)),
                        }
                    )
                    written += 1
                    if written >= int(sample_pairs_per_step):
                        break
                if written >= int(sample_pairs_per_step):
                    break


def compute_one_topology(
    *,
    topology_name: str,
    edge_table: EdgeTable,
    group_data: dict,
    steps: list[int],
    source_group: int,
    target_group: int,
    out_dir: Path,
    sample_steps: int,
    sample_pairs_per_step: int,
) -> list[StateSummary]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[shortest-path] {topology_name}: edges={edge_table.num_edges} precomputing all-pairs hop table...", flush=True)
    dist, next_hop = precompute_hop_and_next(edge_table, G60_CONFIG.total_sats)
    np.save(out_dir / "hop_dist.npy", dist)
    np.save(out_dir / "next_hop.npy", next_hop)
    write_edges_csv(edge_table, out_dir / "edges.csv")

    unique_states, state_ids = build_state_index(
        group_data=group_data,
        steps=steps,
        source_group=int(source_group),
        target_group=int(target_group),
    )
    print(f"[shortest-path] {topology_name}: unique_states={len(unique_states)} writing summaries...", flush=True)
    state_summaries = [
        summarize_state(dist, source_nodes, target_nodes, state_id)
        for state_id, (source_nodes, target_nodes) in enumerate(unique_states)
    ]
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "state_ids.npy", state_ids)
    write_state_definitions(out_dir / "state_definitions.json", unique_states)
    write_state_summary(out_dir / "state_summary.csv", state_summaries)
    write_step_summary(
        path=out_dir / "step_summary.csv",
        steps=steps,
        state_ids=state_ids,
        state_summaries=state_summaries,
        source_group=int(source_group),
        target_group=int(target_group),
    )
    write_path_samples(
        path=out_dir / "path_samples.csv",
        steps=steps,
        group_data=group_data,
        dist=dist,
        next_hop=next_hop,
        source_group=int(source_group),
        target_group=int(target_group),
        sample_steps=int(sample_steps),
        sample_pairs_per_step=int(sample_pairs_per_step),
    )

    means = np.asarray([state_summaries[int(state_id)].mean_shortest_distance_hops for state_id in state_ids], dtype=np.float32)
    meta = {
        "topology": topology_name,
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "source_group_id": int(source_group),
        "source_group_name": group_name(source_group),
        "target_group_id": int(target_group),
        "target_group_name": group_name(target_group),
        "num_steps": int(len(steps)),
        "start": int(min(steps)) if steps else None,
        "end": int(max(steps)) if steps else None,
        "num_edges": int(edge_table.num_edges),
        "unique_states": int(len(unique_states)),
        "mean_hops_min": float(np.nanmin(means)) if means.size else float("nan"),
        "mean_hops_max": float(np.nanmax(means)) if means.size else float("nan"),
        "method": "unweighted all-pairs BFS hop table, then per-step group lookup",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[shortest-path] {topology_name}: mean_hops=({meta['mean_hops_min']:.4f}, {meta['mean_hops_max']:.4f}) "
        f"out_dir={out_dir}",
        flush=True,
    )
    return state_summaries


def read_step_means(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["step"])] = float(row["mean_shortest_distance_hops"])
    return out


def write_comparison(*, out_dir: Path, topology_dirs: dict[str, Path], steps: list[int]) -> None:
    means_by_topology = {name: read_step_means(path / "step_summary.csv") for name, path in topology_dirs.items()}
    comparison_path = out_dir / "compare_step_summary.csv"
    names = list(topology_dirs)
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step"] + [f"{name}_mean_hops" for name in names]
        if len(names) == 2:
            fieldnames.append(f"{names[0]}_minus_{names[1]}")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            row = {"step": int(step)}
            for name in names:
                row[f"{name}_mean_hops"] = means_by_topology[name].get(int(step), float("nan"))
            if len(names) == 2:
                row[f"{names[0]}_minus_{names[1]}"] = row[f"{names[0]}_mean_hops"] - row[f"{names[1]}_mean_hops"]
            writer.writerow(row)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4.8))
        x = np.asarray(steps, dtype=np.int64)
        for name in names:
            y = np.asarray([means_by_topology[name].get(int(step), np.nan) for step in steps], dtype=np.float32)
            ax.plot(x, y, linewidth=1.0, label=name)
        ax.set_xlabel("time step (s)")
        ax.set_ylabel("China-Europe mean shortest path (hops)")
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "mean_shortest_path_timeseries.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[shortest-path] plot skipped: {exc}", flush=True)


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"[shortest-path] steps={len(steps)} group_steps={len(group_data)} out_dir={out_dir}", flush=True)

    motif_edge_table = build_support_motif_edge_table(
        motif_config=Path(args.motif_config),
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
    )
    gridplus_edge_table = build_legacy_gridplus_edge_table(motif_json=Path(args.gridplus_config))

    topology_dirs = {
        "support_motif_DAD_Cxx": out_dir / "support_motif_DAD_Cxx",
        "gridplus": out_dir / "gridplus",
    }
    compute_one_topology(
        topology_name="support_motif_DAD_Cxx",
        edge_table=motif_edge_table,
        group_data=group_data,
        steps=steps,
        source_group=int(args.source_group),
        target_group=int(args.target_group),
        out_dir=topology_dirs["support_motif_DAD_Cxx"],
        sample_steps=int(args.sample_steps),
        sample_pairs_per_step=int(args.sample_pairs_per_step),
    )
    compute_one_topology(
        topology_name="gridplus",
        edge_table=gridplus_edge_table,
        group_data=group_data,
        steps=steps,
        source_group=int(args.source_group),
        target_group=int(args.target_group),
        out_dir=topology_dirs["gridplus"],
        sample_steps=int(args.sample_steps),
        sample_pairs_per_step=int(args.sample_pairs_per_step),
    )
    write_comparison(out_dir=out_dir, topology_dirs=topology_dirs, steps=steps)

    meta = {
        "constellation": G60_CONFIG.name,
        "source_group_id": int(args.source_group),
        "source_group_name": group_name(args.source_group),
        "target_group_id": int(args.target_group),
        "target_group_name": group_name(args.target_group),
        "start": int(args.start),
        "end": int(args.end),
        "end_semantics": "inclusive",
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "motif_config": str(Path(args.motif_config)),
        "gridplus_config": str(Path(args.gridplus_config)),
        "outputs": {name: str(path) for name, path in topology_dirs.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[shortest-path] done | comparison={out_dir / 'compare_step_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
