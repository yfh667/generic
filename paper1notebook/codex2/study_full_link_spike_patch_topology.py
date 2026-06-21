from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra, shortest_path  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv  # noqa: E402
from src.link_delay.module.position_cache import PositionCacheStore  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.edge_betweenness import edge_usage_share_from_counts  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_metrics.module.weighted_edge_betweenness import (  # noqa: E402
    build_weighted_adjacency,
    weighted_edge_betweenness_between_node_sets,
)
from src.topology_workflow.module.edge_tables import (  # noqa: E402
    INTRA_OPTION,
    build_full_option_plus_intra_edge_table,
    build_motif_text_edge_table,
    make_edge_table_from_records,
)
from src.topology_workflow.module.shortest_delay import (  # noqa: E402
    build_bidirectional_sparse_parts,
    build_weight_lookup,
    edge_weights_for_step,
)


GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\_position_cache\cache_0_86164_1s"
)
MOTIF_LIBRARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
METRIC_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3"
    r"\paper_style_808_no_region_grid_t0_86160_stride60"
)
SPIKE_SUMMARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\china_europe_t0_86160_stride60"
    r"\top_common_delay_spikes_summary.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\full_link_guided_spike_patch"
)
FULL_LINK_BETWEENNESS_CACHE = Path(r"E:\paper11\data\linshi\g60_multi_region_weighted_betweenness_t0_86164_stride60")

PAIR_KEY = "china_europe"
SOURCE_GROUP_ID = 2
TARGET_GROUP_ID = 3
BASE_MOTIF_ID = 56
ALT_MOTIF_ID = 61


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study why full_link suppresses motif000056/000061 local spikes, then build a "
            "one-right-neighbor topology guided by full_link shortest-delay edge usage."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--top-spikes", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delay-store-dir", type=Path, default=DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=POSITION_CACHE_DIR)
    parser.add_argument("--group-xml", type=Path, default=GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=GROUP_CACHE_DIR)
    parser.add_argument("--metric-root", type=Path, default=METRIC_ROOT)
    parser.add_argument("--motif-library", type=Path, default=MOTIF_LIBRARY_CSV)
    parser.add_argument("--spike-summary", type=Path, default=SPIKE_SUMMARY_CSV)
    parser.add_argument("--full-link-betweenness-cache", type=Path, default=FULL_LINK_BETWEENNESS_CACHE)
    parser.add_argument("--spike-weight", type=float, default=1000.0)
    parser.add_argument("--keep-base-bonus", type=float, default=1.0)
    parser.add_argument("--keep-alt-bonus", type=float, default=0.10)
    parser.add_argument("--fill-bonus", type=float, default=0.001)
    parser.add_argument("--option0-bonus", type=float, default=0.0002)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src <= dst else (dst, src)


def node_xy(node: int) -> str:
    return f"({int(node) // int(G60_CONFIG.N)},{int(node) % int(G60_CONFIG.N)})"


def load_motif_text(motif_id: int, motif_library_csv: Path) -> str:
    with Path(motif_library_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["motif_id"]) == int(motif_id):
                return str(row["motif"])
    raise KeyError(f"motif_id={motif_id} not found in {motif_library_csv}")


def edge_keys_from_table(edge_table: EdgeTable) -> set[tuple[int, int]]:
    return {edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])) for idx in range(edge_table.num_edges)}


def inter_owner_target(edge_table: EdgeTable, edge_idx: int) -> tuple[int, int] | None:
    if int(edge_table.option[edge_idx]) == INTRA_OPTION:
        return None
    src = int(edge_table.src[edge_idx])
    dst = int(edge_table.dst[edge_idx])
    sp = int(edge_table.src_plane[edge_idx])
    dp = int(edge_table.dst_plane[edge_idx])
    if sp < dp:
        return src, dst
    if dp < sp:
        return dst, src
    return None


def records_from_edge_indices(edge_table: EdgeTable, edge_indices: list[int]) -> list[tuple[int, int, int, int, int]]:
    records: list[tuple[int, int, int, int, int]] = []
    for idx in edge_indices:
        records.append(
            (
                int(edge_table.src_plane[idx]),
                int(edge_table.src_y[idx]),
                int(edge_table.dst_plane[idx]),
                int(edge_table.dst_y[idx]),
                int(edge_table.option[idx]),
            )
        )
    return records


def add_intra_records(records: list[tuple[int, int, int, int, int]]) -> None:
    for plane in range(int(G60_CONFIG.P)):
        for y in range(int(G60_CONFIG.N)):
            records.append((plane, y, plane, (y + 1) % int(G60_CONFIG.N), INTRA_OPTION))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_edge_table_csv(path: Path) -> EdgeTable:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=[str(i + 1) for i in range(int(G60_CONFIG.total_sats))],
    )


def assert_edge_tables_align(left: EdgeTable, right: EdgeTable) -> None:
    if int(left.num_edges) != int(right.num_edges):
        raise ValueError(f"edge count mismatch: {left.num_edges} != {right.num_edges}")
    for name in ("src", "dst", "option"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"edge table column {name!r} does not align with betweenness cache")


def load_group_data_for_steps(args: argparse.Namespace, steps: list[int]) -> dict:
    return load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )


def analyze_full_link_spike_edges(
    *,
    args: argparse.Namespace,
    full_edge_table: EdgeTable,
    base_keys: set[tuple[int, int]],
    alt_keys: set[tuple[int, int]],
    group_data: dict,
    spike_steps: list[int],
    out_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    delay_store = FullLinkDelayStore(Path(args.delay_store_dir))
    position_store = PositionCacheStore(Path(args.position_cache_dir))
    lookup = build_weight_lookup(
        full_edge_table,
        delay_store,
        config=G60_CONFIG,
        allow_intra_fallback=True,
    )
    adjacency = build_weighted_adjacency(full_edge_table, int(G60_CONFIG.total_sats))
    edge_score = np.zeros(int(full_edge_table.num_edges), dtype=np.float64)
    usage_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for step in spike_steps:
        delay_row = delay_store.row_for_time_step(int(step))
        position_row = position_store.row_for_time_step(int(step))
        weights = edge_weights_for_step(
            edge_table=full_edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_row),
            position_row=int(position_row),
        )
        sources = group_nodes_for_step(group_data, int(step), SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), TARGET_GROUP_ID)
        values, summary, samples = weighted_edge_betweenness_between_node_sets(
            full_edge_table,
            total_nodes=int(G60_CONFIG.total_sats),
            source_nodes=sources,
            target_nodes=targets,
            weights=weights,
            adjacency=adjacency,
            sample_path_limit=24,
        )
        shares = edge_usage_share_from_counts(values, summary.reachable_pairs)
        edge_score += np.asarray(shares, dtype=np.float64)

        for option in (-1, 0, 1, 2, 4):
            mask = np.asarray(full_edge_table.option == int(option))
            used = np.asarray(values > 0.0) & mask
            option_rows.append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "option": int(option),
                    "nonzero_edges": int(np.count_nonzero(used)),
                    "usage_sum": float(np.sum(values[mask])),
                    "usage_share_sum": float(np.sum(shares[mask])),
                    "reachable_pairs": int(summary.reachable_pairs),
                    "mean_shortest_delay_ms": float(summary.mean_shortest_weight),
                }
            )

        nonzero = np.flatnonzero(values > 0.0)
        for idx in nonzero:
            owner_target = inter_owner_target(full_edge_table, int(idx))
            src = int(full_edge_table.src[idx])
            dst = int(full_edge_table.dst[idx])
            key = edge_key(src, dst)
            usage_rows.append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "edge_idx": int(idx),
                    "src": src,
                    "src_xy": node_xy(src),
                    "dst": dst,
                    "dst_xy": node_xy(dst),
                    "option": int(full_edge_table.option[idx]),
                    "owner": None if owner_target is None else int(owner_target[0]),
                    "owner_xy": "" if owner_target is None else node_xy(owner_target[0]),
                    "right_target": None if owner_target is None else int(owner_target[1]),
                    "right_target_xy": "" if owner_target is None else node_xy(owner_target[1]),
                    "usage": float(values[idx]),
                    "usage_share": float(shares[idx]),
                    "delay_ms": float(weights[idx]),
                    "in_motif000056": int(key in base_keys),
                    "in_motif000061": int(key in alt_keys),
                }
            )
        for sample in samples:
            sample_rows.append(
                {
                    "step": int(step),
                    "source": int(sample["source"]),
                    "source_xy": node_xy(int(sample["source"])),
                    "target": int(sample["target"]),
                    "target_xy": node_xy(int(sample["target"])),
                    "delay_ms": float(sample["shortest_weight"]),
                    "path": "->".join(str(x) for x in sample["path"]),
                    "path_xy": "->".join(node_xy(int(x)) for x in sample["path"]),
                    "edge_indices": " ".join(str(x) for x in sample["edge_indices"]),
                }
            )

    usage_df = pd.DataFrame(usage_rows)
    option_df = pd.DataFrame(option_rows)
    if not usage_df.empty:
        summary_df = (
            usage_df.groupby(["edge_idx", "src", "src_xy", "dst", "dst_xy", "option"], as_index=False)
            .agg(
                spike_usage_sum=("usage", "sum"),
                spike_usage_share_sum=("usage_share", "sum"),
                spike_steps_used=("step", "nunique"),
                mean_delay_ms_when_used=("delay_ms", "mean"),
                in_motif000056=("in_motif000056", "max"),
                in_motif000061=("in_motif000061", "max"),
                owner=("owner", "first"),
                owner_xy=("owner_xy", "first"),
                right_target=("right_target", "first"),
                right_target_xy=("right_target_xy", "first"),
            )
            .sort_values(["spike_usage_share_sum", "spike_usage_sum"], ascending=False)
        )
    else:
        summary_df = pd.DataFrame()
    usage_df.to_csv(out_dir / "full_link_spike_used_edges_by_step.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "full_link_spike_used_edges_summary.csv", index=False, encoding="utf-8-sig")
    option_df.to_csv(out_dir / "full_link_spike_option_usage_by_step.csv", index=False, encoding="utf-8-sig")
    write_rows(out_dir / "full_link_spike_path_samples.csv", sample_rows)
    return edge_score, usage_df, option_df


def build_patch_topology(
    *,
    full_edge_table: EdgeTable,
    edge_score: np.ndarray,
    base_keys: set[tuple[int, int]],
    alt_keys: set[tuple[int, int]],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[EdgeTable, pd.DataFrame]:
    valid_edges: list[dict[str, Any]] = []
    for idx in range(int(full_edge_table.num_edges)):
        owner_target = inter_owner_target(full_edge_table, idx)
        if owner_target is None:
            continue
        src = int(full_edge_table.src[idx])
        dst = int(full_edge_table.dst[idx])
        key = edge_key(src, dst)
        option = int(full_edge_table.option[idx])
        score = float(args.fill_bonus)
        if option == 0:
            score += float(args.option0_bonus)
        if key in base_keys:
            score += float(args.keep_base_bonus)
        if key in alt_keys:
            score += float(args.keep_alt_bonus)
        score += float(args.spike_weight) * float(edge_score[idx])
        valid_edges.append(
            {
                "edge_idx": int(idx),
                "owner": int(owner_target[0]),
                "right_target": int(owner_target[1]),
                "score": float(score),
                "spike_score": float(edge_score[idx]),
                "option": option,
                "in_motif000056": int(key in base_keys),
                "in_motif000061": int(key in alt_keys),
                "src": src,
                "dst": dst,
            }
        )
    owners = sorted({int(item["owner"]) for item in valid_edges})
    targets = sorted({int(item["right_target"]) for item in valid_edges})
    owner_to_row = {owner: idx for idx, owner in enumerate(owners)}
    target_to_col = {target: idx for idx, target in enumerate(targets)}

    # Each owner gets one assignment. Real target columns enforce one-left-neighbor.
    # Per-owner dummy columns allow the optimizer to skip an owner if every real edge is worse than no edge.
    invalid_cost = 1.0e9
    cost = np.full((len(owners), len(targets) + len(owners)), invalid_cost, dtype=np.float64)
    score_by_owner_target: dict[tuple[int, int], dict[str, Any]] = {}
    for item in valid_edges:
        row = owner_to_row[int(item["owner"])]
        col = target_to_col[int(item["right_target"])]
        # If duplicate undirected records ever appear, keep the higher score.
        current = score_by_owner_target.get((row, col))
        if current is None or float(item["score"]) > float(current["score"]):
            score_by_owner_target[(row, col)] = item
            cost[row, col] = -float(item["score"])
    for row in range(len(owners)):
        cost[row, len(targets) + row] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    selected: list[dict[str, Any]] = []
    selected_edge_indices: list[int] = []
    for row, col in zip(row_ind, col_ind):
        if int(col) >= len(targets):
            continue
        item = score_by_owner_target.get((int(row), int(col)))
        if item is None or float(item["score"]) <= 0.0:
            continue
        selected.append(dict(item))
        selected_edge_indices.append(int(item["edge_idx"]))

    inter_records = records_from_edge_indices(full_edge_table, selected_edge_indices)
    all_records = list(inter_records)
    add_intra_records(all_records)
    patch_edge_table = make_edge_table_from_records(p=int(G60_CONFIG.P), n=int(G60_CONFIG.N), records=all_records)

    selected_df = pd.DataFrame(selected).sort_values(["spike_score", "score"], ascending=False)
    if not selected_df.empty:
        selected_df["owner_xy"] = selected_df["owner"].map(node_xy)
        selected_df["right_target_xy"] = selected_df["right_target"].map(node_xy)
        selected_df["src_xy"] = selected_df["src"].map(node_xy)
        selected_df["dst_xy"] = selected_df["dst"].map(node_xy)
    selected_df.to_csv(out_dir / "spike_patch_selected_inter_edges.csv", index=False, encoding="utf-8-sig")
    write_edges_csv(patch_edge_table, out_dir / "spike_patch_edges.csv")

    owners_count = int(selected_df["owner"].nunique()) if not selected_df.empty else 0
    targets_count = int(selected_df["right_target"].nunique()) if not selected_df.empty else 0
    meta = {
        "topology": "full_link_guided_spike_patch_on_motif000056",
        "selection_rule": (
            "weighted assignment: maximize spike_weight*full_link_spike_usage_share "
            "+ keep bonuses + tiny fill bonuses, subject to one outgoing right-link owner "
            "and one incoming right-link target"
        ),
        "inter_edges": int(len(selected_edge_indices)),
        "intra_edges": int(G60_CONFIG.P * G60_CONFIG.N),
        "total_edges": int(patch_edge_table.num_edges),
        "unique_owners": owners_count,
        "unique_right_targets": targets_count,
        "violates_one_right": bool(owners_count != len(selected_edge_indices)),
        "violates_one_left": bool(targets_count != len(selected_edge_indices)),
        "parameters": {
            "spike_weight": float(args.spike_weight),
            "keep_base_bonus": float(args.keep_base_bonus),
            "keep_alt_bonus": float(args.keep_alt_bonus),
            "fill_bonus": float(args.fill_bonus),
            "option0_bonus": float(args.option0_bonus),
        },
    }
    (out_dir / "spike_patch_selection_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return patch_edge_table, selected_df


def greedy_select_inter_edge_indices(
    *,
    full_edge_table: EdgeTable,
    edge_score: np.ndarray,
    base_keys: set[tuple[int, int]],
    alt_keys: set[tuple[int, int]],
    args: argparse.Namespace,
) -> list[int]:
    candidates: list[tuple[float, int, int, int]] = []
    for idx in range(int(full_edge_table.num_edges)):
        owner_target = inter_owner_target(full_edge_table, idx)
        if owner_target is None:
            continue
        src = int(full_edge_table.src[idx])
        dst = int(full_edge_table.dst[idx])
        key = edge_key(src, dst)
        option = int(full_edge_table.option[idx])
        score = float(args.fill_bonus)
        if option == 0:
            score += float(args.option0_bonus)
        if key in base_keys:
            score += float(args.keep_base_bonus)
        if key in alt_keys:
            score += float(args.keep_alt_bonus)
        score += float(args.spike_weight) * float(edge_score[idx])
        candidates.append((float(score), int(idx), int(owner_target[0]), int(owner_target[1])))
    candidates.sort(key=lambda item: (-item[0], item[2], item[3], item[1]))
    used_owner: set[int] = set()
    used_target: set[int] = set()
    selected: list[int] = []
    for score, idx, owner, target in candidates:
        if score <= 0.0 or owner in used_owner or target in used_target:
            continue
        selected.append(int(idx))
        used_owner.add(int(owner))
        used_target.add(int(target))
    return selected


def evaluate_dynamic_one_right_from_full_link_cache(
    *,
    args: argparse.Namespace,
    full_edge_table: EdgeTable,
    base_keys: set[tuple[int, int]],
    alt_keys: set[tuple[int, int]],
    group_data: dict,
    steps: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    cache_dir = Path(args.full_link_betweenness_cache)
    cache_table = read_edge_table_csv(cache_dir / "edges.csv")
    try:
        assert_edge_tables_align(full_edge_table, cache_table)
    except ValueError as exc:
        print(f"[dynamic-one-right] using cache edge order because current full table differs: {exc}", flush=True)
        full_edge_table = cache_table
    cache_steps = np.asarray(np.load(cache_dir / PAIR_KEY / "time_indices.npy", mmap_mode="r"), dtype=np.int64)
    cache_values = np.load(cache_dir / PAIR_KEY / "edge_betweenness.npy", mmap_mode="r")
    cache_row_by_step = {int(step): idx for idx, step in enumerate(cache_steps.tolist())}

    delay_store = FullLinkDelayStore(Path(args.delay_store_dir))
    position_store = PositionCacheStore(Path(args.position_cache_dir))
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(args.stride))
    position_rows = position_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(args.stride))
    lookup = build_weight_lookup(full_edge_table, delay_store, config=G60_CONFIG, allow_intra_fallback=True)

    intra_indices = np.flatnonzero(np.asarray(full_edge_table.option) == INTRA_OPTION).astype(np.int32)
    all_src = np.asarray(full_edge_table.src, dtype=np.int32)
    all_dst = np.asarray(full_edge_table.dst, dtype=np.int32)

    active_mask = np.zeros((len(steps), int(full_edge_table.num_edges)), dtype=bool)
    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    previous_inter: set[int] | None = None
    started = time.perf_counter()

    for local_idx, step in enumerate(steps):
        cache_row = cache_row_by_step.get(int(step))
        if cache_row is None:
            raise KeyError(f"step {step} not found in full-link betweenness cache {cache_dir}")
        raw_score = np.asarray(cache_values[int(cache_row)], dtype=np.float64)
        selected_inter = greedy_select_inter_edge_indices(
            full_edge_table=full_edge_table,
            edge_score=raw_score,
            base_keys=base_keys,
            alt_keys=alt_keys,
            args=args,
        )
        selected = np.asarray(selected_inter + intra_indices.tolist(), dtype=np.int32)
        active_mask[local_idx, selected] = True
        current_inter = set(int(x) for x in selected_inter)
        if previous_inter is None:
            transition_rows.append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "new_inter_edges": int(len(current_inter)),
                    "removed_inter_edges": 0,
                    "kept_inter_edges": 0,
                    "symmetric_diff_inter_edges": int(len(current_inter)),
                }
            )
        else:
            transition_rows.append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "new_inter_edges": int(len(current_inter - previous_inter)),
                    "removed_inter_edges": int(len(previous_inter - current_inter)),
                    "kept_inter_edges": int(len(previous_inter & current_inter)),
                    "symmetric_diff_inter_edges": int(len(current_inter ^ previous_inter)),
                }
            )
        previous_inter = current_inter

        sources = group_nodes_for_step(group_data, int(step), SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), TARGET_GROUP_ID)
        if not sources or not targets:
            rows.append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "selected_inter_edges": int(len(selected_inter)),
                    "mean_shortest_hops_dynamic_one_right": math.nan,
                    "mean_shortest_delay_ms_dynamic_one_right": math.nan,
                    "reachable_pairs_hops_dynamic_one_right": 0,
                    "reachable_pairs_delay_dynamic_one_right": 0,
                }
            )
            continue

        weights_full = edge_weights_for_step(
            edge_table=full_edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[local_idx]),
            position_row=int(position_rows[local_idx]),
        )
        src = all_src[selected]
        dst = all_dst[selected]
        row_index = np.concatenate([src, dst])
        col_index = np.concatenate([dst, src])
        graph_hops = csr_matrix(
            (np.ones(selected.size * 2, dtype=np.float32), (row_index, col_index)),
            shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
        )
        selected_weights = weights_full[selected]
        graph_delay = csr_matrix(
            (
                np.concatenate([selected_weights, selected_weights]).astype(np.float32, copy=False),
                (row_index, col_index),
            ),
            shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
        )
        src_arr = np.asarray(sources, dtype=np.int32)
        tgt_arr = np.asarray(targets, dtype=np.int32)
        hop_matrix = dijkstra(graph_hops, directed=False, unweighted=True, indices=src_arr)
        hop_values = np.atleast_2d(np.asarray(hop_matrix, dtype=np.float64))[:, tgt_arr]
        hop_finite = hop_values[np.isfinite(hop_values)]
        delay_matrix = dijkstra(graph_delay, directed=False, indices=src_arr)
        delay_values = np.atleast_2d(np.asarray(delay_matrix, dtype=np.float64))[:, tgt_arr]
        delay_finite = delay_values[np.isfinite(delay_values)]
        rows.append(
            {
                "step": int(step),
                "hour": float(step) / 3600.0,
                "selected_inter_edges": int(len(selected_inter)),
                "mean_shortest_hops_dynamic_one_right": float(np.mean(hop_finite)) if hop_finite.size else math.nan,
                "mean_shortest_delay_ms_dynamic_one_right": (
                    float(np.mean(delay_finite)) if delay_finite.size else math.nan
                ),
                "reachable_pairs_hops_dynamic_one_right": int(hop_finite.size),
                "reachable_pairs_delay_dynamic_one_right": int(delay_finite.size),
            }
        )
        if int(args.progress_every) > 0 and (
            local_idx + 1 == len(steps) or (local_idx + 1) % int(args.progress_every) == 0
        ):
            print(
                f"[dynamic-one-right] evaluated {local_idx + 1}/{len(steps)} "
                f"step={step} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    np.save(out_dir / "dynamic_one_right_active_full_edge_mask.npy", active_mask)
    df = pd.DataFrame(rows)
    trans = pd.DataFrame(transition_rows)
    df.to_csv(out_dir / "dynamic_one_right_from_full_link_timeseries.csv", index=False, encoding="utf-8-sig")
    trans.to_csv(out_dir / "dynamic_one_right_transition_counts.csv", index=False, encoding="utf-8-sig")
    return df


def evaluate_patch_timeseries(
    *,
    args: argparse.Namespace,
    edge_table: EdgeTable,
    group_data: dict,
    steps: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    delay_store = FullLinkDelayStore(Path(args.delay_store_dir))
    position_store = PositionCacheStore(Path(args.position_cache_dir))
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(args.stride))
    position_rows = position_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(args.stride))
    lookup = build_weight_lookup(edge_table, delay_store, config=G60_CONFIG, allow_intra_fallback=True)
    row_index, col_index = build_bidirectional_sparse_parts(edge_table)

    unweighted_graph = csr_matrix(
        (
            np.ones(int(edge_table.num_edges) * 2, dtype=np.float32),
            (row_index, col_index),
        ),
        shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
    )
    hop_all = shortest_path(unweighted_graph, directed=False, unweighted=True)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for local_idx, step in enumerate(steps):
        sources = group_nodes_for_step(group_data, int(step), SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), TARGET_GROUP_ID)
        if not sources or not targets:
            rows.append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "source_nodes": len(sources),
                    "target_nodes": len(targets),
                    "reachable_pairs_hops": 0,
                    "mean_shortest_hops_spike_patch": math.nan,
                    "reachable_pairs_delay": 0,
                    "mean_shortest_delay_ms_spike_patch": math.nan,
                }
            )
            continue

        src_arr = np.asarray(sources, dtype=np.int32)
        tgt_arr = np.asarray(targets, dtype=np.int32)
        hop_values = np.asarray(hop_all[np.ix_(src_arr, tgt_arr)], dtype=np.float64)
        hop_finite = hop_values[np.isfinite(hop_values)]

        weights = edge_weights_for_step(
            edge_table=edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[local_idx]),
            position_row=int(position_rows[local_idx]),
        )
        graph = csr_matrix(
            (
                np.concatenate([weights, weights]).astype(np.float32, copy=False),
                (row_index, col_index),
            ),
            shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
        )
        delay_matrix = dijkstra(graph, directed=False, indices=src_arr)
        delay_matrix = np.atleast_2d(np.asarray(delay_matrix, dtype=np.float64))[:, tgt_arr]
        delay_finite = delay_matrix[np.isfinite(delay_matrix)]
        rows.append(
            {
                "step": int(step),
                "hour": float(step) / 3600.0,
                "source_nodes": len(sources),
                "target_nodes": len(targets),
                "reachable_pairs_hops": int(hop_finite.size),
                "mean_shortest_hops_spike_patch": float(np.mean(hop_finite)) if hop_finite.size else math.nan,
                "reachable_pairs_delay": int(delay_finite.size),
                "mean_shortest_delay_ms_spike_patch": float(np.mean(delay_finite)) if delay_finite.size else math.nan,
            }
        )
        if int(args.progress_every) > 0 and (
            local_idx + 1 == len(steps) or (local_idx + 1) % int(args.progress_every) == 0
        ):
            print(
                f"[spike-patch] evaluated {local_idx + 1}/{len(steps)} "
                f"step={step} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "spike_patch_china_europe_timeseries.csv", index=False, encoding="utf-8-sig")
    return df


def read_reference_metrics(metric_root: Path) -> pd.DataFrame:
    delay_path = (
        Path(metric_root)
        / "shortest_delay"
        / PAIR_KEY
        / "compare_mean_shortest_delay_ms_808_strict_reachable.csv"
    )
    hops_path = Path(metric_root) / "shortest_hops" / PAIR_KEY / "compare_mean_shortest_hops_808_strict_reachable.csv"
    usecols = ["step", "full_link", "combined_motif_000056", "combined_motif_000061"]
    delay = pd.read_csv(delay_path, usecols=usecols).rename(
        columns={
            "full_link": "mean_shortest_delay_ms_full_link",
            "combined_motif_000056": "mean_shortest_delay_ms_motif000056",
            "combined_motif_000061": "mean_shortest_delay_ms_motif000061",
        }
    )
    hops = pd.read_csv(hops_path, usecols=usecols).rename(
        columns={
            "full_link": "mean_shortest_hops_full_link",
            "combined_motif_000056": "mean_shortest_hops_motif000056",
            "combined_motif_000061": "mean_shortest_hops_motif000061",
        }
    )
    out = hops.merge(delay, on="step", validate="one_to_one")
    out["hour"] = out["step"] / 3600.0
    return out


def local_jump_summary(df: pd.DataFrame, spike_steps: list[int], out_dir: Path) -> pd.DataFrame:
    by_step = df.set_index("step")
    rows: list[dict[str, Any]] = []
    stride = int(df["step"].iloc[1] - df["step"].iloc[0])
    topologies = ("motif000056", "motif000061", "spike_patch", "dynamic_one_right", "full_link")
    metrics = ("mean_shortest_delay_ms", "mean_shortest_hops")
    for step in spike_steps:
        prev = int(step) - stride
        if prev not in by_step.index or int(step) not in by_step.index:
            continue
        for metric in metrics:
            for topo in topologies:
                col = f"{metric}_{topo}"
                if col not in by_step.columns:
                    continue
                before = float(by_step.loc[prev, col])
                after = float(by_step.loc[int(step), col])
                jump = after - before
                rows.append(
                    {
                        "step": int(step),
                        "hour": float(step) / 3600.0,
                        "metric": metric,
                        "topology": topo,
                        "value_before": before,
                        "value_after": after,
                        "signed_jump": jump,
                        "positive_jump": max(0.0, jump) if math.isfinite(jump) else math.nan,
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "local_spike_jump_detail.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby(["metric", "topology"], as_index=False)
        .agg(
            mean_positive_jump=("positive_jump", "mean"),
            max_positive_jump=("positive_jump", "max"),
            sum_positive_jump=("positive_jump", "sum"),
            mean_after=("value_after", "mean"),
            max_after=("value_after", "max"),
        )
        .sort_values(["metric", "mean_positive_jump"])
    )
    summary.to_csv(out_dir / "local_spike_jump_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_option_usage(option_df: pd.DataFrame, out_path: Path) -> None:
    if option_df.empty:
        return
    pivot = option_df.pivot_table(index="hour", columns="option", values="usage_share_sum", aggfunc="sum").fillna(0.0)
    colors = {-1: "#94a3b8", 0: "#2563eb", 1: "#16a34a", 2: "#d97706", 4: "#be123c"}
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=170)
    bottom = np.zeros(len(pivot), dtype=float)
    x = pivot.index.to_numpy(dtype=float)
    for option in [col for col in (-1, 0, 1, 2, 4) if col in pivot.columns]:
        values = pivot[option].to_numpy(dtype=float)
        ax.bar(x, values, width=0.035, bottom=bottom, color=colors.get(option, "#64748b"), label=f"option {option}")
        bottom += values
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("full_link shortest-delay edge usage share sum")
    ax.set_title("Full-link path option composition at common spike steps")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=5, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric(df: pd.DataFrame, spike_steps: list[int], out_path: Path, *, metric: str, ylabel: str) -> None:
    styles = {
        "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.15, 0.82),
        "motif000061": ("motif000061 DCD | C--", "#C1121F", 1.05, 0.70),
        "spike_patch": ("full-link guided spike patch", "#d97706", 1.7, 0.98),
        "dynamic_one_right": ("dynamic one-right guided by full_link", "#0f766e", 1.55, 0.96),
        "full_link": ("full_link", "#111827", 1.35, 0.85),
    }
    fig, ax = plt.subplots(figsize=(15.8, 6.6), dpi=180)
    x = df["hour"].to_numpy(dtype=float)
    for topo, (label, color, width, alpha) in styles.items():
        col = f"{metric}_{topo}"
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=float)
        ax.plot(x, values, color=color, linewidth=width, alpha=alpha, label=f"{label} | mean={np.nanmean(values):.3f}")
    for step in spike_steps:
        ax.axvline(float(step) / 3600.0, color="#64748b", linewidth=0.55, alpha=0.18)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe: full-link guided spike patch vs static motifs, {ylabel}")
    ax.grid(True, alpha=0.24, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    spike_df = pd.read_csv(args.spike_summary).head(int(args.top_spikes))
    spike_steps = [int(x) for x in spike_df["step"].tolist() if int(args.start) <= int(x) <= int(args.end)]
    if not spike_steps:
        raise ValueError("no spike steps fall inside the requested interval")

    base_motif = load_motif_text(BASE_MOTIF_ID, Path(args.motif_library))
    alt_motif = load_motif_text(ALT_MOTIF_ID, Path(args.motif_library))
    base_table = build_motif_text_edge_table(motif_text=base_motif, config=G60_CONFIG, add_intra_ring=True)
    alt_table = build_motif_text_edge_table(motif_text=alt_motif, config=G60_CONFIG, add_intra_ring=True)
    full_table = build_full_option_plus_intra_edge_table(config=G60_CONFIG, options=(0, 1, 2, 4), add_intra_ring=True)
    base_keys = edge_keys_from_table(base_table)
    alt_keys = edge_keys_from_table(alt_table)

    print(
        f"[spike-patch] spikes={spike_steps} base={base_motif} alt={alt_motif} "
        f"full_edges={full_table.num_edges}",
        flush=True,
    )
    group_data_all = load_group_data_for_steps(args, steps)
    edge_score, _usage_df, option_df = analyze_full_link_spike_edges(
        args=args,
        full_edge_table=full_table,
        base_keys=base_keys,
        alt_keys=alt_keys,
        group_data=group_data_all,
        spike_steps=spike_steps,
        out_dir=out_dir,
    )
    plot_option_usage(option_df, out_dir / "full_link_spike_option_usage_share.png")

    patch_table, selected_df = build_patch_topology(
        full_edge_table=full_table,
        edge_score=edge_score,
        base_keys=base_keys,
        alt_keys=alt_keys,
        args=args,
        out_dir=out_dir,
    )
    print(
        f"[spike-patch] selected inter={len(selected_df)} total_edges={patch_table.num_edges} "
        f"top_spike_edges={int(np.count_nonzero(edge_score > 0.0))}",
        flush=True,
    )

    patch_df = evaluate_patch_timeseries(
        args=args,
        edge_table=patch_table,
        group_data=group_data_all,
        steps=steps,
        out_dir=out_dir,
    )
    dynamic_df = evaluate_dynamic_one_right_from_full_link_cache(
        args=args,
        full_edge_table=full_table,
        base_keys=base_keys,
        alt_keys=alt_keys,
        group_data=group_data_all,
        steps=steps,
        out_dir=out_dir,
    )
    ref_df = read_reference_metrics(Path(args.metric_root))
    compare = ref_df.merge(patch_df, on=["step", "hour"], how="inner", validate="one_to_one")
    compare = compare.merge(
        dynamic_df[
            [
                "step",
                "hour",
                "mean_shortest_hops_dynamic_one_right",
                "mean_shortest_delay_ms_dynamic_one_right",
                "selected_inter_edges",
            ]
        ],
        on=["step", "hour"],
        how="inner",
        validate="one_to_one",
    )
    compare.to_csv(out_dir / "compare_spike_patch_vs_056_061_full_link.csv", index=False, encoding="utf-8-sig")
    jump_summary = local_jump_summary(compare, spike_steps, out_dir)
    plot_metric(
        compare,
        spike_steps,
        out_dir / "china_europe_spike_patch_vs_056_061_full_link_delay_ms.png",
        metric="mean_shortest_delay_ms",
        ylabel="mean shortest delay (ms)",
    )
    plot_metric(
        compare,
        spike_steps,
        out_dir / "china_europe_spike_patch_vs_056_061_full_link_hops.png",
        metric="mean_shortest_hops",
        ylabel="mean shortest hops",
    )

    meta = {
        "out_dir": str(out_dir),
        "steps": {"start": int(args.start), "end": int(args.end), "stride": int(args.stride), "count": len(steps)},
        "spike_steps": spike_steps,
        "base_motif": {"id": BASE_MOTIF_ID, "motif": base_motif},
        "alt_motif": {"id": ALT_MOTIF_ID, "motif": alt_motif},
        "files": {
            "edge_usage_by_step": "full_link_spike_used_edges_by_step.csv",
            "edge_usage_summary": "full_link_spike_used_edges_summary.csv",
            "selected_edges": "spike_patch_selected_inter_edges.csv",
            "patch_edges": "spike_patch_edges.csv",
            "compare_timeseries": "compare_spike_patch_vs_056_061_full_link.csv",
            "jump_summary": "local_spike_jump_summary.csv",
            "delay_plot": "china_europe_spike_patch_vs_056_061_full_link_delay_ms.png",
            "hops_plot": "china_europe_spike_patch_vs_056_061_full_link_hops.png",
        },
        "local_jump_summary": json.loads(jump_summary.to_json(orient="records", force_ascii=False)),
    }
    (out_dir / "study_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}", flush=True)
    print(f"delay_plot={out_dir / 'china_europe_spike_patch_vs_056_061_full_link_delay_ms.png'}", flush=True)
    print(f"hops_plot={out_dir / 'china_europe_spike_patch_vs_056_061_full_link_hops.png'}", flush=True)
    print(f"jump_summary={out_dir / 'local_spike_jump_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
