from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.link_delay.module.position_cache import PositionCacheStore  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.edge_tables import (  # noqa: E402
    INTRA_OPTION,
    build_motif_text_edge_table,
)
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


DEFAULT_FULL_LINK_CACHE = Path(
    r"E:\paper11\data\linshi\g60_full_link_multi_region_weighted_betweenness_t0_86164_stride1"
)
DEFAULT_MOTIF_LIBRARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\full_link_guided_one_right_stride1"
)
DEFAULT_DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\_position_cache\cache_0_86164_1s"
)
DEFAULT_GROUP_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"

PAIR_KEY = "china_europe"
SOURCE_GROUP_ID = 2
TARGET_GROUP_ID = 3
BASE_MOTIF_ID = 56
ALT_MOTIF_ID = 61


CandidateArrays = dict[str, np.ndarray]
SparseTemplate = dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full-link-guided dynamic one-right target topology from a full-link "
            "weighted edge-betweenness cache. This only writes the target edge mask and "
            "transition summaries; metric evaluation is handled by later scripts."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--full-link-cache", type=Path, default=DEFAULT_FULL_LINK_CACHE)
    parser.add_argument("--score-source", choices=["cache", "online-full-link"], default="cache")
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE_DIR)
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--pair-key", type=str, default=PAIR_KEY)
    parser.add_argument("--motif-library", type=Path, default=DEFAULT_MOTIF_LIBRARY_CSV)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--spike-weight", type=float, default=1000.0)
    parser.add_argument("--keep-base-bonus", type=float, default=1.0)
    parser.add_argument("--keep-alt-bonus", type=float, default=0.10)
    parser.add_argument("--fill-bonus", type=float, default=0.001)
    parser.add_argument("--option0-bonus", type=float, default=0.0002)
    parser.add_argument("--progress-every", type=int, default=5000)
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


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src <= dst else (dst, src)


def edge_keys_from_table(edge_table: EdgeTable) -> set[tuple[int, int]]:
    return {edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])) for idx in range(edge_table.num_edges)}


def load_motif_text(motif_id: int, motif_library_csv: Path) -> str:
    with Path(motif_library_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["motif_id"]) == int(motif_id):
                return str(row["motif"])
    raise KeyError(f"motif_id={motif_id} not found in {motif_library_csv}")


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


def build_candidate_arrays(
    *,
    full_edge_table: EdgeTable,
    base_keys: set[tuple[int, int]],
    alt_keys: set[tuple[int, int]],
    args: argparse.Namespace,
) -> CandidateArrays:
    edge_indices: list[int] = []
    owners: list[int] = []
    targets: list[int] = []
    static_bias: list[float] = []
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
        edge_indices.append(int(idx))
        owners.append(int(owner_target[0]))
        targets.append(int(owner_target[1]))
        static_bias.append(float(score))
    return {
        "edge_indices": np.asarray(edge_indices, dtype=np.int32),
        "owners": np.asarray(owners, dtype=np.int32),
        "targets": np.asarray(targets, dtype=np.int32),
        "static_bias": np.asarray(static_bias, dtype=np.float64),
    }


def build_sparse_template(edge_table: EdgeTable, *, total_nodes: int) -> SparseTemplate:
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    edge_idx = np.arange(int(edge_table.num_edges), dtype=np.int32)
    edge_lookup = np.full((int(total_nodes), int(total_nodes)), -1, dtype=np.int32)
    edge_lookup[src, dst] = edge_idx
    edge_lookup[dst, src] = edge_idx
    return {
        "row": np.concatenate([src, dst]).astype(np.int32, copy=False),
        "col": np.concatenate([dst, src]).astype(np.int32, copy=False),
        "edge_lookup": edge_lookup,
    }


def fast_weighted_edge_usage_between_node_sets(
    *,
    edge_table: EdgeTable,
    total_nodes: int,
    weights: np.ndarray,
    source_nodes: tuple[int, ...],
    target_nodes: tuple[int, ...],
    sparse_template: SparseTemplate,
) -> tuple[np.ndarray, dict[str, Any]]:
    sources = tuple(sorted(int(x) for x in source_nodes if 0 <= int(x) < int(total_nodes)))
    targets = tuple(sorted(int(x) for x in target_nodes if 0 <= int(x) < int(total_nodes)))
    edge_values = np.zeros(int(edge_table.num_edges), dtype=np.float32)
    if not sources or not targets:
        return edge_values, {"reachable_pairs": 0, "nonzero_edges": 0, "max_edge_betweenness": 0.0}

    weights = np.asarray(weights, dtype=np.float32)
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    graph = csr_matrix(
        (data, (sparse_template["row"], sparse_template["col"])),
        shape=(int(total_nodes), int(total_nodes)),
    )
    dist, predecessors = dijkstra(
        graph,
        directed=False,
        indices=np.asarray(sources, dtype=np.int32),
        return_predecessors=True,
    )
    dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
    predecessors = np.atleast_2d(np.asarray(predecessors, dtype=np.int32))
    edge_lookup = sparse_template["edge_lookup"]
    reachable = 0
    total_delay = 0.0
    for source_row, source in enumerate(sources):
        for target in targets:
            if int(target) == int(source):
                continue
            value = float(dist[source_row, int(target)])
            if not np.isfinite(value):
                continue
            node = int(target)
            path_edges: list[int] = []
            for _guard in range(int(total_nodes) + 1):
                if node == int(source):
                    break
                parent = int(predecessors[source_row, node])
                if parent < 0:
                    path_edges = []
                    break
                edge_idx = int(edge_lookup[node, parent])
                if edge_idx < 0:
                    path_edges = []
                    break
                path_edges.append(edge_idx)
                node = parent
            if node != int(source) or not path_edges:
                continue
            reachable += 1
            total_delay += value
            for edge_idx in path_edges:
                edge_values[int(edge_idx)] += 1.0

    return edge_values, {
        "reachable_pairs": int(reachable),
        "nonzero_edges": int(np.count_nonzero(edge_values > 0.0)),
        "max_edge_betweenness": float(np.max(edge_values)) if edge_values.size else 0.0,
        "edge_value_sum": float(np.sum(edge_values)),
        "mean_shortest_weight": float(total_delay / reachable) if reachable else float("nan"),
    }


def greedy_select_inter_edge_indices(
    *,
    edge_score: np.ndarray,
    candidates: CandidateArrays,
    spike_weight: float,
    total_sats: int,
) -> tuple[list[int], list[int], list[int]]:
    edge_indices = candidates["edge_indices"]
    owners = candidates["owners"]
    targets = candidates["targets"]
    scores = float(spike_weight) * np.asarray(edge_score[edge_indices], dtype=np.float64) + candidates["static_bias"]
    order = np.lexsort((edge_indices, targets, owners, -scores))
    used_owner = np.zeros(int(total_sats), dtype=bool)
    used_target = np.zeros(int(total_sats), dtype=bool)
    selected: list[int] = []
    selected_owners: list[int] = []
    selected_targets: list[int] = []
    for pos in order.tolist():
        score = float(scores[int(pos)])
        owner = int(owners[int(pos)])
        target = int(targets[int(pos)])
        if score <= 0.0:
            break
        if bool(used_owner[owner]) or bool(used_target[target]):
            continue
        selected.append(int(edge_indices[int(pos)]))
        selected_owners.append(owner)
        selected_targets.append(target)
        used_owner[owner] = True
        used_target[target] = True
    return selected, selected_owners, selected_targets


def summarize_selected(
    *,
    selected_inter: list[int],
    selected_owners: list[int],
    selected_targets: list[int],
    previous_inter: set[int] | None,
    step: int,
) -> dict[str, Any]:
    current = set(int(x) for x in selected_inter)
    owners = set(int(x) for x in selected_owners)
    targets = set(int(x) for x in selected_targets)
    if previous_inter is None:
        new_edges = len(current)
        removed_edges = 0
        kept_edges = 0
        symmetric = len(current)
    else:
        new_edges = len(current - previous_inter)
        removed_edges = len(previous_inter - current)
        kept_edges = len(current & previous_inter)
        symmetric = len(current ^ previous_inter)
    return {
        "step": int(step),
        "hour": float(step) / 3600.0,
        "selected_inter_edges": int(len(selected_inter)),
        "unique_right_owners": int(len(owners)),
        "unique_left_targets": int(len(targets)),
        "violates_one_right": int(len(owners) != len(selected_inter)),
        "violates_one_left": int(len(targets) != len(selected_inter)),
        "new_inter_edges": int(new_edges),
        "removed_inter_edges": int(removed_edges),
        "kept_inter_edges": int(kept_edges),
        "symmetric_diff_inter_edges": int(symmetric),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.out_root) / (
        f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cache = Path(args.full_link_cache)
    edge_table = read_edge_table_csv(full_cache / "edges.csv", total_sats=int(G60_CONFIG.total_sats))
    cache_steps = np.asarray(np.load(full_cache / "time_indices.npy", mmap_mode="r"), dtype=np.int64)
    cache_values = None
    if str(args.score_source) == "cache":
        cache_values = np.load(full_cache / str(args.pair_key) / "edge_betweenness.npy", mmap_mode="r")
        if cache_values.shape != (int(cache_steps.shape[0]), int(edge_table.num_edges)):
            raise ValueError(
                f"cache shape mismatch: {cache_values.shape} vs steps={cache_steps.shape} edges={edge_table.num_edges}"
            )

    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    cache_row_by_step = {int(step): idx for idx, step in enumerate(cache_steps.tolist())}
    missing = [int(step) for step in steps if int(step) not in cache_row_by_step]
    if missing:
        raise KeyError(f"{len(missing)} requested steps are missing from {full_cache}; first={missing[:5]}")

    base_motif = load_motif_text(BASE_MOTIF_ID, Path(args.motif_library))
    alt_motif = load_motif_text(ALT_MOTIF_ID, Path(args.motif_library))
    base_table = build_motif_text_edge_table(motif_text=base_motif, config=G60_CONFIG, add_intra_ring=True)
    alt_table = build_motif_text_edge_table(motif_text=alt_motif, config=G60_CONFIG, add_intra_ring=True)
    base_keys = edge_keys_from_table(base_table)
    alt_keys = edge_keys_from_table(alt_table)
    candidates = build_candidate_arrays(
        full_edge_table=edge_table,
        base_keys=base_keys,
        alt_keys=alt_keys,
        args=args,
    )
    online_context: dict[str, Any] = {}
    if str(args.score_source) == "online-full-link":
        delay_store = FullLinkDelayStore(Path(args.delay_store_dir))
        position_store = PositionCacheStore(Path(args.position_cache_dir))
        online_context["delay_store"] = delay_store
        online_context["position_store"] = position_store
        online_context["delay_rows"] = delay_store.rows_for_interval(int(args.start), int(args.end), int(args.stride))
        online_context["position_rows"] = position_store.rows_for_interval(int(args.start), int(args.end), int(args.stride))
        online_context["lookup"] = build_weight_lookup(edge_table, delay_store, config=G60_CONFIG, allow_intra_fallback=True)
        online_context["sparse_template"] = build_sparse_template(edge_table, total_nodes=int(G60_CONFIG.total_sats))
        online_context["group_data"] = load_or_build_group_data(
            xml_file=Path(args.group_xml),
            group_cache_dir=Path(args.group_cache_dir),
            steps=steps.tolist(),
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(args.stride),
            enabled=True,
            force=False,
        )

    np.save(out_dir / "dynamic_one_right_time_indices.npy", steps)
    active_mask = np.lib.format.open_memmap(
        out_dir / "dynamic_one_right_active_full_edge_mask.npy",
        mode="w+",
        dtype=bool,
        shape=(int(steps.shape[0]), int(edge_table.num_edges)),
    )
    active_mask[:] = False

    intra_indices = np.flatnonzero(np.asarray(edge_table.option) == INTRA_OPTION).astype(np.int32)
    summary_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    previous_inter: set[int] | None = None
    started = time.perf_counter()

    for local_idx, step in enumerate(steps.tolist()):
        if str(args.score_source) == "cache":
            if cache_values is None:
                raise RuntimeError("cache_values is not initialized")
            cache_row = int(cache_row_by_step[int(step)])
            edge_score = np.asarray(cache_values[cache_row], dtype=np.float64)
            reachable_pairs = -1
            nonzero_edges = int(np.count_nonzero(edge_score))
            max_edge_usage = float(np.max(edge_score)) if edge_score.size else 0.0
        else:
            weights = edge_weights_for_step(
                edge_table=edge_table,
                lookup=online_context["lookup"],
                delay_store=online_context["delay_store"],
                position_store=online_context["position_store"],
                delay_row=int(online_context["delay_rows"][local_idx]),
                position_row=int(online_context["position_rows"][local_idx]),
            )
            sources = group_nodes_for_step(online_context["group_data"], int(step), SOURCE_GROUP_ID)
            targets = group_nodes_for_step(online_context["group_data"], int(step), TARGET_GROUP_ID)
            values, usage_summary = fast_weighted_edge_usage_between_node_sets(
                edge_table=edge_table,
                total_nodes=int(G60_CONFIG.total_sats),
                source_nodes=sources,
                target_nodes=targets,
                weights=weights,
                sparse_template=online_context["sparse_template"],
            )
            edge_score = np.asarray(values, dtype=np.float64)
            reachable_pairs = int(usage_summary["reachable_pairs"])
            nonzero_edges = int(usage_summary["nonzero_edges"])
            max_edge_usage = float(usage_summary["max_edge_betweenness"])
        selected_inter, selected_owners, selected_targets = greedy_select_inter_edge_indices(
            edge_score=edge_score,
            candidates=candidates,
            spike_weight=float(args.spike_weight),
            total_sats=int(G60_CONFIG.total_sats),
        )
        selected = np.asarray(selected_inter + intra_indices.tolist(), dtype=np.int32)
        active_mask[local_idx, selected] = True
        summary = summarize_selected(
            selected_inter=selected_inter,
            selected_owners=selected_owners,
            selected_targets=selected_targets,
            previous_inter=previous_inter,
            step=int(step),
        )
        summary_rows.append(summary)
        usage_rows.append(
            {
                "step": int(step),
                "hour": float(step) / 3600.0,
                "reachable_pairs": int(reachable_pairs),
                "nonzero_full_link_edges": int(nonzero_edges),
                "max_full_link_edge_usage": float(max_edge_usage),
                "full_link_edge_usage_sum": float(np.sum(edge_score)),
            }
        )
        previous_inter = set(int(x) for x in selected_inter)
        if int(args.progress_every) > 0 and (
            local_idx + 1 == int(steps.shape[0]) or (local_idx + 1) % int(args.progress_every) == 0
        ):
            print(
                f"[dynamic-one-right-target] {local_idx + 1}/{steps.shape[0]} "
                f"step={step} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    active_mask.flush()
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "dynamic_one_right_transition_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(usage_rows).to_csv(out_dir / "dynamic_one_right_score_summary.csv", index=False, encoding="utf-8-sig")
    aggregate = {
        "target": "full_link_guided_dynamic_one_right",
        "pair_key": str(args.pair_key),
        "score_source": str(args.score_source),
        "time": {"start": int(args.start), "end": int(args.end), "stride": int(args.stride), "count": int(steps.shape[0])},
        "full_link_cache": str(full_cache),
        "base_motif": {"id": BASE_MOTIF_ID, "motif": base_motif},
        "alt_motif": {"id": ALT_MOTIF_ID, "motif": alt_motif},
        "edges": {
            "full_edge_count": int(edge_table.num_edges),
            "candidate_inter_edges": int(candidates["edge_indices"].shape[0]),
            "intra_edges": int(intra_indices.size),
            "selected_inter_mean": float(summary_df["selected_inter_edges"].mean()),
            "selected_inter_min": int(summary_df["selected_inter_edges"].min()),
            "selected_inter_max": int(summary_df["selected_inter_edges"].max()),
            "violation_rows_one_right": int(summary_df["violates_one_right"].sum()),
            "violation_rows_one_left": int(summary_df["violates_one_left"].sum()),
            "symmetric_diff_inter_mean": float(summary_df["symmetric_diff_inter_edges"].mean()),
            "symmetric_diff_inter_p95": float(summary_df["symmetric_diff_inter_edges"].quantile(0.95)),
            "symmetric_diff_inter_max": int(summary_df["symmetric_diff_inter_edges"].max()),
        },
        "selection_parameters": {
            "spike_weight": float(args.spike_weight),
            "keep_base_bonus": float(args.keep_base_bonus),
            "keep_alt_bonus": float(args.keep_alt_bonus),
            "fill_bonus": float(args.fill_bonus),
            "option0_bonus": float(args.option0_bonus),
        },
        "outputs": {
            "time_indices": str(out_dir / "dynamic_one_right_time_indices.npy"),
            "active_mask": str(out_dir / "dynamic_one_right_active_full_edge_mask.npy"),
            "transition_counts": str(out_dir / "dynamic_one_right_transition_counts.csv"),
            "score_summary": str(out_dir / "dynamic_one_right_score_summary.csv"),
        },
    }
    (out_dir / "dynamic_one_right_target_meta.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"out_dir={out_dir}", flush=True)
    print(f"mask={out_dir / 'dynamic_one_right_active_full_edge_mask.npy'}", flush=True)
    print(f"summary={out_dir / 'dynamic_one_right_transition_counts.csv'}", flush=True)
    print(f"meta={out_dir / 'dynamic_one_right_target_meta.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
