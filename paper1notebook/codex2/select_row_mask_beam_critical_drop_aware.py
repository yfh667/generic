from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs, wrap_planes_from_config  # noqa: E402
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from select_row_mask_beam_lst_drop_aware import (  # noqa: E402
    build_port_maps,
    candidate_sets,
    default_include_schedules,
    iter_set_bits,
    read_schedule_actions,
    reduce_beam,
    words_to_int,
)
from select_row_mask_dp_edge_transition_penalty import (  # noqa: E402
    DEFAULT_REWARD_DIR,
    build_or_load_action_edge_masks,
    read_action_table,
    read_wide,
)
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.edge_betweenness import (  # noqa: E402
    build_undirected_adjacency,
    edge_betweenness_between_node_sets,
)
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402
from src.topology_workflow.module.edge_tables import build_full_option_plus_intra_edge_table  # noqa: E402


DEFAULT_MASK_CACHE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_dp_edge_transition_penalty"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_beam_critical_drop_aware"
)


@dataclass(frozen=True)
class CriticalBeamState:
    prev_action: int
    curr_action: int
    cost: float
    path: tuple[int, ...]
    predicted_active_drop: int
    predicted_building: int
    predicted_critical_drop: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Beam-search row-mask policy that penalizes LST-dropped active edges by "
            "their China-Europe shortest-path edge usage, not only by edge count."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--mask-cache-dir", type=Path, default=DEFAULT_MASK_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--base-motif", type=int, default=56)
    parser.add_argument("--patch-motif", type=int, default=40)
    parser.add_argument("--patch-mode", choices=("c", "cb", "all"), default="all")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    parser.add_argument("--critical-drop-weights", nargs="+", type=float, default=(0.0, 0.25, 0.5, 1.0, 2.0, 5.0))
    parser.add_argument("--active-drop-count-weight", type=float, default=0.0)
    parser.add_argument("--building-weight", type=float, default=0.0)
    parser.add_argument("--count-edge-scale", type=float, default=100.0)
    parser.add_argument("--critical-drop-scale", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--include-schedules", nargs="*", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=300)
    parser.add_argument("--usage-progress-every", type=int, default=250)
    parser.add_argument("--reuse-mask-cache", action="store_true", default=True)
    parser.add_argument("--reuse-usage-cache", action="store_true", default=True)
    return parser.parse_args()


def ensure_same_actions(expected: Sequence[str], actual: Sequence[str], label: str) -> None:
    if list(expected) != list(actual):
        raise ValueError(f"{label} action columns differ from action table")


def edge_table_subset(universe: EdgeTable, indices: Sequence[int]) -> EdgeTable:
    idx = np.asarray(indices, dtype=np.int32)
    return EdgeTable(
        src=np.asarray(universe.src[idx], dtype=np.int32),
        dst=np.asarray(universe.dst[idx], dtype=np.int32),
        option=np.asarray(universe.option[idx], dtype=np.int16),
        src_plane=np.asarray(universe.src_plane[idx], dtype=np.int16),
        src_y=np.asarray(universe.src_y[idx], dtype=np.int16),
        dst_plane=np.asarray(universe.dst_plane[idx], dtype=np.int16),
        dst_y=np.asarray(universe.dst_y[idx], dtype=np.int16),
        sat_ids=universe.sat_ids,
    )


def local_lst_drop_edges(
    *,
    e0: int,
    e1: int,
    e2: int,
    edge_port_int: list[int],
    port_to_edges_int: list[int],
) -> tuple[int, int, int, int]:
    # stride=60, LST=120: row r builds edges first active at r+1 and r+2.
    b1 = int(e1) & ~int(e0)
    b2 = int(e2) & ~int(e1) & ~int(e0)
    owner_ports = 0
    building = 0
    building_dropped = 0
    for requested in (b1, b2):
        for edge_idx in iter_set_bits(requested):
            port_mask = int(edge_port_int[edge_idx])
            if port_mask == 0:
                continue
            if owner_ports & port_mask:
                building_dropped += 1
                continue
            owner_ports |= port_mask
            building += 1

    conflict_edges = 0
    for port_idx in iter_set_bits(owner_ports):
        conflict_edges |= int(port_to_edges_int[port_idx])
    dropped_edges = int(e0) & conflict_edges
    return dropped_edges, int(dropped_edges.bit_count()), int(building), int(building_dropped)


def sum_weights_for_bits(bits: int, weights: np.ndarray) -> float:
    total = 0.0
    for edge_idx in iter_set_bits(bits):
        total += float(weights[int(edge_idx)])
    return total


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_or_load_usage_cache(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    raw: dict[str, Any],
    config,
    steps: np.ndarray,
    edge_masks: np.ndarray,
    mask_ints: list[list[int]],
    candidates: list[np.ndarray],
) -> tuple[np.ndarray, dict[tuple[int, int], int], dict[str, Any]]:
    cache_path = out_dir / "candidate_edge_usage_weights_float32.npy"
    key_path = out_dir / "candidate_edge_usage_keys.csv"
    meta_path = out_dir / "candidate_edge_usage_meta.json"
    if bool(args.reuse_usage_cache) and cache_path.exists() and key_path.exists() and meta_path.exists():
        usage = np.load(cache_path, mmap_mode=None)
        keys: dict[tuple[int, int], int] = {}
        with key_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                keys[(int(row["time_index"]), int(row["action_index"]))] = int(row["usage_row"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return usage, keys, meta

    pair_spec = region_pair_specs(raw, subset=[str(args.target_pair)])[0]
    paths = raw.get("paths", {})
    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=[int(x) for x in steps.tolist()],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(raw.get("time", {}).get("stride", 60)),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    universe = build_full_option_plus_intra_edge_table(
        config=config,
        options=(0, 1, 2, 4),
        add_intra_ring=True,
        wrap_planes=bool(wrap_planes_from_config(raw)),
    )
    edge_universe_size = int(universe.num_edges)
    key_pairs = sorted({(t, int(a)) for t, values in enumerate(candidates) for a in values.astype(int).tolist()})
    usage = np.zeros((len(key_pairs), edge_universe_size), dtype=np.float32)
    key_to_row: dict[tuple[int, int], int] = {}
    summary_rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    for usage_row, (t_idx, action_idx) in enumerate(key_pairs):
        key_to_row[(int(t_idx), int(action_idx))] = int(usage_row)
        step = int(steps[int(t_idx)])
        sources = group_nodes_for_step(group_data, step, int(pair_spec.source_group_id))
        targets = group_nodes_for_step(group_data, step, int(pair_spec.target_group_id))
        edge_indices = list(iter_set_bits(mask_ints[int(t_idx)][int(action_idx)]))
        if sources and targets and edge_indices:
            sub_table = edge_table_subset(universe, edge_indices)
            adjacency = build_undirected_adjacency(sub_table, int(config.total_sats))
            values, summary, _samples = edge_betweenness_between_node_sets(
                sub_table,
                total_nodes=int(config.total_sats),
                source_nodes=sources,
                target_nodes=targets,
                adjacency=adjacency,
                sample_path_limit=0,
            )
            norm = max(1.0, float(summary.reachable_pairs))
            usage[int(usage_row), np.asarray(edge_indices, dtype=np.int32)] = np.asarray(values, dtype=np.float32) / norm
            summary_rows.append(
                {
                    "usage_row": int(usage_row),
                    "time_index": int(t_idx),
                    "step": step,
                    "action_index": int(action_idx),
                    "source_nodes": int(summary.source_nodes),
                    "target_nodes": int(summary.target_nodes),
                    "reachable_pairs": int(summary.reachable_pairs),
                    "mean_shortest_hops": float(summary.mean_shortest_distance_hops),
                    "max_edge_usage_per_pair": float(np.max(usage[int(usage_row)])),
                    "nonzero_edges": int(np.count_nonzero(usage[int(usage_row)] > 0)),
                }
            )
        else:
            summary_rows.append(
                {
                    "usage_row": int(usage_row),
                    "time_index": int(t_idx),
                    "step": step,
                    "action_index": int(action_idx),
                    "source_nodes": int(len(sources)),
                    "target_nodes": int(len(targets)),
                    "reachable_pairs": 0,
                    "mean_shortest_hops": 0.0,
                    "max_edge_usage_per_pair": 0.0,
                    "nonzero_edges": 0,
                }
            )
        if (usage_row + 1) % int(args.usage_progress_every) == 0 or usage_row + 1 == len(key_pairs):
            elapsed = time.perf_counter() - started
            print(
                f"[critical-drop] usage {usage_row + 1}/{len(key_pairs)} "
                f"step={step} action={action_idx} elapsed={elapsed:.1f}s",
                flush=True,
            )

    np.save(cache_path, usage)
    write_rows(key_path, [{"time_index": t, "action_index": a, "usage_row": key_to_row[(t, a)]} for (t, a) in key_pairs])
    write_rows(out_dir / "candidate_edge_usage_summary.csv", summary_rows)
    meta = {
        "cache_path": str(cache_path),
        "key_path": str(key_path),
        "target_pair": str(args.target_pair),
        "num_usage_rows": int(usage.shape[0]),
        "edge_universe_size": edge_universe_size,
        "usage_normalization": "edge betweenness divided by reachable source-target pairs",
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return usage, key_to_row, meta


def run_critical_beam(
    *,
    stage_cost: np.ndarray,
    mask_ints: list[list[int]],
    candidates: list[np.ndarray],
    usage_weights: np.ndarray,
    usage_key_to_row: dict[tuple[int, int], int],
    edge_port_int: list[int],
    port_to_edges_int: list[int],
    critical_drop_weight: float,
    active_drop_count_weight: float,
    building_weight: float,
    count_edge_scale: float,
    critical_drop_scale: float,
    beam_width: int,
    progress_every: int,
) -> CriticalBeamState:
    t_count = int(stage_cost.shape[0])
    beam: dict[tuple[int, int], CriticalBeamState] = {}
    for a0 in candidates[0]:
        for a1 in candidates[1]:
            a0_i = int(a0)
            a1_i = int(a1)
            state = CriticalBeamState(
                prev_action=a0_i,
                curr_action=a1_i,
                cost=float(stage_cost[0, a0_i] + stage_cost[1, a1_i]),
                path=(a0_i, a1_i),
                predicted_active_drop=0,
                predicted_building=0,
                predicted_critical_drop=0.0,
            )
            old = beam.get((a0_i, a1_i))
            if old is None or state.cost < old.cost:
                beam[(a0_i, a1_i)] = state
    beam = reduce_beam(beam, int(beam_width))
    count_norm = max(1e-9, float(count_edge_scale))
    critical_norm = max(1e-9, float(critical_drop_scale))
    started = time.perf_counter()
    for t in range(2, t_count):
        next_states: dict[tuple[int, int], CriticalBeamState] = {}
        t0 = int(t - 2)
        for state in beam.values():
            a = int(state.prev_action)
            b = int(state.curr_action)
            e0 = mask_ints[t0][a]
            e1 = mask_ints[t - 1][b]
            usage = usage_weights[usage_key_to_row[(t0, a)]]
            for c_raw in candidates[t]:
                c = int(c_raw)
                e2 = mask_ints[t][c]
                dropped_edges, active_drop, building, _building_dropped = local_lst_drop_edges(
                    e0=e0,
                    e1=e1,
                    e2=e2,
                    edge_port_int=edge_port_int,
                    port_to_edges_int=port_to_edges_int,
                )
                critical_drop = sum_weights_for_bits(dropped_edges, usage)
                lst_cost = (
                    float(critical_drop_weight) * critical_drop / critical_norm
                    + float(active_drop_count_weight) * float(active_drop) / count_norm
                    + float(building_weight) * float(building) / count_norm
                )
                new_cost = float(state.cost) + float(stage_cost[t, c]) + lst_cost
                key = (b, c)
                old = next_states.get(key)
                if old is None or new_cost < old.cost:
                    next_states[key] = CriticalBeamState(
                        prev_action=b,
                        curr_action=c,
                        cost=new_cost,
                        path=state.path + (c,),
                        predicted_active_drop=int(state.predicted_active_drop + active_drop),
                        predicted_building=int(state.predicted_building + building),
                        predicted_critical_drop=float(state.predicted_critical_drop + critical_drop),
                    )
        beam = reduce_beam(next_states, int(beam_width))
        if (t + 1) % int(progress_every) == 0 or t + 1 == t_count:
            best = min(beam.values(), key=lambda s: s.cost)
            print(
                f"[critical-drop] t={t + 1}/{t_count} beam={len(beam)} "
                f"cost={best.cost:.4f} drop={best.predicted_active_drop} "
                f"critical={best.predicted_critical_drop:.3f} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    zero_future = 0
    final_states: dict[tuple[int, int], CriticalBeamState] = {}
    t0 = t_count - 2
    for state in beam.values():
        a = int(state.prev_action)
        b = int(state.curr_action)
        dropped_edges, active_drop, building, _building_dropped = local_lst_drop_edges(
            e0=mask_ints[t_count - 2][a],
            e1=mask_ints[t_count - 1][b],
            e2=zero_future,
            edge_port_int=edge_port_int,
            port_to_edges_int=port_to_edges_int,
        )
        critical_drop = sum_weights_for_bits(dropped_edges, usage_weights[usage_key_to_row[(t0, a)]])
        lst_cost = (
            float(critical_drop_weight) * critical_drop / critical_norm
            + float(active_drop_count_weight) * float(active_drop) / count_norm
            + float(building_weight) * float(building) / count_norm
        )
        final_states[(a, b)] = CriticalBeamState(
            prev_action=a,
            curr_action=b,
            cost=float(state.cost) + lst_cost,
            path=state.path,
            predicted_active_drop=int(state.predicted_active_drop + active_drop),
            predicted_building=int(state.predicted_building + building),
            predicted_critical_drop=float(state.predicted_critical_drop + critical_drop),
        )
    return min(final_states.values(), key=lambda s: s.cost)


def schedule_rows(
    *,
    steps: np.ndarray,
    selected: np.ndarray,
    policy: str,
    lambda_hop: float,
    critical_drop_weight: float,
    action_names: Sequence[str],
    action_masks: np.ndarray,
    mask_tokens: Sequence[str],
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(selected.astype(int).tolist()):
        mask = action_masks[action_idx]
        row: dict[str, Any] = {
            "step": int(steps[t]),
            "policy": policy,
            "lambda_hop": float(lambda_hop),
            "critical_drop_weight": float(critical_drop_weight),
            "action": str(action_names[action_idx]),
            "action_index": int(action_idx),
            "mask": str(mask_tokens[action_idx]),
            "row_count": int(mask.sum()),
            "mean_hops": float(hops[t, action_idx]),
            "mean_delay_ms": float(delay[t, action_idx]),
            "reference_hop_envelope": float(hop_env[t]),
            "reference_delay_envelope_ms": float(delay_env[t]),
            "gap_hop_env": float(hops[t, action_idx] - hop_env[t]),
            "gap_delay_env_ms": float(delay[t, action_idx] - delay_env[t]),
        }
        for y in range(mask.shape[0]):
            row[f"y{y:02d}"] = int(mask[y])
        rows.append(row)
    return rows


def summarize_policy(
    *,
    selected: np.ndarray,
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
    final_state: CriticalBeamState,
) -> dict[str, Any]:
    idx = selected.astype(int)
    rows = np.arange(idx.size)
    return {
        "mean_hops": float(np.mean(hops[rows, idx])),
        "mean_delay_ms": float(np.mean(delay[rows, idx])),
        "mean_gap_hop_env": float(np.mean(hops[rows, idx] - hop_env)),
        "mean_gap_delay_env_ms": float(np.mean(delay[rows, idx] - delay_env)),
        "switches": int(np.count_nonzero(idx[1:] != idx[:-1])),
        "num_segments": int(np.count_nonzero(idx[1:] != idx[:-1]) + 1),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
        "beam_objective_cost": float(final_state.cost),
        "predicted_active_drop_sum": int(final_state.predicted_active_drop),
        "predicted_building_sum": int(final_state.predicted_building),
        "predicted_critical_drop_sum": float(final_state.predicted_critical_drop),
        "predicted_active_drop_mean": float(final_state.predicted_active_drop / max(1, int(selected.size))),
        "predicted_building_mean": float(final_state.predicted_building / max(1, int(selected.size))),
        "predicted_critical_drop_mean": float(final_state.predicted_critical_drop / max(1, int(selected.size))),
    }


def main() -> int:
    args = parse_args()
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_names, action_masks, mask_tokens = read_action_table(Path(args.reward_dir) / "row_mask_action_library_actions.csv")
    steps_hop, names_hop, hops = read_wide(Path(args.reward_dir) / "row_mask_action_library_mean_hops_wide.csv")
    steps_delay, names_delay, delay = read_wide(Path(args.reward_dir) / "row_mask_action_library_mean_delay_ms_wide.csv")
    if not np.array_equal(steps_hop, steps_delay):
        raise ValueError("hop/delay tables have different steps")
    ensure_same_actions(action_names, names_hop, "hop table")
    ensure_same_actions(action_names, names_delay, "delay table")

    edge_masks, _edge_universe_size = build_or_load_action_edge_masks(
        args=args,
        out_dir=Path(args.mask_cache_dir),
        raw=raw,
        config=config,
        steps=steps_hop,
        action_masks=action_masks,
    )
    mask_ints = [[words_to_int(edge_masks[t, a]) for a in range(edge_masks.shape[1])] for t in range(edge_masks.shape[0])]
    edge_port_int, port_to_edges_int = build_port_maps(config, wrap_planes=bool(wrap_planes_from_config(raw)))

    hop_env = np.nanmin(hops, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hops) - np.nanmin(hops)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    stage_cost = (
        float(args.lambda_hop) * ((hops - hop_env[:, None]) / hop_scale)
        + (1.0 - float(args.lambda_hop)) * ((delay - delay_env[:, None]) / delay_scale)
    )

    include_paths = args.include_schedules if args.include_schedules else default_include_schedules()
    include_actions = []
    for path in include_paths:
        values = read_schedule_actions(Path(path), num_steps=int(steps_hop.size))
        if values is not None:
            include_actions.append(values)
    candidates = candidate_sets(stage_cost=stage_cost, include_actions=include_actions, top_k=int(args.top_k_actions))
    usage_weights, usage_key_to_row, usage_meta = build_or_load_usage_cache(
        args=args,
        out_dir=out_dir,
        raw=raw,
        config=config,
        steps=steps_hop,
        edge_masks=edge_masks,
        mask_ints=mask_ints,
        candidates=candidates,
    )

    summary_rows: list[dict[str, Any]] = []
    for critical_drop_weight in args.critical_drop_weights:
        policy = (
            f"beam_critdrop_lambda{float(args.lambda_hop):.2f}"
            f"_crit{float(critical_drop_weight):g}"
            f"_cnt{float(args.active_drop_count_weight):g}"
            f"_build{float(args.building_weight):g}"
            f"_k{int(args.top_k_actions)}_b{int(args.beam_width)}"
        )
        final_state = run_critical_beam(
            stage_cost=stage_cost,
            mask_ints=mask_ints,
            candidates=candidates,
            usage_weights=usage_weights,
            usage_key_to_row=usage_key_to_row,
            edge_port_int=edge_port_int,
            port_to_edges_int=port_to_edges_int,
            critical_drop_weight=float(critical_drop_weight),
            active_drop_count_weight=float(args.active_drop_count_weight),
            building_weight=float(args.building_weight),
            count_edge_scale=float(args.count_edge_scale),
            critical_drop_scale=float(args.critical_drop_scale),
            beam_width=int(args.beam_width),
            progress_every=int(args.progress_every),
        )
        selected = np.asarray(final_state.path, dtype=np.int16)
        schedule = schedule_rows(
            steps=steps_hop,
            selected=selected,
            policy=policy,
            lambda_hop=float(args.lambda_hop),
            critical_drop_weight=float(critical_drop_weight),
            action_names=action_names,
            action_masks=action_masks,
            mask_tokens=mask_tokens,
            hops=hops,
            delay=delay,
            hop_env=hop_env,
            delay_env=delay_env,
        )
        schedule_path = out_dir / f"{policy}.csv"
        write_rows(schedule_path, schedule)
        summary_rows.append(
            {
                "policy": policy,
                "schedule_csv": str(schedule_path),
                "lambda_hop": float(args.lambda_hop),
                "critical_drop_weight": float(critical_drop_weight),
                "active_drop_count_weight": float(args.active_drop_count_weight),
                "building_weight": float(args.building_weight),
                "count_edge_scale": float(args.count_edge_scale),
                "critical_drop_scale": float(args.critical_drop_scale),
                "beam_width": int(args.beam_width),
                "top_k_actions": int(args.top_k_actions),
                **summarize_policy(
                    selected=selected,
                    hops=hops,
                    delay=delay,
                    hop_env=hop_env,
                    delay_env=delay_env,
                    final_state=final_state,
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "beam_critical_drop_aware_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta = {
        "config": str(Path(args.config)),
        "reward_dir": str(Path(args.reward_dir)),
        "mask_cache_dir": str(Path(args.mask_cache_dir)),
        "out_dir": str(out_dir),
        "usage_meta": usage_meta,
        "hop_scale": hop_scale,
        "delay_scale": delay_scale,
        "mean_hop_envelope": float(np.mean(hop_env)),
        "mean_delay_envelope_ms": float(np.mean(delay_env)),
        "summary_csv": str(summary_path),
    }
    (out_dir / "beam_critical_drop_aware_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
