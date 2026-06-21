# -*- coding: utf-8 -*-
"""Export transition-level labels for 000040/000056 row-mask fusion.

The one-step reward tables describe how good an action is at a time slice.
This exporter adds the missing LST part: for triples
``(action[t-2], action[t-1], action[t])`` it records the active edges dropped
by setup, building edges, and critical dropped edge usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from learn_hybrid_candidate_policy import DEFAULT_GROUP_CACHE, build_state_features  # noqa: E402
from run_paper1_motif_shortest_hops import load_yaml, wrap_planes_from_config  # noqa: E402
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from select_row_mask_beam_critical_drop_aware import (  # noqa: E402
    build_port_maps,
    local_lst_drop_edges,
    sum_weights_for_bits,
)
from select_row_mask_beam_lst_drop_aware import (  # noqa: E402
    candidate_sets,
    default_include_schedules,
    read_schedule_actions,
    words_to_int,
)
from select_row_mask_dp_edge_transition_penalty import (  # noqa: E402
    DEFAULT_REWARD_DIR,
    build_or_load_action_edge_masks,
    read_action_table,
    read_wide,
)
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402


G60_RUN_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
DEFAULT_MASK_CACHE_DIR = (
    G60_RUN_ROOT / "motif0040_0056_region_internal_plus_grid_row_mask_dp_edge_transition_penalty"
)
DEFAULT_USAGE_CACHE_DIR = (
    G60_RUN_ROOT / "motif0040_0056_region_internal_plus_grid_row_mask_beam_critical_drop_aware_topk3"
)
DEFAULT_OUT_DIR = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_transition_value_dataset"
    / "topk3_lam080_crit005_cnt003_build003"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export candidate transition triples with exact local LST labels. "
            "The default candidate pool is top3 actions plus included teacher/DP schedules."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--mask-cache-dir", type=Path, default=DEFAULT_MASK_CACHE_DIR)
    parser.add_argument("--usage-cache-dir", type=Path, default=DEFAULT_USAGE_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k-actions", type=int, default=3)
    parser.add_argument("--include-schedules", nargs="*", type=Path, default=None)
    parser.add_argument("--lambda-hop", type=float, default=0.8)
    parser.add_argument("--critical-drop-weight", type=float, default=0.05)
    parser.add_argument("--active-drop-count-weight", type=float, default=0.03)
    parser.add_argument("--building-weight", type=float, default=0.03)
    parser.add_argument("--count-edge-scale", type=float, default=100.0)
    parser.add_argument("--critical-drop-scale", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=("time", "group"), default="group")
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--sample-csv-rows", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--reuse-mask-cache", action="store_true", default=True)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def ensure_same_actions(expected: Sequence[str], actual: Sequence[str], label: str) -> None:
    if list(expected) != list(actual):
        raise ValueError(f"{label} action columns differ from action table")


def load_usage_cache(path: Path) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    weights_path = Path(path) / "candidate_edge_usage_weights_float32.npy"
    keys_path = Path(path) / "candidate_edge_usage_keys.csv"
    if not weights_path.exists() or not keys_path.exists():
        raise FileNotFoundError(
            f"missing usage cache under {path}; run select_row_mask_beam_critical_drop_aware.py "
            "with the same top-k candidate setting first"
        )
    weights = np.load(weights_path, mmap_mode="r")
    key_to_row: dict[tuple[int, int], int] = {}
    with keys_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key_to_row[(int(row["time_index"]), int(row["action_index"]))] = int(row["usage_row"])
    return weights, key_to_row


def write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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

    edge_masks, edge_universe_size = build_or_load_action_edge_masks(
        args=args,
        out_dir=Path(args.mask_cache_dir),
        raw=raw,
        config=config,
        steps=steps_hop,
        action_masks=action_masks,
    )
    mask_ints = [[words_to_int(edge_masks[t, a]) for a in range(edge_masks.shape[1])] for t in range(edge_masks.shape[0])]
    edge_port_int, port_to_edges_int = build_port_maps(config, wrap_planes=bool(wrap_planes_from_config(raw)))
    usage_weights, usage_key_to_row = load_usage_cache(Path(args.usage_cache_dir))

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

    state_features, state_meta = build_state_features(
        steps=steps_hop,
        feature_mode=str(args.feature_mode),
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )

    count_norm = max(1e-9, float(args.count_edge_scale))
    critical_norm = max(1e-9, float(args.critical_drop_scale))
    time_index: list[int] = []
    step_values: list[int] = []
    prev2_actions: list[int] = []
    prev_actions: list[int] = []
    curr_actions: list[int] = []
    stage_cost_values: list[float] = []
    hop_values: list[float] = []
    delay_values: list[float] = []
    active_drop_values: list[int] = []
    building_values: list[int] = []
    building_dropped_values: list[int] = []
    critical_drop_values: list[float] = []
    local_cost_values: list[float] = []
    missing_usage_keys: set[tuple[int, int]] = set()
    sample_rows: list[dict[str, Any]] = []

    for t in range(2, int(steps_hop.size)):
        if (t + 1) % int(args.progress_every) == 0 or t + 1 == int(steps_hop.size):
            print(f"[transition-dataset] t={t + 1}/{int(steps_hop.size)} samples={len(time_index)}", flush=True)
        for a in candidates[t - 2].astype(int).tolist():
            key = (int(t - 2), int(a))
            usage_row = usage_key_to_row.get(key)
            if usage_row is None:
                missing_usage_keys.add(key)
                continue
            usage = usage_weights[int(usage_row)]
            e0 = mask_ints[t - 2][int(a)]
            for b in candidates[t - 1].astype(int).tolist():
                e1 = mask_ints[t - 1][int(b)]
                for c in candidates[t].astype(int).tolist():
                    e2 = mask_ints[t][int(c)]
                    dropped_edges, active_drop, building, building_dropped = local_lst_drop_edges(
                        e0=e0,
                        e1=e1,
                        e2=e2,
                        edge_port_int=edge_port_int,
                        port_to_edges_int=port_to_edges_int,
                    )
                    critical_drop = sum_weights_for_bits(dropped_edges, usage)
                    local_cost = (
                        float(stage_cost[t, c])
                        + float(args.critical_drop_weight) * float(critical_drop) / critical_norm
                        + float(args.active_drop_count_weight) * float(active_drop) / count_norm
                        + float(args.building_weight) * float(building) / count_norm
                    )
                    time_index.append(int(t))
                    step_values.append(int(steps_hop[t]))
                    prev2_actions.append(int(a))
                    prev_actions.append(int(b))
                    curr_actions.append(int(c))
                    stage_cost_values.append(float(stage_cost[t, c]))
                    hop_values.append(float(hops[t, c]))
                    delay_values.append(float(delay[t, c]))
                    active_drop_values.append(int(active_drop))
                    building_values.append(int(building))
                    building_dropped_values.append(int(building_dropped))
                    critical_drop_values.append(float(critical_drop))
                    local_cost_values.append(float(local_cost))
                    if len(sample_rows) < int(args.sample_csv_rows):
                        sample_rows.append(
                            {
                                "time_index": int(t),
                                "step": int(steps_hop[t]),
                                "prev2_action": str(action_names[int(a)]),
                                "prev_action": str(action_names[int(b)]),
                                "action": str(action_names[int(c)]),
                                "prev2_action_index": int(a),
                                "prev_action_index": int(b),
                                "action_index": int(c),
                                "active_drop": int(active_drop),
                                "building": int(building),
                                "building_dropped": int(building_dropped),
                                "critical_drop": float(critical_drop),
                                "stage_cost": float(stage_cost[t, c]),
                                "local_cost": float(local_cost),
                                "mean_hops": float(hops[t, c]),
                                "mean_delay_ms": float(delay[t, c]),
                                "mask": str(mask_tokens[int(c)]),
                            }
                        )

    if missing_usage_keys:
        first = sorted(missing_usage_keys)[:10]
        raise KeyError(
            f"usage cache is missing {len(missing_usage_keys)} (time_index, action_index) keys; "
            f"examples={first}. Use a usage cache built with --top-k-actions {int(args.top_k_actions)}."
        )

    npz_path = out_dir / "row_mask_transition_value_dataset.npz"
    np.savez_compressed(
        npz_path,
        steps=steps_hop.astype(np.int64),
        state_features=np.asarray(state_features, dtype=np.float32),
        time_index=np.asarray(time_index, dtype=np.int16),
        step=np.asarray(step_values, dtype=np.int64),
        prev2_action_index=np.asarray(prev2_actions, dtype=np.int16),
        prev_action_index=np.asarray(prev_actions, dtype=np.int16),
        action_index=np.asarray(curr_actions, dtype=np.int16),
        action_bits=np.asarray(action_masks, dtype=np.int8),
        action_names=np.asarray(action_names, dtype=object),
        mask_tokens=np.asarray(mask_tokens, dtype=object),
        stage_cost=np.asarray(stage_cost_values, dtype=np.float32),
        mean_hops=np.asarray(hop_values, dtype=np.float32),
        mean_delay_ms=np.asarray(delay_values, dtype=np.float32),
        active_drop_count=np.asarray(active_drop_values, dtype=np.int16),
        building_count=np.asarray(building_values, dtype=np.int16),
        building_dropped_count=np.asarray(building_dropped_values, dtype=np.int16),
        critical_drop=np.asarray(critical_drop_values, dtype=np.float32),
        local_transition_cost=np.asarray(local_cost_values, dtype=np.float32),
        hop_table=np.asarray(hops, dtype=np.float32),
        delay_table_ms=np.asarray(delay, dtype=np.float32),
        edge_universe_size=np.asarray([int(edge_universe_size)], dtype=np.int32),
    )

    sample_csv = out_dir / "row_mask_transition_value_sample.csv"
    write_sample_csv(sample_csv, sample_rows)
    meta = {
        "config": str(Path(args.config)),
        "reward_dir": str(Path(args.reward_dir)),
        "mask_cache_dir": str(Path(args.mask_cache_dir)),
        "usage_cache_dir": str(Path(args.usage_cache_dir)),
        "out_dir": str(out_dir),
        "npz": str(npz_path),
        "sample_csv": str(sample_csv),
        "top_k_actions": int(args.top_k_actions),
        "include_schedules": [str(Path(x)) for x in include_paths],
        "lambda_hop": float(args.lambda_hop),
        "critical_drop_weight": float(args.critical_drop_weight),
        "active_drop_count_weight": float(args.active_drop_count_weight),
        "building_weight": float(args.building_weight),
        "count_edge_scale": float(args.count_edge_scale),
        "critical_drop_scale": float(args.critical_drop_scale),
        "num_steps": int(steps_hop.size),
        "num_actions": int(len(action_names)),
        "num_transition_samples": int(len(time_index)),
        "candidate_size": {
            "mean": float(np.mean([len(x) for x in candidates])),
            "min": int(np.min([len(x) for x in candidates])),
            "max": int(np.max([len(x) for x in candidates])),
        },
        "state_meta": state_meta,
        "arrays": {
            "state_features": [int(x) for x in state_features.shape],
            "time_index": [int(len(time_index))],
            "action_bits": [int(x) for x in action_masks.shape],
        },
    }
    meta_path = out_dir / "row_mask_transition_value_dataset_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(npz_path), "meta": str(meta_path), "samples": int(len(time_index))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
