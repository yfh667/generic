from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module.group_states import group_nodes_for_step
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step
from src.topology_learning.module.config_io import load_yaml_dict, optional_path
from src.topology_learning.module.lst_aware_beam_selector import (
    LstAwarePairSpec,
    _active_mask_from_history,
    _build_union_table_and_action_masks,
    _candidate_actions_for_row,
    _load_config_with_g60_fallback,
    _mean_shortest_delay,
    _mean_shortest_hops,
    _metric_weights,
    _relative_gap,
    parse_pair_spec,
)


def _read_by_step(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "step" not in rows[0] or "topology" not in rows[0]:
        raise ValueError("by-step CSV must contain step and topology columns")
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _setup_counts(selected: np.ndarray, transition_counts: np.ndarray) -> np.ndarray:
    out = np.zeros(selected.shape[0], dtype=np.int32)
    if selected.size > 1:
        out[1:] = transition_counts[selected[:-1], selected[1:]].astype(np.int32, copy=False)
    return out


def _segment_rows(
    *,
    selected: np.ndarray,
    steps: np.ndarray,
    topology_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    for idx in range(1, selected.size + 1):
        if idx == selected.size or int(selected[idx]) != int(selected[start]):
            action = int(selected[start])
            rows.append(
                {
                    "left_row": int(start),
                    "right_row": int(idx - 1),
                    "left_step": int(steps[start]),
                    "right_step": int(steps[idx - 1]),
                    "length_rows": int(idx - start),
                    "action_idx": action,
                    "topology": str(topology_names[action]),
                }
            )
            start = idx
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locally replace one window of a by-step topology schedule using "
            "the same active-after-LST scoring as the full LST-aware beam selector."
        )
    )
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-by-step-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--left-row", type=int, required=True)
    parser.add_argument("--right-row", type=int, required=True)
    parser.add_argument("--pair", action="append", required=True, help="source_group:target_group:metric_prefix[:label]")
    parser.add_argument("--setup-time", type=float, required=True)
    parser.add_argument("--delay-store-dir", type=Path, required=True)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--group-xml", type=Path, required=True)
    parser.add_argument("--group-cache-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--setup-penalty", type=float, default=0.0)
    parser.add_argument("--burst-penalty", type=float, default=0.0)
    parser.add_argument("--max-transition-count", type=int, default=None)
    parser.add_argument("--extra-tail-rows", type=int, default=None)
    parser.add_argument("--include-action", action="append", default=[], help="Topology name or action index always allowed in the window.")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selector_dir = Path(args.selector_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_yaml_dict(selector_dir / "selector_config.yaml")
    config = _load_config_with_g60_fallback(raw_config)
    topology_library_csv = optional_path(raw_config.get("inputs", {}).get("topology_library_csv"))
    if topology_library_csv is None:
        raise ValueError("selector_config.yaml inputs.topology_library_csv is required")

    with np.load(selector_dir / "selector_arrays.npz", allow_pickle=False) as arrays:
        steps = np.asarray(arrays["steps"], dtype=np.int64)
        topology_names = tuple(str(x) for x in np.asarray(arrays["topology_names"]))
        stage_cost = np.asarray(arrays["stage_cost"], dtype=np.float32)
        transition_counts = np.asarray(arrays["transition_counts"], dtype=np.float32)
        metric_refs_all = {
            str(key): np.asarray(arrays[key], dtype=np.float32)
            for key in arrays.files
            if str(key).startswith("metric_") and str(key).endswith("_full_link")
        }

    by_step_rows = _read_by_step(args.base_by_step_csv)
    if len(by_step_rows) != int(steps.size):
        raise ValueError(f"base by-step rows={len(by_step_rows)} does not match selector steps={steps.size}")
    name_to_action = {name: idx for idx, name in enumerate(topology_names)}
    base_selected = np.asarray([name_to_action[str(row["topology"])] for row in by_step_rows], dtype=np.int32)

    left = int(args.left_row)
    right = int(args.right_row)
    if not (0 < left <= right < base_selected.size - 1):
        raise ValueError("window must satisfy 0 < left <= right < num_rows - 1")

    pair_specs = [parse_pair_spec(text) for text in args.pair]
    metric_weights = _metric_weights(raw_config, pair_specs)

    stride = int(steps[1] - steps[0]) if steps.size > 1 else 1
    lag_steps = int(math.ceil(float(args.setup_time) / float(stride))) if float(args.setup_time) > 0 else 0
    history_len = max(1, lag_steps + 1)
    extra_tail = lag_steps if args.extra_tail_rows is None else int(args.extra_tail_rows)
    score_right = min(int(steps.size) - 1, right + max(0, extra_tail))
    prefix_left = max(0, left - history_len)
    local_steps = steps[prefix_left : score_right + 1]

    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=[int(x) for x in local_steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    pair_nodes_by_global_row: dict[int, list[tuple[LstAwarePairSpec, tuple[int, ...], tuple[int, ...]]]] = {}
    for step in local_steps:
        row_pairs = []
        for pair in pair_specs:
            row_pairs.append(
                (
                    pair,
                    tuple(group_nodes_for_step(group_data, int(step), int(pair.source_group_id))),
                    tuple(group_nodes_for_step(group_data, int(step), int(pair.target_group_id))),
                )
            )
        row = int((int(step) - int(steps[0])) // stride)
        pair_nodes_by_global_row[row] = row_pairs

    union_table, action_masks = _build_union_table_and_action_masks(
        config=config,
        topology_names=topology_names,
        topology_library_csv=topology_library_csv,
    )
    src_all = np.asarray(union_table.src, dtype=np.int32)
    dst_all = np.asarray(union_table.dst, dtype=np.int32)

    delay_store = open_delay_store_for_interval(
        int(local_steps[0]),
        int(local_steps[-1]),
        stride=int(stride),
        store_dir=Path(args.delay_store_dir),
        constellation_name=config.name,
    )
    delay_rows = delay_store.rows_for_interval(int(local_steps[0]), int(local_steps[-1]), int(stride))
    position_store = None
    position_rows = None
    if args.position_cache_dir not in (None, "", False):
        position_store = open_position_cache_for_interval(
            int(local_steps[0]),
            int(local_steps[-1]),
            stride=int(stride),
            cache_dir=Path(args.position_cache_dir),
        )
        position_rows = position_store.rows_for_interval(int(local_steps[0]), int(local_steps[-1]), int(stride))
    lookup = build_weight_lookup(
        union_table,
        delay_store,
        config=config,
        allow_intra_fallback=position_store is not None,
    )
    weights_by_global_row: dict[int, np.ndarray] = {}
    for local_idx, step in enumerate(local_steps):
        row = int((int(step) - int(steps[0])) // stride)
        weights_by_global_row[row] = edge_weights_for_step(
            edge_table=union_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[local_idx]),
            position_row=int(position_rows[local_idx]) if position_rows is not None else None,
        )

    eval_cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def score_active_graph(global_row: int, history: tuple[int, ...]) -> float:
        key = (int(global_row), tuple(int(x) for x in history[-history_len:]))
        cached = eval_cache.get(key)
        if cached is not None:
            return float(cached)
        active = _active_mask_from_history(action_masks, key[1])
        cols = np.flatnonzero(active)
        src_edges = src_all[cols]
        dst_edges = dst_all[cols]
        weights = np.asarray(weights_by_global_row[int(global_row)], dtype=np.float32)[cols]
        total = 0.0
        for pair, sources, targets in pair_nodes_by_global_row[int(global_row)]:
            delay_name = f"{pair.metric_prefix}_delay_ms"
            hops_name = f"{pair.metric_prefix}_hops"
            delay_value = _mean_shortest_delay(
                n_nodes=int(config.total_sats),
                src_edges=src_edges,
                dst_edges=dst_edges,
                weights=weights,
                sources=sources,
                targets=targets,
                require_all_pairs=True,
            )
            hops_value = _mean_shortest_hops(
                n_nodes=int(config.total_sats),
                src_edges=src_edges,
                dst_edges=dst_edges,
                sources=sources,
                targets=targets,
                require_all_pairs=True,
            )
            delay_ref = float(metric_refs_all[f"metric_{delay_name}_full_link"][int(global_row)])
            hops_ref = float(metric_refs_all[f"metric_{hops_name}_full_link"][int(global_row)])
            total += float(metric_weights.get(delay_name, 0.0)) * _relative_gap(delay_value, delay_ref)
            total += float(metric_weights.get(hops_name, 0.0)) * _relative_gap(hops_value, hops_ref)
        if not math.isfinite(total):
            total = 1.0e6
        eval_cache[key] = float(total)
        return float(total)

    extra_actions: set[int] = set()
    for token in args.include_action:
        raw = str(token).strip()
        if raw.isdigit():
            extra_actions.add(int(raw))
        else:
            extra_actions.add(int(name_to_action[raw]))
    for row in range(left, right + 1):
        extra_actions.add(int(base_selected[row]))
    extra_actions.add(int(base_selected[left - 1]))
    extra_actions.add(int(base_selected[right + 1]))

    max_transition = float(np.max(transition_counts)) or 1.0
    initial_history = tuple(int(x) for x in base_selected[max(0, left - history_len) : left])
    if len(initial_history) < history_len:
        initial_history = tuple([int(base_selected[0])] * (history_len - len(initial_history)) + list(initial_history))
    beam: list[tuple[float, tuple[int, ...], list[int], list[float], int, int]] = [
        (0.0, initial_history[-history_len:], [], [], 0, 0)
    ]

    for global_row in range(left, score_right + 1):
        if global_row <= right:
            row_candidates = _candidate_actions_for_row(
                row=int(global_row),
                stage_cost=stage_cost,
                top_k=int(args.top_k),
                include_schedules=[],
            )
            candidate_set = set(int(x) for x in row_candidates.tolist())
            candidate_set.update(extra_actions)
        else:
            candidate_set = {int(base_selected[global_row])}

        next_beam: list[tuple[float, tuple[int, ...], list[int], list[float], int, int]] = []
        for total_cost, history, path, row_costs, total_setup, max_setup in beam:
            prev_action = int(history[-1])
            candidates = sorted(candidate_set | {prev_action})
            for action in candidates:
                setup = int(round(float(transition_counts[prev_action, int(action)])))
                if args.max_transition_count is not None and setup > int(args.max_transition_count):
                    continue
                next_history = tuple((list(history) + [int(action)])[-history_len:])
                graph_cost = score_active_graph(global_row, next_history)
                transition_cost = float(args.setup_penalty) * (float(setup) / max_transition)
                if float(args.burst_penalty) != 0.0:
                    transition_cost += float(args.burst_penalty) * (float(setup) / max_transition) ** 2
                next_cost = float(total_cost) + float(graph_cost) + float(transition_cost)
                next_path = path + ([int(action)] if global_row <= right else [])
                next_beam.append(
                    (
                        next_cost,
                        next_history,
                        next_path,
                        row_costs + [float(graph_cost)],
                        int(total_setup + (setup if global_row <= right + 1 else 0)),
                        max(int(max_setup), int(setup)),
                    )
                )
        if not next_beam:
            raise RuntimeError(f"beam became empty at global_row={global_row}")
        next_beam.sort(key=lambda item: item[0])
        beam = next_beam[: int(args.beam_width)]
        print(
            f"[lst-aware-local] row={global_row}/{score_right} "
            f"beam={len(beam)} best={beam[0][0]:.6f} cache={len(eval_cache)}",
            flush=True,
        )

    best_cost, _history, local_path, best_row_costs, local_setup, local_max_setup = beam[0]
    if len(local_path) != right - left + 1:
        raise RuntimeError(f"local path length {len(local_path)} != window length {right - left + 1}")

    selected = base_selected.copy()
    selected[left : right + 1] = np.asarray(local_path, dtype=np.int32)
    setup_counts = _setup_counts(selected, transition_counts)
    by_step_out = []
    for row, action in enumerate(selected):
        by_step_out.append(
            {
                "step": int(steps[row]),
                "action_idx": int(action),
                "topology": str(topology_names[int(action)]),
                "setup_commands": int(setup_counts[row]),
            }
        )
    label = f"lst_aware_local_window_{left}_{right}_k{int(args.top_k)}_b{int(args.beam_width)}_sp{float(args.setup_penalty):g}"
    by_step_path = out_dir / f"{label}_by_step.csv"
    _write_rows(by_step_path, by_step_out)
    np.save(out_dir / "selected_action.npy", selected.astype(np.int32, copy=False))
    np.save(out_dir / "local_selected_action.npy", np.asarray(local_path, dtype=np.int32))
    _write_rows(out_dir / "segments.csv", _segment_rows(selected=selected, steps=steps, topology_names=topology_names))

    meta = {
        "selector_dir": str(selector_dir),
        "base_by_step_csv": str(args.base_by_step_csv),
        "schedule_name": label,
        "left_row": int(left),
        "right_row": int(right),
        "left_step": int(steps[left]),
        "right_step": int(steps[right]),
        "score_right_row": int(score_right),
        "score_right_step": int(steps[score_right]),
        "setup_time_seconds": float(args.setup_time),
        "stride_seconds": int(stride),
        "history_len": int(history_len),
        "top_k": int(args.top_k),
        "beam_width": int(args.beam_width),
        "setup_penalty": float(args.setup_penalty),
        "burst_penalty": float(args.burst_penalty),
        "max_transition_count": None if args.max_transition_count is None else int(args.max_transition_count),
        "best_local_objective": float(best_cost),
        "best_local_row_cost_sum": float(np.sum(best_row_costs)),
        "local_window_setup_accumulator": int(local_setup),
        "local_max_setup": int(local_max_setup),
        "full_total_setup_commands": int(np.sum(setup_counts, dtype=np.int64)),
        "full_max_setup_commands_per_step": int(np.max(setup_counts)) if setup_counts.size else 0,
        "full_num_switches": int(np.count_nonzero(selected[1:] != selected[:-1])) if selected.size > 1 else 0,
        "eval_cache_entries": int(len(eval_cache)),
        "outputs": {
            "by_step_csv": str(by_step_path),
            "selected_action_npy": str(out_dir / "selected_action.npy"),
            "local_selected_action_npy": str(out_dir / "local_selected_action.npy"),
            "segments_csv": str(out_dir / "segments.csv"),
            "meta_json": str(out_dir / "meta.json"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
