# -*- coding: utf-8 -*-
"""Export exact Q/value-to-go labels for transition-level row-mask fusion.

This reads ``row_mask_transition_value_dataset.npz`` and performs an exact
backward Bellman recursion over the candidate triple graph:

    Q_t(a,b,c) = local_cost_t(a,b,c) + V_{t+1}(b,c)
    V_t(a,b)   = min_c Q_t(a,b,c)

The resulting Q/advantage labels are the natural supervised target for a
value-to-go model or fitted-Q style policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


G60_RUN_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
DEFAULT_DATASET = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_transition_value_dataset"
    / "topk3_lam080_crit005_cnt003_build003"
    / "row_mask_transition_value_dataset.npz"
)
DEFAULT_OUT_DIR = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_transition_q_value_dataset"
    / "topk3_lam080_crit005_cnt003_build003"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact Bellman Q labels from candidate transition triples.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lambda-hop", type=float, default=0.8)
    parser.add_argument("--sample-csv-rows", type=int, default=3000)
    return parser.parse_args()


def build_stage_matrix(hops: np.ndarray, delay: np.ndarray, lambda_hop: float) -> np.ndarray:
    hop_env = np.nanmin(hops, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hops) - np.nanmin(hops)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    return (
        float(lambda_hop) * ((hops - hop_env[:, None]) / hop_scale)
        + (1.0 - float(lambda_hop)) * ((delay - delay_env[:, None]) / delay_scale)
    )


def mask_token(bits: np.ndarray) -> str:
    rows = [idx for idx, value in enumerate(bits.tolist()) if int(value) != 0]
    return "{" + ",".join(f"{idx:02d}" for idx in rows) + "}"


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def write_schedule(
    path: Path,
    *,
    steps: np.ndarray,
    selected: np.ndarray,
    action_names: list[str],
    action_bits: np.ndarray,
    hops: np.ndarray,
    delay: np.ndarray,
    stage_matrix: np.ndarray,
) -> None:
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(selected.astype(int).tolist()):
        bits = action_bits[int(action_idx)]
        row: dict[str, Any] = {
            "step": int(steps[t]),
            "policy": "exact_transition_q_topk3",
            "action": str(action_names[int(action_idx)]),
            "action_index": int(action_idx),
            "mask": mask_token(bits),
            "row_count": int(np.sum(bits)),
            "mean_hops": float(hops[t, int(action_idx)]),
            "mean_delay_ms": float(delay[t, int(action_idx)]),
            "stage_cost": float(stage_matrix[t, int(action_idx)]),
        }
        for y in range(bits.shape[0]):
            row[f"y{y:02d}"] = int(bits[y])
        rows.append(row)
    write_rows(path, rows)


def summarize_selected(selected: np.ndarray, hops: np.ndarray, delay: np.ndarray, stage_matrix: np.ndarray) -> dict[str, Any]:
    idx = selected.astype(int)
    row = np.arange(idx.size)
    return {
        "mean_hops_no_lst": float(np.mean(hops[row, idx])),
        "mean_delay_ms_no_lst": float(np.mean(delay[row, idx])),
        "mean_stage_cost_no_lst": float(np.mean(stage_matrix[row, idx])),
        "switches": int(np.count_nonzero(idx[1:] != idx[:-1])),
        "num_segments": int(np.count_nonzero(idx[1:] != idx[:-1]) + 1),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(Path(args.dataset), allow_pickle=True)

    steps = data["steps"].astype(np.int64)
    time_idx = data["time_index"].astype(np.int64)
    a0 = data["prev2_action_index"].astype(np.int64)
    a1 = data["prev_action_index"].astype(np.int64)
    a2 = data["action_index"].astype(np.int64)
    local_cost = data["local_transition_cost"].astype(np.float64)
    action_bits = data["action_bits"].astype(np.int8)
    action_names = [str(x) for x in data["action_names"].tolist()]
    hops = data["hop_table"].astype(np.float64)
    delay = data["delay_table_ms"].astype(np.float64)
    stage_matrix = build_stage_matrix(hops, delay, float(args.lambda_hop))

    order = np.argsort(time_idx, kind="stable")
    sorted_time = time_idx[order]
    t_count = int(steps.size)
    q_value = np.full(time_idx.shape, np.inf, dtype=np.float64)
    q_advantage = np.full(time_idx.shape, np.inf, dtype=np.float64)
    is_best = np.zeros(time_idx.shape, dtype=np.int8)
    best_next_by_time_state: dict[tuple[int, int, int], int] = {}

    offsets: dict[int, tuple[int, int]] = {}
    start = 0
    while start < order.size:
        t = int(sorted_time[start])
        end = start + 1
        while end < order.size and int(sorted_time[end]) == t:
            end += 1
        offsets[t] = (start, end)
        start = end

    v_next: dict[tuple[int, int], float] = {}
    value_sizes: dict[int, int] = {}
    for t in range(t_count - 1, 1, -1):
        if t not in offsets:
            raise ValueError(f"transition dataset has no rows for t={t}")
        start, end = offsets[t]
        state_rows: dict[tuple[int, int], list[int]] = {}
        for pos in range(start, end):
            idx = int(order[pos])
            future = 0.0 if t == t_count - 1 else v_next.get((int(a1[idx]), int(a2[idx])), math.inf)
            q = float(local_cost[idx]) + float(future)
            q_value[idx] = q
            state_rows.setdefault((int(a0[idx]), int(a1[idx])), []).append(idx)

        v_curr: dict[tuple[int, int], float] = {}
        for state, rows in state_rows.items():
            finite_rows = [idx for idx in rows if math.isfinite(float(q_value[idx]))]
            if not finite_rows:
                continue
            best_idx = min(finite_rows, key=lambda idx: float(q_value[idx]))
            best_q = float(q_value[best_idx])
            v_curr[state] = best_q
            best_next_by_time_state[(int(t), int(state[0]), int(state[1]))] = int(a2[best_idx])
            for idx in finite_rows:
                q_advantage[idx] = float(q_value[idx] - best_q)
                if abs(float(q_value[idx] - best_q)) <= 1e-10:
                    is_best[idx] = 1
        v_next = v_curr
        value_sizes[t] = int(len(v_curr))

    initial_candidates = [
        (state, float(stage_matrix[0, state[0]] + stage_matrix[1, state[1]] + value))
        for state, value in v_next.items()
    ]
    if not initial_candidates:
        raise ValueError("no feasible initial state after Bellman recursion")
    initial_state, initial_objective = min(initial_candidates, key=lambda item: item[1])
    selected = np.full(t_count, -1, dtype=np.int16)
    selected[0] = int(initial_state[0])
    selected[1] = int(initial_state[1])
    for t in range(2, t_count):
        key = (int(t), int(selected[t - 2]), int(selected[t - 1]))
        if key not in best_next_by_time_state:
            raise ValueError(f"cannot reconstruct policy at t={t}, state={key[1:]}")
        selected[t] = int(best_next_by_time_state[key])

    npz_path = out_dir / "row_mask_transition_q_value_dataset.npz"
    np.savez_compressed(
        npz_path,
        steps=steps.astype(np.int64),
        state_features=data["state_features"].astype(np.float32),
        time_index=time_idx.astype(np.int16),
        step=data["step"].astype(np.int64),
        prev2_action_index=a0.astype(np.int16),
        prev_action_index=a1.astype(np.int16),
        action_index=a2.astype(np.int16),
        action_bits=action_bits.astype(np.int8),
        action_names=np.asarray(action_names, dtype=object),
        stage_cost=data["stage_cost"].astype(np.float32),
        local_transition_cost=local_cost.astype(np.float32),
        q_value=np.asarray(q_value, dtype=np.float32),
        q_advantage=np.asarray(q_advantage, dtype=np.float32),
        is_best_transition=is_best.astype(np.int8),
        exact_selected_action_index=selected.astype(np.int16),
        mean_hops=data["mean_hops"].astype(np.float32),
        mean_delay_ms=data["mean_delay_ms"].astype(np.float32),
        hop_table=hops.astype(np.float32),
        delay_table_ms=delay.astype(np.float32),
    )

    sample_rows: list[dict[str, Any]] = []
    finite_idx = np.flatnonzero(np.isfinite(q_value))[: int(args.sample_csv_rows)]
    for idx in finite_idx.astype(int).tolist():
        sample_rows.append(
            {
                "time_index": int(time_idx[idx]),
                "step": int(data["step"][idx]),
                "prev2_action_index": int(a0[idx]),
                "prev_action_index": int(a1[idx]),
                "action_index": int(a2[idx]),
                "local_transition_cost": float(local_cost[idx]),
                "q_value": float(q_value[idx]),
                "q_advantage": float(q_advantage[idx]),
                "is_best_transition": int(is_best[idx]),
            }
        )
    sample_csv = out_dir / "row_mask_transition_q_value_sample.csv"
    write_rows(sample_csv, sample_rows)
    schedule_csv = out_dir / "exact_transition_q_topk3.csv"
    write_schedule(
        schedule_csv,
        steps=steps,
        selected=selected,
        action_names=action_names,
        action_bits=action_bits,
        hops=hops,
        delay=delay,
        stage_matrix=stage_matrix,
    )

    finite_q = q_value[np.isfinite(q_value)]
    finite_adv = q_advantage[np.isfinite(q_advantage)]
    meta = {
        "dataset": str(Path(args.dataset)),
        "out_dir": str(out_dir),
        "npz": str(npz_path),
        "sample_csv": str(sample_csv),
        "exact_schedule_csv": str(schedule_csv),
        "lambda_hop": float(args.lambda_hop),
        "num_transition_rows": int(time_idx.size),
        "num_finite_q_rows": int(finite_q.size),
        "q_value_mean": float(np.mean(finite_q)),
        "q_value_min": float(np.min(finite_q)),
        "q_value_max": float(np.max(finite_q)),
        "q_advantage_mean": float(np.mean(finite_adv)),
        "q_advantage_p95": float(np.percentile(finite_adv, 95)),
        "best_transition_rows": int(np.count_nonzero(is_best)),
        "initial_state": [int(initial_state[0]), int(initial_state[1])],
        "initial_objective": float(initial_objective),
        "value_state_counts_by_time": {str(k): int(v) for k, v in value_sizes.items()},
        "exact_policy_summary": summarize_selected(selected, hops, delay, stage_matrix),
    }
    meta_path = out_dir / "row_mask_transition_q_value_dataset_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(npz_path), "meta": str(meta_path), "schedule": str(schedule_csv), "summary": meta["exact_policy_summary"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
