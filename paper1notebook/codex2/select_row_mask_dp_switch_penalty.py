from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_REWARD_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_action_scorer_hard96"
    r"\action_library_reward_table_full1437_merged"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_dp_switch_penalty"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic-programming action selector over row-mask 000040/000056 actions. "
            "The stage cost is normalized hop/delay regret to the table envelope, plus "
            "a transition penalty proportional to row-mask Hamming distance."
        )
    )
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    parser.add_argument(
        "--switch-penalties",
        nargs="+",
        type=float,
        default=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
        help="Penalty multiplier applied to normalized row-mask Hamming distance.",
    )
    parser.add_argument("--min-dwell-steps", type=int, default=1, help="Optional minimum dwell in sampled steps.")
    return parser.parse_args()


def read_action_table(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(path)
    action_names = [str(x) for x in df["action"].tolist()]
    y_cols = [f"y{i:02d}" for i in range(36)]
    masks = df[y_cols].to_numpy(dtype=np.int8)
    mask_tokens = [str(x) for x in df["mask"].tolist()]
    return action_names, masks, mask_tokens


def read_wide(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(path)
    steps = df["step"].to_numpy(dtype=np.int64)
    names = [name for name in df.columns if name != "step"]
    values = df[names].to_numpy(dtype=np.float64)
    return steps, names, values


def ensure_same_actions(expected: Sequence[str], actual: Sequence[str], label: str) -> None:
    if list(expected) != list(actual):
        raise ValueError(f"{label} action columns differ from action table")


def transition_cost_matrix(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.int8)
    # normalized Hamming distance in y-row space; values are in [0, 1].
    diff = np.abs(masks[:, None, :] - masks[None, :, :]).sum(axis=2).astype(np.float64)
    return diff / float(masks.shape[1])


def run_dp(
    stage_cost: np.ndarray,
    transition_cost: np.ndarray,
    *,
    switch_penalty: float,
    min_dwell_steps: int,
) -> np.ndarray:
    t_count, a_count = stage_cost.shape
    if min_dwell_steps <= 1:
        dp = np.empty((t_count, a_count), dtype=np.float64)
        parent = np.empty((t_count, a_count), dtype=np.int16)
        dp[0] = stage_cost[0]
        parent[0] = -1
        trans = float(switch_penalty) * transition_cost
        for t in range(1, t_count):
            prev = dp[t - 1][:, None] + trans
            parent[t] = np.argmin(prev, axis=0).astype(np.int16)
            dp[t] = stage_cost[t] + prev[parent[t], np.arange(a_count)]
        out = np.empty(t_count, dtype=np.int16)
        out[-1] = int(np.argmin(dp[-1]))
        for t in range(t_count - 1, 0, -1):
            out[t - 1] = int(parent[t, out[t]])
        return out

    # State augments action with dwell age, capped at min_dwell_steps.
    dwell_cap = int(min_dwell_steps)
    inf = 1e100
    dp = np.full((t_count, a_count, dwell_cap), inf, dtype=np.float64)
    parent_action = np.full((t_count, a_count, dwell_cap), -1, dtype=np.int16)
    parent_age = np.full((t_count, a_count, dwell_cap), -1, dtype=np.int16)
    dp[0, :, 0] = stage_cost[0]
    trans = float(switch_penalty) * transition_cost
    for t in range(1, t_count):
        for a_prev in range(a_count):
            for age_prev in range(dwell_cap):
                base = dp[t - 1, a_prev, age_prev]
                if not np.isfinite(base):
                    continue
                # stay
                age_next = min(dwell_cap - 1, age_prev + 1)
                value = base + stage_cost[t, a_prev]
                if value < dp[t, a_prev, age_next]:
                    dp[t, a_prev, age_next] = value
                    parent_action[t, a_prev, age_next] = a_prev
                    parent_age[t, a_prev, age_next] = age_prev
                # switch only after satisfying dwell.
                if age_prev < dwell_cap - 1:
                    continue
                for a_next in range(a_count):
                    if a_next == a_prev:
                        continue
                    value = base + trans[a_prev, a_next] + stage_cost[t, a_next]
                    if value < dp[t, a_next, 0]:
                        dp[t, a_next, 0] = value
                        parent_action[t, a_next, 0] = a_prev
                        parent_age[t, a_next, 0] = age_prev
    flat = int(np.argmin(dp[-1].reshape(-1)))
    action = flat // dwell_cap
    age = flat % dwell_cap
    out = np.empty(t_count, dtype=np.int16)
    out[-1] = action
    for t in range(t_count - 1, 0, -1):
        prev_action = int(parent_action[t, action, age])
        prev_age = int(parent_age[t, action, age])
        out[t - 1] = prev_action
        action, age = prev_action, prev_age
    return out


def schedule_rows(
    *,
    steps: np.ndarray,
    selected: np.ndarray,
    policy: str,
    lambda_hop: float,
    switch_penalty: float,
    action_names: Sequence[str],
    masks: np.ndarray,
    mask_tokens: Sequence[str],
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(selected.astype(int).tolist()):
        mask = masks[action_idx]
        row: dict[str, Any] = {
            "step": int(steps[t]),
            "policy": policy,
            "lambda_hop": float(lambda_hop),
            "switch_penalty": float(switch_penalty),
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


def summarize_policy(
    *,
    selected: np.ndarray,
    masks: np.ndarray,
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
    transition_cost: np.ndarray,
) -> dict[str, Any]:
    idx = selected.astype(int)
    rows = np.arange(idx.size)
    switches = int(np.count_nonzero(idx[1:] != idx[:-1]))
    hamming = [float(transition_cost[int(idx[t - 1]), int(idx[t])]) for t in range(1, idx.size)]
    return {
        "mean_hops": float(np.mean(hops[rows, idx])),
        "mean_delay_ms": float(np.mean(delay[rows, idx])),
        "mean_gap_hop_env": float(np.mean(hops[rows, idx] - hop_env)),
        "mean_gap_delay_env_ms": float(np.mean(delay[rows, idx] - delay_env)),
        "switches": switches,
        "num_segments": int(switches + 1),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
        "mean_row_hamming_transition": float(np.mean(hamming)) if hamming else 0.0,
        "max_row_hamming_transition": float(np.max(hamming)) if hamming else 0.0,
        "total_row_hamming_transition": float(np.sum(hamming)) if hamming else 0.0,
    }


def main() -> int:
    args = parse_args()
    reward_dir = Path(args.reward_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_names, masks, mask_tokens = read_action_table(reward_dir / "row_mask_action_library_actions.csv")
    steps_hop, names_hop, hops = read_wide(reward_dir / "row_mask_action_library_mean_hops_wide.csv")
    steps_delay, names_delay, delay = read_wide(reward_dir / "row_mask_action_library_mean_delay_ms_wide.csv")
    if not np.array_equal(steps_hop, steps_delay):
        raise ValueError("hop/delay tables have different steps")
    ensure_same_actions(action_names, names_hop, "hop table")
    ensure_same_actions(action_names, names_delay, "delay table")

    hop_env = np.nanmin(hops, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hops) - np.nanmin(hops)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    stage_cost = (
        float(args.lambda_hop) * ((hops - hop_env[:, None]) / hop_scale)
        + (1.0 - float(args.lambda_hop)) * ((delay - delay_env[:, None]) / delay_scale)
    )
    transition = transition_cost_matrix(masks)

    summary_rows: list[dict[str, Any]] = []
    for penalty in args.switch_penalties:
        selected = run_dp(
            stage_cost,
            transition,
            switch_penalty=float(penalty),
            min_dwell_steps=int(args.min_dwell_steps),
        )
        policy = f"dp_lambda{float(args.lambda_hop):.2f}_sw{float(penalty):g}_dwell{int(args.min_dwell_steps)}"
        rows = schedule_rows(
            steps=steps_hop,
            selected=selected,
            policy=policy,
            lambda_hop=float(args.lambda_hop),
            switch_penalty=float(penalty),
            action_names=action_names,
            masks=masks,
            mask_tokens=mask_tokens,
            hops=hops,
            delay=delay,
            hop_env=hop_env,
            delay_env=delay_env,
        )
        schedule_path = out_dir / f"{policy}.csv"
        write_rows(schedule_path, rows)
        summary = summarize_policy(
            selected=selected,
            masks=masks,
            hops=hops,
            delay=delay,
            hop_env=hop_env,
            delay_env=delay_env,
            transition_cost=transition,
        )
        summary_rows.append(
            {
                "policy": policy,
                "lambda_hop": float(args.lambda_hop),
                "switch_penalty": float(penalty),
                "min_dwell_steps": int(args.min_dwell_steps),
                "schedule_csv": str(schedule_path),
                **summary,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "dp_switch_penalty_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta = {
        "reward_dir": str(reward_dir),
        "out_dir": str(out_dir),
        "lambda_hop": float(args.lambda_hop),
        "min_dwell_steps": int(args.min_dwell_steps),
        "hop_scale": hop_scale,
        "delay_scale": delay_scale,
        "mean_hop_envelope": float(np.mean(hop_env)),
        "mean_delay_envelope_ms": float(np.mean(delay_env)),
        "summary_csv": str(summary_path),
    }
    (out_dir / "dp_switch_penalty_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
