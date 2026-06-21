from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_STATIC_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\static_motif0040_0056_china_europe_region_internal_plus_grid"
    r"\region_internal_grid_metrics_t0_86160_stride60"
    r"\china_europe\topologies"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_pure_advantage_policy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build interpretable pure-000040/pure-000056 schedules from static China-Europe "
            "metrics. Action 0 is pure motif000056; action 1 is pure motif000040. The output "
            "CSV format is compatible with build_tabular_bandit_oracle_dynamic_topology.py."
        )
    )
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument(
        "--delay-eps-ms",
        nargs="+",
        type=float,
        default=(0.0, 0.5, 1.0, 2.0, 3.0),
        help="Use 000040 when hop40<hop56 and delay40-delay56 <= eps.",
    )
    parser.add_argument(
        "--dp-switch-penalties",
        nargs="+",
        type=float,
        default=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        help="Binary DP transition penalties on action changes.",
    )
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    return parser.parse_args()


def read_static_pair(static_dir: Path) -> pd.DataFrame:
    p40 = pd.read_csv(Path(static_dir) / "selected_motif_000040" / "step_metrics.csv")
    p56 = pd.read_csv(Path(static_dir) / "selected_motif_000056" / "step_metrics.csv")
    df = p40[
        [
            "step",
            "mean_shortest_hops",
            "mean_shortest_delay_ms",
        ]
    ].rename(
        columns={
            "mean_shortest_hops": "hop40",
            "mean_shortest_delay_ms": "delay40",
        }
    )
    df = df.merge(
        p56[
            [
                "step",
                "mean_shortest_hops",
                "mean_shortest_delay_ms",
            ]
        ].rename(
            columns={
                "mean_shortest_hops": "hop56",
                "mean_shortest_delay_ms": "delay56",
            }
        ),
        on="step",
        validate="one_to_one",
    )
    df = df.sort_values("step").reset_index(drop=True)
    df["dhop_40_minus_56"] = df["hop40"] - df["hop56"]
    df["ddelay_ms_40_minus_56"] = df["delay40"] - df["delay56"]
    return df


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def contiguous_segments(actions: np.ndarray, steps: np.ndarray) -> list[dict[str, Any]]:
    if actions.size == 0:
        return []
    segments: list[dict[str, Any]] = []
    start_idx = 0
    for idx in range(1, actions.size + 1):
        if idx == actions.size or int(actions[idx]) != int(actions[start_idx]):
            segments.append(
                {
                    "start_step": int(steps[start_idx]),
                    "end_step": int(steps[idx - 1]),
                    "rows": int(idx - start_idx),
                    "duration_s": int((idx - start_idx) * (steps[1] - steps[0] if steps.size > 1 else 0)),
                    "action_index": int(actions[start_idx]),
                    "action": "pure_000040" if int(actions[start_idx]) == 1 else "pure_000056",
                }
            )
            start_idx = idx
    return segments


def run_binary_dp(
    *,
    df: pd.DataFrame,
    lambda_hop: float,
    switch_penalty: float,
) -> np.ndarray:
    hops = df[["hop56", "hop40"]].to_numpy(dtype=np.float64)
    delays = df[["delay56", "delay40"]].to_numpy(dtype=np.float64)
    hop_env = np.min(hops, axis=1)
    delay_env = np.min(delays, axis=1)
    hop_scale = max(1e-9, float(np.max(hops) - np.min(hops)))
    delay_scale = max(1e-9, float(np.max(delays) - np.min(delays)))
    cost = (
        float(lambda_hop) * ((hops - hop_env[:, None]) / hop_scale)
        + (1.0 - float(lambda_hop)) * ((delays - delay_env[:, None]) / delay_scale)
    )
    t_count = cost.shape[0]
    dp = np.empty((t_count, 2), dtype=np.float64)
    parent = np.empty((t_count, 2), dtype=np.int8)
    dp[0] = cost[0]
    parent[0] = -1
    transition = np.asarray([[0.0, float(switch_penalty)], [float(switch_penalty), 0.0]], dtype=np.float64)
    for t in range(1, t_count):
        prev = dp[t - 1][:, None] + transition
        parent[t] = np.argmin(prev, axis=0).astype(np.int8)
        dp[t] = cost[t] + prev[parent[t], np.arange(2)]
    out = np.empty(t_count, dtype=np.int8)
    out[-1] = int(np.argmin(dp[-1]))
    for t in range(t_count - 1, 0, -1):
        out[t - 1] = int(parent[t, out[t]])
    return out


def schedule_rows(
    *,
    df: pd.DataFrame,
    actions: np.ndarray,
    n: int,
    policy: str,
    lambda_hop: float | None = None,
    switch_penalty: float | None = None,
    delay_eps_ms: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, src in df.iterrows():
        action_idx = int(actions[int(idx)])
        is_40 = action_idx == 1
        row: dict[str, Any] = {
            "step": int(src["step"]),
            "policy": policy,
            "action": "pure_000040" if is_40 else "pure_000056",
            "action_index": action_idx,
            "mask": "{all}" if is_40 else "{}",
            "row_count": int(n) if is_40 else 0,
            "mean_hops": float(src["hop40"] if is_40 else src["hop56"]),
            "mean_delay_ms": float(src["delay40"] if is_40 else src["delay56"]),
            "hop40": float(src["hop40"]),
            "hop56": float(src["hop56"]),
            "delay40_ms": float(src["delay40"]),
            "delay56_ms": float(src["delay56"]),
            "dhop_40_minus_56": float(src["dhop_40_minus_56"]),
            "ddelay_ms_40_minus_56": float(src["ddelay_ms_40_minus_56"]),
        }
        if lambda_hop is not None:
            row["lambda_hop"] = float(lambda_hop)
        if switch_penalty is not None:
            row["switch_penalty"] = float(switch_penalty)
        if delay_eps_ms is not None:
            row["delay_eps_ms"] = float(delay_eps_ms)
        for y in range(int(n)):
            row[f"y{y:02d}"] = int(is_40)
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame, actions: np.ndarray, steps: np.ndarray) -> dict[str, Any]:
    idx = actions.astype(int)
    hop = np.where(idx == 1, df["hop40"].to_numpy(), df["hop56"].to_numpy())
    delay = np.where(idx == 1, df["delay40"].to_numpy(), df["delay56"].to_numpy())
    hop_env = np.minimum(df["hop40"].to_numpy(), df["hop56"].to_numpy())
    delay_env = np.minimum(df["delay40"].to_numpy(), df["delay56"].to_numpy())
    return {
        "mean_hops": float(np.mean(hop)),
        "mean_delay_ms": float(np.mean(delay)),
        "mean_gap_hop_two_action_env": float(np.mean(hop - hop_env)),
        "mean_gap_delay_ms_two_action_env": float(np.mean(delay - delay_env)),
        "selected_000040_rows": int(np.count_nonzero(idx == 1)),
        "selected_000056_rows": int(np.count_nonzero(idx == 0)),
        "selected_000040_ratio": float(np.mean(idx == 1)),
        "switches": int(np.count_nonzero(idx[1:] != idx[:-1])),
        "segments": int(np.count_nonzero(idx[1:] != idx[:-1]) + 1),
        "longest_000040_segment_rows": int(
            max((seg["rows"] for seg in contiguous_segments(idx, steps) if int(seg["action_index"]) == 1), default=0)
        ),
    }


def main() -> int:
    args = parse_args()
    df = read_static_pair(Path(args.static_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = df["step"].to_numpy(dtype=np.int64)
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for action_idx, policy in ((0, "pure_000056_all_day"), (1, "pure_000040_all_day")):
        actions = np.full(df.shape[0], int(action_idx), dtype=np.int8)
        path = out_dir / f"{policy}.csv"
        write_rows(path, schedule_rows(df=df, actions=actions, n=int(args.n), policy=policy))
        for seg in contiguous_segments(actions, steps):
            segment_rows.append({"policy": policy, **seg})
        summary_rows.append(
            {
                "policy": policy,
                "selection_rule": "constant pure motif baseline",
                "schedule_csv": str(path),
                "delay_eps_ms": "",
                "lambda_hop": "",
                "switch_penalty": "",
                **summarize(df, actions, steps),
            }
        )

    for eps in args.delay_eps_ms:
        actions = ((df["dhop_40_minus_56"].to_numpy() < 0.0) & (df["ddelay_ms_40_minus_56"].to_numpy() <= float(eps))).astype(
            np.int8
        )
        policy = f"threshold_eps{float(eps):g}ms"
        path = out_dir / f"{policy}.csv"
        write_rows(path, schedule_rows(df=df, actions=actions, n=int(args.n), policy=policy, delay_eps_ms=float(eps)))
        for seg in contiguous_segments(actions, steps):
            segment_rows.append({"policy": policy, **seg})
        summary_rows.append(
            {
                "policy": policy,
                "selection_rule": "hop40<hop56 and delay40-delay56<=eps",
                "schedule_csv": str(path),
                "delay_eps_ms": float(eps),
                "lambda_hop": "",
                "switch_penalty": "",
                **summarize(df, actions, steps),
            }
        )

    for penalty in args.dp_switch_penalties:
        actions = run_binary_dp(df=df, lambda_hop=float(args.lambda_hop), switch_penalty=float(penalty))
        policy = f"binary_dp_lambda{float(args.lambda_hop):.2f}_sw{float(penalty):g}"
        path = out_dir / f"{policy}.csv"
        write_rows(
            path,
            schedule_rows(
                df=df,
                actions=actions,
                n=int(args.n),
                policy=policy,
                lambda_hop=float(args.lambda_hop),
                switch_penalty=float(penalty),
            ),
        )
        for seg in contiguous_segments(actions, steps):
            segment_rows.append({"policy": policy, **seg})
        summary_rows.append(
            {
                "policy": policy,
                "selection_rule": "binary DP over pure 000056/000040",
                "schedule_csv": str(path),
                "delay_eps_ms": "",
                "lambda_hop": float(args.lambda_hop),
                "switch_penalty": float(penalty),
                **summarize(df, actions, steps),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_hops", "mean_delay_ms"]).reset_index(drop=True)
    summary_path = out_dir / "pure_advantage_policy_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(out_dir / "pure_advantage_policy_segments.csv", index=False, encoding="utf-8-sig")
    meta = {
        "static_dir": str(Path(args.static_dir)),
        "out_dir": str(out_dir),
        "n": int(args.n),
        "rows": int(df.shape[0]),
        "step_start": int(df["step"].min()),
        "step_end": int(df["step"].max()),
        "summary_csv": str(summary_path),
    }
    (out_dir / "pure_advantage_policy_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
