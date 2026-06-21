from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
for path in (GENERIC_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_row_mask_knn_reward_policy import split_indices  # noqa: E402
from learn_hybrid_action_aware_policy import DEFAULT_GROUP_CACHE, build_state_features  # noqa: E402
from select_row_mask_dp_switch_penalty import transition_cost_matrix  # noqa: E402
from train_row_mask_reward_model import DEFAULT_REWARD_TABLE_DIR, read_action_csv, read_wide_csv  # noqa: E402


DEFAULT_TEACHER = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_dp_switch_penalty"
    r"\dp_lambda0.50_sw0.5_dwell1.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_lst_teacher_knn_imitation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Imitate an LST-verified row-mask teacher with a no-torch KNN contextual policy, "
            "then smooth predicted actions with a DP transition penalty."
        )
    )
    parser.add_argument("--teacher-schedule", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--reward-table-dir", type=Path, default=DEFAULT_REWARD_TABLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--feature-mode", choices=("time", "group"), default="group")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mode", choices=("loo", "train-val"), default="loo")
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--switch-penalties", nargs="+", type=float, default=(0.0, 0.02, 0.05, 0.1, 0.2, 0.5))
    return parser.parse_args()


def read_teacher(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    df = pd.read_csv(path)
    steps = df["step"].to_numpy(dtype=np.int64)
    action_idx = df["action_index"].to_numpy(dtype=np.int64)
    bits = df[[f"y{i:02d}" for i in range(int(n))]].to_numpy(dtype=np.int8)
    return steps, action_idx, bits, df.to_dict("records")


def pairwise_sqdist(x: np.ndarray) -> np.ndarray:
    norms = np.sum(x * x, axis=1, keepdims=True)
    out = norms + norms.T - 2.0 * (x @ x.T)
    out[out < 0.0] = 0.0
    return out


def knn_action_costs(
    *,
    dist: np.ndarray,
    teacher_idx: np.ndarray,
    action_count: int,
    k: int,
    temperature: float,
    mode: str,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = int(dist.shape[0])
    costs = np.zeros((rows, int(action_count)), dtype=np.float64)
    neighbor_rows = np.full((rows, int(k)), -1, dtype=np.int64)
    train_set = set(int(x) for x in train_idx.tolist())
    for row in range(rows):
        if str(mode) == "train-val" and row in train_set:
            probs = np.zeros(int(action_count), dtype=np.float64)
            probs[int(teacher_idx[row])] = 1.0
            costs[row] = -np.log(probs + 1e-9)
            neighbor_rows[row, 0] = int(row)
            continue
        if str(mode) == "train-val":
            candidates = train_idx
        else:
            candidates = np.asarray([idx for idx in range(rows) if idx != row], dtype=np.int64)
        order = candidates[np.argsort(dist[row, candidates])[: max(1, int(k))]]
        d = dist[row, order].astype(np.float64)
        positive = d[d > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        weights = np.exp(-d / (max(1e-9, float(temperature)) * max(1e-12, scale)))
        weights = weights / max(1e-12, float(np.sum(weights)))
        probs = np.zeros(int(action_count), dtype=np.float64)
        for nbr, weight in zip(order.tolist(), weights.tolist()):
            probs[int(teacher_idx[int(nbr)])] += float(weight)
        costs[row] = -np.log(probs + 1e-9)
        neighbor_rows[row, : len(order)] = order
    return costs, neighbor_rows


def run_dp(stage_cost: np.ndarray, transition: np.ndarray, penalty: float) -> np.ndarray:
    steps, actions = stage_cost.shape
    dp = np.empty((steps, actions), dtype=np.float64)
    parent = np.empty((steps, actions), dtype=np.int16)
    dp[0] = stage_cost[0]
    parent[0] = -1
    trans = float(penalty) * transition
    for t in range(1, steps):
        prev = dp[t - 1][:, None] + trans
        parent[t] = np.argmin(prev, axis=0).astype(np.int16)
        dp[t] = stage_cost[t] + prev[parent[t], np.arange(actions)]
    out = np.empty(steps, dtype=np.int16)
    out[-1] = int(np.argmin(dp[-1]))
    for t in range(steps - 1, 0, -1):
        out[t - 1] = int(parent[t, out[t]])
    return out


def mask_token(bits: np.ndarray) -> str:
    rows = [idx for idx, value in enumerate(bits.tolist()) if int(value) != 0]
    return "{" + ",".join(f"{idx:02d}" for idx in rows) + "}"


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


def schedule_rows(
    *,
    steps: np.ndarray,
    selected: np.ndarray,
    policy: str,
    action_names: Sequence[str],
    action_bits: np.ndarray,
    hops: np.ndarray,
    delay_ms: np.ndarray,
    teacher_idx: np.ndarray,
    teacher_bits: np.ndarray,
    neighbor_rows: np.ndarray,
    n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(selected.astype(int).tolist()):
        bits = action_bits[int(action_idx)]
        teacher = int(teacher_idx[t])
        row: dict[str, Any] = {
            "step": int(steps[t]),
            "policy": policy,
            "action": str(action_names[int(action_idx)]),
            "action_index": int(action_idx),
            "mask": mask_token(bits),
            "row_count": int(np.sum(bits)),
            "mean_hops": float(hops[t, int(action_idx)]),
            "mean_delay_ms": float(delay_ms[t, int(action_idx)]),
            "teacher_action": str(action_names[teacher]),
            "teacher_action_index": teacher,
            "teacher_mask": mask_token(teacher_bits[t]),
            "teacher_match_action": int(action_idx == teacher),
            "teacher_match_bits": int(np.all(bits == teacher_bits[t])),
            "neighbors": " ".join(str(int(x)) for x in neighbor_rows[t].tolist() if int(x) >= 0),
        }
        for y in range(int(n)):
            row[f"y{y:02d}"] = int(bits[y])
        rows.append(row)
    return rows


def summarize(
    *,
    selected: np.ndarray,
    teacher_idx: np.ndarray,
    action_bits: np.ndarray,
    teacher_bits: np.ndarray,
    hops: np.ndarray,
    delay_ms: np.ndarray,
    transition: np.ndarray,
) -> dict[str, Any]:
    idx = selected.astype(int)
    rows = np.arange(idx.size)
    changes = [float(transition[int(idx[t - 1]), int(idx[t])]) for t in range(1, idx.size)]
    return {
        "mean_hops": float(np.mean(hops[rows, idx])),
        "mean_delay_ms": float(np.mean(delay_ms[rows, idx])),
        "action_match_teacher": float(np.mean(idx == teacher_idx)),
        "bit_match_teacher": float(np.mean(action_bits[idx] == teacher_bits)),
        "exact_bits_match_teacher": float(np.mean(np.all(action_bits[idx] == teacher_bits, axis=1))),
        "switches": int(np.count_nonzero(idx[1:] != idx[:-1])),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
        "mean_row_hamming_transition": float(np.mean(changes)) if changes else 0.0,
        "max_row_hamming_transition": float(np.max(changes)) if changes else 0.0,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, teacher_idx, teacher_bits, _teacher_rows = read_teacher(Path(args.teacher_schedule), int(args.n))
    reward_dir = Path(args.reward_table_dir)
    reward_steps, action_names, hops = read_wide_csv(reward_dir / "row_mask_action_library_mean_hops_wide.csv")
    delay_steps, delay_names, delay_ms = read_wide_csv(reward_dir / "row_mask_action_library_mean_delay_ms_wide.csv")
    action_meta_names, action_bits = read_action_csv(reward_dir / "row_mask_action_library_actions.csv", int(args.n))
    if not np.array_equal(steps, reward_steps) or not np.array_equal(steps, delay_steps):
        raise ValueError("teacher/reward steps differ")
    if list(action_names) != list(delay_names) or list(action_names) != list(action_meta_names):
        raise ValueError("action names differ")
    features, feature_meta = build_state_features(
        steps=steps,
        feature_mode=str(args.feature_mode),
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )
    train_idx, val_idx = split_indices(len(steps), float(args.val_fraction), int(args.seed))
    mean = np.mean(features[train_idx], axis=0)
    std = np.std(features[train_idx], axis=0)
    std[std < 1e-8] = 1.0
    x = ((features - mean) / std).astype(np.float32)
    dist = pairwise_sqdist(x)
    base_cost, neighbor_rows = knn_action_costs(
        dist=dist,
        teacher_idx=teacher_idx,
        action_count=len(action_names),
        k=int(args.k),
        temperature=float(args.temperature),
        mode=str(args.mode),
        train_idx=train_idx,
    )
    transition = transition_cost_matrix(action_bits)
    summary_rows: list[dict[str, Any]] = []
    for penalty in args.switch_penalties:
        selected = run_dp(base_cost, transition, float(penalty))
        policy = f"lst_teacher_knn_{Path(args.teacher_schedule).stem}_k{int(args.k)}_sw{float(penalty):g}_{args.feature_mode}"
        csv_path = out_dir / f"{policy}.csv"
        rows = schedule_rows(
            steps=steps,
            selected=selected,
            policy=policy,
            action_names=action_names,
            action_bits=action_bits,
            hops=hops,
            delay_ms=delay_ms,
            teacher_idx=teacher_idx,
            teacher_bits=teacher_bits,
            neighbor_rows=neighbor_rows,
            n=int(args.n),
        )
        write_rows(csv_path, rows)
        summary_rows.append(
            {
                "policy": policy,
                "schedule_csv": str(csv_path),
                "teacher_schedule": str(Path(args.teacher_schedule)),
                "k": int(args.k),
                "temperature": float(args.temperature),
                "mode": str(args.mode),
                "feature_mode": str(args.feature_mode),
                "switch_penalty": float(penalty),
                **summarize(
                    selected=selected,
                    teacher_idx=teacher_idx,
                    action_bits=action_bits,
                    teacher_bits=teacher_bits,
                    hops=hops,
                    delay_ms=delay_ms,
                    transition=transition,
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "lst_teacher_knn_imitation_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta = {
        "teacher_schedule": str(Path(args.teacher_schedule)),
        "reward_table_dir": str(reward_dir),
        "out_dir": str(out_dir),
        "summary_csv": str(summary_path),
        "feature_meta": feature_meta,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
    }
    (out_dir / "lst_teacher_knn_imitation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
