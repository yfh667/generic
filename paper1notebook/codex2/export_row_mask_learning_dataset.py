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
for path in (GENERIC_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from learn_hybrid_candidate_policy import DEFAULT_GROUP_CACHE, build_state_features  # noqa: E402
from train_row_mask_reward_model import (  # noqa: E402
    DEFAULT_REWARD_TABLE_DIR,
    read_action_csv,
    read_wide_csv,
)


DEFAULT_TEACHER = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_beam_critical_drop_aware"
    r"\beam_critdrop_lambda0.80_crit0.05_cnt0.03_build0.03_k0_b64.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_learning_dataset"
    r"\teacher_critical_count_lam080_crit005_cnt003_build003"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a compact offline-learning dataset for row-mask 000040/000056 fusion. "
            "The dataset contains state features, teacher actions, action masks, and the "
            "full hop/delay reward tables."
        )
    )
    parser.add_argument("--teacher-schedule", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--reward-table-dir", type=Path, default=DEFAULT_REWARD_TABLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--feature-mode", choices=("time", "group"), default="group")
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--lambda-hop", type=float, default=0.8)
    return parser.parse_args()


def read_teacher(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty teacher schedule: {path}")
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    action_idx = np.asarray([int(float(row["action_index"])) for row in rows], dtype=np.int64)
    bits = np.asarray(
        [[int(float(row[f"y{idx:02d}"])) for idx in range(int(n))] for row in rows],
        dtype=np.int8,
    )
    return steps, action_idx, bits, rows


def ensure_same_steps(expected: np.ndarray, actual: np.ndarray, label: str) -> None:
    if not np.array_equal(expected, actual):
        raise ValueError(f"{label} steps differ from teacher schedule")


def ensure_same_names(expected: Sequence[str], actual: Sequence[str], label: str) -> None:
    if list(expected) != list(actual):
        raise ValueError(f"{label} action names differ from action table")


def transition_hamming_matrix(action_bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(action_bits, dtype=np.int8)
    diff = np.abs(bits[:, None, :] - bits[None, :, :]).sum(axis=2).astype(np.float32)
    return diff / float(bits.shape[1])


def write_teacher_summary(
    path: Path,
    *,
    steps: np.ndarray,
    action_names: Sequence[str],
    teacher_idx: np.ndarray,
    teacher_bits: np.ndarray,
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "teacher_action",
        "teacher_action_index",
        "row_count",
        "mean_hops",
        "mean_delay_ms",
        "gap_to_hop_env",
        "gap_to_delay_env_ms",
        "mask",
    ]
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(teacher_idx.astype(int).tolist()):
        active_rows = [f"{idx:02d}" for idx, value in enumerate(teacher_bits[t].tolist()) if int(value) != 0]
        rows.append(
            {
                "step": int(steps[t]),
                "teacher_action": str(action_names[action_idx]),
                "teacher_action_index": int(action_idx),
                "row_count": int(np.sum(teacher_bits[t])),
                "mean_hops": float(hops[t, action_idx]),
                "mean_delay_ms": float(delay[t, action_idx]),
                "gap_to_hop_env": float(hops[t, action_idx] - hop_env[t]),
                "gap_to_delay_env_ms": float(delay[t, action_idx] - delay_env[t]),
                "mask": "{" + ",".join(active_rows) + "}",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_steps, teacher_idx, teacher_bits, _teacher_rows = read_teacher(Path(args.teacher_schedule), int(args.n))
    action_names, action_bits = read_action_csv(Path(args.reward_table_dir) / "row_mask_action_library_actions.csv", int(args.n))
    steps_hop, names_hop, hops = read_wide_csv(Path(args.reward_table_dir) / "row_mask_action_library_mean_hops_wide.csv")
    steps_delay, names_delay, delay = read_wide_csv(Path(args.reward_table_dir) / "row_mask_action_library_mean_delay_ms_wide.csv")
    ensure_same_steps(teacher_steps, steps_hop, "hop table")
    ensure_same_steps(teacher_steps, steps_delay, "delay table")
    ensure_same_names(action_names, names_hop, "hop table")
    ensure_same_names(action_names, names_delay, "delay table")

    if np.any(teacher_idx < 0) or np.any(teacher_idx >= len(action_names)):
        raise ValueError("teacher schedule contains action indices outside action table")
    if not np.array_equal(action_bits[teacher_idx].astype(np.int8), teacher_bits):
        raise ValueError("teacher schedule y-bit masks do not match action table")

    state_features, state_meta = build_state_features(
        steps=teacher_steps,
        feature_mode=str(args.feature_mode),
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )
    hop_env = np.nanmin(hops, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hops) - np.nanmin(hops)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    normalized_hop_gap = (hops - hop_env[:, None]) / hop_scale
    normalized_delay_gap = (delay - delay_env[:, None]) / delay_scale
    stage_cost = (
        float(args.lambda_hop) * normalized_hop_gap
        + (1.0 - float(args.lambda_hop)) * normalized_delay_gap
    )
    transition_hamming = transition_hamming_matrix(action_bits)

    row_idx = np.arange(teacher_idx.size)
    teacher_hops = hops[row_idx, teacher_idx]
    teacher_delay = delay[row_idx, teacher_idx]
    teacher_summary = {
        "mean_hops": float(np.mean(teacher_hops)),
        "mean_delay_ms": float(np.mean(teacher_delay)),
        "mean_gap_to_hop_env": float(np.mean(teacher_hops - hop_env)),
        "mean_gap_to_delay_env_ms": float(np.mean(teacher_delay - delay_env)),
        "switches": int(np.count_nonzero(teacher_idx[1:] != teacher_idx[:-1])),
        "num_used_actions": int(len(set(int(x) for x in teacher_idx.tolist()))),
        "mean_stage_cost": float(np.mean(stage_cost[row_idx, teacher_idx])),
    }

    npz_path = out_dir / "row_mask_learning_dataset.npz"
    np.savez_compressed(
        npz_path,
        steps=teacher_steps.astype(np.int64),
        state_features=np.asarray(state_features, dtype=np.float32),
        teacher_action_index=teacher_idx.astype(np.int16),
        teacher_bits=teacher_bits.astype(np.int8),
        action_bits=np.asarray(action_bits, dtype=np.int8),
        action_names=np.asarray(action_names, dtype=object),
        hop_table=np.asarray(hops, dtype=np.float32),
        delay_table_ms=np.asarray(delay, dtype=np.float32),
        hop_env=np.asarray(hop_env, dtype=np.float32),
        delay_env_ms=np.asarray(delay_env, dtype=np.float32),
        normalized_hop_gap=np.asarray(normalized_hop_gap, dtype=np.float32),
        normalized_delay_gap=np.asarray(normalized_delay_gap, dtype=np.float32),
        stage_cost=np.asarray(stage_cost, dtype=np.float32),
        transition_hamming=np.asarray(transition_hamming, dtype=np.float32),
    )

    summary_csv = out_dir / "teacher_schedule_summary.csv"
    write_teacher_summary(
        summary_csv,
        steps=teacher_steps,
        action_names=action_names,
        teacher_idx=teacher_idx,
        teacher_bits=teacher_bits,
        hops=hops,
        delay=delay,
        hop_env=hop_env,
        delay_env=delay_env,
    )
    meta = {
        "teacher_schedule": str(Path(args.teacher_schedule)),
        "reward_table_dir": str(Path(args.reward_table_dir)),
        "group_cache": str(Path(args.group_cache)),
        "out_dir": str(out_dir),
        "npz": str(npz_path),
        "teacher_summary_csv": str(summary_csv),
        "source_group_id": int(args.source_group_id),
        "target_group_id": int(args.target_group_id),
        "p": int(args.p),
        "n": int(args.n),
        "lambda_hop": float(args.lambda_hop),
        "num_steps": int(teacher_steps.size),
        "num_actions": int(len(action_names)),
        "state_meta": state_meta,
        "teacher_summary": teacher_summary,
        "arrays": {
            "state_features": [int(x) for x in state_features.shape],
            "teacher_action_index": [int(x) for x in teacher_idx.shape],
            "action_bits": [int(x) for x in action_bits.shape],
            "hop_table": [int(x) for x in hops.shape],
            "delay_table_ms": [int(x) for x in delay.shape],
            "transition_hamming": [int(x) for x in transition_hamming.shape],
        },
    }
    meta_path = out_dir / "row_mask_learning_dataset_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(npz_path), "meta": str(meta_path), "teacher_summary": teacher_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
