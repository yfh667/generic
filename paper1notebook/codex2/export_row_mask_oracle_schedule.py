from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
for path in (GENERIC_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_row_mask_action_scorer import mask_token  # noqa: E402
from train_row_mask_reward_model import (  # noqa: E402
    DEFAULT_REFERENCE_DIR,
    DEFAULT_REWARD_TABLE_DIR,
    build_reward_score,
    read_action_csv,
    read_wide_csv,
)


DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_tabular_bandit_oracle_schedules"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export executable tabular/contextual-bandit row-mask schedules.")
    parser.add_argument("--reward-table-dir", type=Path, default=DEFAULT_REWARD_TABLE_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lambdas", nargs="*", type=float, default=[0.4, 0.5, 0.6])
    parser.add_argument("--n", type=int, default=36)
    return parser.parse_args()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    reward_dir = Path(args.reward_table_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, hop_names, hops = read_wide_csv(reward_dir / "row_mask_action_library_mean_hops_wide.csv")
    steps_delay, delay_names, delay_ms = read_wide_csv(reward_dir / "row_mask_action_library_mean_delay_ms_wide.csv")
    action_names, action_bits = read_action_csv(reward_dir / "row_mask_action_library_actions.csv", int(args.n))
    if not np.array_equal(steps, steps_delay):
        raise ValueError("hop/delay steps differ")
    if hop_names != delay_names or hop_names != action_names:
        raise ValueError("action metadata does not match reward tables")

    schedules: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for lam in [float(x) for x in args.lambdas]:
        score, reward_meta = build_reward_score(
            steps=steps,
            action_names=action_names,
            hops=hops,
            delay_ms=delay_ms,
            reference_dir=Path(args.reference_dir),
            lambda_hop=float(lam),
        )
        selected = np.nanargmin(score, axis=1).astype(np.int64)
        selected_hops = hops[np.arange(len(steps)), selected]
        selected_delay = delay_ms[np.arange(len(steps)), selected]
        # The reference envelopes in reward_meta are means; recompute per-row values for useful gaps.
        # Using score's reference normalization keeps this file independent of candidate names.
        ref_steps, _ref_names, ref_hops = read_wide_csv(Path(args.reference_dir) / "compare_mean_shortest_hops.csv")
        ref_steps2, _ref_names2, ref_delay = read_wide_csv(Path(args.reference_dir) / "compare_mean_shortest_delay_ms.csv")
        if not np.array_equal(ref_steps, ref_steps2):
            raise ValueError("reference steps differ")
        step_to_ref = {int(step): idx for idx, step in enumerate(ref_steps.tolist())}
        ref_rows = np.asarray([step_to_ref[int(step)] for step in steps.tolist()], dtype=np.int64)
        ref_hop_env = np.nanmin(ref_hops[ref_rows], axis=1)
        ref_delay_env = np.nanmin(ref_delay[ref_rows], axis=1)
        rows: list[dict[str, Any]] = []
        for idx, step in enumerate(steps.tolist()):
            action_idx = int(selected[idx])
            bits = action_bits[action_idx]
            row = {
                "step": int(step),
                "policy": f"lambda_hop_{lam:.2f}",
                "lambda_hop": float(lam),
                "action": action_names[action_idx],
                "action_index": action_idx,
                "mask": mask_token(bits),
                "row_count": int(np.sum(bits)),
                "mean_hops": float(selected_hops[idx]),
                "mean_delay_ms": float(selected_delay[idx]),
                "reference_hop_envelope": float(ref_hop_env[idx]),
                "reference_delay_envelope_ms": float(ref_delay_env[idx]),
                "gap_hop_env": float(selected_hops[idx] - ref_hop_env[idx]),
                "gap_delay_env_ms": float(selected_delay[idx] - ref_delay_env[idx]),
            }
            for y in range(int(args.n)):
                row[f"y{y:02d}"] = int(bits[y])
            rows.append(row)
        schedule_csv = out_dir / f"row_mask_tabular_bandit_oracle_lambda{lam:.2f}.csv"
        _write_rows(schedule_csv, rows)
        summary = {
            "policy": f"lambda_hop_{lam:.2f}",
            "lambda_hop": float(lam),
            "schedule_csv": str(schedule_csv),
            "num_steps": int(len(steps)),
            "num_actions_used": int(len(set(selected.tolist()))),
            "mean_hops": float(np.nanmean(selected_hops)),
            "mean_delay_ms": float(np.nanmean(selected_delay)),
            "mean_reference_hop_envelope": float(np.nanmean(ref_hop_env)),
            "mean_reference_delay_envelope_ms": float(np.nanmean(ref_delay_env)),
            "mean_gap_hop_env": float(np.nanmean(selected_hops - ref_hop_env)),
            "mean_gap_delay_env_ms": float(np.nanmean(selected_delay - ref_delay_env)),
            "steps_on_hop_env": int(np.sum(np.isclose(selected_hops, ref_hop_env, atol=1e-9))),
            "steps_on_delay_env": int(np.sum(np.isclose(selected_delay, ref_delay_env, atol=1e-9))),
            "reward_meta": reward_meta,
        }
        summary_path = out_dir / f"row_mask_tabular_bandit_oracle_lambda{lam:.2f}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
        summaries.append(summary)
        schedules.extend(rows)

    summary_csv = out_dir / "row_mask_tabular_bandit_oracle_summary.csv"
    _write_rows(
        summary_csv,
        [
            {
                key: value
                for key, value in summary.items()
                if key not in {"reward_meta"}
            }
            for summary in summaries
        ],
    )
    print(json.dumps({"out_dir": str(out_dir), "summary_csv": str(summary_csv), "policies": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
