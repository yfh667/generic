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

from learn_hybrid_action_aware_policy import DEFAULT_GROUP_CACHE, build_state_features  # noqa: E402
from train_row_mask_action_scorer import binary_metrics, mask_token  # noqa: E402
from train_row_mask_reward_model import (  # noqa: E402
    DEFAULT_REFERENCE_DIR,
    DEFAULT_REWARD_TABLE_DIR,
    build_reward_score,
    read_action_csv,
    read_wide_csv,
)


DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_knn_reward_policy_lambda050"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KNN contextual-bandit baseline over row-mask reward table.")
    parser.add_argument("--reward-table-dir", type=Path, default=DEFAULT_REWARD_TABLE_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    parser.add_argument("--feature-mode", choices=["time", "group"], default="group")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mode", choices=["loo", "train-val"], default="loo")
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def split_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import random

    indices = list(range(int(num_rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_size = max(1, int(round(float(val_fraction) * int(num_rows))))
    val = np.asarray(sorted(indices[:val_size]), dtype=np.int64)
    train = np.asarray(sorted(indices[val_size:]), dtype=np.int64)
    return train, val


def pairwise_sqdist(x: np.ndarray) -> np.ndarray:
    norms = np.sum(x * x, axis=1, keepdims=True)
    d = norms + norms.T - 2.0 * (x @ x.T)
    d[d < 0] = 0.0
    return d


def predict_knn_scores(
    *,
    dist: np.ndarray,
    reward_score: np.ndarray,
    k: int,
    temperature: float,
    mode: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_rows, num_actions = reward_score.shape
    pred_idx = np.zeros(num_rows, dtype=np.int64)
    pred_score = np.zeros((num_rows, num_actions), dtype=np.float32)
    neighbor_rows = np.full((num_rows, int(k)), -1, dtype=np.int64)
    train_set = set(int(x) for x in train_idx.tolist())
    for row in range(num_rows):
        if mode == "train-val":
            if row in train_set:
                pred_score[row] = reward_score[row]
                pred_idx[row] = int(np.nanargmin(pred_score[row]))
                neighbor_rows[row, 0] = row
                continue
            candidates = train_idx
        else:
            candidates = np.asarray([idx for idx in range(num_rows) if idx != row], dtype=np.int64)
        order = candidates[np.argsort(dist[row, candidates])[: max(1, int(k))]]
        d = dist[row, order].astype(np.float64)
        scale = max(1e-12, float(np.median(d[d > 0])) if np.any(d > 0) else 1.0)
        weights = np.exp(-d / (max(1e-9, float(temperature)) * scale))
        weights = weights / max(1e-12, float(np.sum(weights)))
        pred_score[row] = np.sum(reward_score[order] * weights[:, None], axis=0)
        pred_idx[row] = int(np.nanargmin(pred_score[row]))
        neighbor_rows[row, : len(order)] = order
    return pred_idx, pred_score, neighbor_rows


def policy_lookup_metrics(
    *,
    hops: np.ndarray,
    delay_ms: np.ndarray,
    target_idx: np.ndarray,
    pred_idx: np.ndarray,
    action_bits: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, Any]:
    all_idx = np.arange(hops.shape[0], dtype=np.int64)
    val_set = set(int(x) for x in val_idx.tolist())
    train_idx = np.asarray([idx for idx in all_idx.tolist() if idx not in val_set], dtype=np.int64)

    def one(rows: np.ndarray) -> dict[str, float]:
        pred = pred_idx[rows]
        target = target_idx[rows]
        return {
            "rows": int(len(rows)),
            "action_exact": float(np.mean(pred == target)),
            "mean_pred_hops": float(np.nanmean(hops[rows, pred])),
            "mean_pred_delay_ms": float(np.nanmean(delay_ms[rows, pred])),
            "mean_target_hops": float(np.nanmean(hops[rows, target])),
            "mean_target_delay_ms": float(np.nanmean(delay_ms[rows, target])),
            **binary_metrics(action_bits[target], action_bits[pred]),
        }

    return {"train": one(train_idx), "val": one(val_idx), "all": one(all_idx)}


def write_prediction_csv(
    path: Path,
    *,
    steps: np.ndarray,
    action_names: list[str],
    action_bits: np.ndarray,
    target_idx: np.ndarray,
    pred_idx: np.ndarray,
    val_idx: np.ndarray,
    hops: np.ndarray,
    delay_ms: np.ndarray,
    neighbor_rows: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    val_set = set(int(x) for x in val_idx.tolist())
    rows: list[dict[str, Any]] = []
    for row_idx, step in enumerate(steps.tolist()):
        true_bits = action_bits[int(target_idx[row_idx])]
        pred_bits = action_bits[int(pred_idx[row_idx])]
        item: dict[str, Any] = {
            "step": int(step),
            "split": "val" if row_idx in val_set else "train",
            "true_action": action_names[int(target_idx[row_idx])],
            "pred_action": action_names[int(pred_idx[row_idx])],
            "true_action_index": int(target_idx[row_idx]),
            "pred_action_index": int(pred_idx[row_idx]),
            "true_mask": mask_token(true_bits),
            "pred_mask": mask_token(pred_bits),
            "exact": bool(np.all(true_bits == pred_bits)),
            "action_exact": bool(int(target_idx[row_idx]) == int(pred_idx[row_idx])),
            "true_count": int(np.sum(true_bits)),
            "pred_count": int(np.sum(pred_bits)),
            "target_hops": float(hops[row_idx, int(target_idx[row_idx])]),
            "target_delay_ms": float(delay_ms[row_idx, int(target_idx[row_idx])]),
            "pred_hops_lookup": float(hops[row_idx, int(pred_idx[row_idx])]),
            "pred_delay_ms_lookup": float(delay_ms[row_idx, int(pred_idx[row_idx])]),
            "neighbors": " ".join(str(int(x)) for x in neighbor_rows[row_idx].tolist() if int(x) >= 0),
        }
        for idx in range(action_bits.shape[1]):
            item[f"pred_y{idx:02d}"] = int(pred_bits[idx])
            item[f"true_y{idx:02d}"] = int(true_bits[idx])
        rows.append(item)
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
        raise ValueError("action names differ")
    if int(args.max_rows) > 0:
        limit = int(args.max_rows)
        steps = steps[:limit]
        hops = hops[:limit]
        delay_ms = delay_ms[:limit]
    reward_score, reward_meta = build_reward_score(
        steps=steps,
        action_names=action_names,
        hops=hops,
        delay_ms=delay_ms,
        reference_dir=Path(args.reference_dir),
        lambda_hop=float(args.lambda_hop),
    )
    target_idx = np.nanargmin(reward_score, axis=1).astype(np.int64)
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
    pred_idx, pred_score, neighbor_rows = predict_knn_scores(
        dist=dist,
        reward_score=reward_score,
        k=int(args.k),
        temperature=float(args.temperature),
        mode=str(args.mode),
        train_idx=train_idx,
        val_idx=val_idx,
    )
    metrics = policy_lookup_metrics(
        hops=hops,
        delay_ms=delay_ms,
        target_idx=target_idx,
        pred_idx=pred_idx,
        action_bits=action_bits,
        val_idx=val_idx,
    )
    prediction_csv = out_dir / "row_mask_knn_reward_policy_predictions.csv"
    write_prediction_csv(
        prediction_csv,
        steps=steps,
        action_names=action_names,
        action_bits=action_bits,
        target_idx=target_idx,
        pred_idx=pred_idx,
        val_idx=val_idx,
        hops=hops,
        delay_ms=delay_ms,
        neighbor_rows=neighbor_rows,
    )
    meta = {
        "reward_table_dir": str(reward_dir),
        "reference_dir": str(Path(args.reference_dir)),
        "out_dir": str(out_dir),
        "prediction_csv": str(prediction_csv),
        "mode": str(args.mode),
        "k": int(args.k),
        "temperature": float(args.temperature),
        "num_rows": int(len(steps)),
        "num_actions": int(len(action_names)),
        "lambda_hop": float(args.lambda_hop),
        "feature_mode": str(args.feature_mode),
        "seed": int(args.seed),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "feature_meta": feature_meta,
        "reward_meta": reward_meta,
        "lookup_metrics": metrics,
    }
    (out_dir / "row_mask_knn_reward_policy_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
