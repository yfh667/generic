from __future__ import annotations

import argparse
import csv
import json
import random
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
from train_row_mask_action_scorer import action_features, binary_metrics, mask_token  # noqa: E402
from train_row_mask_reward_model import (  # noqa: E402
    DEFAULT_REFERENCE_DIR,
    DEFAULT_REWARD_TABLE_DIR,
    build_reward_score,
    read_action_csv,
    read_wide_csv,
)


DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_explicit_pair_model_lambda050"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit state-action feature reward model for row-mask fusion.")
    parser.add_argument("--reward-table-dir", type=Path, default=DEFAULT_REWARD_TABLE_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def split_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(int(num_rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_size = max(1, int(round(float(val_fraction) * int(num_rows))))
    val = np.asarray(sorted(indices[:val_size]), dtype=np.int64)
    train = np.asarray(sorted(indices[val_size:]), dtype=np.int64)
    return train, val


def _roll_corr(hist: np.ndarray, mask: np.ndarray, shifts: range) -> np.ndarray:
    cols = []
    for shift in shifts:
        rolled = np.roll(mask, int(shift), axis=1)
        cols.append(np.sum(hist[:, None, :] * rolled[None, :, :], axis=2))
    return np.stack(cols, axis=2)


def build_pair_features(
    *,
    state_features: np.ndarray,
    action_bits: np.ndarray,
    n: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    state = np.asarray(state_features, dtype=np.float32)
    masks = np.asarray(action_bits, dtype=np.float32)
    action_feat = action_features(masks).astype(np.float32)
    num_steps = state.shape[0]
    num_actions = masks.shape[0]
    time_dim = 17
    src_y = state[:, time_dim : time_dim + int(n)]
    dst_y = state[:, time_dim + int(n) : time_dim + 2 * int(n)]
    both_y = src_y + dst_y
    mask = masks[None, :, :]
    src = src_y[:, None, :]
    dst = dst_y[:, None, :]
    both = both_y[:, None, :]
    mask_count = np.maximum(1.0, np.sum(mask, axis=2, keepdims=True))
    overlap = np.concatenate(
        [
            np.sum(src * mask, axis=2, keepdims=True),
            np.sum(dst * mask, axis=2, keepdims=True),
            np.sum(both * mask, axis=2, keepdims=True),
            np.sum(src * (1.0 - mask), axis=2, keepdims=True),
            np.sum(dst * (1.0 - mask), axis=2, keepdims=True),
            np.sum((src * mask), axis=2, keepdims=True) / mask_count,
            np.sum((dst * mask), axis=2, keepdims=True) / mask_count,
        ],
        axis=2,
    )
    masked_src = src * mask
    masked_dst = dst * mask
    shifts = range(-4, 5)
    corr = np.concatenate([_roll_corr(src_y, masks, shifts), _roll_corr(dst_y, masks, shifts)], axis=2)
    state_rep = np.repeat(state[:, None, :], num_actions, axis=1)
    action_rep = np.repeat(action_feat[None, :, :], num_steps, axis=0)
    mask_rep = np.repeat(masks[None, :, :], num_steps, axis=0)
    pair = np.concatenate(
        [
            state_rep,
            action_rep,
            masked_src,
            masked_dst,
            mask_rep * both,
            overlap,
            corr.astype(np.float32),
        ],
        axis=2,
    ).astype(np.float32)
    meta = {
        "pair_feature_dim": int(pair.shape[2]),
        "parts": [
            f"state_{state.shape[1]}",
            f"action_features_{action_feat.shape[1]}",
            f"mask_times_source_y_{n}",
            f"mask_times_target_y_{n}",
            f"mask_times_source_plus_target_y_{n}",
            "overlap_scalars_7",
            "source_target_shift_corr_18",
        ],
    }
    return pair, meta


def policy_metrics(
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
    import torch

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    reward_dir = Path(args.reward_table_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, hop_names, hops = read_wide_csv(reward_dir / "row_mask_action_library_mean_hops_wide.csv")
    steps_delay, delay_names, delay_ms = read_wide_csv(reward_dir / "row_mask_action_library_mean_delay_ms_wide.csv")
    action_names, action_bits = read_action_csv(reward_dir / "row_mask_action_library_actions.csv", int(args.n))
    if not np.array_equal(steps, steps_delay):
        raise ValueError("hop/delay reward tables have different steps")
    if hop_names != delay_names or hop_names != action_names:
        raise ValueError("reward table actions do not match action metadata")
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
    train_idx, val_idx = split_indices(len(steps), float(args.val_fraction), int(args.seed))
    state_features, state_meta = build_state_features(
        steps=steps,
        feature_mode="group",
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )
    pair_features, pair_meta = build_pair_features(
        state_features=state_features,
        action_bits=action_bits,
        n=int(args.n),
    )
    train_pairs = pair_features[train_idx].reshape(-1, pair_features.shape[2])
    mean = np.mean(train_pairs, axis=0)
    std = np.std(train_pairs, axis=0)
    std[std < 1e-8] = 1.0
    pair_norm = ((pair_features - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    score_mean = float(np.mean(reward_score[train_idx]))
    score_std = float(np.std(reward_score[train_idx]))
    if score_std < 1e-8:
        score_std = 1.0
    utility = -((reward_score - score_mean) / score_std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(pair_norm.reshape(-1, pair_norm.shape[2]), dtype=torch.float32, device=device)
    y = torch.as_tensor(utility.reshape(-1), dtype=torch.float32, device=device)
    num_steps, num_actions = reward_score.shape
    train_flat = np.concatenate(
        [
            np.arange(int(row) * num_actions, int(row + 1) * num_actions, dtype=np.int64)
            for row in train_idx.tolist()
        ]
    )
    rng = np.random.default_rng(int(args.seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(pair_norm.shape[2], int(args.hidden_dim)),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(int(args.hidden_dim)),
        torch.nn.Dropout(0.08),
        torch.nn.Linear(int(args.hidden_dim), int(args.hidden_dim)),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(int(args.hidden_dim)),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(int(args.hidden_dim), 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    batch_size = max(1, int(args.batch_size))
    for _epoch in range(int(args.epochs)):
        shuffled = rng.permutation(train_flat)
        for start in range(0, len(shuffled), batch_size):
            idx = torch.as_tensor(shuffled[start : start + batch_size], dtype=torch.long, device=device)
            pred = model(x[idx]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        pred_utility = model(x).squeeze(-1).detach().cpu().numpy().reshape(num_steps, num_actions)
    pred_idx = np.nanargmax(pred_utility, axis=1).astype(np.int64)
    metrics = policy_metrics(
        hops=hops,
        delay_ms=delay_ms,
        target_idx=target_idx,
        pred_idx=pred_idx,
        action_bits=action_bits,
        val_idx=val_idx,
    )
    prediction_csv = out_dir / "row_mask_explicit_pair_model_predictions.csv"
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
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "score_mean": np.asarray(score_mean, dtype=np.float32),
            "score_std": np.asarray(score_std, dtype=np.float32),
            "action_names": action_names,
            "action_bits": action_bits.astype(np.int8),
            "state_meta": state_meta,
            "pair_meta": pair_meta,
            "reward_meta": reward_meta,
        },
        out_dir / "row_mask_explicit_pair_model.pt",
    )
    meta = {
        "reward_table_dir": str(reward_dir),
        "reference_dir": str(Path(args.reference_dir)),
        "out_dir": str(out_dir),
        "prediction_csv": str(prediction_csv),
        "num_rows": int(num_steps),
        "num_actions": int(num_actions),
        "lambda_hop": float(args.lambda_hop),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "seed": int(args.seed),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "",
        "state_meta": state_meta,
        "pair_meta": pair_meta,
        "reward_meta": reward_meta,
        "lookup_metrics": metrics,
    }
    (out_dir / "row_mask_explicit_pair_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
