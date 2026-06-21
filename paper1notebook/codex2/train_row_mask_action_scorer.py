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


DEFAULT_LABELS = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_training_all1437_pseudo_hard96"
    r"\row_mask_training_labels_all1437_pseudo_hard96.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_action_scorer_hard96"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an action scorer over valid 000040/000056 row-mask actions.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=192)
    return parser.parse_args()


def read_labels(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty labels CSV: {path}")
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    labels = np.asarray([[int(float(row[f"y{idx:02d}"])) for idx in range(int(n))] for row in rows], dtype=np.int64)
    return steps, labels, rows


def split_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(int(num_rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_size = max(1, int(round(float(val_fraction) * int(num_rows))))
    val = np.asarray(sorted(indices[:val_size]), dtype=np.int64)
    train = np.asarray(sorted(indices[val_size:]), dtype=np.int64)
    return train, val


def make_action_library(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys: dict[tuple[int, ...], int] = {}
    actions: list[np.ndarray] = []
    y = np.zeros(labels.shape[0], dtype=np.int64)
    for idx, row in enumerate(labels):
        key = tuple(int(x) for x in row.tolist())
        if key not in keys:
            keys[key] = len(actions)
            actions.append(row.astype(np.float32))
        y[idx] = keys[key]
    return np.asarray(actions, dtype=np.float32), y


def action_features(actions: np.ndarray) -> np.ndarray:
    n = actions.shape[1]
    theta = np.arange(n, dtype=np.float64) * (2.0 * np.pi / float(n))
    feats: list[np.ndarray] = []
    for row in actions.astype(np.float64):
        count = float(np.sum(row))
        if count > 0:
            sin_mean = float(np.sum(row * np.sin(theta)) / count)
            cos_mean = float(np.sum(row * np.cos(theta)) / count)
        else:
            sin_mean = 0.0
            cos_mean = 0.0
        transitions = float(np.sum(np.abs(row - np.roll(row, 1)))) / float(n)
        feats.append(
            np.concatenate(
                [
                    row.astype(np.float32),
                    np.asarray([count / float(n), sin_mean, cos_mean, transitions], dtype=np.float32),
                ]
            )
        )
    return np.asarray(feats, dtype=np.float32)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    bit_acc = float(np.mean(y_true == y_pred))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(1e-9, tp + fp)
    recall = tp / max(1e-9, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "exact_match": exact,
        "bit_accuracy": bit_acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_true_rows": float(np.mean(np.sum(y_true, axis=1))),
        "mean_pred_rows": float(np.mean(np.sum(y_pred, axis=1))),
    }


def mask_token(bits: np.ndarray) -> str:
    rows = [idx for idx, value in enumerate(bits.tolist()) if int(value) == 1]
    return "{" + ",".join(f"{idx:02d}" for idx in rows) + "}"


def write_prediction_csv(
    path: Path,
    *,
    steps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    action_index: np.ndarray,
    split: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_set = set(int(x) for x in split.tolist())
    rows: list[dict[str, Any]] = []
    for row_idx, step in enumerate(steps.tolist()):
        item: dict[str, Any] = {
            "step": int(step),
            "split": "val" if row_idx in split_set else "train",
            "true_mask": mask_token(y_true[row_idx]),
            "pred_mask": mask_token(y_pred[row_idx]),
            "exact": bool(np.all(y_true[row_idx] == y_pred[row_idx])),
            "true_count": int(np.sum(y_true[row_idx])),
            "pred_count": int(np.sum(y_pred[row_idx])),
            "pred_action_index": int(action_index[row_idx]),
        }
        for idx in range(y_true.shape[1]):
            item[f"pred_y{idx:02d}"] = int(y_pred[row_idx, idx])
            item[f"true_y{idx:02d}"] = int(y_true[row_idx, idx])
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps, labels, _label_rows = read_labels(Path(args.labels), int(args.n))
    actions, target_index = make_action_library(labels)
    cand_features = action_features(actions)
    state_features, feature_meta = build_state_features(
        steps=steps,
        feature_mode="group",
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )
    train_idx, val_idx = split_indices(len(steps), float(args.val_fraction), int(args.seed))
    state_mean = np.mean(state_features[train_idx], axis=0)
    state_std = np.std(state_features[train_idx], axis=0)
    state_std[state_std < 1e-8] = 1.0
    cand_mean = np.mean(cand_features, axis=0)
    cand_std = np.std(cand_features, axis=0)
    cand_std[cand_std < 1e-8] = 1.0

    x_np = ((state_features - state_mean) / state_std).astype(np.float32)
    a_np = ((cand_features - cand_mean) / cand_std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    a = torch.as_tensor(a_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(target_index, dtype=torch.long, device=device)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_tensor = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    hidden = int(args.hidden_dim)
    state_net = torch.nn.Sequential(
        torch.nn.Linear(x_np.shape[1], hidden),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(hidden),
        torch.nn.Linear(hidden, hidden),
    ).to(device)
    action_net = torch.nn.Sequential(
        torch.nn.Linear(a_np.shape[1], hidden),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(hidden),
        torch.nn.Linear(hidden, hidden),
    ).to(device)
    action_bias = torch.nn.Linear(a_np.shape[1], 1).to(device)
    params = list(state_net.parameters()) + list(action_net.parameters()) + list(action_bias.parameters())
    optimizer = torch.optim.AdamW(params, lr=1.2e-3, weight_decay=1e-4)
    for _epoch in range(int(args.epochs)):
        s = torch.nn.functional.normalize(state_net(x[train_tensor]), dim=1)
        c = torch.nn.functional.normalize(action_net(a), dim=1)
        logits = s @ c.t() * 12.0 + action_bias(a).view(1, -1)
        loss = torch.nn.functional.cross_entropy(logits, y[train_tensor])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        s_all = torch.nn.functional.normalize(state_net(x), dim=1)
        c_all = torch.nn.functional.normalize(action_net(a), dim=1)
        logits_all = s_all @ c_all.t() * 12.0 + action_bias(a).view(1, -1)
        pred_index = torch.argmax(logits_all, dim=1).detach().cpu().numpy().astype(np.int64)
    pred_labels = actions[pred_index].astype(np.int32)
    true_labels = labels.astype(np.int32)
    metrics = {
        "train": binary_metrics(true_labels[train_idx], pred_labels[train_idx]),
        "val": binary_metrics(true_labels[val_idx], pred_labels[val_idx]),
        "all": binary_metrics(true_labels, pred_labels),
    }
    torch.save(
        {
            "state_net": state_net.state_dict(),
            "action_net": action_net.state_dict(),
            "action_bias": action_bias.state_dict(),
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "cand_mean": cand_mean.astype(np.float32),
            "cand_std": cand_std.astype(np.float32),
            "actions": actions.astype(np.float32),
            "feature_meta": feature_meta,
        },
        out_dir / "row_mask_action_scorer.pt",
    )
    np.save(out_dir / "row_mask_action_library.npy", actions.astype(np.int8))
    write_prediction_csv(
        out_dir / "row_mask_action_scorer_predictions.csv",
        steps=steps,
        y_true=true_labels,
        y_pred=pred_labels,
        action_index=pred_index,
        split=val_idx,
    )
    meta = {
        "labels": str(Path(args.labels)),
        "out_dir": str(out_dir),
        "num_rows": int(len(steps)),
        "num_actions": int(actions.shape[0]),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "feature_meta": feature_meta,
        "action_feature_dim": int(cand_features.shape[1]),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "",
        "metrics": metrics,
    }
    (out_dir / "row_mask_action_scorer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
