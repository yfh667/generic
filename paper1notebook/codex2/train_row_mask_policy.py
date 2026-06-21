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
    r"\motif0040_0056_region_internal_plus_grid_row_mask_training_hard48"
    r"\row_mask_training_labels_hard48.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_policy_hard48"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 36-bit row-mask policy from row-level oracle labels.")
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
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def read_labels(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty labels CSV: {path}")
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    labels = np.asarray([[int(float(row[f"y{idx:02d}"])) for idx in range(int(n))] for row in rows], dtype=np.float32)
    return steps, labels, rows


def split_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(int(num_rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_size = max(1, int(round(float(val_fraction) * int(num_rows))))
    val = np.asarray(sorted(indices[:val_size]), dtype=np.int64)
    train = np.asarray(sorted(indices[val_size:]), dtype=np.int64)
    return train, val


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    exact = np.mean(np.all(y_true == y_pred, axis=1))
    bit_acc = np.mean(y_true == y_pred)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(1e-9, tp + fp)
    recall = tp / max(1e-9, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    pred_count = float(np.mean(np.sum(y_pred, axis=1)))
    true_count = float(np.mean(np.sum(y_true, axis=1)))
    return {
        "exact_match": float(exact),
        "bit_accuracy": float(bit_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_true_rows": true_count,
        "mean_pred_rows": pred_count,
    }


def mask_token(bits: np.ndarray) -> str:
    rows = [idx for idx, value in enumerate(bits.tolist()) if int(value) == 1]
    return "{" + ",".join(f"{idx:02d}" for idx in rows) + "}"


def write_prediction_csv(
    path: Path,
    *,
    steps: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
    split: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    split_set = set(int(x) for x in split.tolist())
    for row_idx, step in enumerate(steps.tolist()):
        item: dict[str, Any] = {
            "step": int(step),
            "split": "val" if row_idx in split_set else "train",
            "true_mask": mask_token(y_true[row_idx]),
            "pred_mask": mask_token(pred[row_idx]),
            "exact": bool(np.all(y_true[row_idx] == pred[row_idx])),
            "true_count": int(np.sum(y_true[row_idx])),
            "pred_count": int(np.sum(pred[row_idx])),
        }
        for idx in range(y_true.shape[1]):
            item[f"prob_y{idx:02d}"] = float(probs[row_idx, idx])
            item[f"pred_y{idx:02d}"] = int(pred[row_idx, idx])
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
    steps, labels, label_rows = read_labels(Path(args.labels), int(args.n))
    features, feature_meta = build_state_features(
        steps=steps,
        feature_mode="group",
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
    x_np = ((features - mean) / std).astype(np.float32)
    y_np = labels.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_np.shape[1], 128),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(128),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(128),
        torch.nn.Linear(128, int(args.n)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_tensor = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    for _epoch in range(int(args.epochs)):
        logits = model(x[train_tensor])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[train_tensor])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_all = model(x)
        probs = torch.sigmoid(logits_all).detach().cpu().numpy()
    pred = (probs >= float(args.threshold)).astype(np.int32)
    metrics = {
        "train": binary_metrics(y_np[train_idx], pred[train_idx]),
        "val": binary_metrics(y_np[val_idx], pred[val_idx]),
        "all": binary_metrics(y_np, pred),
    }
    meta = {
        "labels": str(Path(args.labels)),
        "out_dir": str(out_dir),
        "num_rows": int(len(steps)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "feature_meta": feature_meta,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "threshold": float(args.threshold),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "",
        "metrics": metrics,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "feature_meta": feature_meta,
            "n": int(args.n),
        },
        out_dir / "row_mask_mlp.pt",
    )
    write_prediction_csv(
        out_dir / "row_mask_policy_predictions.csv",
        steps=steps,
        y_true=y_np.astype(np.int32),
        probs=probs,
        pred=pred,
        split=val_idx,
    )
    (out_dir / "row_mask_policy_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
