from __future__ import annotations

import argparse
import copy
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


DEFAULT_REWARD_TABLE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_action_scorer_hard96"
    r"\action_library_reward_table_full1437_merged"
)
DEFAULT_REFERENCE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_hybrid_search_all110"
    r"\full_t0_86160_s60"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_reward_model_lambda050"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reward-aware contextual-bandit scorer for 000040/000056 row-mask actions."
    )
    parser.add_argument("--reward-table-dir", type=Path, default=DEFAULT_REWARD_TABLE_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--ce-weight", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--select-best-val", action="store_true")
    return parser.parse_args()


def read_wide_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty wide CSV: {path}")
    names = [name for name in rows[0].keys() if name != "step"]
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    matrix = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    return steps, names, matrix


def read_reference_tables(reference_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    steps_hop, names_hop, hop = read_wide_csv(Path(reference_dir) / "compare_mean_shortest_hops.csv")
    steps_delay, names_delay, delay = read_wide_csv(Path(reference_dir) / "compare_mean_shortest_delay_ms.csv")
    if not np.array_equal(steps_hop, steps_delay):
        raise ValueError("reference hop/delay steps differ")
    if names_hop != names_delay:
        raise ValueError("reference hop/delay action names differ")
    return steps_hop, names_hop, hop, delay


def read_action_csv(path: Path, n: int) -> tuple[list[str], np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    names = [str(row["action"]) for row in rows]
    bits = np.asarray([[int(float(row[f"y{idx:02d}"])) for idx in range(int(n))] for row in rows], dtype=np.int32)
    return names, bits


def split_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(int(num_rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_size = max(1, int(round(float(val_fraction) * int(num_rows))))
    val = np.asarray(sorted(indices[:val_size]), dtype=np.int64)
    train = np.asarray(sorted(indices[val_size:]), dtype=np.int64)
    return train, val


def build_reward_score(
    *,
    steps: np.ndarray,
    action_names: list[str],
    hops: np.ndarray,
    delay_ms: np.ndarray,
    reference_dir: Path,
    lambda_hop: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    ref_steps, ref_names, ref_hops, ref_delay = read_reference_tables(Path(reference_dir))
    step_to_ref = {int(step): idx for idx, step in enumerate(ref_steps.tolist())}
    missing = [int(step) for step in steps.tolist() if int(step) not in step_to_ref]
    if missing:
        raise ValueError(f"{len(missing)} steps missing in reference table, first={missing[:5]}")
    ref_rows = np.asarray([step_to_ref[int(step)] for step in steps.tolist()], dtype=np.int64)
    hop_env = np.nanmin(ref_hops[ref_rows], axis=1)
    delay_env = np.nanmin(ref_delay[ref_rows], axis=1)
    hop_scale = max(1e-9, float(np.nanmax(ref_hops) - np.nanmin(ref_hops)))
    delay_scale = max(1e-9, float(np.nanmax(ref_delay) - np.nanmin(ref_delay)))
    lam = float(lambda_hop)
    score = lam * ((hops - hop_env[:, None]) / hop_scale) + (1.0 - lam) * (
        (delay_ms - delay_env[:, None]) / delay_scale
    )
    meta = {
        "reference_dir": str(Path(reference_dir)),
        "lambda_hop": lam,
        "hop_scale": hop_scale,
        "delay_scale": delay_scale,
        "mean_reference_hop_envelope": float(np.nanmean(hop_env)),
        "mean_reference_delay_envelope_ms": float(np.nanmean(delay_env)),
        "num_reference_candidates": int(len(ref_names)),
        "num_actions": int(len(action_names)),
    }
    return score.astype(np.float32), meta


class PairRewardModel:
    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int, device: Any):
        import torch

        hidden = int(hidden_dim)
        self.state_net = torch.nn.Sequential(
            torch.nn.Linear(int(state_dim), hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        ).to(device)
        self.action_net = torch.nn.Sequential(
            torch.nn.Linear(int(action_dim), hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        ).to(device)
        self.pair_net = torch.nn.Sequential(
            torch.nn.Linear(hidden * 4, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden // 2, 1),
        ).to(device)

    def parameters(self):
        return list(self.state_net.parameters()) + list(self.action_net.parameters()) + list(self.pair_net.parameters())

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_net": self.state_net.state_dict(),
            "action_net": self.action_net.state_dict(),
            "pair_net": self.pair_net.state_dict(),
        }

    def __call__(self, x_state: Any, x_action: Any) -> Any:
        import torch

        s = self.state_net(x_state)
        a = self.action_net(x_action)
        batch = s.shape[0]
        actions = a.shape[0]
        s2 = s[:, None, :].expand(batch, actions, s.shape[1])
        a2 = a[None, :, :].expand(batch, actions, a.shape[1])
        pair = torch.cat([s2, a2, s2 * a2, torch.abs(s2 - a2)], dim=2)
        return self.pair_net(pair).squeeze(-1)


def lookup_policy_metrics(
    *,
    hops: np.ndarray,
    delay_ms: np.ndarray,
    pred_idx: np.ndarray,
    target_idx: np.ndarray,
    split: np.ndarray,
    action_bits: np.ndarray,
) -> dict[str, Any]:
    all_idx = np.arange(hops.shape[0])

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

    split_set = set(int(x) for x in split.tolist())
    train_rows = np.asarray([idx for idx in all_idx.tolist() if idx not in split_set], dtype=np.int64)
    return {
        "train": one(train_rows),
        "val": one(split),
        "all": one(all_idx),
    }


def selected_reward_objective(reward_score: np.ndarray, pred_idx: np.ndarray, rows: np.ndarray) -> float:
    return float(np.nanmean(reward_score[rows, pred_idx[rows]]))


def write_prediction_csv(
    path: Path,
    *,
    steps: np.ndarray,
    action_names: list[str],
    action_bits: np.ndarray,
    target_idx: np.ndarray,
    pred_idx: np.ndarray,
    split: np.ndarray,
    pred_score: np.ndarray,
    target_score: np.ndarray,
    hops: np.ndarray,
    delay_ms: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_set = set(int(x) for x in split.tolist())
    rows: list[dict[str, Any]] = []
    for row_idx, step in enumerate(steps.tolist()):
        true_bits = action_bits[int(target_idx[row_idx])]
        pred_bits = action_bits[int(pred_idx[row_idx])]
        item: dict[str, Any] = {
            "step": int(step),
            "split": "val" if row_idx in split_set else "train",
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
            "target_score": float(target_score[row_idx, int(target_idx[row_idx])]),
            "predicted_utility": float(pred_score[row_idx, int(pred_idx[row_idx])]),
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_dir = Path(args.reward_table_dir)

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

    features, feature_meta = build_state_features(
        steps=steps,
        feature_mode="group",
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )
    cand_features = action_features(action_bits.astype(np.float32))
    state_mean = np.mean(features[train_idx], axis=0)
    state_std = np.std(features[train_idx], axis=0)
    state_std[state_std < 1e-8] = 1.0
    cand_mean = np.mean(cand_features, axis=0)
    cand_std = np.std(cand_features, axis=0)
    cand_std[cand_std < 1e-8] = 1.0
    x_np = ((features - state_mean) / state_std).astype(np.float32)
    a_np = ((cand_features - cand_mean) / cand_std).astype(np.float32)

    score_mean = float(np.mean(reward_score[train_idx]))
    score_std = float(np.std(reward_score[train_idx]))
    if score_std < 1e-8:
        score_std = 1.0
    target_utility_np = -((reward_score - score_mean) / score_std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    a = torch.as_tensor(a_np, dtype=torch.float32, device=device)
    utility = torch.as_tensor(target_utility_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(target_idx, dtype=torch.long, device=device)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    model = PairRewardModel(
        state_dim=x_np.shape[1],
        action_dim=a_np.shape[1],
        hidden_dim=int(args.hidden_dim),
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    batch_size = max(1, int(args.batch_size))
    val_tensor = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_val_objective = float("inf")
    stale_evals = 0
    eval_every = max(1, int(args.eval_every))
    patience = max(1, int(args.patience))
    history: list[dict[str, Any]] = []
    epochs_run = 0
    for epoch in range(int(args.epochs)):
        epochs_run = epoch + 1
        shuffled = rng.permutation(train_idx)
        for start in range(0, len(shuffled), batch_size):
            rows = torch.as_tensor(shuffled[start : start + batch_size], dtype=torch.long, device=device)
            pred = model(x[rows], a)
            mse = torch.nn.functional.mse_loss(pred, utility[rows])
            ce = torch.nn.functional.cross_entropy(pred, y[rows])
            loss = mse + float(args.ce_weight) * ce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if bool(args.select_best_val) and ((epoch + 1) % eval_every == 0 or epoch + 1 == int(args.epochs)):
            with torch.no_grad():
                pred_val = model(x[val_tensor], a).detach().cpu().numpy()
            pred_val_idx = np.nanargmax(pred_val, axis=1).astype(np.int64)
            full_pred_idx = np.zeros(len(steps), dtype=np.int64)
            full_pred_idx[val_idx] = pred_val_idx
            val_objective = selected_reward_objective(reward_score, full_pred_idx, val_idx)
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "val_reward_objective": val_objective,
                    "val_action_exact": float(np.mean(pred_val_idx == target_idx[val_idx])),
                }
            )
            if val_objective + 1e-12 < best_val_objective:
                best_val_objective = val_objective
                best_epoch = int(epoch + 1)
                best_state = copy.deepcopy(model.state_dict())
                stale_evals = 0
            else:
                stale_evals += 1
                if stale_evals >= patience:
                    break

    if bool(args.select_best_val) and best_state is not None:
        model.state_net.load_state_dict(best_state["state_net"])
        model.action_net.load_state_dict(best_state["action_net"])
        model.pair_net.load_state_dict(best_state["pair_net"])

    with torch.no_grad():
        pred_utility = model(x, a).detach().cpu().numpy()
    pred_idx = np.nanargmax(pred_utility, axis=1).astype(np.int64)
    lookup_metrics = lookup_policy_metrics(
        hops=hops,
        delay_ms=delay_ms,
        pred_idx=pred_idx,
        target_idx=target_idx,
        split=val_idx,
        action_bits=action_bits,
    )
    prediction_csv = out_dir / "row_mask_reward_model_predictions.csv"
    write_prediction_csv(
        prediction_csv,
        steps=steps,
        action_names=action_names,
        action_bits=action_bits,
        target_idx=target_idx,
        pred_idx=pred_idx,
        split=val_idx,
        pred_score=pred_utility,
        target_score=reward_score,
        hops=hops,
        delay_ms=delay_ms,
    )
    torch.save(
        {
            **model.state_dict(),
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "cand_mean": cand_mean.astype(np.float32),
            "cand_std": cand_std.astype(np.float32),
            "score_mean": np.asarray(score_mean, dtype=np.float32),
            "score_std": np.asarray(score_std, dtype=np.float32),
            "action_bits": action_bits.astype(np.int8),
            "action_names": action_names,
            "feature_meta": feature_meta,
            "reward_meta": reward_meta,
        },
        out_dir / "row_mask_reward_model.pt",
    )
    meta = {
        "reward_table_dir": str(reward_dir),
        "reference_dir": str(Path(args.reference_dir)),
        "out_dir": str(out_dir),
        "prediction_csv": str(prediction_csv),
        "num_rows": int(len(steps)),
        "num_actions": int(len(action_names)),
        "lambda_hop": float(args.lambda_hop),
        "epochs": int(args.epochs),
        "epochs_run": int(epochs_run),
        "select_best_val": bool(args.select_best_val),
        "eval_every": int(args.eval_every),
        "patience": int(args.patience),
        "best_epoch": int(best_epoch),
        "best_val_reward_objective": float(best_val_objective),
        "training_history_tail": history[-20:],
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "ce_weight": float(args.ce_weight),
        "seed": int(args.seed),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "",
        "feature_meta": feature_meta,
        "reward_meta": reward_meta,
        "lookup_metrics": lookup_metrics,
    }
    (out_dir / "row_mask_reward_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
