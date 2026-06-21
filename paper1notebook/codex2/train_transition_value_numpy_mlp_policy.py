# -*- coding: utf-8 -*-
"""Train a NumPy transition-value model for 000040/000056 fusion.

The model learns the residual LST cost of a transition triple:

    (action[t-2], action[t-1], action[t]) -> local_transition_cost - stage_cost[t, action[t]]

Known instantaneous hop/delay cost is kept exact. The learned residual is then
decoded by a second-order dynamic program over the candidate triple set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


G60_RUN_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
DEFAULT_DATASET = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_transition_value_dataset"
    / "topk3_lam080_crit005_cnt003_build003"
    / "row_mask_transition_value_dataset.npz"
)
DEFAULT_OUT_DIR = G60_RUN_ROOT / "motif0040_0056_region_internal_plus_grid_transition_value_mlp_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a lightweight transition-value MLP and decode a row-mask schedule by second-order DP."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--validation-mod", type=int, default=5)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-scales", nargs="*", type=float, default=None)
    parser.add_argument("--lambda-hop", type=float, default=0.8)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def standardize_train(x_train: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    return (x_all - mean) / std, mean, std


def make_features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    time_idx = data["time_index"].astype(np.int64)
    a0 = data["prev2_action_index"].astype(np.int64)
    a1 = data["prev_action_index"].astype(np.int64)
    a2 = data["action_index"].astype(np.int64)
    state = data["state_features"].astype(np.float32)[time_idx]
    bits = data["action_bits"].astype(np.float32)
    b0 = bits[a0]
    b1 = bits[a1]
    b2 = bits[a2]
    row_counts = np.stack([b0.sum(axis=1), b1.sum(axis=1), b2.sum(axis=1)], axis=1) / float(bits.shape[1])
    h01 = np.mean(np.abs(b0 - b1), axis=1, keepdims=True)
    h12 = np.mean(np.abs(b1 - b2), axis=1, keepdims=True)
    h02 = np.mean(np.abs(b0 - b2), axis=1, keepdims=True)
    known = np.stack(
        [
            data["stage_cost"].astype(np.float32),
            data["mean_hops"].astype(np.float32),
            data["mean_delay_ms"].astype(np.float32),
        ],
        axis=1,
    )
    known[:, 1] /= max(1.0, float(np.nanmax(known[:, 1])))
    known[:, 2] /= max(1.0, float(np.nanmax(known[:, 2])))
    return np.concatenate([state, b0, b1, b2, row_counts, h01, h12, h02, known], axis=1).astype(np.float32)


def init_params(dim: int, hidden: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / max(1, dim)), size=(dim, hidden)).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": rng.normal(0.0, np.sqrt(2.0 / max(1, hidden)), size=(hidden, hidden)).astype(np.float32),
        "b2": np.zeros(hidden, dtype=np.float32),
        "w3": rng.normal(0.0, np.sqrt(2.0 / max(1, hidden)), size=(hidden, 1)).astype(np.float32),
        "b3": np.zeros(1, dtype=np.float32),
    }


def forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ params["w1"] + params["b1"]
    h1 = relu(z1)
    z2 = h1 @ params["w2"] + params["b2"]
    h2 = relu(z2)
    y = h2 @ params["w3"] + params["b3"]
    return y[:, 0], {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    params = init_params(int(x.shape[1]), int(hidden_dim), rng)
    adam_m = {k: np.zeros_like(v) for k, v in params.items()}
    adam_v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(~train_mask)
    history: list[dict[str, Any]] = []
    step = 0
    for epoch in range(1, int(epochs) + 1):
        rng.shuffle(train_idx)
        for start in range(0, train_idx.size, int(batch_size)):
            step += 1
            idx = train_idx[start : start + int(batch_size)]
            xb = x[idx]
            yb = y[idx]
            pred, cache = forward(xb, params)
            err = pred - yb
            grad_y = (2.0 / float(idx.size)) * err[:, None]
            grads: dict[str, np.ndarray] = {}
            grads["w3"] = cache["h2"].T @ grad_y + float(weight_decay) * params["w3"]
            grads["b3"] = grad_y.sum(axis=0)
            gh2 = grad_y @ params["w3"].T
            gz2 = gh2 * (cache["z2"] > 0.0)
            grads["w2"] = cache["h1"].T @ gz2 + float(weight_decay) * params["w2"]
            grads["b2"] = gz2.sum(axis=0)
            gh1 = gz2 @ params["w2"].T
            gz1 = gh1 * (cache["z1"] > 0.0)
            grads["w1"] = cache["x"].T @ gz1 + float(weight_decay) * params["w1"]
            grads["b1"] = gz1.sum(axis=0)
            for key in params:
                adam_m[key] = beta1 * adam_m[key] + (1.0 - beta1) * grads[key]
                adam_v[key] = beta2 * adam_v[key] + (1.0 - beta2) * (grads[key] * grads[key])
                m_hat = adam_m[key] / (1.0 - beta1**step)
                v_hat = adam_v[key] / (1.0 - beta2**step)
                params[key] -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)

        if epoch == 1 or epoch == int(epochs) or (int(progress_every) > 0 and epoch % int(progress_every) == 0):
            train_pred, _ = forward(x[train_idx], params)
            val_pred, _ = forward(x[val_idx], params)
            train_mae = float(np.mean(np.abs(train_pred - y[train_idx])))
            val_mae = float(np.mean(np.abs(val_pred - y[val_idx])))
            val_rmse = float(np.sqrt(np.mean((val_pred - y[val_idx]) ** 2)))
            denom = float(np.sum((y[val_idx] - float(np.mean(y[val_idx]))) ** 2))
            r2 = 1.0 - float(np.sum((val_pred - y[val_idx]) ** 2)) / max(1e-12, denom)
            item = {
                "epoch": int(epoch),
                "train_mae_scaled": train_mae,
                "val_mae_scaled": val_mae,
                "val_rmse_scaled": val_rmse,
                "val_r2": float(r2),
            }
            history.append(item)
            print(json.dumps(item), flush=True)
    return params, history


def build_stage_matrix(hops: np.ndarray, delay: np.ndarray, lambda_hop: float) -> np.ndarray:
    hop_env = np.nanmin(hops, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hops) - np.nanmin(hops)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    return (
        float(lambda_hop) * ((hops - hop_env[:, None]) / hop_scale)
        + (1.0 - float(lambda_hop)) * ((delay - delay_env[:, None]) / delay_scale)
    )


def decode_second_order_dp(
    *,
    time_idx: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    predicted_local_cost: np.ndarray,
    stage_matrix: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    t_count = int(steps.size)
    candidates = [set() for _ in range(t_count)]
    for t, p0, p1, c in zip(time_idx.tolist(), a0.tolist(), a1.tolist(), a2.tolist()):
        candidates[int(t) - 2].add(int(p0))
        candidates[int(t) - 1].add(int(p1))
        candidates[int(t)].add(int(c))
    for t, values in enumerate(candidates):
        if not values:
            values.add(int(np.argmin(stage_matrix[t])))

    dp: dict[tuple[int, int], float] = {}
    for p0 in candidates[0]:
        for p1 in candidates[1]:
            dp[(int(p0), int(p1))] = float(stage_matrix[0, int(p0)] + stage_matrix[1, int(p1)])
    parents: list[dict[tuple[int, int], tuple[int, int]]] = [dict() for _ in range(t_count)]

    order = np.argsort(time_idx, kind="stable")
    start = 0
    for t in range(2, t_count):
        while start < order.size and int(time_idx[order[start]]) < t:
            start += 1
        end = start
        while end < order.size and int(time_idx[order[end]]) == t:
            end += 1
        next_dp: dict[tuple[int, int], float] = {}
        parent_t: dict[tuple[int, int], tuple[int, int]] = {}
        for idx in order[start:end]:
            prev_state = (int(a0[idx]), int(a1[idx]))
            prev_cost = dp.get(prev_state)
            if prev_cost is None:
                continue
            new_state = (int(a1[idx]), int(a2[idx]))
            value = float(prev_cost + predicted_local_cost[idx])
            old = next_dp.get(new_state)
            if old is None or value < old:
                next_dp[new_state] = value
                parent_t[new_state] = prev_state
        if not next_dp:
            raise ValueError(f"DP has no reachable state at t={t}")
        dp = next_dp
        parents[t] = parent_t
        start = end

    state = min(dp, key=dp.get)
    selected = np.full(t_count, -1, dtype=np.int16)
    selected[-2] = int(state[0])
    selected[-1] = int(state[1])
    for t in range(t_count - 1, 1, -1):
        prev_state = parents[t][state]
        selected[t - 2] = int(prev_state[0])
        state = prev_state
    if np.any(selected < 0):
        raise ValueError("failed to reconstruct selected action sequence")
    return selected


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
    mask_tokens: Sequence[str],
    hops: np.ndarray,
    delay: np.ndarray,
    stage_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(selected.astype(int).tolist()):
        bits = action_bits[int(action_idx)]
        row: dict[str, Any] = {
            "step": int(steps[t]),
            "policy": policy,
            "action": str(action_names[int(action_idx)]),
            "action_index": int(action_idx),
            "mask": str(mask_tokens[int(action_idx)]),
            "row_count": int(np.sum(bits)),
            "mean_hops": float(hops[t, int(action_idx)]),
            "mean_delay_ms": float(delay[t, int(action_idx)]),
            "stage_cost": float(stage_matrix[t, int(action_idx)]),
        }
        for y in range(bits.shape[0]):
            row[f"y{y:02d}"] = int(bits[y])
        rows.append(row)
    return rows


def summarize_selected(selected: np.ndarray, hops: np.ndarray, delay: np.ndarray, stage: np.ndarray) -> dict[str, Any]:
    idx = selected.astype(int)
    rows = np.arange(idx.size)
    return {
        "mean_hops_no_lst": float(np.mean(hops[rows, idx])),
        "mean_delay_ms_no_lst": float(np.mean(delay[rows, idx])),
        "mean_stage_cost_no_lst": float(np.mean(stage[rows, idx])),
        "switches": int(np.count_nonzero(idx[1:] != idx[:-1])),
        "num_segments": int(np.count_nonzero(idx[1:] != idx[:-1]) + 1),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(Path(args.dataset), allow_pickle=True)

    steps = data["steps"].astype(np.int64)
    time_idx = data["time_index"].astype(np.int64)
    a0 = data["prev2_action_index"].astype(np.int64)
    a1 = data["prev_action_index"].astype(np.int64)
    a2 = data["action_index"].astype(np.int64)
    action_bits = data["action_bits"].astype(np.int8)
    action_names = [str(x) for x in data["action_names"].tolist()]
    mask_tokens = [str(x) for x in data["mask_tokens"].tolist()]
    hops = data["hop_table"].astype(np.float64)
    delay = data["delay_table_ms"].astype(np.float64)
    stage_matrix = build_stage_matrix(hops, delay, float(args.lambda_hop))

    x_raw = make_features(data).astype(np.float32)
    residual = data["local_transition_cost"].astype(np.float32) - data["stage_cost"].astype(np.float32)
    residual = np.maximum(residual, 0.0)
    train_mask = (time_idx % int(args.validation_mod)) != 0
    x_scaled, x_mean, x_std = standardize_train(x_raw[train_mask], x_raw)
    y_mean = float(np.mean(residual[train_mask]))
    y_std = float(np.std(residual[train_mask]))
    y_std = y_std if y_std > 1e-9 else 1.0
    y_scaled = ((residual - y_mean) / y_std).astype(np.float32)

    params, history = train_mlp(
        x_scaled,
        y_scaled,
        train_mask,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        progress_every=int(args.progress_every),
    )
    pred_scaled, _ = forward(x_scaled, params)
    val = ~train_mask
    base_pred_residual = np.maximum(0.0, pred_scaled.astype(np.float64) * y_std + y_mean)
    residual_mae = float(np.mean(np.abs(base_pred_residual[val] - residual[val])))
    residual_rmse = float(np.sqrt(np.mean((base_pred_residual[val] - residual[val]) ** 2)))
    denom = float(np.sum((residual[val] - float(np.mean(residual[val]))) ** 2))
    residual_r2 = 1.0 - float(np.sum((base_pred_residual[val] - residual[val]) ** 2)) / max(1e-12, denom)

    base_policy = (
        f"transition_value_mlp_h{int(args.hidden_dim)}"
        f"_e{int(args.epochs)}"
        f"_topk3_lam{float(args.lambda_hop):.2f}"
    )
    model_path = out_dir / f"{base_policy}_model.npz"
    np.savez_compressed(
        model_path,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=np.asarray([y_mean], dtype=np.float32),
        y_std=np.asarray([y_std], dtype=np.float32),
        **{key: value.astype(np.float32) for key, value in params.items()},
    )

    residual_scales = (
        [float(x) for x in args.residual_scales]
        if args.residual_scales is not None and len(args.residual_scales) > 0
        else [float(args.residual_scale)]
    )
    summaries: list[dict[str, Any]] = []
    for residual_scale in residual_scales:
        predicted_local = data["stage_cost"].astype(np.float64) + base_pred_residual * float(residual_scale)
        selected = decode_second_order_dp(
            time_idx=time_idx,
            a0=a0,
            a1=a1,
            a2=a2,
            predicted_local_cost=predicted_local,
            stage_matrix=stage_matrix,
            steps=steps,
        )
        policy = f"{base_policy}_rs{float(residual_scale):g}"
        schedule_path = out_dir / f"{policy}.csv"
        write_rows(
            schedule_path,
            schedule_rows(
                steps=steps,
                selected=selected,
                policy=policy,
                action_names=action_names,
                action_bits=action_bits,
                mask_tokens=mask_tokens,
                hops=hops,
                delay=delay,
                stage_matrix=stage_matrix,
            ),
        )
        summaries.append(
            {
                "policy": policy,
                "dataset": str(Path(args.dataset)),
                "schedule_csv": str(schedule_path),
                "model_npz": str(model_path),
                "hidden_dim": int(args.hidden_dim),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "validation_mod": int(args.validation_mod),
                "residual_scale": float(residual_scale),
                "lambda_hop": float(args.lambda_hop),
                "val_residual_mae": residual_mae,
                "val_residual_rmse": residual_rmse,
                "val_residual_r2": float(residual_r2),
                **summarize_selected(selected, hops, delay, stage_matrix),
            }
        )

    summary_csv = out_dir / "transition_value_mlp_policy_summary.csv"
    old_rows: list[dict[str, Any]] = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
            old_rows = list(csv.DictReader(f))
    old_rows.extend(summaries)
    write_rows(summary_csv, old_rows)
    meta_path = out_dir / f"{base_policy}_meta.json"
    meta = {
        "summaries": summaries,
        "history": history,
        "feature_dim": int(x_raw.shape[1]),
        "num_samples": int(x_raw.shape[0]),
        "train_samples": int(np.count_nonzero(train_mask)),
        "validation_samples": int(np.count_nonzero(~train_mask)),
        "target": "max(0, local_transition_cost - stage_cost)",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "meta": str(meta_path)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
