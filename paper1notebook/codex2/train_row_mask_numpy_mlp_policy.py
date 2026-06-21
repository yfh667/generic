from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_DATASET = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_learning_dataset"
    r"\teacher_critical_count_lam080_crit005_cnt003_build003"
    r"\row_mask_learning_dataset.npz"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_numpy_mlp_policy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a NumPy MLP policy to imitate the LST-aware row-mask teacher. "
            "The learned logits are converted to a topology schedule with optional "
            "objective-cost mixing and DP transition smoothing."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3500)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    parser.add_argument("--objective-weight", type=float, default=0.08)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument(
        "--switch-penalties",
        nargs="+",
        type=float,
        default=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    )
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    return (x - mean) / std, mean, std


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_actions: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    seed: int,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    rows, dim = x.shape
    hidden = int(hidden_dim)
    scale1 = np.sqrt(2.0 / max(1, dim))
    scale2 = np.sqrt(2.0 / max(1, hidden))
    params = {
        "w1": rng.normal(0.0, scale1, size=(dim, hidden)).astype(np.float64),
        "b1": np.zeros(hidden, dtype=np.float64),
        "w2": rng.normal(0.0, scale2, size=(hidden, int(num_actions))).astype(np.float64),
        "b2": np.zeros(int(num_actions), dtype=np.float64),
    }
    adam_m = {key: np.zeros_like(value) for key, value in params.items()}
    adam_v = {key: np.zeros_like(value) for key, value in params.items()}
    history: list[dict[str, Any]] = []
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    row_idx = np.arange(rows)
    for epoch in range(1, int(epochs) + 1):
        z1 = x @ params["w1"] + params["b1"]
        h1 = relu(z1)
        logits = h1 @ params["w2"] + params["b2"]
        probs = softmax(logits)
        smooth = min(0.99, max(0.0, float(label_smoothing)))
        if smooth > 0.0:
            target = np.full_like(probs, smooth / float(num_actions))
            target[row_idx, y] += 1.0 - smooth
            ce = -np.sum(target * np.log(probs + 1e-12), axis=1)
        else:
            target = None
            ce = -np.log(probs[row_idx, y] + 1e-12)
        loss = float(np.mean(ce))
        if weight_decay > 0.0:
            loss += 0.5 * float(weight_decay) * (
                float(np.sum(params["w1"] * params["w1"])) + float(np.sum(params["w2"] * params["w2"]))
            )

        if target is None:
            grad_logits = probs.copy()
            grad_logits[row_idx, y] -= 1.0
        else:
            grad_logits = probs - target
        grad_logits /= float(rows)
        grads: dict[str, np.ndarray] = {}
        grads["w2"] = h1.T @ grad_logits + float(weight_decay) * params["w2"]
        grads["b2"] = np.sum(grad_logits, axis=0)
        grad_h1 = grad_logits @ params["w2"].T
        grad_z1 = grad_h1 * (z1 > 0.0)
        grads["w1"] = x.T @ grad_z1 + float(weight_decay) * params["w1"]
        grads["b1"] = np.sum(grad_z1, axis=0)

        for key in params:
            adam_m[key] = beta1 * adam_m[key] + (1.0 - beta1) * grads[key]
            adam_v[key] = beta2 * adam_v[key] + (1.0 - beta2) * (grads[key] * grads[key])
            m_hat = adam_m[key] / (1.0 - beta1**epoch)
            v_hat = adam_v[key] / (1.0 - beta2**epoch)
            params[key] -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)

        if epoch == 1 or epoch == int(epochs) or (int(progress_every) > 0 and epoch % int(progress_every) == 0):
            pred = np.argmax(probs, axis=1)
            acc = float(np.mean(pred == y))
            item = {"epoch": int(epoch), "loss": float(loss), "ce": float(np.mean(ce)), "train_accuracy": acc}
            history.append(item)
            print(json.dumps(item), flush=True)

    return params, history


def predict_logits(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    return relu(x @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]


def run_dp(stage_cost: np.ndarray, transition_cost: np.ndarray, switch_penalty: float) -> np.ndarray:
    steps, actions = stage_cost.shape
    dp = np.empty((steps, actions), dtype=np.float64)
    parent = np.empty((steps, actions), dtype=np.int16)
    dp[0] = stage_cost[0]
    parent[0] = -1
    trans = float(switch_penalty) * transition_cost
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
    probs: np.ndarray,
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
            "model_prob": float(probs[t, int(action_idx)]),
            "teacher_prob": float(probs[t, teacher]),
        }
        for y in range(bits.shape[0]):
            row[f"y{y:02d}"] = int(bits[y])
        rows.append(row)
    return rows


def summarize_policy(
    *,
    selected: np.ndarray,
    teacher_idx: np.ndarray,
    action_bits: np.ndarray,
    teacher_bits: np.ndarray,
    hops: np.ndarray,
    delay_ms: np.ndarray,
    stage_cost: np.ndarray,
    transition: np.ndarray,
    probs: np.ndarray,
) -> dict[str, Any]:
    idx = selected.astype(int)
    row = np.arange(idx.size)
    changes = [float(transition[int(idx[t - 1]), int(idx[t])]) for t in range(1, idx.size)]
    return {
        "mean_hops": float(np.mean(hops[row, idx])),
        "mean_delay_ms": float(np.mean(delay_ms[row, idx])),
        "mean_stage_cost": float(np.mean(stage_cost[row, idx])),
        "mean_model_prob": float(np.mean(probs[row, idx])),
        "action_match_teacher": float(np.mean(idx == teacher_idx)),
        "exact_bits_match_teacher": float(np.mean(np.all(action_bits[idx] == teacher_bits, axis=1))),
        "switches": int(np.count_nonzero(idx[1:] != idx[:-1])),
        "num_segments": int(np.count_nonzero(idx[1:] != idx[:-1]) + 1),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
        "mean_row_hamming_transition": float(np.mean(changes)) if changes else 0.0,
        "max_row_hamming_transition": float(np.max(changes)) if changes else 0.0,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(Path(args.dataset), allow_pickle=True)
    steps = data["steps"].astype(np.int64)
    x_raw = data["state_features"].astype(np.float64)
    teacher_idx = data["teacher_action_index"].astype(np.int64)
    teacher_bits = data["teacher_bits"].astype(np.int8)
    action_bits = data["action_bits"].astype(np.int8)
    action_names = [str(x) for x in data["action_names"].tolist()]
    hops = data["hop_table"].astype(np.float64)
    delay_ms = data["delay_table_ms"].astype(np.float64)
    objective_stage_cost = data["stage_cost"].astype(np.float64)
    transition = data["transition_hamming"].astype(np.float64)

    x, x_mean, x_std = standardize(x_raw)
    params, history = train_mlp(
        x,
        teacher_idx,
        num_actions=len(action_names),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        label_smoothing=float(args.label_smoothing),
        seed=int(args.seed),
        progress_every=int(args.progress_every),
    )
    logits = predict_logits(x, params)
    probs = softmax(logits / max(1e-9, float(args.logit_temperature)))
    bc_cost = -np.log(probs + 1e-12)
    learned_stage_cost = float(args.bc_weight) * bc_cost + float(args.objective_weight) * objective_stage_cost
    greedy = np.argmin(learned_stage_cost, axis=1).astype(np.int16)

    schedules: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    variants = [("greedy", greedy)]
    for penalty in args.switch_penalties:
        selected = run_dp(learned_stage_cost, transition, float(penalty))
        variants.append((f"dp_sw{float(penalty):g}", selected))

    for label, selected in variants:
        policy = (
            f"numpy_mlp_hidden{int(args.hidden_dim)}"
            f"_smooth{float(args.label_smoothing):g}"
            f"_temp{float(args.logit_temperature):g}"
            f"_obj{float(args.objective_weight):g}_{label}"
        )
        schedule_path = out_dir / f"{policy}.csv"
        write_rows(
            schedule_path,
            schedule_rows(
                steps=steps,
                selected=selected,
                policy=policy,
                action_names=action_names,
                action_bits=action_bits,
                hops=hops,
                delay_ms=delay_ms,
                teacher_idx=teacher_idx,
                teacher_bits=teacher_bits,
                probs=probs,
            ),
        )
        summary = summarize_policy(
            selected=selected,
            teacher_idx=teacher_idx,
            action_bits=action_bits,
            teacher_bits=teacher_bits,
            hops=hops,
            delay_ms=delay_ms,
            stage_cost=objective_stage_cost,
            transition=transition,
            probs=probs,
        )
        summary_rows.append(
            {
                "policy": policy,
                "schedule_csv": str(schedule_path),
                "dataset": str(Path(args.dataset)),
                "hidden_dim": int(args.hidden_dim),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "label_smoothing": float(args.label_smoothing),
                "logit_temperature": float(args.logit_temperature),
                "bc_weight": float(args.bc_weight),
                "objective_weight": float(args.objective_weight),
                "switch_penalty": "" if label == "greedy" else float(label.removeprefix("dp_sw")),
                **summary,
            }
        )

    model_path = out_dir / "numpy_mlp_model.npz"
    np.savez_compressed(model_path, x_mean=x_mean, x_std=x_std, **{key: value.astype(np.float32) for key, value in params.items()})
    summary_csv = out_dir / "numpy_mlp_policy_summary.csv"
    write_rows(summary_csv, summary_rows)
    meta = {
        "dataset": str(Path(args.dataset)),
        "out_dir": str(out_dir),
        "model_npz": str(model_path),
        "summary_csv": str(summary_csv),
        "history": history,
        "input_shape": [int(x) for x in x_raw.shape],
        "num_actions": int(len(action_names)),
        "teacher_used_actions": int(len(set(int(x) for x in teacher_idx.tolist()))),
    }
    meta_path = out_dir / "numpy_mlp_policy_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "model": str(model_path)}, ensure_ascii=False, indent=2))
    print("\n".join(json.dumps(row, ensure_ascii=False) for row in summary_rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
