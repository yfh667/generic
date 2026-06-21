from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_HYBRID_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_hybrid_search"
    r"\full_t0_86160_s60"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_hybrid_policy"
)
DEFAULT_GROUP_CACHE = Path(
    r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache"
    r"\station_visible_satellites_20250106_G60_t0_86160_stride60.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn/evaluate dynamic policies over searched 000040/000056 hybrid candidates."
    )
    parser.add_argument("--hybrid-dir", type=Path, default=DEFAULT_HYBRID_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delay-eps-ms", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--hop-weight", type=float, default=1.0)
    parser.add_argument("--delay-weight", type=float, default=1.0)
    parser.add_argument("--switch-penalty", type=float, default=0.0)
    parser.add_argument("--torch-epochs", type=int, default=0)
    parser.add_argument("--torch-seed", type=int, default=667)
    parser.add_argument("--feature-mode", choices=("time", "group"), default="time")
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def read_wide_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    names = [name for name in fieldnames if name != "step"]
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    matrix = np.asarray(
        [
            [float(row[name]) if row.get(name) not in ("", None) else math.nan for name in names]
            for row in rows
        ],
        dtype=np.float64,
    )
    return steps, names, matrix


def read_metrics(hybrid_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    steps_hop, names_hop, hop = read_wide_csv(hybrid_dir / "compare_mean_shortest_hops.csv")
    steps_delay, names_delay, delay = read_wide_csv(hybrid_dir / "compare_mean_shortest_delay_ms.csv")
    if not np.array_equal(steps_hop, steps_delay):
        raise ValueError("hops and delay files have different steps")
    if names_hop != names_delay:
        raise ValueError("hops and delay files have different candidate columns")
    return steps_hop, names_hop, hop, delay


def finite_minmax(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalized_gap_cost(
    hop: np.ndarray,
    delay: np.ndarray,
    *,
    hop_weight: float,
    delay_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop_env = np.nanmin(hop, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_lo, hop_hi = finite_minmax(hop)
    delay_lo, delay_hi = finite_minmax(delay)
    hop_gap = (hop - hop_env[:, None]) / (hop_hi - hop_lo)
    delay_gap = (delay - delay_env[:, None]) / (delay_hi - delay_lo)
    cost = float(hop_weight) * hop_gap + float(delay_weight) * delay_gap
    return cost, hop_env, delay_env


def choose_delay_eps_policy(hop: np.ndarray, delay: np.ndarray, eps_ms: float) -> np.ndarray:
    delay_env = np.nanmin(delay, axis=1)
    chosen = np.empty(hop.shape[0], dtype=np.int32)
    for row in range(hop.shape[0]):
        candidates = np.flatnonzero(delay[row] <= delay_env[row] + float(eps_ms))
        if candidates.size == 0:
            chosen[row] = int(np.nanargmin(delay[row]))
            continue
        best_local = candidates[np.lexsort((delay[row, candidates], hop[row, candidates]))][0]
        chosen[row] = int(best_local)
    return chosen


def choose_switch_dp(cost: np.ndarray, switch_penalty: float) -> np.ndarray:
    values = np.asarray(cost, dtype=np.float64)
    n_steps, n_actions = values.shape
    dp = np.full((n_steps, n_actions), np.inf, dtype=np.float64)
    prev = np.full((n_steps, n_actions), -1, dtype=np.int32)
    dp[0] = values[0]
    for row in range(1, n_steps):
        for action in range(n_actions):
            transition = dp[row - 1] + float(switch_penalty) * (np.arange(n_actions) != action)
            best_prev = int(np.argmin(transition))
            dp[row, action] = float(transition[best_prev] + values[row, action])
            prev[row, action] = best_prev
    chosen = np.empty(n_steps, dtype=np.int32)
    chosen[-1] = int(np.argmin(dp[-1]))
    for row in range(n_steps - 1, 0, -1):
        chosen[row - 1] = int(prev[row, chosen[row]])
    return chosen


def fourier_features(steps: np.ndarray, period_s: float = 86400.0) -> np.ndarray:
    phase = np.asarray(steps, dtype=np.float64) / float(period_s)
    feats = [np.ones_like(phase)]
    for k in range(1, 9):
        feats.append(np.sin(2.0 * np.pi * k * phase))
        feats.append(np.cos(2.0 * np.pi * k * phase))
    return np.column_stack(feats).astype(np.float64)


def _histogram(values: list[int], size: int) -> np.ndarray:
    out = np.zeros(int(size), dtype=np.float64)
    if not values:
        return out
    for value in values:
        out[int(value) % int(size)] += 1.0
    total = float(np.sum(out))
    return out / total if total > 0 else out


def _circular_stats(values: list[int], size: int) -> np.ndarray:
    if not values:
        return np.zeros(2, dtype=np.float64)
    angles = np.asarray(values, dtype=np.float64) / float(size) * 2.0 * np.pi
    return np.asarray([float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))], dtype=np.float64)


def _lookup_group_nodes(groups_by_step: dict[str, Any], step: int, row_idx: int, group_id: int, *, start: int, stride: int) -> list[int]:
    keys = [str(int(step)), str(int(row_idx))]
    if int(stride) > 0:
        keys.append(str(int((int(step) - int(start)) // int(stride))))
    for key in keys:
        payload = groups_by_step.get(key)
        if isinstance(payload, dict):
            nodes = payload.get(str(int(group_id)), payload.get(int(group_id), []))
            return [int(x) for x in (nodes or [])]
    return []


def group_distribution_features(
    *,
    steps: np.ndarray,
    group_cache: Path,
    source_group_id: int,
    target_group_id: int,
    p: int,
    n: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    with Path(group_cache).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    source = payload.get("source", {}) if isinstance(payload, dict) else {}
    groups_by_step = payload.get("groups_by_step", {}) if isinstance(payload, dict) else {}
    start = int(source.get("start", int(np.min(steps))))
    stride = int(source.get("stride", 1))
    rows: list[np.ndarray] = []
    for row_idx, step in enumerate(steps):
        source_nodes = _lookup_group_nodes(
            groups_by_step,
            int(step),
            int(row_idx),
            int(source_group_id),
            start=start,
            stride=stride,
        )
        target_nodes = _lookup_group_nodes(
            groups_by_step,
            int(step),
            int(row_idx),
            int(target_group_id),
            start=start,
            stride=stride,
        )
        source_y = [node % int(n) for node in source_nodes]
        target_y = [node % int(n) for node in target_nodes]
        source_x = [node // int(n) for node in source_nodes]
        target_x = [node // int(n) for node in target_nodes]
        feature = np.concatenate(
            [
                _histogram(source_y, int(n)),
                _histogram(target_y, int(n)),
                _histogram(source_x, int(p)),
                _histogram(target_x, int(p)),
                _circular_stats(source_y, int(n)),
                _circular_stats(target_y, int(n)),
                _circular_stats(source_x, int(p)),
                _circular_stats(target_x, int(p)),
                np.asarray(
                    [
                        len(source_nodes) / float(max(1, int(p) * int(n))),
                        len(target_nodes) / float(max(1, int(p) * int(n))),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        rows.append(feature)
    meta = {
        "group_cache": str(Path(group_cache)),
        "source_group_id": int(source_group_id),
        "target_group_id": int(target_group_id),
        "p": int(p),
        "n": int(n),
        "feature_parts": [
            f"source_y_hist_{n}",
            f"target_y_hist_{n}",
            f"source_x_hist_{p}",
            f"target_x_hist_{p}",
            "source/target circular y/x stats",
            "source/target node count fractions",
        ],
    }
    return np.asarray(rows, dtype=np.float64), meta


def build_state_features(
    *,
    steps: np.ndarray,
    feature_mode: str,
    group_cache: Path,
    source_group_id: int,
    target_group_id: int,
    p: int,
    n: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    time_features = fourier_features(steps)
    if str(feature_mode) == "time":
        return time_features, {"feature_mode": "time", "feature_dim": int(time_features.shape[1])}
    group_features, group_meta = group_distribution_features(
        steps=steps,
        group_cache=Path(group_cache),
        source_group_id=int(source_group_id),
        target_group_id=int(target_group_id),
        p=int(p),
        n=int(n),
    )
    features = np.concatenate([time_features, group_features], axis=1)
    return features, {
        "feature_mode": "group",
        "feature_dim": int(features.shape[1]),
        "time_feature_dim": int(time_features.shape[1]),
        "group_feature_dim": int(group_features.shape[1]),
        **group_meta,
    }


def train_linear_bandit(
    features: np.ndarray,
    rewards: np.ndarray,
    *,
    episodes: int = 1200,
    seed: int = 667,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = random.Random(int(seed))
    features = np.asarray(features, dtype=np.float64)
    n_steps, n_actions = rewards.shape
    weights = np.zeros((n_actions, features.shape[1]), dtype=np.float64)
    lr0 = 0.06
    eps0 = 0.35
    for episode in range(int(episodes)):
        lr = lr0 / math.sqrt(1.0 + episode / 250.0)
        eps = max(0.02, eps0 * (1.0 - episode / max(1, int(episodes))))
        order = list(range(n_steps))
        rng.shuffle(order)
        for row in order:
            q_values = weights @ features[row]
            if rng.random() < eps:
                action = rng.randrange(n_actions)
            else:
                action = int(np.argmax(q_values))
            target = float(rewards[row, action])
            pred = float(q_values[action])
            weights[action] += lr * (target - pred) * features[row]
    q_all = features @ weights.T
    chosen = np.argmax(q_all, axis=1).astype(np.int32)
    oracle = np.argmax(rewards, axis=1).astype(np.int32)
    meta = {
        "method": "linear_contextual_bandit",
        "episodes": int(episodes),
        "seed": int(seed),
        "oracle_match_accuracy": float(np.mean(chosen == oracle)),
        "feature_dim": int(features.shape[1]),
    }
    return chosen, meta


def train_torch_mlp(
    *,
    steps: np.ndarray,
    features: np.ndarray,
    rewards: np.ndarray,
    names: Sequence[str],
    epochs: int,
    seed: int,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_np = np.asarray(features, dtype=np.float64)
    x = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(features_np.shape[1], 96),
        torch.nn.SiLU(),
        torch.nn.Linear(96, 96),
        torch.nn.SiLU(),
        torch.nn.Linear(96, len(names)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    for _epoch in range(int(epochs)):
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred_np = model(x).detach().cpu().numpy()
    chosen = np.argmax(pred_np, axis=1).astype(np.int32)
    oracle = np.argmax(rewards, axis=1).astype(np.int32)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "candidate_names": list(names),
            "feature_dim": int(features_np.shape[1]),
        },
        model_path,
    )
    meta = {
        "method": "torch_mlp_contextual_q",
        "epochs": int(epochs),
        "seed": int(seed),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "",
        "oracle_match_accuracy": float(np.mean(chosen == oracle)),
        "final_mse": float(np.mean((pred_np - rewards) ** 2)),
        "model_path": str(model_path),
    }
    return chosen, meta


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def selected_values(values: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0]), np.asarray(chosen, dtype=np.int32)]


def count_switches(chosen: np.ndarray) -> int:
    return int(np.count_nonzero(chosen[1:] != chosen[:-1])) if chosen.size > 1 else 0


def summarize_policy(
    *,
    name: str,
    chosen: np.ndarray,
    candidate_names: Sequence[str],
    hop: np.ndarray,
    delay: np.ndarray,
    hybrid_hop_env: np.ndarray,
    hybrid_delay_env: np.ndarray,
    pure_hop_env: np.ndarray,
    pure_delay_env: np.ndarray,
) -> dict[str, Any]:
    hop_values = selected_values(hop, chosen)
    delay_values = selected_values(delay, chosen)
    counts = np.bincount(np.asarray(chosen, dtype=np.int32), minlength=len(candidate_names))
    used = {candidate_names[idx]: int(value) for idx, value in enumerate(counts.tolist()) if int(value) > 0}
    return {
        "policy": name,
        "mean_hops": float(np.mean(hop_values)),
        "mean_delay_ms": float(np.mean(delay_values)),
        "mean_hop_gap_to_hybrid_envelope": float(np.mean(hop_values - hybrid_hop_env)),
        "mean_delay_gap_to_hybrid_envelope_ms": float(np.mean(delay_values - hybrid_delay_env)),
        "mean_hop_gap_to_pure_envelope": float(np.mean(hop_values - pure_hop_env)),
        "mean_delay_gap_to_pure_envelope_ms": float(np.mean(delay_values - pure_delay_env)),
        "switches": count_switches(chosen),
        "num_segments": count_switches(chosen) + 1,
        "num_used_candidates": int(np.count_nonzero(counts)),
        "used_candidates": json.dumps(used, ensure_ascii=False),
    }


def compress_segments(
    *,
    steps: np.ndarray,
    chosen: np.ndarray,
    names: Sequence[str],
    hop_values: np.ndarray,
    delay_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if steps.size == 0:
        return rows
    start = 0
    for idx in range(1, int(steps.size) + 1):
        if idx < int(steps.size) and int(chosen[idx]) == int(chosen[start]):
            continue
        rows.append(
            {
                "segment_id": len(rows),
                "start_step": int(steps[start]),
                "end_step": int(steps[idx - 1]),
                "num_points": int(idx - start),
                "candidate": str(names[int(chosen[start])]),
                "mean_hops": float(np.mean(hop_values[start:idx])),
                "mean_delay_ms": float(np.mean(delay_values[start:idx])),
            }
        )
        start = idx
    return rows


def write_policy_choice(path: Path, *, steps: np.ndarray, chosen: np.ndarray, names: Sequence[str], policy: str) -> None:
    write_csv_rows(
        path,
        [
            {
                "step": int(step),
                "policy": str(policy),
                "action_idx": int(action),
                "candidate": str(names[int(action)]),
            }
            for step, action in zip(steps, chosen)
        ],
    )


def read_policy_choice(path: Path, *, expected_steps: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != int(expected_steps.size):
        return None
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    if not np.array_equal(steps, expected_steps):
        return None
    return np.asarray([int(row["action_idx"]) for row in rows], dtype=np.int32)


def plot_results(
    *,
    out_dir: Path,
    steps: np.ndarray,
    candidate_names: Sequence[str],
    hop: np.ndarray,
    delay: np.ndarray,
    policies: dict[str, np.ndarray],
    pure_hop_env: np.ndarray,
    pure_delay_env: np.ndarray,
    hybrid_hop_env: np.ndarray,
    hybrid_delay_env: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = steps.astype(np.float64) / 3600.0
    best_static = int(np.nanargmin(np.nanmean(hop + delay * 0.0, axis=0)))
    colors = {
        "pure_envelope": "#111827",
        "hybrid_envelope": "#059669",
        "eps_0p5ms": "#7c3aed",
        "eps_1ms": "#f97316",
        "scalar_oracle": "#0f766e",
        "switch_dp": "#dc2626",
        "linear_bandit": "#2563eb",
        "torch_mlp_q": "#1d4ed8",
    }

    for values, pure_env, hybrid_env, ylabel, filename in (
        (hop, pure_hop_env, hybrid_hop_env, "China-Europe mean shortest hops", "hybrid_policy_hops.png"),
        (delay, pure_delay_env, hybrid_delay_env, "China-Europe mean shortest delay (ms)", "hybrid_policy_delay_ms.png"),
    ):
        fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=180)
        ax.plot(x, pure_env, color=colors["pure_envelope"], linestyle="--", linewidth=1.8, label="pure 040/056 envelope")
        ax.plot(x, hybrid_env, color=colors["hybrid_envelope"], linestyle="--", linewidth=2.0, label="hybrid candidate envelope")
        ax.plot(x, values[:, best_static], color="#9ca3af", linewidth=0.9, alpha=0.65, label=f"one candidate: {candidate_names[best_static]}")
        for name, chosen in policies.items():
            if name.startswith("eps_") and name not in ("eps_0p5ms", "eps_1ms"):
                continue
            ax.plot(
                x,
                selected_values(values, chosen),
                linewidth=1.7,
                alpha=0.9,
                color=colors.get(name, None),
                label=name,
            )
        ax.set_title(f"G60 China-Europe region-internal +grid hybrid policy: {filename.replace('hybrid_policy_', '').replace('.png', '')}")
        ax.set_xlabel("time (hour)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, names, hop, delay = read_metrics(Path(args.hybrid_dir))
    features, feature_meta = build_state_features(
        steps=steps,
        feature_mode=str(args.feature_mode),
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )
    pure_indices = [idx for idx, name in enumerate(names) if str(name).startswith("pure_")]
    if len(pure_indices) < 2:
        raise ValueError("expected at least two pure candidates")
    pure_hop_env = np.nanmin(hop[:, pure_indices], axis=1)
    pure_delay_env = np.nanmin(delay[:, pure_indices], axis=1)
    cost, hybrid_hop_env, hybrid_delay_env = normalized_gap_cost(
        hop,
        delay,
        hop_weight=float(args.hop_weight),
        delay_weight=float(args.delay_weight),
    )
    rewards = -cost
    hop_arg = np.nanargmin(hop, axis=1).astype(np.int32)
    delay_arg = np.nanargmin(delay, axis=1).astype(np.int32)
    scalar_oracle = np.nanargmin(cost, axis=1).astype(np.int32)

    policies: dict[str, np.ndarray] = {}
    for eps in args.delay_eps_ms:
        label = f"eps_{float(eps):g}ms".replace(".", "p")
        policies[label] = choose_delay_eps_policy(hop, delay, float(eps))
    policies["scalar_oracle"] = scalar_oracle
    policies["switch_dp"] = choose_switch_dp(cost, float(args.switch_penalty))
    linear_chosen, linear_meta = train_linear_bandit(features, rewards, episodes=1200, seed=int(args.torch_seed))
    linear_name = "linear_bandit" if str(args.feature_mode) == "time" else "linear_bandit_group"
    policies[linear_name] = linear_chosen

    torch_meta: dict[str, Any] | None = None
    torch_policy_name = "torch_mlp_q" if str(args.feature_mode) == "time" else "torch_mlp_q_group"
    torch_policy_path = out_dir / f"{torch_policy_name}_policy.csv"
    torch_meta_path = out_dir / f"{torch_policy_name}_meta.json"
    if int(args.torch_epochs) > 0:
        try:
            torch_chosen, torch_meta = train_torch_mlp(
                steps=steps,
                features=features,
                rewards=rewards,
                names=names,
                epochs=int(args.torch_epochs),
                seed=int(args.torch_seed),
                model_path=out_dir / "models" / "torch_mlp_q.pt",
            )
            policies[torch_policy_name] = torch_chosen
            write_policy_choice(torch_policy_path, steps=steps, chosen=torch_chosen, names=names, policy=torch_policy_name)
            torch_meta_path.write_text(json.dumps(torch_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            torch_meta = {"error": repr(exc), "requested_epochs": int(args.torch_epochs)}
    else:
        torch_chosen = read_policy_choice(torch_policy_path, expected_steps=steps)
        if torch_chosen is not None:
            policies[torch_policy_name] = torch_chosen
            if torch_meta_path.exists():
                try:
                    torch_meta = json.loads(torch_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    torch_meta = {"warning": f"failed to read {torch_meta_path}"}

    summary_rows = [
        {
            "policy": "pure_envelope_reference",
            "mean_hops": float(np.mean(pure_hop_env)),
            "mean_delay_ms": float(np.mean(pure_delay_env)),
            "mean_hop_gap_to_hybrid_envelope": float(np.mean(pure_hop_env - hybrid_hop_env)),
            "mean_delay_gap_to_hybrid_envelope_ms": float(np.mean(pure_delay_env - hybrid_delay_env)),
            "mean_hop_gap_to_pure_envelope": 0.0,
            "mean_delay_gap_to_pure_envelope_ms": 0.0,
            "switches": "",
            "num_segments": "",
            "num_used_candidates": "",
            "used_candidates": "",
        },
        {
            "policy": "hybrid_hop_envelope_reference",
            "mean_hops": float(np.mean(hybrid_hop_env)),
            "mean_delay_ms": float(np.mean(selected_values(delay, hop_arg))),
            "mean_hop_gap_to_hybrid_envelope": 0.0,
            "mean_delay_gap_to_hybrid_envelope_ms": float(np.mean(selected_values(delay, hop_arg) - hybrid_delay_env)),
            "mean_hop_gap_to_pure_envelope": float(np.mean(hybrid_hop_env - pure_hop_env)),
            "mean_delay_gap_to_pure_envelope_ms": float(np.mean(selected_values(delay, hop_arg) - pure_delay_env)),
            "switches": count_switches(hop_arg),
            "num_segments": count_switches(hop_arg) + 1,
            "num_used_candidates": int(len(set(hop_arg.tolist()))),
            "used_candidates": "",
        },
        {
            "policy": "hybrid_delay_envelope_reference",
            "mean_hops": float(np.mean(selected_values(hop, delay_arg))),
            "mean_delay_ms": float(np.mean(hybrid_delay_env)),
            "mean_hop_gap_to_hybrid_envelope": float(np.mean(selected_values(hop, delay_arg) - hybrid_hop_env)),
            "mean_delay_gap_to_hybrid_envelope_ms": 0.0,
            "mean_hop_gap_to_pure_envelope": float(np.mean(selected_values(hop, delay_arg) - pure_hop_env)),
            "mean_delay_gap_to_pure_envelope_ms": float(np.mean(hybrid_delay_env - pure_delay_env)),
            "switches": count_switches(delay_arg),
            "num_segments": count_switches(delay_arg) + 1,
            "num_used_candidates": int(len(set(delay_arg.tolist()))),
            "used_candidates": "",
        },
    ]
    for name, chosen in policies.items():
        summary_rows.append(
            summarize_policy(
                name=name,
                chosen=chosen,
                candidate_names=names,
                hop=hop,
                delay=delay,
                hybrid_hop_env=hybrid_hop_env,
                hybrid_delay_env=hybrid_delay_env,
                pure_hop_env=pure_hop_env,
                pure_delay_env=pure_delay_env,
            )
        )
    write_csv_rows(out_dir / "hybrid_policy_summary.csv", summary_rows)

    by_step_rows: list[dict[str, Any]] = []
    for row, step in enumerate(steps):
        item: dict[str, Any] = {
            "step": int(step),
            "hour": float(step / 3600.0),
            "pure_hop_envelope": float(pure_hop_env[row]),
            "pure_delay_envelope_ms": float(pure_delay_env[row]),
            "hybrid_hop_envelope": float(hybrid_hop_env[row]),
            "hybrid_delay_envelope_ms": float(hybrid_delay_env[row]),
            "hop_winner": names[int(hop_arg[row])],
            "delay_winner": names[int(delay_arg[row])],
            "winner_conflict": bool(int(hop_arg[row]) != int(delay_arg[row])),
        }
        for name, chosen in policies.items():
            action = int(chosen[row])
            item[f"{name}_candidate"] = names[action]
            item[f"{name}_hops"] = float(hop[row, action])
            item[f"{name}_delay_ms"] = float(delay[row, action])
        by_step_rows.append(item)
    write_csv_rows(out_dir / "hybrid_policy_by_step.csv", by_step_rows)

    segment_dir = out_dir / "segments"
    for name, chosen in policies.items():
        write_csv_rows(
            segment_dir / f"{name}_segments.csv",
            compress_segments(
                steps=steps,
                chosen=chosen,
                names=names,
                hop_values=selected_values(hop, chosen),
                delay_values=selected_values(delay, chosen),
            ),
        )

    meta = {
        "hybrid_dir": str(Path(args.hybrid_dir)),
        "out_dir": str(out_dir),
        "num_steps": int(steps.size),
        "num_candidates": int(len(names)),
        "candidate_names": list(names),
        "hop_delay_winner_conflict_steps": int(np.count_nonzero(hop_arg != delay_arg)),
        "hop_delay_winner_conflict_ratio": float(np.mean(hop_arg != delay_arg)),
        "same_candidate_dual_envelope_steps": int(np.count_nonzero(hop_arg == delay_arg)),
        "same_candidate_dual_envelope_ratio": float(np.mean(hop_arg == delay_arg)),
        "mean_pure_hop_envelope": float(np.mean(pure_hop_env)),
        "mean_hybrid_hop_envelope": float(np.mean(hybrid_hop_env)),
        "mean_pure_delay_envelope_ms": float(np.mean(pure_delay_env)),
        "mean_hybrid_delay_envelope_ms": float(np.mean(hybrid_delay_env)),
        "linear_meta": linear_meta,
        "torch_meta": torch_meta,
        "feature_meta": feature_meta,
        "summary_csv": str(out_dir / "hybrid_policy_summary.csv"),
        "by_step_csv": str(out_dir / "hybrid_policy_by_step.csv"),
    }
    (out_dir / "hybrid_policy_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not bool(args.skip_plots):
        plot_results(
            out_dir=out_dir,
            steps=steps,
            candidate_names=names,
            hop=hop,
            delay=delay,
            policies=policies,
            pure_hop_env=pure_hop_env,
            pure_delay_env=pure_delay_env,
            hybrid_hop_env=hybrid_hop_env,
            hybrid_delay_env=hybrid_delay_env,
        )

    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
