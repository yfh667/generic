from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRIC_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\static_motif0040_0056_china_europe_region_internal_plus_grid"
    r"\region_internal_grid_metrics_t0_86160_stride60\china_europe"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_fusion_policy"
)

TOPO_040 = "selected_motif_000040"
TOPO_056 = "selected_motif_000056"
ACTIONS = (TOPO_040, TOPO_056)


@dataclass(frozen=True)
class MetricSeries:
    steps: np.ndarray
    hops: dict[str, np.ndarray]
    delay_ms: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study a dynamic fusion policy for motif 000040 and 000056 under "
            "China-Europe region-internal +grid constraints."
        )
    )
    parser.add_argument("--metric-dir", type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delay-eps-ms", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0, 3.0))
    parser.add_argument("--hop-weight", type=float, default=1.0)
    parser.add_argument("--switch-penalty", type=float, default=0.0)
    parser.add_argument("--rl-episodes", type=int, default=800)
    parser.add_argument("--rl-seed", type=int, default=667)
    parser.add_argument("--torch-epochs", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def read_wide_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    for name in ACTIONS:
        values[name] = np.asarray(
            [float(row[name]) if row.get(name) not in ("", None) else math.nan for row in rows],
            dtype=np.float64,
        )
    return steps, values


def read_metric_series(metric_dir: Path) -> MetricSeries:
    hops_steps, hops = read_wide_csv(metric_dir / "compare_mean_shortest_hops.csv")
    delay_steps, delay_ms = read_wide_csv(metric_dir / "compare_mean_shortest_delay_ms.csv")
    if not np.array_equal(hops_steps, delay_steps):
        raise ValueError("hops and delay CSVs have different time axes")
    return MetricSeries(steps=hops_steps, hops=hops, delay_ms=delay_ms)


def finite_minmax(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - float(lo)) / (float(hi) - float(lo))


def action_matrix(series: MetricSeries) -> tuple[np.ndarray, np.ndarray]:
    hop = np.column_stack([series.hops[name] for name in ACTIONS]).astype(np.float64)
    delay = np.column_stack([series.delay_ms[name] for name in ACTIONS]).astype(np.float64)
    return hop, delay


def choose_delay_epsilon_hop_policy(hop: np.ndarray, delay: np.ndarray, eps_ms: float) -> np.ndarray:
    """Choose the lower-hop motif if its delay stays within eps_ms of best delay."""

    best_delay = np.nanmin(delay, axis=1)
    min_hop = np.nanmin(hop, axis=1)
    chosen = np.full(hop.shape[0], 1, dtype=np.int32)
    for row in range(hop.shape[0]):
        candidates = [
            col
            for col in range(hop.shape[1])
            if np.isfinite(hop[row, col])
            and np.isfinite(delay[row, col])
            and delay[row, col] <= best_delay[row] + float(eps_ms)
        ]
        if candidates:
            chosen[row] = min(candidates, key=lambda col: (hop[row, col], delay[row, col], col))
        else:
            chosen[row] = int(np.nanargmin(delay[row]))
    return chosen


def scalar_cost_matrix(hop: np.ndarray, delay: np.ndarray, hop_weight: float) -> np.ndarray:
    hop_lo, hop_hi = finite_minmax(hop)
    delay_lo, delay_hi = finite_minmax(delay)
    return normalize(delay, delay_lo, delay_hi) + float(hop_weight) * normalize(hop, hop_lo, hop_hi)


def choose_weighted_oracle(hop: np.ndarray, delay: np.ndarray, hop_weight: float) -> np.ndarray:
    cost = scalar_cost_matrix(hop, delay, hop_weight)
    return np.nanargmin(cost, axis=1).astype(np.int32)


def choose_weighted_dp(
    hop: np.ndarray,
    delay: np.ndarray,
    hop_weight: float,
    switch_penalty: float,
) -> np.ndarray:
    cost = scalar_cost_matrix(hop, delay, hop_weight)
    n_steps, n_actions = cost.shape
    dp = np.full((n_steps, n_actions), np.inf, dtype=np.float64)
    prev = np.full((n_steps, n_actions), -1, dtype=np.int32)
    dp[0] = cost[0]
    for t in range(1, n_steps):
        for action in range(n_actions):
            transition = dp[t - 1] + float(switch_penalty) * (np.arange(n_actions) != action)
            best_prev = int(np.argmin(transition))
            dp[t, action] = float(transition[best_prev] + cost[t, action])
            prev[t, action] = best_prev
    chosen = np.empty(n_steps, dtype=np.int32)
    chosen[-1] = int(np.argmin(dp[-1]))
    for t in range(n_steps - 1, 0, -1):
        chosen[t - 1] = int(prev[t, chosen[t]])
    return chosen


def fourier_features(steps: np.ndarray, period_s: float = 86400.0) -> np.ndarray:
    phase = np.asarray(steps, dtype=np.float64) / float(period_s)
    feats = [np.ones_like(phase)]
    for k in range(1, 7):
        feats.append(np.sin(2.0 * np.pi * k * phase))
        feats.append(np.cos(2.0 * np.pi * k * phase))
    return np.column_stack(feats).astype(np.float64)


def train_linear_contextual_bandit(
    *,
    steps: np.ndarray,
    hop: np.ndarray,
    delay: np.ndarray,
    hop_weight: float,
    episodes: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A tiny RL baseline: epsilon-greedy linear Q contextual bandit.

    The state is a Fourier encoding of the orbital time phase and the action is
    one of {motif000040, motif000056}. Reward is the negative normalized
    multi-objective cost. This is intentionally lightweight; it is a sanity
    baseline before moving to GNN/RL over node-level states.
    """

    rng = random.Random(int(seed))
    features = fourier_features(steps)
    cost = scalar_cost_matrix(hop, delay, hop_weight)
    rewards = -cost
    n_steps, n_actions = cost.shape
    weights = np.zeros((n_actions, features.shape[1]), dtype=np.float64)
    lr0 = 0.08
    eps0 = 0.35
    for episode in range(int(episodes)):
        lr = lr0 / math.sqrt(1.0 + episode / 200.0)
        eps = max(0.02, eps0 * (1.0 - episode / max(1, int(episodes))))
        order = list(range(n_steps))
        rng.shuffle(order)
        for t in order:
            q_values = weights @ features[t]
            if rng.random() < eps:
                action = rng.randrange(n_actions)
            else:
                action = int(np.argmax(q_values))
            target = float(rewards[t, action])
            pred = float(q_values[action])
            weights[action] += lr * (target - pred) * features[t]
    q_all = features @ weights.T
    chosen = np.argmax(q_all, axis=1).astype(np.int32)
    oracle = np.argmax(rewards, axis=1).astype(np.int32)
    accuracy = float(np.mean(chosen == oracle))
    meta = {
        "method": "linear_contextual_bandit_q_learning",
        "episodes": int(episodes),
        "seed": int(seed),
        "state_features": "Fourier time phase k=1..6",
        "action_names": list(ACTIONS),
        "oracle_match_accuracy": accuracy,
        "weights": weights.tolist(),
    }
    return chosen, meta


def train_torch_contextual_q(
    *,
    steps: np.ndarray,
    hop: np.ndarray,
    delay: np.ndarray,
    hop_weight: float,
    epochs: int,
    seed: int,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    features_np = fourier_features(steps)
    cost_np = scalar_cost_matrix(hop, delay, hop_weight)
    reward_np = -cost_np
    x = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(reward_np, dtype=torch.float32, device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(features_np.shape[1], 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, len(ACTIONS)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _epoch in range(int(epochs)):
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        q_np = model(x).detach().cpu().numpy()
    chosen = np.argmax(q_np, axis=1).astype(np.int32)
    oracle = np.argmax(reward_np, axis=1).astype(np.int32)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "actions": list(ACTIONS),
            "feature": "Fourier time phase k=1..6",
            "hop_weight": float(hop_weight),
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
        "model_path": str(model_path),
        "final_mse": float(np.mean((q_np - reward_np) ** 2)),
    }
    return chosen, meta


def write_policy_choice_csv(path: Path, *, steps: np.ndarray, chosen: np.ndarray, policy_name: str) -> None:
    rows = [
        {
            "step": int(step),
            "policy": str(policy_name),
            "action_idx": int(action),
            "topology": ACTIONS[int(action)],
        }
        for step, action in zip(steps, chosen)
    ]
    write_csv_rows(path, rows)


def read_policy_choice_csv(path: Path, *, expected_steps: np.ndarray) -> np.ndarray | None:
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


def selected_values(values: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0]), np.asarray(chosen, dtype=np.int32)]


def count_switches(chosen: np.ndarray) -> int:
    if chosen.size <= 1:
        return 0
    return int(np.count_nonzero(chosen[1:] != chosen[:-1]))


def compress_segments(steps: np.ndarray, chosen: np.ndarray, hop_values: np.ndarray, delay_values: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if steps.size == 0:
        return rows
    start = 0
    for idx in range(1, steps.size + 1):
        if idx < steps.size and int(chosen[idx]) == int(chosen[start]):
            continue
        rows.append(
            {
                "segment_id": len(rows),
                "start_step": int(steps[start]),
                "end_step": int(steps[idx - 1]),
                "num_points": int(idx - start),
                "topology": ACTIONS[int(chosen[start])],
                "mean_hops": float(np.mean(hop_values[start:idx])),
                "mean_delay_ms": float(np.mean(delay_values[start:idx])),
            }
        )
        start = idx
    return rows


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
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_policy(name: str, chosen: np.ndarray, hop: np.ndarray, delay: np.ndarray, hop_env: np.ndarray, delay_env: np.ndarray) -> dict[str, Any]:
    hop_values = selected_values(hop, chosen)
    delay_values = selected_values(delay, chosen)
    return {
        "policy": name,
        "mean_hops": float(np.mean(hop_values)),
        "mean_delay_ms": float(np.mean(delay_values)),
        "mean_hop_gap_to_hop_envelope": float(np.mean(hop_values - hop_env)),
        "mean_delay_gap_to_delay_envelope_ms": float(np.mean(delay_values - delay_env)),
        "switches": count_switches(chosen),
        "num_segments": count_switches(chosen) + 1,
        "uses_000040_steps": int(np.count_nonzero(chosen == 0)),
        "uses_000056_steps": int(np.count_nonzero(chosen == 1)),
    }


def plot_series(
    *,
    out_dir: Path,
    steps: np.ndarray,
    hop: np.ndarray,
    delay: np.ndarray,
    policies: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = steps.astype(np.float64) / 3600.0
    colors = {
        TOPO_040: "#dc2626",
        TOPO_056: "#2563eb",
        "hop_envelope": "#111827",
        "delay_envelope": "#111827",
        "eps_2ms": "#059669",
        "weighted_oracle": "#7c3aed",
        "weighted_dp": "#f97316",
        "rl_bandit": "#0f766e",
        "torch_mlp_q": "#1d4ed8",
    }
    labels = {
        TOPO_040: "motif 000040",
        TOPO_056: "motif 000056",
        "hop_envelope": "per-step hop lower envelope",
        "delay_envelope": "per-step delay lower envelope",
        "eps_2ms": "fusion eps=2ms",
        "weighted_oracle": "weighted oracle",
        "weighted_dp": "weighted DP",
        "rl_bandit": "RL contextual bandit",
        "torch_mlp_q": "GPU MLP Q policy",
    }

    hop_lines: dict[str, np.ndarray] = {
        TOPO_040: hop[:, 0],
        TOPO_056: hop[:, 1],
        "hop_envelope": np.min(hop, axis=1),
    }
    delay_lines: dict[str, np.ndarray] = {
        TOPO_040: delay[:, 0],
        TOPO_056: delay[:, 1],
        "delay_envelope": np.min(delay, axis=1),
    }
    for name, chosen in policies.items():
        hop_lines[name] = selected_values(hop, chosen)
        delay_lines[name] = selected_values(delay, chosen)

    for metric_name, lines, ylabel, path_name in (
        ("Mean Shortest Hops", hop_lines, "China-Europe mean shortest hops", "fusion_policy_hops.png"),
        ("Mean Shortest Delay", delay_lines, "China-Europe mean shortest delay (ms)", "fusion_policy_delay_ms.png"),
    ):
        fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=180)
        for name, values in lines.items():
            linewidth = 2.0 if "envelope" in name or name in policies else 1.1
            linestyle = "--" if "envelope" in name else "-"
            alpha = 0.95 if name in policies or "envelope" in name else 0.55
            ax.plot(
                x,
                values,
                label=labels.get(name, name),
                color=colors.get(name, None),
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )
        ax.set_title(f"G60 China-Europe region-internal +grid 040/056 fusion: {metric_name}")
        ax.set_xlabel("time (hour)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(out_dir / path_name)
        plt.close(fig)

    # Tradeoff diagnostic.
    dhop = hop[:, 0] - hop[:, 1]
    ddelay = delay[:, 0] - delay[:, 1]
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=180)
    ax.scatter(ddelay, dhop, s=10, c=x, cmap="viridis", alpha=0.75)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel("delay(000040) - delay(000056) ms")
    ax.set_ylabel("hops(000040) - hops(000056)")
    ax.set_title("040 vs 056 per-step tradeoff")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    fig.tight_layout()
    fig.savefig(out_dir / "fusion_tradeoff_scatter.png")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = read_metric_series(Path(args.metric_dir))
    hop, delay = action_matrix(series)
    hop_env = np.nanmin(hop, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_arg = np.nanargmin(hop, axis=1).astype(np.int32)
    delay_arg = np.nanargmin(delay, axis=1).astype(np.int32)

    policies: dict[str, np.ndarray] = {}
    for eps in args.delay_eps_ms:
        label = f"eps_{float(eps):g}ms".replace(".", "p")
        policies[label] = choose_delay_epsilon_hop_policy(hop, delay, float(eps))
    policies["eps_2ms"] = choose_delay_epsilon_hop_policy(hop, delay, 2.0)
    policies["weighted_oracle"] = choose_weighted_oracle(hop, delay, float(args.hop_weight))
    policies["weighted_dp"] = choose_weighted_dp(
        hop,
        delay,
        float(args.hop_weight),
        float(args.switch_penalty),
    )
    rl_chosen, rl_meta = train_linear_contextual_bandit(
        steps=series.steps,
        hop=hop,
        delay=delay,
        hop_weight=float(args.hop_weight),
        episodes=int(args.rl_episodes),
        seed=int(args.rl_seed),
    )
    policies["rl_bandit"] = rl_chosen
    torch_meta: dict[str, Any] | None = None
    torch_policy_csv = out_dir / "torch_mlp_q_policy.csv"
    torch_meta_path = out_dir / "torch_mlp_q_meta.json"
    if int(args.torch_epochs) > 0:
        try:
            torch_chosen, torch_meta = train_torch_contextual_q(
                steps=series.steps,
                hop=hop,
                delay=delay,
                hop_weight=float(args.hop_weight),
                epochs=int(args.torch_epochs),
                seed=int(args.rl_seed),
                model_path=out_dir / "models" / "torch_mlp_q.pt",
            )
            policies["torch_mlp_q"] = torch_chosen
            write_policy_choice_csv(torch_policy_csv, steps=series.steps, chosen=torch_chosen, policy_name="torch_mlp_q")
            torch_meta_path.write_text(json.dumps(torch_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            torch_meta = {"error": repr(exc), "requested_epochs": int(args.torch_epochs)}
    else:
        torch_chosen = read_policy_choice_csv(torch_policy_csv, expected_steps=series.steps)
        if torch_chosen is not None:
            policies["torch_mlp_q"] = torch_chosen
            if torch_meta_path.exists():
                try:
                    torch_meta = json.loads(torch_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    torch_meta = {"warning": f"failed to load {torch_meta_path}"}

    diagnostic_rows: list[dict[str, Any]] = []
    for row, step in enumerate(series.steps):
        h40, h56 = hop[row]
        d40, d56 = delay[row]
        relation = "tradeoff"
        if h40 <= h56 and d40 <= d56 and (h40 < h56 or d40 < d56):
            relation = "040_dominates"
        elif h56 <= h40 and d56 <= d40 and (h56 < h40 or d56 < d40):
            relation = "056_dominates"
        elif int(hop_arg[row]) == int(delay_arg[row]):
            relation = "same_metric_winner"
        diagnostic_rows.append(
            {
                "step": int(step),
                "hour": float(step / 3600.0),
                "hops_000040": float(h40),
                "hops_000056": float(h56),
                "delay_ms_000040": float(d40),
                "delay_ms_000056": float(d56),
                "hop_lower_envelope": float(hop_env[row]),
                "delay_lower_envelope_ms": float(delay_env[row]),
                "hop_winner": ACTIONS[int(hop_arg[row])],
                "delay_winner": ACTIONS[int(delay_arg[row])],
                "winner_conflict": bool(int(hop_arg[row]) != int(delay_arg[row])),
                "pareto_relation": relation,
            }
        )
    write_csv_rows(out_dir / "fusion_feasibility_by_step.csv", diagnostic_rows)

    summary_rows: list[dict[str, Any]] = [
        {
            "policy": TOPO_040,
            "mean_hops": float(np.mean(hop[:, 0])),
            "mean_delay_ms": float(np.mean(delay[:, 0])),
            "mean_hop_gap_to_hop_envelope": float(np.mean(hop[:, 0] - hop_env)),
            "mean_delay_gap_to_delay_envelope_ms": float(np.mean(delay[:, 0] - delay_env)),
            "switches": 0,
            "num_segments": 1,
            "uses_000040_steps": int(hop.shape[0]),
            "uses_000056_steps": 0,
        },
        {
            "policy": TOPO_056,
            "mean_hops": float(np.mean(hop[:, 1])),
            "mean_delay_ms": float(np.mean(delay[:, 1])),
            "mean_hop_gap_to_hop_envelope": float(np.mean(hop[:, 1] - hop_env)),
            "mean_delay_gap_to_delay_envelope_ms": float(np.mean(delay[:, 1] - delay_env)),
            "switches": 0,
            "num_segments": 1,
            "uses_000040_steps": 0,
            "uses_000056_steps": int(hop.shape[0]),
        },
        {
            "policy": "hop_lower_envelope",
            "mean_hops": float(np.mean(hop_env)),
            "mean_delay_ms": float(np.mean(selected_values(delay, hop_arg))),
            "mean_hop_gap_to_hop_envelope": 0.0,
            "mean_delay_gap_to_delay_envelope_ms": float(np.mean(selected_values(delay, hop_arg) - delay_env)),
            "switches": count_switches(hop_arg),
            "num_segments": count_switches(hop_arg) + 1,
            "uses_000040_steps": int(np.count_nonzero(hop_arg == 0)),
            "uses_000056_steps": int(np.count_nonzero(hop_arg == 1)),
        },
        {
            "policy": "delay_lower_envelope",
            "mean_hops": float(np.mean(selected_values(hop, delay_arg))),
            "mean_delay_ms": float(np.mean(delay_env)),
            "mean_hop_gap_to_hop_envelope": float(np.mean(selected_values(hop, delay_arg) - hop_env)),
            "mean_delay_gap_to_delay_envelope_ms": 0.0,
            "switches": count_switches(delay_arg),
            "num_segments": count_switches(delay_arg) + 1,
            "uses_000040_steps": int(np.count_nonzero(delay_arg == 0)),
            "uses_000056_steps": int(np.count_nonzero(delay_arg == 1)),
        },
    ]
    for name, chosen in policies.items():
        summary_rows.append(summarize_policy(name, chosen, hop, delay, hop_env, delay_env))
    write_csv_rows(out_dir / "fusion_policy_summary.csv", summary_rows)

    by_step_rows: list[dict[str, Any]] = []
    for row, step in enumerate(series.steps):
        item: dict[str, Any] = {
            "step": int(step),
            "hour": float(step / 3600.0),
            "hops_000040": float(hop[row, 0]),
            "hops_000056": float(hop[row, 1]),
            "delay_ms_000040": float(delay[row, 0]),
            "delay_ms_000056": float(delay[row, 1]),
            "hop_lower_envelope": float(hop_env[row]),
            "delay_lower_envelope_ms": float(delay_env[row]),
        }
        for name, chosen in policies.items():
            action = int(chosen[row])
            item[f"{name}_topology"] = ACTIONS[action]
            item[f"{name}_hops"] = float(hop[row, action])
            item[f"{name}_delay_ms"] = float(delay[row, action])
        by_step_rows.append(item)
    write_csv_rows(out_dir / "fusion_policy_by_step.csv", by_step_rows)

    segment_root = out_dir / "segments"
    for name, chosen in policies.items():
        write_csv_rows(
            segment_root / f"{name}_segments.csv",
            compress_segments(series.steps, chosen, selected_values(hop, chosen), selected_values(delay, chosen)),
        )

    relation_counts: dict[str, int] = {}
    for row in diagnostic_rows:
        relation = str(row["pareto_relation"])
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    conflict_count = int(np.count_nonzero(hop_arg != delay_arg))
    meta = {
        "metric_dir": str(Path(args.metric_dir)),
        "out_dir": str(out_dir),
        "num_steps": int(series.steps.size),
        "actions": list(ACTIONS),
        "hop_delay_winner_conflict_steps": conflict_count,
        "hop_delay_winner_conflict_ratio": float(conflict_count / max(1, series.steps.size)),
        "pareto_relation_counts": relation_counts,
        "hop_weight": float(args.hop_weight),
        "switch_penalty": float(args.switch_penalty),
        "delay_eps_ms": [float(x) for x in args.delay_eps_ms],
        "rl_meta": rl_meta,
        "torch_meta": torch_meta,
        "summary_csv": str(out_dir / "fusion_policy_summary.csv"),
        "by_step_csv": str(out_dir / "fusion_policy_by_step.csv"),
    }
    (out_dir / "fusion_policy_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not bool(args.skip_plots):
        plot_policies = {
            "eps_2ms": policies["eps_2ms"],
            "weighted_oracle": policies["weighted_oracle"],
            "weighted_dp": policies["weighted_dp"],
            "rl_bandit": policies["rl_bandit"],
        }
        if "torch_mlp_q" in policies:
            plot_policies["torch_mlp_q"] = policies["torch_mlp_q"]
        plot_series(
            out_dir=out_dir,
            steps=series.steps,
            hop=hop,
            delay=delay,
            policies=plot_policies,
        )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
