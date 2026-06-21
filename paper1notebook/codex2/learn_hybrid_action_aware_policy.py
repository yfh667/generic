from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from learn_hybrid_candidate_policy import (
    DEFAULT_GROUP_CACHE,
    DEFAULT_HYBRID_DIR,
    build_state_features,
    choose_delay_eps_policy,
    count_switches,
    normalized_gap_cost,
    read_metrics,
    selected_values,
    write_csv_rows,
)


DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_action_aware_policy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train action-aware neural policies over 000040/000056 hybrid candidates. "
            "Each candidate is represented by its band/action semantics instead of only an action index."
        )
    )
    parser.add_argument("--hybrid-dir", type=Path, default=DEFAULT_HYBRID_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-cache", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group-id", type=int, default=2)
    parser.add_argument("--target-group-id", type=int, default=3)
    parser.add_argument("--feature-mode", choices=("time", "group"), default="group")
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--hop-weight", type=float, default=1.0)
    parser.add_argument("--delay-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=667)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def read_candidate_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        if not name:
            continue
        item: dict[str, Any] = dict(row)
        for key in ("candidate_idx", "band_start", "band_end", "band_len", "changed_edges"):
            value = item.get(key)
            item[key] = int(float(value)) if value not in ("", None) else None
        out[name] = item
    return out


def _band_indices(start: int | None, length: int | None, n: int) -> list[int]:
    if start is None or length is None or int(length) <= 0:
        return []
    return [int((int(start) + offset) % int(n)) for offset in range(int(length))]


def _band_indices_from_name(name: str, n: int) -> list[int]:
    out: set[int] = set()
    for match in re.finditer(r"y(\d{2})_(\d{2})(?:_l(\d{2}))?", str(name)):
        start = int(match.group(1))
        if match.group(3) is not None:
            length = int(match.group(3))
            out.update(_band_indices(start, length, int(n)))
            continue
        end = int(match.group(2))
        if start <= end:
            out.update(range(start, end + 1))
        else:
            out.update(list(range(start, int(n))) + list(range(0, end + 1)))
    return sorted(out)


def _candidate_kind(name: str) -> str:
    if name == "pure_000056":
        return "pure56"
    if name == "pure_000040":
        return "pure40"
    if name.startswith("hybrid_"):
        return "hybrid"
    return "other"


def build_candidate_features(
    *,
    names: Sequence[str],
    summary: dict[str, dict[str, Any]],
    n: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    static_rows: list[np.ndarray] = []
    band_masks: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    max_changed = max(
        [float(row.get("changed_edges") or 0) for row in summary.values()] + [1.0]
    )
    for name in names:
        row = summary.get(str(name), {})
        kind = _candidate_kind(str(name))
        band_start = row.get("band_start")
        band_len = row.get("band_len")
        indices = _band_indices_from_name(str(name), int(n))
        if not indices and band_start is None:
            match = re.search(r"_y(\d{2})_(\d{2})_l(\d{2})", str(name))
            if match:
                band_start = int(match.group(1))
                band_len = int(match.group(3))
        if not indices:
            indices = _band_indices(band_start, band_len, int(n))
        mask = np.zeros(int(n), dtype=np.float64)
        for idx in indices:
            mask[idx] = 1.0
        if indices:
            center = (float(indices[0]) + (len(indices) - 1) / 2.0) % float(n)
            start_phase = float(indices[0]) / float(n) * 2.0 * np.pi
            center_phase = center / float(n) * 2.0 * np.pi
        else:
            start_phase = 0.0
            center_phase = 0.0
        kind_vec = np.asarray(
            [
                1.0 if kind == "pure56" else 0.0,
                1.0 if kind == "pure40" else 0.0,
                1.0 if kind == "hybrid" else 0.0,
            ],
            dtype=np.float64,
        )
        static = np.concatenate(
            [
                kind_vec,
                np.asarray(
                    [
                        math.sin(start_phase),
                        math.cos(start_phase),
                        math.sin(center_phase),
                        math.cos(center_phase),
                        len(indices) / float(max(1, int(n))),
                        float(row.get("changed_edges") or 0) / float(max_changed),
                    ],
                    dtype=np.float64,
                ),
                mask,
            ]
        )
        static_rows.append(static)
        band_masks.append(mask)
        meta_rows.append(
            {
                "name": str(name),
                "kind": kind,
                "band_start": int(band_start) if band_start is not None else "",
                "band_len": int(band_len) if band_len is not None else 0,
                "band_indices": ",".join(str(x) for x in indices),
            }
        )
    return np.asarray(static_rows, dtype=np.float64), np.asarray(band_masks, dtype=np.float64), meta_rows


def _histogram(values: list[int], size: int) -> np.ndarray:
    out = np.zeros(int(size), dtype=np.float64)
    for value in values:
        out[int(value) % int(size)] += 1.0
    total = float(np.sum(out))
    return out / total if total > 0 else out


def _lookup_group_nodes(
    groups_by_step: dict[str, Any],
    step: int,
    row_idx: int,
    group_id: int,
    *,
    start: int,
    stride: int,
) -> list[int]:
    keys = [str(int(step)), str(int(row_idx))]
    if int(stride) > 0:
        keys.append(str(int((int(step) - int(start)) // int(stride))))
    for key in keys:
        payload = groups_by_step.get(key)
        if isinstance(payload, dict):
            nodes = payload.get(str(int(group_id)), payload.get(int(group_id), []))
            return [int(x) for x in (nodes or [])]
    return []


def load_group_y_hists(
    *,
    steps: np.ndarray,
    group_cache: Path,
    source_group_id: int,
    target_group_id: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    with Path(group_cache).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    source = payload.get("source", {}) if isinstance(payload, dict) else {}
    groups_by_step = payload.get("groups_by_step", {}) if isinstance(payload, dict) else {}
    start = int(source.get("start", int(np.min(steps))))
    stride = int(source.get("stride", 1))
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for row_idx, step in enumerate(steps):
        src_nodes = _lookup_group_nodes(
            groups_by_step,
            int(step),
            int(row_idx),
            int(source_group_id),
            start=start,
            stride=stride,
        )
        dst_nodes = _lookup_group_nodes(
            groups_by_step,
            int(step),
            int(row_idx),
            int(target_group_id),
            start=start,
            stride=stride,
        )
        source_rows.append(_histogram([node % int(n) for node in src_nodes], int(n)))
        target_rows.append(_histogram([node % int(n) for node in dst_nodes], int(n)))
    return np.asarray(source_rows, dtype=np.float64), np.asarray(target_rows, dtype=np.float64)


def build_pair_features(
    *,
    state_features: np.ndarray,
    candidate_features: np.ndarray,
    band_masks: np.ndarray,
    source_y_hist: np.ndarray,
    target_y_hist: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_steps, state_dim = state_features.shape
    n_actions, action_dim = candidate_features.shape
    pairs = np.zeros((n_steps, n_actions, state_dim + action_dim + 6), dtype=np.float64)
    for action in range(n_actions):
        mask = band_masks[action]
        src_overlap = source_y_hist @ mask
        dst_overlap = target_y_hist @ mask
        any_overlap = np.maximum(src_overlap, dst_overlap)
        both_overlap = src_overlap * dst_overlap
        outside_src = 1.0 - src_overlap
        outside_dst = 1.0 - dst_overlap
        interaction = np.column_stack(
            [src_overlap, dst_overlap, any_overlap, both_overlap, outside_src, outside_dst]
        )
        pairs[:, action, :] = np.concatenate(
            [
                state_features,
                np.repeat(candidate_features[action][None, :], n_steps, axis=0),
                interaction,
            ],
            axis=1,
        )
    flat = pairs.reshape(n_steps * n_actions, -1)
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std[std < 1e-8] = 1.0
    pairs = ((pairs - mean) / std).astype(np.float32)
    meta = {
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "interaction_dim": 6,
        "pair_dim": int(pairs.shape[-1]),
    }
    return pairs, meta


def train_pair_scorer(
    *,
    pair_features: np.ndarray,
    rewards: np.ndarray,
    labels: dict[str, np.ndarray],
    epochs: int,
    seed: int,
    model_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(pair_features, dtype=torch.float32, device=device)
    y_reward = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    n_steps, n_actions, feat_dim = pair_features.shape

    def make_model() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 1),
        ).to(device)

    policies: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "",
        "epochs": int(epochs),
        "seed": int(seed),
    }
    model_dir.mkdir(parents=True, exist_ok=True)

    q_model = make_model()
    opt = torch.optim.AdamW(q_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    for _ in range(int(epochs)):
        pred = q_model(x).squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, y_reward)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        q_pred = q_model(x).squeeze(-1).detach().cpu().numpy()
    policies["action_aware_q"] = np.argmax(q_pred, axis=1).astype(np.int32)
    meta["action_aware_q"] = {
        "final_mse": float(np.mean((q_pred - rewards) ** 2)),
        "oracle_match_accuracy": float(np.mean(policies["action_aware_q"] == np.argmax(rewards, axis=1))),
    }
    torch.save(q_model.state_dict(), model_dir / "action_aware_q.pt")

    for label_name, label_np in labels.items():
        label = torch.as_tensor(label_np, dtype=torch.long, device=device)
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        for _ in range(int(epochs)):
            logits = model(x).squeeze(-1)
            loss = torch.nn.functional.cross_entropy(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits_np = model(x).squeeze(-1).detach().cpu().numpy()
        chosen = np.argmax(logits_np, axis=1).astype(np.int32)
        policy_name = f"action_aware_ce_{label_name}"
        policies[policy_name] = chosen
        meta[policy_name] = {
            "train_label_accuracy": float(np.mean(chosen == label_np)),
        }
        torch.save(model.state_dict(), model_dir / f"{policy_name}.pt")
    return policies, meta


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


def write_policy_choice(
    path: Path,
    *,
    steps: np.ndarray,
    chosen: np.ndarray,
    names: Sequence[str],
    policy: str,
    hop: np.ndarray,
    delay: np.ndarray,
) -> None:
    write_csv_rows(
        path,
        [
            {
                "step": int(step),
                "policy": str(policy),
                "action_idx": int(action),
                "candidate": str(names[int(action)]),
                "mean_hops": float(hop[row, int(action)]),
                "mean_delay_ms": float(delay[row, int(action)]),
            }
            for row, (step, action) in enumerate(zip(steps, chosen))
        ],
    )


def read_saved_policies(out_dir: Path, steps: np.ndarray) -> dict[str, np.ndarray]:
    policies: dict[str, np.ndarray] = {}
    for path in sorted((out_dir / "policies").glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != int(steps.size):
            continue
        row_steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
        if not np.array_equal(row_steps, steps):
            continue
        policies[path.stem] = np.asarray([int(row["action_idx"]) for row in rows], dtype=np.int32)
    return policies


def plot_results(
    *,
    out_dir: Path,
    steps: np.ndarray,
    names: Sequence[str],
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
    color_map = {
        "pure_000056": "#9ca3af",
        "pure_000040": "#6b7280",
        "pure_envelope": "#111827",
        "hybrid_envelope": "#059669",
        "scalar_oracle": "#0f766e",
        "eps_1ms": "#f97316",
        "eps_2ms": "#7c3aed",
        "dual_envelope_balanced": "#0ea5e9",
        "action_aware_q": "#2563eb",
        "action_aware_ce_scalar": "#dc2626",
        "action_aware_ce_eps1": "#be123c",
        "action_aware_ce_eps2": "#9333ea",
        "action_aware_ce_lagrange_best_both": "#22c55e",
        "action_aware_ce_lagrange_delay_safe_best_hops": "#06b6d4",
        "action_aware_ce_lagrange_hop_safe_best_delay": "#eab308",
        "action_aware_ce_dual_envelope_balanced": "#0284c7",
        "lagrange_best_both": "#16a34a",
        "lagrange_delay_safe_best_hops": "#0891b2",
        "lagrange_hop_safe_best_delay": "#ca8a04",
    }
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for values, pure_env, hybrid_env, ylabel, filename in (
        (hop, pure_hop_env, hybrid_hop_env, "China-Europe mean shortest hops", "action_aware_policy_hops.png"),
        (delay, pure_delay_env, hybrid_delay_env, "China-Europe mean shortest delay (ms)", "action_aware_policy_delay_ms.png"),
    ):
        fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=180)
        ax.plot(x, pure_env, color=color_map["pure_envelope"], linestyle="--", linewidth=1.9, label="pure 040/056 envelope")
        ax.plot(x, hybrid_env, color=color_map["hybrid_envelope"], linestyle="--", linewidth=2.0, label="hybrid candidate envelope")
        for candidate in ("pure_000056", "pure_000040"):
            if candidate in name_to_idx:
                ax.plot(
                    x,
                    values[:, name_to_idx[candidate]],
                    color=color_map[candidate],
                    linewidth=0.9,
                    alpha=0.55,
                    label=candidate,
                )
        for policy_name in (
            "scalar_oracle",
            "eps_1ms",
            "eps_2ms",
            "dual_envelope_balanced",
            "lagrange_best_both",
            "lagrange_delay_safe_best_hops",
            "lagrange_hop_safe_best_delay",
            "action_aware_q",
            "action_aware_ce_dual_envelope_balanced",
            "action_aware_ce_lagrange_best_both",
            "action_aware_ce_lagrange_delay_safe_best_hops",
            "action_aware_ce_lagrange_hop_safe_best_delay",
            "action_aware_ce_scalar",
            "action_aware_ce_eps1",
            "action_aware_ce_eps2",
        ):
            if policy_name not in policies:
                continue
            ax.plot(
                x,
                selected_values(values, policies[policy_name]),
                linewidth=1.55,
                alpha=0.9,
                color=color_map.get(policy_name),
                label=policy_name,
            )
        ax.set_title(f"G60 China-Europe action-aware hybrid policy: {filename.replace('action_aware_policy_', '').replace('.png', '')}")
        ax.set_xlabel("time (hour)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)


def write_lagrange_pareto(
    *,
    out_dir: Path,
    hop: np.ndarray,
    delay: np.ndarray,
    pure_hop_env: np.ndarray,
    pure_delay_env: np.ndarray,
    make_plot: bool,
) -> dict[str, Any]:
    ref_hop = float(np.mean(pure_hop_env))
    ref_delay = float(np.mean(pure_delay_env))
    weights = np.concatenate([np.linspace(0.0, 2.0, 801), np.geomspace(1e-4, 100.0, 2000)])
    points: list[tuple[float, float, float, str]] = []
    seen: set[tuple[float, float]] = set()
    for weight in weights:
        for mode, values in (
            ("hop_plus_weight_delay", hop + float(weight) * delay),
            ("delay_plus_weight_hop", delay + float(weight) * hop),
        ):
            chosen = np.argmin(values, axis=1)
            mean_hops = float(np.mean(selected_values(hop, chosen)))
            mean_delay = float(np.mean(selected_values(delay, chosen)))
            key = (round(mean_hops, 9), round(mean_delay, 9))
            if key in seen:
                continue
            seen.add(key)
            points.append((mean_hops, mean_delay, float(weight), mode))
    points.sort()
    feasible = [point for point in points if point[0] < ref_hop and point[1] < ref_delay]
    best_under_delay = [point for point in points if point[1] <= ref_delay]
    best_under_hop = [point for point in points if point[0] <= ref_hop]
    csv_path = out_dir / "candidate_lagrange_pareto.csv"
    write_csv_rows(
        csv_path,
        [
            {
                "mean_hops": point[0],
                "mean_delay_ms": point[1],
                "weight": point[2],
                "mode": point[3],
                "beats_both_pure_envelope": bool(point[0] < ref_hop and point[1] < ref_delay),
            }
            for point in points
        ],
    )
    plot_path = out_dir / "candidate_lagrange_pareto.png"
    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=12, alpha=0.65)
        ax.axvline(ref_hop, color="black", linestyle="--", linewidth=1.2, label="pure hop envelope mean")
        ax.axhline(ref_delay, color="black", linestyle=":", linewidth=1.2, label="pure delay envelope mean")
        ax.set_xlabel("mean hops")
        ax.set_ylabel("mean delay (ms)")
        ax.set_title("Feasible tradeoff within current candidate pool")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
    return {
        "csv": str(csv_path),
        "plot": str(plot_path) if make_plot else "",
        "num_lagrange_points": int(len(points)),
        "num_points_beating_both_pure_envelopes": int(len(feasible)),
        "best_hops_with_delay_not_worse_than_pure": {
            "mean_hops": float(min(best_under_delay, key=lambda x: x[0])[0]) if best_under_delay else None,
            "mean_delay_ms": float(min(best_under_delay, key=lambda x: x[0])[1]) if best_under_delay else None,
        },
        "best_delay_with_hops_not_worse_than_pure": {
            "mean_hops": float(min(best_under_hop, key=lambda x: x[1])[0]) if best_under_hop else None,
            "mean_delay_ms": float(min(best_under_hop, key=lambda x: x[1])[1]) if best_under_hop else None,
        },
    }


def choose_lagrange_policies(
    *,
    hop: np.ndarray,
    delay: np.ndarray,
    pure_hop_env: np.ndarray,
    pure_delay_env: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ref_hop = float(np.mean(pure_hop_env))
    ref_delay = float(np.mean(pure_delay_env))
    weights = np.concatenate([np.linspace(0.0, 2.0, 801), np.geomspace(1e-4, 100.0, 2000)])
    records: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for weight in weights:
        for mode, values in (
            ("hop_plus_weight_delay", hop + float(weight) * delay),
            ("delay_plus_weight_hop", delay + float(weight) * hop),
        ):
            chosen = np.argmin(values, axis=1).astype(np.int32)
            mean_hops = float(np.mean(selected_values(hop, chosen)))
            mean_delay = float(np.mean(selected_values(delay, chosen)))
            key = (round(mean_hops, 9), round(mean_delay, 9))
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "mean_hops": mean_hops,
                    "mean_delay_ms": mean_delay,
                    "weight": float(weight),
                    "mode": mode,
                    "chosen": chosen,
                }
            )
    policies: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}

    feasible = [r for r in records if r["mean_hops"] < ref_hop and r["mean_delay_ms"] < ref_delay]
    if feasible:
        # Balanced within the strictly better region: maximize the smaller normalized improvement.
        def balanced_key(row: dict[str, Any]) -> tuple[float, float]:
            hop_improve = (ref_hop - float(row["mean_hops"])) / max(1e-9, ref_hop)
            delay_improve = (ref_delay - float(row["mean_delay_ms"])) / max(1e-9, ref_delay)
            return (min(hop_improve, delay_improve), hop_improve + delay_improve)

        best = max(feasible, key=balanced_key)
        policies["lagrange_best_both"] = np.asarray(best["chosen"], dtype=np.int32)
        meta["lagrange_best_both"] = {
            "mean_hops": float(best["mean_hops"]),
            "mean_delay_ms": float(best["mean_delay_ms"]),
            "weight": float(best["weight"]),
            "mode": str(best["mode"]),
        }

    under_delay = [r for r in records if r["mean_delay_ms"] <= ref_delay]
    if under_delay:
        best = min(under_delay, key=lambda r: float(r["mean_hops"]))
        policies["lagrange_delay_safe_best_hops"] = np.asarray(best["chosen"], dtype=np.int32)
        meta["lagrange_delay_safe_best_hops"] = {
            "mean_hops": float(best["mean_hops"]),
            "mean_delay_ms": float(best["mean_delay_ms"]),
            "weight": float(best["weight"]),
            "mode": str(best["mode"]),
        }

    under_hop = [r for r in records if r["mean_hops"] <= ref_hop]
    if under_hop:
        best = min(under_hop, key=lambda r: float(r["mean_delay_ms"]))
        policies["lagrange_hop_safe_best_delay"] = np.asarray(best["chosen"], dtype=np.int32)
        meta["lagrange_hop_safe_best_delay"] = {
            "mean_hops": float(best["mean_hops"]),
            "mean_delay_ms": float(best["mean_delay_ms"]),
            "weight": float(best["weight"]),
            "mode": str(best["mode"]),
        }

    return policies, meta


def choose_dual_envelope_balanced_policy(hop: np.ndarray, delay: np.ndarray) -> np.ndarray:
    hop_env = np.nanmin(hop, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_lo = float(np.nanmin(hop))
    hop_hi = float(np.nanmax(hop))
    delay_lo = float(np.nanmin(delay))
    delay_hi = float(np.nanmax(delay))
    hop_scale = max(1e-9, hop_hi - hop_lo)
    delay_scale = max(1e-9, delay_hi - delay_lo)
    hop_gap = (hop - hop_env[:, None]) / hop_scale
    delay_gap = (delay - delay_env[:, None]) / delay_scale
    balanced_gap = np.maximum(hop_gap, delay_gap)
    return np.nanargmin(balanced_gap, axis=1).astype(np.int32)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "action_aware_policy_meta.json"
    previous_train_meta: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            previous_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(previous_meta, dict) and previous_meta.get("train_meta") is not None:
                previous_train_meta = previous_meta.get("train_meta")
        except Exception:
            previous_train_meta = None
    steps, names, hop, delay = read_metrics(Path(args.hybrid_dir))
    state_features, state_meta = build_state_features(
        steps=steps,
        feature_mode=str(args.feature_mode),
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        p=int(args.p),
        n=int(args.n),
    )
    summary = read_candidate_summary(Path(args.hybrid_dir) / "candidate_full_summary.csv")
    candidate_features, band_masks, candidate_meta = build_candidate_features(
        names=names,
        summary=summary,
        n=int(args.n),
    )
    source_y_hist, target_y_hist = load_group_y_hists(
        steps=steps,
        group_cache=Path(args.group_cache),
        source_group_id=int(args.source_group_id),
        target_group_id=int(args.target_group_id),
        n=int(args.n),
    )
    pair_features, pair_meta = build_pair_features(
        state_features=state_features,
        candidate_features=candidate_features,
        band_masks=band_masks,
        source_y_hist=source_y_hist,
        target_y_hist=target_y_hist,
    )
    pure_indices = [idx for idx, name in enumerate(names) if str(name).startswith("pure_")]
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
    policies: dict[str, np.ndarray] = {
        "scalar_oracle": scalar_oracle,
        "eps_1ms": choose_delay_eps_policy(hop, delay, 1.0),
        "eps_2ms": choose_delay_eps_policy(hop, delay, 2.0),
        "dual_envelope_balanced": choose_dual_envelope_balanced_policy(hop, delay),
    }
    lagrange_policies, lagrange_policy_meta = choose_lagrange_policies(
        hop=hop,
        delay=delay,
        pure_hop_env=pure_hop_env,
        pure_delay_env=pure_delay_env,
    )
    policies.update(lagrange_policies)
    labels = {
        "scalar": scalar_oracle,
        "eps1": policies["eps_1ms"],
        "eps2": policies["eps_2ms"],
        "dual_envelope_balanced": policies["dual_envelope_balanced"],
    }
    for policy_name, chosen in lagrange_policies.items():
        labels[policy_name.replace("lagrange_", "lagrange_")] = chosen
    train_meta: dict[str, Any] | None = None
    if int(args.epochs) > 0:
        learned, train_meta = train_pair_scorer(
            pair_features=pair_features,
            rewards=rewards,
            labels=labels,
            epochs=int(args.epochs),
            seed=int(args.seed),
            model_dir=out_dir / "models",
        )
        policies.update(learned)
        for policy_name, chosen in learned.items():
            write_policy_choice(
                out_dir / "policies" / f"{policy_name}.csv",
                steps=steps,
                chosen=chosen,
                names=names,
                policy=policy_name,
                hop=hop,
                delay=delay,
            )
    else:
        policies.update(read_saved_policies(out_dir, steps))
        train_meta = previous_train_meta

    for policy_name in ("scalar_oracle", "eps_1ms", "eps_2ms", "dual_envelope_balanced", *lagrange_policies.keys()):
        write_policy_choice(
            out_dir / "policies" / f"{policy_name}.csv",
            steps=steps,
            chosen=policies[policy_name],
            names=names,
            policy=policy_name,
            hop=hop,
            delay=delay,
        )

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
        summarize_policy(
            name="hybrid_hop_envelope_reference",
            chosen=hop_arg,
            candidate_names=names,
            hop=hop,
            delay=delay,
            hybrid_hop_env=hybrid_hop_env,
            hybrid_delay_env=hybrid_delay_env,
            pure_hop_env=pure_hop_env,
            pure_delay_env=pure_delay_env,
        ),
        summarize_policy(
            name="hybrid_delay_envelope_reference",
            chosen=delay_arg,
            candidate_names=names,
            hop=hop,
            delay=delay,
            hybrid_hop_env=hybrid_hop_env,
            hybrid_delay_env=hybrid_delay_env,
            pure_hop_env=pure_hop_env,
            pure_delay_env=pure_delay_env,
        ),
    ]
    for policy_name, chosen in policies.items():
        summary_rows.append(
            summarize_policy(
                name=policy_name,
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
    write_csv_rows(out_dir / "action_aware_policy_summary.csv", summary_rows)
    write_csv_rows(out_dir / "candidate_action_features.csv", candidate_meta)
    pareto_meta = write_lagrange_pareto(
        out_dir=out_dir,
        hop=hop,
        delay=delay,
        pure_hop_env=pure_hop_env,
        pure_delay_env=pure_delay_env,
        make_plot=not bool(args.skip_plots),
    )

    meta = {
        "hybrid_dir": str(Path(args.hybrid_dir)),
        "out_dir": str(out_dir),
        "num_steps": int(steps.size),
        "num_candidates": int(len(names)),
        "candidate_names": list(names),
        "state_meta": state_meta,
        "pair_meta": pair_meta,
        "train_meta": train_meta,
        "mean_pure_hop_envelope": float(np.mean(pure_hop_env)),
        "mean_pure_delay_envelope_ms": float(np.mean(pure_delay_env)),
        "mean_hybrid_hop_envelope": float(np.mean(hybrid_hop_env)),
        "mean_hybrid_delay_envelope_ms": float(np.mean(hybrid_delay_env)),
        "hop_delay_winner_conflict_steps": int(np.count_nonzero(hop_arg != delay_arg)),
        "lagrange_pareto": pareto_meta,
        "lagrange_policy_meta": lagrange_policy_meta,
        "summary_csv": str(out_dir / "action_aware_policy_summary.csv"),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not bool(args.skip_plots):
        plot_results(
            out_dir=out_dir,
            steps=steps,
            names=names,
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
