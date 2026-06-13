from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path(
    r"E:\paper11\data\linshi\g60_w4h3_selected100_shortest_delay_t0_86164_stride60"
)
DEFAULT_INPUT_CSV = DEFAULT_INPUT_DIR / "compare_100motifs_gridplus_full_link.csv"
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\linshi\g60_w4h3_dynamic_schedule_selected100_t0_86164_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dynamic motif schedules from a precomputed delay comparison CSV. "
            "The optimizer selects among motif_* columns only; full_link/gridplus are references."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--motif-prefix", default="motif_")
    parser.add_argument(
        "--min-dwell-minutes",
        nargs="*",
        type=float,
        default=[10.0, 30.0, 60.0, 120.0],
        help="Minimum time that a selected motif must remain active.",
    )
    return parser.parse_args()


def motif_id_from_name(name: str) -> int | None:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def load_motif_info(input_csv: Path) -> dict[str, dict[str, str]]:
    info_csv = input_csv.with_name("selected_motifs_original_rows.csv")
    if not info_csv.exists():
        return {}
    df = pd.read_csv(info_csv)
    info: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        motif_name = f"motif_{int(row.motif_id):06d}"
        info[motif_name] = {
            "motif_id": str(int(row.motif_id)),
            "motif": str(row.motif),
            "edges": str(row.edges),
        }
    return info


def infer_sample_seconds(steps: np.ndarray) -> float:
    if steps.size < 2:
        return 1.0
    diffs = np.diff(steps.astype(float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def compress_segments(
    steps: np.ndarray,
    topology_by_index: np.ndarray,
    delay_by_step: np.ndarray,
    motif_columns: list[str],
    motif_info: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    if len(steps) == 0:
        return pd.DataFrame(rows)

    start_idx = 0
    for idx in range(1, len(steps) + 1):
        if idx < len(steps) and topology_by_index[idx] == topology_by_index[start_idx]:
            continue
        motif_name = motif_columns[int(topology_by_index[start_idx])]
        meta = motif_info.get(motif_name, {})
        segment_delay = delay_by_step[start_idx:idx]
        rows.append(
            {
                "segment_id": len(rows),
                "start_index": start_idx,
                "end_index_exclusive": idx,
                "start_step": int(steps[start_idx]),
                "end_step": int(steps[idx - 1]),
                "num_points": idx - start_idx,
                "topology": motif_name,
                "motif_id": meta.get("motif_id", motif_id_from_name(motif_name)),
                "motif": meta.get("motif", ""),
                "edges": meta.get("edges", ""),
                "mean_delay_ms": float(np.mean(segment_delay)),
                "min_delay_ms": float(np.min(segment_delay)),
                "max_delay_ms": float(np.max(segment_delay)),
            }
        )
        start_idx = idx
    return pd.DataFrame(rows)


def build_segment_cost_tables(delay_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return best per-segment cost and argmin topology for every [i, j)."""
    num_steps, num_motifs = delay_matrix.shape
    prefix = np.vstack(
        [
            np.zeros((1, num_motifs), dtype=np.float64),
            np.cumsum(delay_matrix.astype(np.float64), axis=0),
        ]
    )
    seg_cost = np.full((num_steps + 1, num_steps + 1), np.inf, dtype=np.float64)
    seg_arg = np.full((num_steps + 1, num_steps + 1), -1, dtype=np.int32)

    for start in range(num_steps):
        sums = prefix[start + 1 :] - prefix[start]
        arg = np.argmin(sums, axis=1)
        cost = sums[np.arange(sums.shape[0]), arg]
        seg_cost[start, start + 1 :] = cost
        seg_arg[start, start + 1 :] = arg
    return seg_cost, seg_arg


def optimize_min_dwell(
    delay_matrix: np.ndarray,
    min_len: int,
    seg_cost: np.ndarray,
    seg_arg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the best piecewise-constant schedule with every segment length >= min_len."""
    num_steps = delay_matrix.shape[0]
    dp = np.full(num_steps + 1, np.inf, dtype=np.float64)
    prev = np.full(num_steps + 1, -1, dtype=np.int32)
    dp[0] = 0.0

    for end in range(min_len, num_steps + 1):
        starts = np.arange(0, end - min_len + 1, dtype=np.int32)
        valid = np.isfinite(dp[starts])
        if not np.any(valid):
            continue
        starts = starts[valid]
        costs = dp[starts] + seg_cost[starts, end]
        best_pos = int(np.argmin(costs))
        dp[end] = float(costs[best_pos])
        prev[end] = int(starts[best_pos])

    if not np.isfinite(dp[num_steps]):
        raise RuntimeError(f"No feasible schedule for min_len={min_len}")

    topo_by_index = np.full(num_steps, -1, dtype=np.int32)
    end = num_steps
    while end > 0:
        start = int(prev[end])
        if start < 0:
            raise RuntimeError(f"Broken DP path for min_len={min_len}, end={end}")
        topo = int(seg_arg[start, end])
        topo_by_index[start:end] = topo
        end = start

    chosen_delay = delay_matrix[np.arange(num_steps), topo_by_index]
    return topo_by_index, chosen_delay, float(dp[num_steps])


def plot_delay_lines(
    out_path: Path,
    steps: np.ndarray,
    static_delay: np.ndarray,
    oracle_delay: np.ndarray,
    schedules: dict[str, np.ndarray],
    gridplus: np.ndarray | None,
    full_link: np.ndarray | None,
) -> None:
    plt.figure(figsize=(14, 7))
    x_hours = steps / 3600.0
    if full_link is not None:
        plt.plot(x_hours, full_link, color="0.4", linewidth=1.2, label="full_link reference")
    if gridplus is not None:
        plt.plot(x_hours, gridplus, color="#6aa84f", linewidth=1.2, alpha=0.8, label="gridplus reference")
    plt.plot(x_hours, static_delay, color="#2f5597", linewidth=1.6, label="static best motif")
    plt.plot(x_hours, oracle_delay, color="#c00000", linewidth=1.8, label="per-step oracle motif")
    colors = ["#f39c12", "#8e44ad", "#0099a8", "#111111"]
    for (label, delay), color in zip(schedules.items(), colors):
        plt.plot(x_hours, delay, linewidth=1.5, alpha=0.9, label=label, color=color)
    plt.xlabel("time (hour)")
    plt.ylabel("mean shortest delay (ms)")
    plt.title("Dynamic motif scheduling under minimum dwell constraints")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_summary_bars(out_path: Path, summary: pd.DataFrame) -> None:
    labels = summary["scheme"].tolist()
    means = summary["mean_delay_ms"].to_numpy(dtype=float)
    switches = summary["num_segments"].to_numpy(dtype=float) - 1.0

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = ax1.bar(x, means, color="#4f81bd", alpha=0.85)
    ax1.set_ylabel("mean delay (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, switches, color="#c00000", marker="o", linewidth=2)
    ax2.set_ylabel("switch count")
    for xpos, value in zip(x, switches):
        ax2.text(xpos, value, f"{int(value)}", ha="center", va="bottom", fontsize=8, color="#8b0000")

    plt.title("Mean delay and switching frequency")
    fig.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "step" not in df.columns:
        raise ValueError("Input CSV must contain a 'step' column.")
    motif_columns = [col for col in df.columns if col.startswith(args.motif_prefix)]
    if not motif_columns:
        raise ValueError(f"No motif columns found with prefix {args.motif_prefix!r}.")

    steps = df["step"].to_numpy(dtype=np.int64)
    sample_seconds = infer_sample_seconds(steps)
    delay_matrix = df[motif_columns].to_numpy(dtype=np.float64)
    motif_info = load_motif_info(args.input_csv)

    static_idx = int(np.argmin(np.mean(delay_matrix, axis=0)))
    static_name = motif_columns[static_idx]
    static_delay = delay_matrix[:, static_idx]

    oracle_idx = np.argmin(delay_matrix, axis=1).astype(np.int32)
    oracle_delay = delay_matrix[np.arange(len(steps)), oracle_idx]
    oracle_segments = compress_segments(steps, oracle_idx, oracle_delay, motif_columns, motif_info)

    seg_cost, seg_arg = build_segment_cost_tables(delay_matrix)

    schedule_delay_for_plot: dict[str, np.ndarray] = {}
    summary_rows = [
        {
            "scheme": "static_best",
            "min_dwell_minutes": None,
            "mean_delay_ms": float(np.mean(static_delay)),
            "min_delay_ms": float(np.min(static_delay)),
            "max_delay_ms": float(np.max(static_delay)),
            "num_segments": 1,
            "num_unique_motifs": 1,
            "primary_topology": static_name,
        },
        {
            "scheme": "per_step_oracle",
            "min_dwell_minutes": 0.0,
            "mean_delay_ms": float(np.mean(oracle_delay)),
            "min_delay_ms": float(np.min(oracle_delay)),
            "max_delay_ms": float(np.max(oracle_delay)),
            "num_segments": int(len(oracle_segments)),
            "num_unique_motifs": int(len(np.unique(oracle_idx))),
            "primary_topology": "",
        },
    ]

    oracle_by_step = pd.DataFrame(
        {
            "step": steps,
            "topology": [motif_columns[int(idx)] for idx in oracle_idx],
            "delay_ms": oracle_delay,
            "static_best_delay_ms": static_delay,
            "gain_vs_static_ms": static_delay - oracle_delay,
        }
    )
    oracle_by_step.to_csv(args.out_dir / "per_step_oracle_by_step.csv", index=False)
    oracle_segments.to_csv(args.out_dir / "per_step_oracle_segments.csv", index=False)

    for dwell_minutes in args.min_dwell_minutes:
        min_len = max(1, int(math.ceil(dwell_minutes * 60.0 / sample_seconds)))
        topo_idx, chosen_delay, total_cost = optimize_min_dwell(delay_matrix, min_len, seg_cost, seg_arg)
        label = f"min_dwell_{dwell_minutes:g}min"
        schedule_delay_for_plot[label] = chosen_delay

        by_step = pd.DataFrame(
            {
                "step": steps,
                "topology": [motif_columns[int(idx)] for idx in topo_idx],
                "delay_ms": chosen_delay,
                "static_best_delay_ms": static_delay,
                "oracle_delay_ms": oracle_delay,
                "gain_vs_static_ms": static_delay - chosen_delay,
                "gap_to_oracle_ms": chosen_delay - oracle_delay,
            }
        )
        segments = compress_segments(steps, topo_idx, chosen_delay, motif_columns, motif_info)
        by_step.to_csv(args.out_dir / f"{label}_by_step.csv", index=False)
        segments.to_csv(args.out_dir / f"{label}_segments.csv", index=False)

        summary_rows.append(
            {
                "scheme": label,
                "min_dwell_minutes": float(dwell_minutes),
                "mean_delay_ms": float(np.mean(chosen_delay)),
                "min_delay_ms": float(np.min(chosen_delay)),
                "max_delay_ms": float(np.max(chosen_delay)),
                "num_segments": int(len(segments)),
                "num_unique_motifs": int(len(np.unique(topo_idx))),
                "primary_topology": "",
                "total_cost_ms_sum": float(total_cost),
                "mean_gain_vs_static_ms": float(np.mean(static_delay - chosen_delay)),
                "mean_gap_to_oracle_ms": float(np.mean(chosen_delay - oracle_delay)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "schedule_summary.csv", index=False)

    full_link = df["full_link"].to_numpy(dtype=float) if "full_link" in df.columns else None
    gridplus = df["gridplus"].to_numpy(dtype=float) if "gridplus" in df.columns else None
    plot_delay_lines(
        args.out_dir / "dynamic_schedules_vs_references.png",
        steps,
        static_delay,
        oracle_delay,
        schedule_delay_for_plot,
        gridplus,
        full_link,
    )
    plot_summary_bars(args.out_dir / "dynamic_schedule_summary.png", summary)

    meta = {
        "input_csv": str(args.input_csv),
        "out_dir": str(args.out_dir),
        "candidate_scope": "motif_* columns only; gridplus and full_link are reference baselines",
        "num_steps": int(len(steps)),
        "actual_start_step": int(steps[0]),
        "actual_end_step": int(steps[-1]),
        "sample_seconds": sample_seconds,
        "num_candidate_motifs": int(len(motif_columns)),
        "static_best_topology": static_name,
        "static_best_mean_delay_ms": float(np.mean(static_delay)),
        "per_step_oracle_mean_delay_ms": float(np.mean(oracle_delay)),
        "per_step_oracle_num_segments": int(len(oracle_segments)),
        "min_dwell_minutes": args.min_dwell_minutes,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"input_csv={args.input_csv}")
    print(f"out_dir={args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
