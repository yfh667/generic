from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_setup_plan_vs_static_metrics as base  # noqa: E402
import plan_motif0056_0061_roundtrip_link_setup as roundtrip  # noqa: E402
from run_motif0056_0061_roundtrip_setup_plan_viewer import build_masks_and_values, read_events  # noqa: E402
from run_motif0056_to_0061_setup_plan_viewer import union_edge_table  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import load_position_cache  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\m0056_to_m0061_to_m0056_china_europe\t0_86160_stride60\switch36000_54000_dur600_c20"
)

SERIES_STYLE = {
    "actual_roundtrip_setup_056_061_056": ("actual setup 056->061->056, building unusable", "#d97706", 2.25, 1.0),
    "ideal_instant_056_061_056": ("ideal instant 056->061->056 @10h/15h", "#0f766e", 1.55, 0.9),
    "ideal_splice_056_040_10h_15h": ("ideal splice 056/040, 10h-15h uses 040", "#7c3aed", 1.35, 0.86),
    "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.15, 0.82),
    "motif000040": ("motif000040 CBD | CB-", "#C1121F", 1.10, 0.74),
    "motif000061": ("motif000061 DCD | C--", "#64748b", 0.95, 0.50),
    "full_link": ("full-link upper bound", "#111827", 1.25, 0.82),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare round-trip setup plan against static and ideal baselines.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def build_comparison_frame(
    *,
    actual_df: pd.DataFrame,
    steps: list[int],
    switch_step: int,
    return_switch_step: int,
) -> pd.DataFrame:
    hops = base.read_static_metric(base.STATIC_METRIC_ROOT, "hops", steps)
    delay = base.read_static_metric(base.STATIC_METRIC_ROOT, "delay_ms", steps)
    df = pd.DataFrame({"step": steps, "hour": np.asarray(steps, dtype=np.float64) / 3600.0})
    df["hops_actual_roundtrip_setup_056_061_056"] = actual_df["mean_shortest_hops"].to_numpy(dtype=np.float64)
    df["delay_ms_actual_roundtrip_setup_056_061_056"] = actual_df["mean_shortest_delay_ms"].to_numpy(dtype=np.float64)
    for name in base.BASELINE_COLUMNS:
        df[f"hops_{name}"] = hops[name].to_numpy(dtype=np.float64)
        df[f"delay_ms_{name}"] = delay[name].to_numpy(dtype=np.float64)

    steps_np = df["step"].to_numpy(dtype=np.int64)
    middle = (steps_np >= int(switch_step)) & (steps_np < int(return_switch_step))
    for metric in ("hops", "delay_ms"):
        df[f"{metric}_ideal_instant_056_061_056"] = np.where(
            middle,
            df[f"{metric}_motif000061"],
            df[f"{metric}_motif000056"],
        )
        df[f"{metric}_ideal_splice_056_040_10h_15h"] = np.where(
            middle,
            df[f"{metric}_motif000040"],
            df[f"{metric}_motif000056"],
        )
        df[f"{metric}_setup_penalty_vs_ideal_instant_056_061_056"] = (
            df[f"{metric}_actual_roundtrip_setup_056_061_056"] - df[f"{metric}_ideal_instant_056_061_056"]
        )
    return df


def summarize(df: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for metric in ("hops", "delay_ms"):
        actual = df[f"{metric}_actual_roundtrip_setup_056_061_056"].to_numpy(dtype=np.float64)
        for name, (label, _color, _width, _alpha) in SERIES_STYLE.items():
            values = df[f"{metric}_{name}"].to_numpy(dtype=np.float64)
            diff = values - actual
            rows.append(
                {
                    "metric": metric,
                    "series": name,
                    "label": label,
                    "mean": float(np.nanmean(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "p95": float(np.nanpercentile(values, 95)),
                    "mean_minus_actual_roundtrip_setup": float(np.nanmean(diff)),
                    "actual_setup_better_steps": int(np.count_nonzero(actual < values)),
                    "actual_setup_worse_steps": int(np.count_nonzero(actual > values)),
                    "equal_steps": int(np.count_nonzero(np.isclose(actual, values, rtol=0.0, atol=1e-9))),
                    "steps": int(values.size),
                }
            )
    path = out_dir / "roundtrip_setup_vs_static_and_ideal_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def plot_metric(df: pd.DataFrame, out_path: Path, *, metric: str, ylabel: str, switch_step: int, return_switch_step: int) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    x = df["hour"].to_numpy(dtype=np.float64)
    for name, (label, color, width, alpha) in SERIES_STYLE.items():
        values = df[f"{metric}_{name}"].to_numpy(dtype=np.float64)
        ax.plot(x, values, color=color, linewidth=float(width), alpha=float(alpha), label=f"{label} | mean={np.nanmean(values):.3f}")
    ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvspan(float(switch_step) / 3600.0, float(return_switch_step) / 3600.0, color="#7c3aed", alpha=0.06, linewidth=0)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe round-trip setup vs ideal/static baselines: {ylabel}")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_penalty(df: pd.DataFrame, out_path: Path, *, switch_step: int, return_switch_step: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15.8, 7.2), dpi=180, sharex=True)
    x = df["hour"].to_numpy(dtype=np.float64)
    for ax, metric, ylabel in zip(axes, ("hops", "delay_ms"), ("hops penalty", "delay penalty (ms)")):
        values = df[f"{metric}_setup_penalty_vs_ideal_instant_056_061_056"].to_numpy(dtype=np.float64)
        ax.plot(x, values, color="#d97706", linewidth=1.7)
        ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
        ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.set_title(f"actual round-trip setup minus ideal instant: mean={np.nanmean(values):.4f}, max={np.nanmax(values):.4f}")
    axes[-1].set_xlabel("time (hour)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.plan_dir) / "metric_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    topo56, topo61 = roundtrip.build_topologies()
    first_events, second_events = read_events(Path(args.plan_dir) / "link_setup_events.csv")
    edge_table = union_edge_table(topo56, topo61)
    active_mask, _building_mask, _values = build_masks_and_values(
        steps=steps,
        edge_table=edge_table,
        topo56=topo56,
        topo61=topo61,
        first_events=first_events,
        second_events=second_events,
        return_switch_step=int(args.return_switch_step),
    )
    np.save(out_dir / "actual_roundtrip_setup_active_mask.npy", active_mask)

    group_data = load_or_build_group_data(
        xml_file=base.GROUP_XML,
        group_cache_dir=base.GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )
    group_data = {int(step): data for step, data in group_data.items() if int(step) in set(steps)}
    actual_df = base.compute_actual_setup_metrics(
        steps=steps,
        edge_table=edge_table,
        active_mask=active_mask,
        group_data=group_data,
        delay_store=FullLinkDelayStore(base.DELAY_STORE_DIR),
        position_store=load_position_cache(base.POSITION_CACHE_DIR),
        out_dir=out_dir,
        progress_every=int(args.progress_every),
    )
    # Rename one-way output files so the directory is self-describing.
    for old_name, new_name in (
        ("actual_setup_056_to_061_step_metrics.csv", "actual_roundtrip_setup_056_061_056_step_metrics.csv"),
        ("actual_setup_056_to_061_mean_shortest_hops.npy", "actual_roundtrip_setup_056_061_056_mean_shortest_hops.npy"),
        ("actual_setup_056_to_061_mean_shortest_delay_ms.npy", "actual_roundtrip_setup_056_061_056_mean_shortest_delay_ms.npy"),
        ("actual_setup_056_to_061_active_edge_counts.npy", "actual_roundtrip_setup_056_061_056_active_edge_counts.npy"),
        ("actual_setup_056_to_061_reachable_pairs_hops.npy", "actual_roundtrip_setup_056_061_056_reachable_pairs_hops.npy"),
        ("actual_setup_056_to_061_reachable_pairs_delay.npy", "actual_roundtrip_setup_056_061_056_reachable_pairs_delay.npy"),
    ):
        old = out_dir / old_name
        if old.exists():
            old.replace(out_dir / new_name)

    compare_df = build_comparison_frame(
        actual_df=actual_df,
        steps=steps,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    timeseries = out_dir / "roundtrip_setup_vs_static_and_ideal_timeseries.csv"
    compare_df.to_csv(timeseries, index=False, encoding="utf-8-sig")
    summary = summarize(compare_df, out_dir)
    plot_metric(
        compare_df,
        out_dir / "roundtrip_setup_vs_static_and_ideal_hops.png",
        metric="hops",
        ylabel="mean shortest hops",
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    plot_metric(
        compare_df,
        out_dir / "roundtrip_setup_vs_static_and_ideal_delay_ms.png",
        metric="delay_ms",
        ylabel="mean shortest delay (ms)",
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    plot_penalty(
        compare_df,
        out_dir / "roundtrip_setup_penalty_vs_ideal_instant_056_061_056.png",
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    meta = {
        "plan_dir": str(args.plan_dir),
        "out_dir": str(out_dir),
        "dynamic_actual_definition": "active edges only; building edges are not usable",
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "runner": str(THIS_FILE),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[roundtrip-setup-compare] out_dir={out_dir}", flush=True)
    print(f"[roundtrip-setup-compare] timeseries={timeseries}", flush=True)
    print(f"[roundtrip-setup-compare] summary={summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
