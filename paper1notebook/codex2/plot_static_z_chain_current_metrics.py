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
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import load_position_cache  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.full_link_node_usage_viewer import read_edge_table_csv  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\static_z_chain_056_061_056_china_europe"
    r"\t0_86160_stride1\seq0056_0061_0056_tau36000_54000_lst060_latest"
)

SERIES_STYLE = {
    "current_static_z_chain": ("current static-z chain, building unusable", "#d97706", 2.4, 1.0),
    "ideal_instant_056_061_056": ("ideal instant 056->061->056", "#0f766e", 1.45, 0.88),
    "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.15, 0.82),
    "motif000061": ("motif000061 DCD | C--", "#64748b", 1.0, 0.62),
    "motif000040": ("motif000040 CBD | CB-", "#C1121F", 1.0, 0.70),
    "full_link": ("full-link upper bound", "#111827", 1.25, 0.82),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot shortest-hop and shortest-delay metrics for current static-z chain.")
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def selected_rows(steps_all: np.ndarray, *, start: int, end: int, stride: int) -> np.ndarray:
    rows = np.flatnonzero(
        (steps_all >= int(start))
        & (steps_all <= int(end))
        & (((steps_all - int(start)) % int(stride)) == 0)
    )
    if rows.size == 0:
        raise ValueError(f"no selected rows for start={start}, end={end}, stride={stride}")
    return rows


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
    df["hops_current_static_z_chain"] = actual_df["mean_shortest_hops"].to_numpy(dtype=np.float64)
    df["delay_ms_current_static_z_chain"] = actual_df["mean_shortest_delay_ms"].to_numpy(dtype=np.float64)
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
        df[f"{metric}_penalty_vs_ideal_instant"] = (
            df[f"{metric}_current_static_z_chain"] - df[f"{metric}_ideal_instant_056_061_056"]
        )
    return df


def write_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for metric in ("hops", "delay_ms"):
        current = df[f"{metric}_current_static_z_chain"].to_numpy(dtype=np.float64)
        for name, (label, _color, _width, _alpha) in SERIES_STYLE.items():
            values = df[f"{metric}_{name}"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "metric": metric,
                    "series": name,
                    "label": label,
                    "mean": float(np.nanmean(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "p95": float(np.nanpercentile(values, 95)),
                    "mean_minus_current": float(np.nanmean(values - current)),
                    "current_better_steps": int(np.count_nonzero(current < values)),
                    "current_worse_steps": int(np.count_nonzero(current > values)),
                    "equal_steps": int(np.count_nonzero(np.isclose(current, values, rtol=0.0, atol=1e-9))),
                    "steps": int(values.size),
                }
            )
    path = out_dir / "current_static_z_chain_metric_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def plot_metric(
    df: pd.DataFrame,
    out_path: Path,
    *,
    metric: str,
    ylabel: str,
    switch_step: int,
    return_switch_step: int,
) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    x = df["hour"].to_numpy(dtype=np.float64)
    for name, (label, color, width, alpha) in SERIES_STYLE.items():
        values = df[f"{metric}_{name}"].to_numpy(dtype=np.float64)
        ax.plot(x, values, color=color, linewidth=float(width), alpha=float(alpha), label=f"{label} | mean={np.nanmean(values):.3f}")
    for step in (switch_step, return_switch_step):
        ax.axvline(float(step) / 3600.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axvspan(float(switch_step) / 3600.0, float(return_switch_step) / 3600.0, color="#0f766e", alpha=0.055, linewidth=0)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe current static-z chain: {ylabel}")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_penalty(df: pd.DataFrame, out_path: Path, *, switch_step: int, return_switch_step: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15.8, 7.2), dpi=180, sharex=True)
    x = df["hour"].to_numpy(dtype=np.float64)
    for ax, metric, ylabel in zip(axes, ("hops", "delay_ms"), ("hops penalty", "delay penalty (ms)")):
        values = df[f"{metric}_penalty_vs_ideal_instant"].to_numpy(dtype=np.float64)
        ax.plot(x, values, color="#d97706", linewidth=1.55)
        ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
        for step in (switch_step, return_switch_step):
            ax.axvline(float(step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.72)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.set_title(f"current minus ideal instant: mean={np.nanmean(values):.4f}, max={np.nanmax(values):.4f}")
    axes[-1].set_xlabel("time (hour)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    plan_dir = Path(args.plan_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else plan_dir / f"metric_compare_t{args.start}_{args.end}_stride{args.stride}"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_all = np.asarray(np.load(plan_dir / "steps.npy", mmap_mode="r"), dtype=np.int64)
    rows = selected_rows(steps_all, start=int(args.start), end=int(args.end), stride=int(args.stride))
    steps = [int(x) for x in steps_all[rows]]
    edge_table = read_edge_table_csv(plan_dir / "union_edges.csv", total_nodes=int(G60_CONFIG.total_sats))
    active_mask = np.asarray(np.load(plan_dir / "edge_active_mask.npy", mmap_mode="r")[rows, :], dtype=bool)
    np.save(out_dir / "current_static_z_chain_active_mask.npy", active_mask)

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
    actual_path = out_dir / "current_static_z_chain_step_metrics.csv"
    actual_df.to_csv(actual_path, index=False, encoding="utf-8-sig")

    compare_df = build_comparison_frame(
        actual_df=actual_df,
        steps=steps,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    timeseries_path = out_dir / "current_static_z_chain_vs_baselines_timeseries.csv"
    compare_df.to_csv(timeseries_path, index=False, encoding="utf-8-sig")
    summary_path = write_summary(compare_df, out_dir)

    hops_plot = out_dir / "current_static_z_chain_vs_baselines_hops.png"
    delay_plot = out_dir / "current_static_z_chain_vs_baselines_delay_ms.png"
    penalty_plot = out_dir / "current_static_z_chain_penalty_vs_ideal_instant.png"
    plot_metric(
        compare_df,
        hops_plot,
        metric="hops",
        ylabel="mean shortest hops",
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    plot_metric(
        compare_df,
        delay_plot,
        metric="delay_ms",
        ylabel="mean shortest delay (ms)",
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    plot_penalty(
        compare_df,
        penalty_plot,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )

    meta = {
        "plan_dir": str(plan_dir),
        "out_dir": str(out_dir),
        "steps": len(steps),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "dynamic_actual_definition": "active edges only; building edges are not usable",
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "timeseries": str(timeseries_path),
        "summary": str(summary_path),
        "hops_plot": str(hops_plot),
        "delay_plot": str(delay_plot),
        "penalty_plot": str(penalty_plot),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}", flush=True)
    print(f"timeseries={timeseries_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(f"hops_plot={hops_plot}", flush=True)
    print(f"delay_plot={delay_plot}", flush=True)
    print(f"penalty_plot={penalty_plot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
