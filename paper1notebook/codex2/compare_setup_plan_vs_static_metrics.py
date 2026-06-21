from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra, shortest_path  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as plan  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import load_position_cache  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION, make_edge_table_from_records  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\_position_cache\cache_0_86164_1s"
)
STATIC_METRIC_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_w_le4_h_le3\paper_style_808_no_region_grid_t0_86160_stride60"
)
DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\m0056_to_m0061_china_europe\t0_86160_stride60\switch36000_dur600_c20"
)

PAIR_KEY = "china_europe"
SOURCE_GROUP_ID = 2
TARGET_GROUP_ID = 3
BASELINE_COLUMNS = {
    "full_link": "full_link",
    "motif000056": "combined_motif_000056",
    "motif000040": "combined_motif_000040",
    "motif000061": "combined_motif_000061",
}
SERIES_STYLE = {
    "actual_setup_056_to_061": ("actual setup 056->061, building unusable", "#d97706", 2.25, 1.0),
    "ideal_instant_056_to_061_at_10h": ("ideal instant 056->061 @10h", "#0f766e", 1.55, 0.9),
    "ideal_splice_056_040_10h_15h": ("ideal splice 056/040, 10h-15h uses 040", "#7c3aed", 1.35, 0.86),
    "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.15, 0.82),
    "motif000040": ("motif000040 CBD | CB-", "#C1121F", 1.10, 0.74),
    "motif000061": ("motif000061 DCD | C--", "#64748b", 0.95, 0.55),
    "full_link": ("full-link upper bound", "#111827", 1.25, 0.82),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare planned link-setup dynamic topology against static and ideal-splice baselines."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--splice-window-end", type=int, default=54000)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def read_plan_events(path: Path) -> dict[int, tuple[int, int]]:
    events: dict[int, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            events[int(row["owner"])] = (int(row["plan_start"]), int(row["plan_end"]))
    return events


def build_source_target() -> tuple[plan.TopologyData, plan.TopologyData]:
    rows = plan.read_motif_rows(plan.MOTIF_LIBRARY_CSV)
    source_spec = plan.build_topology_spec(56, rows[56])
    target_spec = plan.build_topology_spec(61, rows[61])
    source = plan.load_topology_data(source_spec, start=0, end=86160, stride=60)
    target = plan.load_topology_data(target_spec, start=0, end=86160, stride=60)
    return source, target


def union_edge_table(source: plan.TopologyData, target: plan.TopologyData):
    records: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge_table in (source.spec.edge_table, target.spec.edge_table):
        for idx in range(edge_table.num_edges):
            src = int(edge_table.src[idx])
            dst = int(edge_table.dst[idx])
            key = plan.edge_key(src, dst)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                (
                    int(edge_table.src_plane[idx]),
                    int(edge_table.src_y[idx]),
                    int(edge_table.dst_plane[idx]),
                    int(edge_table.dst_y[idx]),
                    int(edge_table.option[idx]),
                )
            )
    return make_edge_table_from_records(p=int(G60_CONFIG.P), n=int(G60_CONFIG.N), records=records)


def edge_key_index(edge_table) -> dict[tuple[int, int], int]:
    return {
        plan.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(edge_table.num_edges)
    }


def all_intra_keys(edge_table) -> set[tuple[int, int]]:
    return {
        plan.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))
        for idx in range(edge_table.num_edges)
        if int(edge_table.option[idx]) == INTRA_OPTION
    }


def build_active_mask(
    *,
    steps: list[int],
    edge_table,
    source: plan.TopologyData,
    target: plan.TopologyData,
    events_by_owner: dict[int, tuple[int, int]],
) -> np.ndarray:
    key_to_idx = edge_key_index(edge_table)
    intra_keys = all_intra_keys(edge_table)
    active_mask = np.zeros((len(steps), edge_table.num_edges), dtype=bool)
    all_owners = sorted(set(source.right_by_owner) | set(target.right_by_owner))
    for row, step in enumerate(steps):
        active_keys = set(intra_keys)
        for owner in all_owners:
            old_link = source.right_by_owner.get(owner)
            new_link = target.right_by_owner.get(owner)
            event = events_by_owner.get(owner)
            if event is None:
                link = old_link or new_link
                if link is not None:
                    active_keys.add(link.edge_key)
                continue
            start, end = event
            if int(step) < int(start):
                if old_link is not None:
                    active_keys.add(old_link.edge_key)
            elif int(start) <= int(step) < int(end):
                # Building links are explicitly not usable.
                continue
            else:
                if new_link is not None:
                    active_keys.add(new_link.edge_key)
        for key in active_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                active_mask[row, idx] = True
    return active_mask


def metric_mean(values: np.ndarray) -> tuple[float, int, float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), 0, float("nan"), float("nan")
    return float(np.mean(finite)), int(finite.size), float(np.min(finite)), float(np.max(finite))


def compute_actual_setup_metrics(
    *,
    steps: list[int],
    edge_table,
    active_mask: np.ndarray,
    group_data: dict,
    delay_store: FullLinkDelayStore,
    position_store,
    out_dir: Path,
    progress_every: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(steps[1] - steps[0]))
    position_rows = delay_rows
    lookup = build_weight_lookup(
        edge_table,
        delay_store,
        config=G60_CONFIG,
        allow_intra_fallback=True,
    )
    src_all = np.asarray(edge_table.src, dtype=np.int32)
    dst_all = np.asarray(edge_table.dst, dtype=np.int32)
    total_nodes = int(G60_CONFIG.total_sats)
    rows: list[dict[str, float | int]] = []
    delay_values_out = np.full(len(steps), np.nan, dtype=np.float32)
    hops_values_out = np.full(len(steps), np.nan, dtype=np.float32)
    active_counts = np.zeros(len(steps), dtype=np.int32)
    reachable_counts_hops = np.zeros(len(steps), dtype=np.int32)
    reachable_counts_delay = np.zeros(len(steps), dtype=np.int32)
    started = time.time()

    for row, step in enumerate(steps):
        mask = np.asarray(active_mask[row], dtype=bool)
        active_counts[row] = int(np.count_nonzero(mask))
        sources = group_nodes_for_step(group_data, int(step), SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), TARGET_GROUP_ID)
        if not sources or not targets or not np.any(mask):
            rows.append(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "active_edges": int(active_counts[row]),
                    "source_nodes": int(len(sources)),
                    "target_nodes": int(len(targets)),
                    "reachable_pairs_hops": 0,
                    "mean_shortest_hops": np.nan,
                    "min_shortest_hops": np.nan,
                    "max_shortest_hops": np.nan,
                    "reachable_pairs_delay": 0,
                    "mean_shortest_delay_ms": np.nan,
                    "min_shortest_delay_ms": np.nan,
                    "max_shortest_delay_ms": np.nan,
                }
            )
            continue

        active_src = src_all[mask]
        active_dst = dst_all[mask]
        hop_data = np.ones(active_src.size * 2, dtype=np.float32)
        hop_graph = csr_matrix(
            (hop_data, (np.concatenate([active_src, active_dst]), np.concatenate([active_dst, active_src]))),
            shape=(total_nodes, total_nodes),
        )
        hop_dist = shortest_path(
            hop_graph,
            directed=False,
            unweighted=True,
            indices=np.asarray(sources, dtype=np.int32),
        )
        hop_dist = np.atleast_2d(hop_dist)[:, np.asarray(targets, dtype=np.int32)]
        hop_mean, hop_count, hop_min, hop_max = metric_mean(hop_dist)
        hops_values_out[row] = hop_mean
        reachable_counts_hops[row] = hop_count

        weights = edge_weights_for_step(
            edge_table=edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[row]),
            position_row=int(position_rows[row]),
        )
        active_weights = np.asarray(weights[mask], dtype=np.float32)
        delay_graph = csr_matrix(
            (
                np.concatenate([active_weights, active_weights]),
                (np.concatenate([active_src, active_dst]), np.concatenate([active_dst, active_src])),
            ),
            shape=(total_nodes, total_nodes),
        )
        delay_dist = dijkstra(
            delay_graph,
            directed=False,
            indices=np.asarray(sources, dtype=np.int32),
        )
        delay_dist = np.atleast_2d(delay_dist)[:, np.asarray(targets, dtype=np.int32)]
        delay_mean, delay_count, delay_min, delay_max = metric_mean(delay_dist)
        delay_values_out[row] = delay_mean
        reachable_counts_delay[row] = delay_count

        rows.append(
            {
                "step": int(step),
                "hour": float(step) / 3600.0,
                "active_edges": int(active_counts[row]),
                "source_nodes": int(len(sources)),
                "target_nodes": int(len(targets)),
                "reachable_pairs_hops": int(hop_count),
                "mean_shortest_hops": float(hop_mean),
                "min_shortest_hops": float(hop_min),
                "max_shortest_hops": float(hop_max),
                "reachable_pairs_delay": int(delay_count),
                "mean_shortest_delay_ms": float(delay_mean),
                "min_shortest_delay_ms": float(delay_min),
                "max_shortest_delay_ms": float(delay_max),
            }
        )
        if int(progress_every) > 0 and ((row + 1) % int(progress_every) == 0 or row + 1 == len(steps)):
            print(
                f"[setup-metrics] {row + 1}/{len(steps)} step={step} "
                f"hops={hop_mean:.3f} delay={delay_mean:.3f} ms elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "actual_setup_056_to_061_step_metrics.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "actual_setup_056_to_061_mean_shortest_hops.npy", hops_values_out)
    np.save(out_dir / "actual_setup_056_to_061_mean_shortest_delay_ms.npy", delay_values_out)
    np.save(out_dir / "actual_setup_056_to_061_active_edge_counts.npy", active_counts)
    np.save(out_dir / "actual_setup_056_to_061_reachable_pairs_hops.npy", reachable_counts_hops)
    np.save(out_dir / "actual_setup_056_to_061_reachable_pairs_delay.npy", reachable_counts_delay)
    return df


def static_metric_csv(metric_root: Path, metric: str) -> Path:
    if metric == "hops":
        return metric_root / "shortest_hops" / PAIR_KEY / "compare_mean_shortest_hops_808_strict_reachable.csv"
    if metric == "delay_ms":
        return metric_root / "shortest_delay" / PAIR_KEY / "compare_mean_shortest_delay_ms_808_strict_reachable.csv"
    raise ValueError(metric)


def read_static_metric(metric_root: Path, metric: str, steps: list[int]) -> pd.DataFrame:
    path = static_metric_csv(metric_root, metric)
    usecols = ["step", *BASELINE_COLUMNS.values()]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["step"].isin(steps)].sort_values("step").reset_index(drop=True)
    if df["step"].tolist() != list(steps):
        raise ValueError(f"static metric steps do not match requested steps: {path}")
    return df.rename(columns={v: k for k, v in BASELINE_COLUMNS.items()})


def build_comparison_frame(
    *,
    actual_df: pd.DataFrame,
    metric_root: Path,
    steps: list[int],
    switch_step: int,
    splice_window_end: int,
) -> pd.DataFrame:
    hops = read_static_metric(metric_root, "hops", steps)
    delay = read_static_metric(metric_root, "delay_ms", steps)
    df = pd.DataFrame({"step": steps, "hour": np.asarray(steps, dtype=np.float64) / 3600.0})
    df["hops_actual_setup_056_to_061"] = actual_df["mean_shortest_hops"].to_numpy(dtype=np.float64)
    df["delay_ms_actual_setup_056_to_061"] = actual_df["mean_shortest_delay_ms"].to_numpy(dtype=np.float64)
    for name in BASELINE_COLUMNS:
        df[f"hops_{name}"] = hops[name].to_numpy(dtype=np.float64)
        df[f"delay_ms_{name}"] = delay[name].to_numpy(dtype=np.float64)

    before_switch = df["step"].to_numpy(dtype=np.int64) < int(switch_step)
    splice_040 = (df["step"].to_numpy(dtype=np.int64) >= int(switch_step)) & (
        df["step"].to_numpy(dtype=np.int64) <= int(splice_window_end)
    )
    for metric in ("hops", "delay_ms"):
        df[f"{metric}_ideal_instant_056_to_061_at_10h"] = np.where(
            before_switch,
            df[f"{metric}_motif000056"],
            df[f"{metric}_motif000061"],
        )
        df[f"{metric}_ideal_splice_056_040_10h_15h"] = np.where(
            splice_040,
            df[f"{metric}_motif000040"],
            df[f"{metric}_motif000056"],
        )
        df[f"{metric}_setup_penalty_vs_instant_056_061"] = (
            df[f"{metric}_actual_setup_056_to_061"] - df[f"{metric}_ideal_instant_056_to_061_at_10h"]
        )
    return df


def summarize_comparison(df: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for metric in ("hops", "delay_ms"):
        actual = df[f"{metric}_actual_setup_056_to_061"].to_numpy(dtype=np.float64)
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
                    "mean_minus_actual_setup": float(np.nanmean(diff)),
                    "actual_setup_better_steps": int(np.count_nonzero(actual < values)),
                    "actual_setup_worse_steps": int(np.count_nonzero(actual > values)),
                    "equal_steps": int(np.count_nonzero(np.isclose(actual, values, rtol=0.0, atol=1e-9))),
                    "steps": int(values.size),
                }
            )
    summary = out_dir / "setup_vs_static_and_ideal_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False, encoding="utf-8-sig")
    return summary


def plot_metric(df: pd.DataFrame, out_path: Path, *, metric: str, ylabel: str, switch_step: int, splice_window_end: int) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    x = df["hour"].to_numpy(dtype=np.float64)
    for name, (label, color, width, alpha) in SERIES_STYLE.items():
        values = df[f"{metric}_{name}"].to_numpy(dtype=np.float64)
        ax.plot(
            x,
            values,
            color=color,
            linewidth=float(width),
            alpha=float(alpha),
            label=f"{label} | mean={np.nanmean(values):.3f}",
        )
    ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvspan(float(switch_step) / 3600.0, float(splice_window_end) / 3600.0, color="#7c3aed", alpha=0.06, linewidth=0)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe setup plan vs ideal/static baselines: {ylabel}")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_penalty(df: pd.DataFrame, out_path: Path, *, switch_step: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15.8, 7.2), dpi=180, sharex=True)
    x = df["hour"].to_numpy(dtype=np.float64)
    for ax, metric, ylabel in zip(axes, ("hops", "delay_ms"), ("hops penalty", "delay penalty (ms)")):
        values = df[f"{metric}_setup_penalty_vs_instant_056_061"].to_numpy(dtype=np.float64)
        ax.plot(x, values, color="#d97706", linewidth=1.7)
        ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
        ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.set_title(f"actual setup minus ideal instant 056->061: mean={np.nanmean(values):.4f}, max={np.nanmax(values):.4f}")
    axes[-1].set_xlabel("time (hour)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.plan_dir) / "metric_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    source, target = build_source_target()
    edge_table = union_edge_table(source, target)
    events = read_plan_events(Path(args.plan_dir) / "link_setup_events.csv")
    active_mask = build_active_mask(
        steps=steps,
        edge_table=edge_table,
        source=source,
        target=target,
        events_by_owner=events,
    )
    np.save(out_dir / "actual_setup_active_mask.npy", active_mask)

    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )
    group_data = {int(step): data for step, data in group_data.items() if int(step) in set(steps)}
    delay_store = FullLinkDelayStore(DELAY_STORE_DIR)
    position_store = load_position_cache(POSITION_CACHE_DIR)

    actual_df = compute_actual_setup_metrics(
        steps=steps,
        edge_table=edge_table,
        active_mask=active_mask,
        group_data=group_data,
        delay_store=delay_store,
        position_store=position_store,
        out_dir=out_dir,
        progress_every=int(args.progress_every),
    )
    compare_df = build_comparison_frame(
        actual_df=actual_df,
        metric_root=STATIC_METRIC_ROOT,
        steps=steps,
        switch_step=int(args.switch_step),
        splice_window_end=int(args.splice_window_end),
    )
    compare_path = out_dir / "setup_vs_static_and_ideal_timeseries.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    summary_path = summarize_comparison(compare_df, out_dir)
    plot_metric(
        compare_df,
        out_dir / "setup_vs_static_and_ideal_hops.png",
        metric="hops",
        ylabel="mean shortest hops",
        switch_step=int(args.switch_step),
        splice_window_end=int(args.splice_window_end),
    )
    plot_metric(
        compare_df,
        out_dir / "setup_vs_static_and_ideal_delay_ms.png",
        metric="delay_ms",
        ylabel="mean shortest delay (ms)",
        switch_step=int(args.switch_step),
        splice_window_end=int(args.splice_window_end),
    )
    plot_penalty(compare_df, out_dir / "setup_penalty_vs_ideal_instant_056_061.png", switch_step=int(args.switch_step))

    meta = {
        "plan_dir": str(args.plan_dir),
        "out_dir": str(out_dir),
        "steps": len(steps),
        "switch_step": int(args.switch_step),
        "splice_window_end": int(args.splice_window_end),
        "dynamic_actual_definition": "active edges only; building edges are not usable",
        "baselines": list(SERIES_STYLE.keys()),
        "runner": str(THIS_FILE),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[setup-compare] out_dir={out_dir}", flush=True)
    print(f"[setup-compare] timeseries={compare_path}", flush=True)
    print(f"[setup-compare] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
