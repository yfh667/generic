from __future__ import annotations

import argparse
import json
import sys
import time
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

import compare_lst_dynamic_setup_vs_static_0056_0061 as one_way  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import PositionCacheStore  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


FORWARD_LST_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_iterative_exemption_lst_sweep_056_061_china_europe_1s_any_direct"
    r"\t0_86160_stride1\switch36000_54000\releaseguard1\lst060"
)
REVERSE_LST_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_iterative_exemption_lst_sweep_061_056_china_europe_1s_any_direct"
    r"\t0_86160_stride1\switch54000_86160\releaseguard1\lst060"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\lst60_metric_compare_056_061_056"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LST=60 roundtrip dynamic setup 000056->000061->000056 with static motifs."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--forward-lst-dir", type=Path, default=FORWARD_LST_DIR)
    parser.add_argument("--reverse-lst-dir", type=Path, default=REVERSE_LST_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def roundtrip_mask_for_step(
    *,
    step: int,
    edge_table,
    motif56: one_way.TopologyLite,
    motif61: one_way.TopologyLite,
    forward_events: dict[int, tuple[int, int]],
    reverse_events: dict[int, tuple[int, int]],
    key_to_idx: dict[tuple[int, int], int],
    intra_keys: set[tuple[int, int]],
) -> np.ndarray:
    active_keys = set(intra_keys)
    owners = sorted(set(motif56.right_by_owner) | set(motif61.right_by_owner))
    for owner in owners:
        link56 = motif56.right_by_owner.get(owner)
        link61 = motif61.right_by_owner.get(owner)
        forward = forward_events.get(owner)

        if forward is None:
            # This owner never left the 000056 state in the forward transition, so the reverse
            # transition should not consume its port again.
            link = link56 or link61
            if link is not None:
                active_keys.add(link.edge_key)
            continue

        f_start, f_end = forward
        if int(step) < int(f_start):
            if link56 is not None:
                active_keys.add(link56.edge_key)
            continue
        if int(f_start) <= int(step) < int(f_end):
            # Forward setup is still building; it is intentionally not routable.
            continue

        reverse = reverse_events.get(owner)
        if reverse is None:
            if link61 is not None:
                active_keys.add(link61.edge_key)
            continue

        r_start, r_end = reverse
        if int(step) < int(r_start):
            if link61 is not None:
                active_keys.add(link61.edge_key)
        elif int(r_start) <= int(step) < int(r_end):
            # Reverse setup is still building; it is intentionally not routable.
            continue
        else:
            if link56 is not None:
                active_keys.add(link56.edge_key)

    mask = np.zeros(edge_table.num_edges, dtype=bool)
    for key in active_keys:
        idx = key_to_idx.get(key)
        if idx is not None:
            mask[idx] = True
    return mask


def compute_roundtrip_metrics(
    *,
    steps: list[int],
    edge_table,
    motif56: one_way.TopologyLite,
    motif61: one_way.TopologyLite,
    forward_events: dict[int, tuple[int, int]],
    reverse_events: dict[int, tuple[int, int]],
    group_data: dict,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    progress_every: int,
) -> pd.DataFrame:
    key_to_idx = one_way.edge_key_index(edge_table)
    intra_keys = one_way.all_intra_keys(edge_table)
    masks_static = {
        "motif000056": one_way.static_mask(edge_table, motif56),
        "motif000061": one_way.static_mask(edge_table, motif61),
    }
    src_all = np.asarray(edge_table.src, dtype=np.int32)
    dst_all = np.asarray(edge_table.dst, dtype=np.int32)
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(steps[1] - steps[0]))
    lookup = build_weight_lookup(edge_table, delay_store, config=G60_CONFIG, allow_intra_fallback=True)
    rows: list[dict[str, float | int]] = []
    started = time.time()

    for row_idx, step in enumerate(steps):
        dynamic_mask = roundtrip_mask_for_step(
            step=int(step),
            edge_table=edge_table,
            motif56=motif56,
            motif61=motif61,
            forward_events=forward_events,
            reverse_events=reverse_events,
            key_to_idx=key_to_idx,
            intra_keys=intra_keys,
        )
        masks = {"dynamic_roundtrip_lst60": dynamic_mask, **masks_static}
        sources = group_nodes_for_step(group_data, int(step), one_way.SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), one_way.TARGET_GROUP_ID)
        weights = edge_weights_for_step(
            edge_table=edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[row_idx]),
            position_row=int(delay_rows[row_idx]),
        )

        out: dict[str, float | int] = {
            "step": int(step),
            "hour": float(step) / 3600.0,
            "source_nodes": int(len(sources)),
            "target_nodes": int(len(targets)),
        }
        for name, mask in masks.items():
            hop_mean, hop_pairs, hop_min, hop_max = one_way.hop_summary_for_mask(
                mask=mask,
                src_all=src_all,
                dst_all=dst_all,
                total_nodes=int(G60_CONFIG.total_sats),
                sources=sources,
                targets=targets,
            )
            delay_mean, delay_pairs, delay_min, delay_max = one_way.delay_summary_for_mask(
                mask=mask,
                weights=weights,
                src_all=src_all,
                dst_all=dst_all,
                total_nodes=int(G60_CONFIG.total_sats),
                sources=sources,
                targets=targets,
            )
            out[f"{name}_active_edges"] = int(np.count_nonzero(mask))
            out[f"{name}_reachable_pairs_hops"] = int(hop_pairs)
            out[f"{name}_mean_shortest_hops"] = float(hop_mean)
            out[f"{name}_min_shortest_hops"] = float(hop_min)
            out[f"{name}_max_shortest_hops"] = float(hop_max)
            out[f"{name}_reachable_pairs_delay"] = int(delay_pairs)
            out[f"{name}_mean_shortest_delay_ms"] = float(delay_mean)
            out[f"{name}_min_shortest_delay_ms"] = float(delay_min)
            out[f"{name}_max_shortest_delay_ms"] = float(delay_max)
        rows.append(out)
        if int(progress_every) > 0 and ((row_idx + 1) % int(progress_every) == 0 or row_idx + 1 == len(steps)):
            print(
                f"[roundtrip-metrics] {row_idx + 1}/{len(steps)} step={step} "
                f"hops={out['dynamic_roundtrip_lst60_mean_shortest_hops']:.3f} "
                f"delay={out['dynamic_roundtrip_lst60_mean_shortest_delay_ms']:.3f}ms "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_periods(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    periods = [
        ("all", 0, 86160),
        ("before_10h", 0, 35999),
        ("10h_15h", 36000, 54000),
        ("after_15h", 54001, 86160),
    ]
    names = ("dynamic_roundtrip_lst60", "motif000056", "motif000061")
    rows: list[dict[str, float | int | str]] = []
    for label, start, end in periods:
        sub = df[(df["step"] >= int(start)) & (df["step"] <= int(end))]
        row: dict[str, float | int | str] = {"period": label, "steps": int(len(sub))}
        for name in names:
            row[f"{name}_hops_mean"] = float(np.nanmean(sub[f"{name}_mean_shortest_hops"].to_numpy(dtype=float)))
            row[f"{name}_delay_mean"] = float(
                np.nanmean(sub[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float))
            )
        row["dynamic_minus_056_hops"] = float(row["dynamic_roundtrip_lst60_hops_mean"]) - float(
            row["motif000056_hops_mean"]
        )
        row["dynamic_minus_056_delay"] = float(row["dynamic_roundtrip_lst60_delay_mean"]) - float(
            row["motif000056_delay_mean"]
        )
        row["dynamic_minus_061_hops"] = float(row["dynamic_roundtrip_lst60_hops_mean"]) - float(
            row["motif000061_hops_mean"]
        )
        row["dynamic_minus_061_delay"] = float(row["dynamic_roundtrip_lst60_delay_mean"]) - float(
            row["motif000061_delay_mean"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "period_summary_roundtrip_lst60_vs_static_0056_0061.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_setup(forward_dir: Path, reverse_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_counts = None
    for direction, lst_dir in (("056_to_061", forward_dir), ("061_to_056", reverse_dir)):
        summary = json.loads((Path(lst_dir) / "summary.json").read_text(encoding="utf-8"))
        counts = np.load(Path(lst_dir) / "building_concurrency_1s.npy")
        total_counts = counts.copy() if total_counts is None else total_counts + counts
        rows.append(
            {
                "direction": direction,
                "changed_links_with_target": int(summary["changed_links_with_target"]),
                "scheduled_links": int(summary["schedule_jobs"]),
                "unscheduled_links": int(summary["iterative_unscheduled_links"]),
                "seed_exemptions": int(summary["seed_exemptions_36000"]),
                "propagated_exemption_owners": int(summary["propagated_exemption_owners"]),
                "remaining_conflict_rows": int(summary["remaining_conflict_rows"]),
                "schedule_window_infeasible": int(summary["schedule_window_infeasible"]),
                "naive_peak_concurrency": int(summary["naive_peak_concurrency"]),
                "smoothed_peak_concurrency": int(summary["smoothed_peak_concurrency"]),
                "busy_seconds": int(summary["busy_seconds"]),
                "total_build_work_seconds": int(summary["total_build_work_seconds"]),
            }
        )
    assert total_counts is not None
    positive = total_counts[total_counts > 0]
    rows.append(
        {
            "direction": "combined_roundtrip",
            "changed_links_with_target": int(sum(int(row["changed_links_with_target"]) for row in rows)),
            "scheduled_links": int(sum(int(row["scheduled_links"]) for row in rows)),
            "unscheduled_links": int(sum(int(row["unscheduled_links"]) for row in rows)),
            "seed_exemptions": int(sum(int(row["seed_exemptions"]) for row in rows)),
            "propagated_exemption_owners": int(sum(int(row["propagated_exemption_owners"]) for row in rows)),
            "remaining_conflict_rows": int(sum(int(row["remaining_conflict_rows"]) for row in rows)),
            "schedule_window_infeasible": int(sum(int(row["schedule_window_infeasible"]) for row in rows)),
            "naive_peak_concurrency": "",
            "smoothed_peak_concurrency": int(np.max(total_counts)) if total_counts.size else 0,
            "busy_seconds": int(positive.size),
            "total_build_work_seconds": int(np.sum(total_counts)),
        }
    )
    np.save(out_dir / "combined_building_concurrency_1s.npy", total_counts.astype(np.int32, copy=False))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "setup_summary_roundtrip_lst60.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14.2, 3.8), dpi=150)
    seconds = np.arange(total_counts.size, dtype=np.float64)
    ax.plot(seconds / 3600.0, total_counts, color="#d97706", linewidth=1.2)
    ax.axvline(10.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.axvline(15.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.set_title("G60 China-Europe LST=60 roundtrip setup concurrency")
    ax.grid(True, alpha=0.25, linewidth=0.55)
    fig.tight_layout()
    fig.savefig(out_dir / "roundtrip_lst60_building_concurrency.png")
    plt.close(fig)
    return df


def write_metric_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for name in ("dynamic_roundtrip_lst60", "motif000056", "motif000061"):
        rows.append(
            {
                "series": name,
                "mean_hops_over_time": float(np.nanmean(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=float))),
                "min_hops_over_time": float(np.nanmin(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=float))),
                "max_hops_over_time": float(np.nanmax(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=float))),
                "mean_delay_ms_over_time": float(
                    np.nanmean(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float))
                ),
                "min_delay_ms_over_time": float(np.nanmin(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float))),
                "max_delay_ms_over_time": float(np.nanmax(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float))),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "summary_roundtrip_lst60_vs_static_0056_0061.csv", index=False, encoding="utf-8-sig")
    return out


def plot_metric(df: pd.DataFrame, out_dir: Path, *, suffix: str, ylabel: str, filename: str) -> None:
    styles = {
        "dynamic_roundtrip_lst60": ("dynamic 056->061->056 LST=60", "#d97706", 2.2, 0.95),
        "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.35, 0.82),
        "motif000061": ("motif000061 DCD | C--", "#C1121F", 1.30, 0.72),
    }
    fig, ax = plt.subplots(figsize=(14.5, 6.2), dpi=150)
    hours = df["hour"].to_numpy(dtype=float)
    for name, (label, color, linewidth, alpha) in styles.items():
        ax.plot(hours, df[f"{name}_{suffix}"], label=label, color=color, linewidth=linewidth, alpha=alpha)
    ax.axvline(10.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.axvline(15.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.text(10.03, ax.get_ylim()[1], "10h 56->61", fontsize=9, color="#111827", va="top")
    ax.text(15.03, ax.get_ylim()[1], "15h 61->56", fontsize=9, color="#111827", va="top")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe {ylabel}: roundtrip dynamic LST=60 vs static motifs")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else DEFAULT_OUT_ROOT / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    motif56 = one_way.load_topology_lite(56)
    motif61 = one_way.load_topology_lite(61)
    edge_table = one_way.union_edge_table([motif56, motif61])
    forward_events = one_way.read_schedule_events(Path(args.forward_lst_dir) / "schedule_1s.csv")
    reverse_events = one_way.read_schedule_events(Path(args.reverse_lst_dir) / "schedule_1s.csv")
    group_data = load_or_build_group_data(
        xml_file=one_way.GROUP_XML,
        group_cache_dir=one_way.GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )
    delay_store = FullLinkDelayStore(one_way.DELAY_STORE_DIR)
    position_store = PositionCacheStore(one_way.POSITION_CACHE_DIR)

    setup_summary = summarize_setup(Path(args.forward_lst_dir), Path(args.reverse_lst_dir), out_dir)
    df = compute_roundtrip_metrics(
        steps=steps,
        edge_table=edge_table,
        motif56=motif56,
        motif61=motif61,
        forward_events=forward_events,
        reverse_events=reverse_events,
        group_data=group_data,
        delay_store=delay_store,
        position_store=position_store,
        progress_every=int(args.progress_every),
    )
    df.to_csv(out_dir / "timeseries_roundtrip_lst60_vs_static_0056_0061.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "steps.npy", np.asarray(steps, dtype=np.int64))
    for name in ("dynamic_roundtrip_lst60", "motif000056", "motif000061"):
        np.save(out_dir / f"{name}_mean_shortest_hops.npy", df[f"{name}_mean_shortest_hops"].to_numpy(np.float32))
        np.save(
            out_dir / f"{name}_mean_shortest_delay_ms.npy",
            df[f"{name}_mean_shortest_delay_ms"].to_numpy(np.float32),
        )
    metric_summary = write_metric_summary(df, out_dir)
    period_summary = summarize_periods(df, out_dir)
    plot_metric(
        df,
        out_dir,
        suffix="mean_shortest_hops",
        ylabel="mean shortest hops",
        filename="roundtrip_lst60_vs_static_0056_0061_hops.png",
    )
    plot_metric(
        df,
        out_dir,
        suffix="mean_shortest_delay_ms",
        ylabel="mean shortest delay (ms)",
        filename="roundtrip_lst60_vs_static_0056_0061_delay_ms.png",
    )
    meta = {
        "pair": one_way.PAIR_KEY,
        "forward_lst_dir": str(args.forward_lst_dir),
        "reverse_lst_dir": str(args.reverse_lst_dir),
        "steps": {"start": int(args.start), "end": int(args.end), "stride": int(args.stride), "count": len(steps)},
        "roundtrip_state_rule": (
            "Start in motif000056. A scheduled 056->061 link is unroutable during its setup window, "
            "then becomes motif000061. A 061->056 schedule is applied only to owners that actually "
            "completed the forward switch; owners exempted in the forward switch remain in motif000056."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("setup summary:")
    print(setup_summary.to_string(index=False))
    print("\nmetric summary:")
    print(metric_summary.to_string(index=False))
    print("\nperiod summary:")
    print(period_summary.to_string(index=False))
    print(f"\nout_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
