from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION  # noqa: E402


DEFAULT_EXPERIMENT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\full_link_guided_spike_patch"
    r"\t0_86160_stride60"
)
DEFAULT_FULL_LINK_CACHE = Path(r"E:\paper11\data\linshi\g60_multi_region_weighted_betweenness_t0_86164_stride60")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and validate the full-link-guided spike patch / dynamic one-right study."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--full-link-cache", type=Path, default=DEFAULT_FULL_LINK_CACHE)
    return parser.parse_args()


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src <= dst else (dst, src)


def read_edge_table_csv(path: Path) -> EdgeTable:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=[str(i + 1) for i in range(int(G60_CONFIG.total_sats))],
    )


def inter_owner_target(edge_table: EdgeTable, idx: int) -> tuple[int, int] | None:
    if int(edge_table.option[idx]) == INTRA_OPTION:
        return None
    src = int(edge_table.src[idx])
    dst = int(edge_table.dst[idx])
    sp = int(edge_table.src_plane[idx])
    dp = int(edge_table.dst_plane[idx])
    if sp < dp:
        return src, dst
    if dp < sp:
        return dst, src
    return None


def validate_dynamic_mask(*, edge_table: EdgeTable, active_mask: np.ndarray, steps: np.ndarray, out_dir: Path) -> dict[str, Any]:
    owner_by_edge: dict[int, int] = {}
    target_by_edge: dict[int, int] = {}
    inter_edge_indices: list[int] = []
    for idx in range(int(edge_table.num_edges)):
        owner_target = inter_owner_target(edge_table, idx)
        if owner_target is None:
            continue
        owner_by_edge[int(idx)] = int(owner_target[0])
        target_by_edge[int(idx)] = int(owner_target[1])
        inter_edge_indices.append(int(idx))

    rows: list[dict[str, Any]] = []
    max_owner_degree = 0
    max_target_degree = 0
    violating_rows = 0
    for row_idx, step in enumerate(np.asarray(steps, dtype=np.int64)):
        selected = [int(x) for x in np.flatnonzero(active_mask[int(row_idx)]) if int(x) in owner_by_edge]
        owners: dict[int, int] = {}
        targets: dict[int, int] = {}
        for edge_idx in selected:
            owners[owner_by_edge[edge_idx]] = owners.get(owner_by_edge[edge_idx], 0) + 1
            targets[target_by_edge[edge_idx]] = targets.get(target_by_edge[edge_idx], 0) + 1
        owner_max = max(owners.values(), default=0)
        target_max = max(targets.values(), default=0)
        max_owner_degree = max(max_owner_degree, owner_max)
        max_target_degree = max(max_target_degree, target_max)
        violates = owner_max > 1 or target_max > 1
        violating_rows += int(violates)
        rows.append(
            {
                "step": int(step),
                "hour": float(step) / 3600.0,
                "selected_inter_edges": int(len(selected)),
                "unique_owners": int(len(owners)),
                "unique_right_targets": int(len(targets)),
                "max_owner_right_degree": int(owner_max),
                "max_target_left_degree": int(target_max),
                "violates_one_right_or_left": int(violates),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dynamic_one_right_constraint_check.csv", index=False, encoding="utf-8-sig")
    return {
        "steps": int(len(rows)),
        "violating_rows": int(violating_rows),
        "max_owner_right_degree": int(max_owner_degree),
        "max_target_left_degree": int(max_target_degree),
        "mean_selected_inter_edges": float(df["selected_inter_edges"].mean()),
        "min_selected_inter_edges": int(df["selected_inter_edges"].min()),
        "max_selected_inter_edges": int(df["selected_inter_edges"].max()),
    }


def summarize_metrics(compare: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ("mean_shortest_delay_ms", "mean_shortest_hops"):
        for topo in ("full_link", "dynamic_one_right", "motif000056", "motif000061", "spike_patch"):
            col = f"{metric}_{topo}"
            if col not in compare.columns:
                continue
            values = compare[col].to_numpy(dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "topology": topo,
                    "mean": float(np.nanmean(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "p95": float(np.nanpercentile(values, 95)),
                    "gap_mean_vs_full_link": (
                        0.0
                        if topo == "full_link"
                        else float(np.nanmean(values - compare[f"{metric}_full_link"].to_numpy(dtype=float)))
                    ),
                }
            )
    return rows


def summarize_transitions(transitions: pd.DataFrame) -> dict[str, Any]:
    sub = transitions.iloc[1:].copy() if len(transitions) > 1 else transitions.copy()
    return {
        "mean_new_inter_edges_per_step": float(sub["new_inter_edges"].mean()),
        "p95_new_inter_edges_per_step": float(sub["new_inter_edges"].quantile(0.95)),
        "max_new_inter_edges_per_step": int(sub["new_inter_edges"].max()),
        "mean_symmetric_diff_inter_edges_per_step": float(sub["symmetric_diff_inter_edges"].mean()),
        "p95_symmetric_diff_inter_edges_per_step": float(sub["symmetric_diff_inter_edges"].quantile(0.95)),
        "max_symmetric_diff_inter_edges_per_step": int(sub["symmetric_diff_inter_edges"].max()),
    }


def summarize_option_usage(option_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for option, part in option_df.groupby("option"):
        rows.append(
            {
                "option": int(option),
                "mean_usage_share_sum_at_spikes": float(part["usage_share_sum"].mean()),
                "max_usage_share_sum_at_spikes": float(part["usage_share_sum"].max()),
                "mean_nonzero_edges_at_spikes": float(part["nonzero_edges"].mean()),
                "steps_with_nonzero": int((part["usage_share_sum"] > 0).sum()),
            }
        )
    return sorted(rows, key=lambda item: item["option"])


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metric_rows = payload["metric_summary"]
    jump_rows = payload["local_spike_jump_summary"]
    option_rows = payload["option_usage_summary"]
    transition = payload["transition_summary"]
    constraint = payload["constraint_check"]

    def metric_line(metric: str, topo: str) -> str:
        row = next(item for item in metric_rows if item["metric"] == metric and item["topology"] == topo)
        return f"{row['mean']:.3f} (gap vs full_link {row['gap_mean_vs_full_link']:.3f})"

    def jump_line(metric: str, topo: str) -> str:
        row = next(item for item in jump_rows if item["metric"] == metric and item["topology"] == topo)
        return f"mean jump {row['mean_positive_jump']:.3f}, max jump {row['max_positive_jump']:.3f}"

    lines = [
        "# Full-link guided spike patch study",
        "",
        "## Conclusion",
        "",
        (
            "The common China-Europe spikes in motif000056 and motif000061 can be suppressed by a "
            "dynamic one-right topology guided by full_link shortest-delay edge usage. A single static "
            "patch can reduce the selected spike jumps, but it introduces a large mid-day side effect; "
            "the dynamic one-right version follows the moving full_link corridor and avoids that side effect."
        ),
        "",
        "## Metric evidence",
        "",
        f"- Mean delay full_link: {metric_line('mean_shortest_delay_ms', 'full_link')}",
        f"- Mean delay dynamic_one_right: {metric_line('mean_shortest_delay_ms', 'dynamic_one_right')}",
        f"- Mean delay motif000056: {metric_line('mean_shortest_delay_ms', 'motif000056')}",
        f"- Mean hops full_link: {metric_line('mean_shortest_hops', 'full_link')}",
        f"- Mean hops dynamic_one_right: {metric_line('mean_shortest_hops', 'dynamic_one_right')}",
        f"- Mean hops motif000056: {metric_line('mean_shortest_hops', 'motif000056')}",
        "",
        "## Local spike evidence",
        "",
        f"- Delay dynamic_one_right: {jump_line('mean_shortest_delay_ms', 'dynamic_one_right')}",
        f"- Delay full_link: {jump_line('mean_shortest_delay_ms', 'full_link')}",
        f"- Delay motif000056: {jump_line('mean_shortest_delay_ms', 'motif000056')}",
        f"- Hops dynamic_one_right: {jump_line('mean_shortest_hops', 'dynamic_one_right')}",
        f"- Hops full_link: {jump_line('mean_shortest_hops', 'full_link')}",
        f"- Hops motif000056: {jump_line('mean_shortest_hops', 'motif000056')}",
        "",
        "## Constraint check",
        "",
        f"- Violating rows: {constraint['violating_rows']} / {constraint['steps']}",
        f"- Max owner right degree: {constraint['max_owner_right_degree']}",
        f"- Max target left degree: {constraint['max_target_left_degree']}",
        (
            f"- Selected inter edges: mean {constraint['mean_selected_inter_edges']:.1f}, "
            f"range {constraint['min_selected_inter_edges']}..{constraint['max_selected_inter_edges']}"
        ),
        "",
        "## Switching cost",
        "",
        f"- Mean new inter edges per 60s: {transition['mean_new_inter_edges_per_step']:.2f}",
        f"- 95th percentile new inter edges per 60s: {transition['p95_new_inter_edges_per_step']:.1f}",
        f"- Max new inter edges per 60s: {transition['max_new_inter_edges_per_step']}",
        f"- Mean symmetric-difference inter edges per 60s: {transition['mean_symmetric_diff_inter_edges_per_step']:.2f}",
        "",
        "## Full-link mechanism",
        "",
        (
            "At the selected common spike steps, full_link routes are not using a fixed motif. "
            "Early spikes are dominated by option 1 diagonal links plus intra links, while later "
            "spikes mix option 1 with option 2, option 0, and some option 4. This is why a fixed "
            "periodic motif can hit a phase mismatch during region handover, while full_link can "
            "switch to the currently favorable corridor."
        ),
        "",
        "| option | mean usage-share sum | max usage-share sum | mean nonzero edges | nonzero steps |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in option_rows:
        lines.append(
            f"| {row['option']} | {row['mean_usage_share_sum_at_spikes']:.3f} | "
            f"{row['max_usage_share_sum_at_spikes']:.3f} | "
            f"{row['mean_nonzero_edges_at_spikes']:.1f} | {row['steps_with_nonzero']} |"
        )
    lines.extend(
        [
            "",
            "## Key files",
            "",
            "- `china_europe_spike_patch_vs_056_061_full_link_delay_ms.png`",
            "- `china_europe_spike_patch_vs_056_061_full_link_hops.png`",
            "- `dynamic_one_right_viewer_t5220.png`",
            "- `dynamic_one_right_active_full_edge_mask.npy`",
            "- `dynamic_one_right_constraint_check.csv`",
            "- `full_link_spike_option_usage_by_step.csv`",
            "- `full_link_spike_used_edges_summary.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.experiment_dir)
    cache_dir = Path(args.full_link_cache)
    edge_table = read_edge_table_csv(cache_dir / "edges.csv")
    steps = np.asarray(np.load(cache_dir / "time_indices.npy", mmap_mode="r"), dtype=np.int64)
    dynamic_mask = np.load(out_dir / "dynamic_one_right_active_full_edge_mask.npy", mmap_mode="r")
    constraint = validate_dynamic_mask(edge_table=edge_table, active_mask=dynamic_mask, steps=steps, out_dir=out_dir)

    compare = pd.read_csv(out_dir / "compare_spike_patch_vs_056_061_full_link.csv")
    metrics = summarize_metrics(compare)
    pd.DataFrame(metrics).to_csv(out_dir / "global_metric_summary.csv", index=False, encoding="utf-8-sig")

    jumps = pd.read_csv(out_dir / "local_spike_jump_summary.csv").to_dict(orient="records")
    transitions = summarize_transitions(pd.read_csv(out_dir / "dynamic_one_right_transition_counts.csv"))
    options = summarize_option_usage(pd.read_csv(out_dir / "full_link_spike_option_usage_by_step.csv"))

    payload = {
        "experiment_dir": str(out_dir),
        "constraint_check": constraint,
        "metric_summary": metrics,
        "local_spike_jump_summary": jumps,
        "transition_summary": transitions,
        "option_usage_summary": options,
    }
    (out_dir / "mechanism_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir / "full_link_spike_mechanism_report.md", payload)
    print(f"report={out_dir / 'full_link_spike_mechanism_report.md'}", flush=True)
    print(f"constraint={out_dir / 'dynamic_one_right_constraint_check.csv'}", flush=True)
    print(f"summary_json={out_dir / 'mechanism_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
