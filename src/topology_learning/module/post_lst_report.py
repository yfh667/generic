from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _source_run_name(summary_path: str | Path, row: Mapping[str, Any]) -> str:
    """Return a namespace that distinguishes same-named schedules.

    Batch reports often merge sweep summaries from many selector experiments.
    Several experiments can produce the same schedule label, for example
    ``budget_dp_b2628_cap408_meanref``, while the actual selected topology
    sequence differs. The per-row ``eval_dir`` has the layout
    ``<run_root>/<pair_key>/<schedule_eval_dir>``; ``run_root`` is therefore a
    stable namespace across the three pair summaries of the same experiment.
    """

    eval_dir = row.get("eval_dir")
    if eval_dir not in (None, ""):
        path = Path(str(eval_dir))
        pair_key = str(row.get("pair_key", ""))
        if pair_key and path.parent.name == pair_key and len(path.parts) >= 3:
            return path.parent.parent.name
        if len(path.parts) >= 2:
            return path.parent.name
    path = Path(summary_path)
    if len(path.parts) >= 3:
        return path.parent.parent.name
    return path.parent.name


def _write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _is_dominated(row: Mapping[str, Any], others: Sequence[Mapping[str, Any]]) -> bool:
    quality = float(row["mean_all_gap"])
    setup = float(row["total_setup_commands"])
    max_setup = float(row["max_setup_commands_per_step"])
    for other in others:
        if other is row:
            continue
        other_quality = float(other["mean_all_gap"])
        other_setup = float(other["total_setup_commands"])
        other_max = float(other["max_setup_commands_per_step"])
        no_worse = other_quality <= quality and other_setup <= setup and other_max <= max_setup
        strictly_better = other_quality < quality or other_setup < setup or other_max < max_setup
        if no_worse and strictly_better:
            return True
    return False


def aggregate_post_lst_summaries(summary_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Aggregate one or more post-LST sweep summary CSVs by schedule."""

    grouped: dict[str, dict[str, Any]] = {}
    for path in summary_paths:
        for row in _read_csv(path):
            schedule = str(row.get("schedule", ""))
            if not schedule:
                continue
            source_run = _source_run_name(path, row)
            schedule_key = f"{source_run}::{schedule}"
            item = grouped.setdefault(
                schedule_key,
                {
                    "schedule": schedule_key,
                    "source_run": source_run,
                    "original_schedule": schedule,
                    "pair_keys": [],
                    "delay_gaps": [],
                    "hop_gaps": [],
                    "total_setup_commands": _float_or_none(row.get("total_setup_commands")) or 0.0,
                    "max_setup_commands_per_step": _float_or_none(row.get("max_setup_commands_per_step")) or 0.0,
                    "mean_setup_commands_per_step": _float_or_none(row.get("mean_setup_commands_per_step")) or 0.0,
                    "mean_building_edges": _float_or_none(row.get("mean_building_edges")) or 0.0,
                    "max_building_edges": _float_or_none(row.get("max_building_edges")) or 0.0,
                    "building_edge_seconds": _float_or_none(row.get("building_edge_seconds")) or 0.0,
                    "num_switches": _float_or_none(row.get("num_switches")) or 0.0,
                },
            )
            pair_key = str(row.get("pair_key", Path(path).parent.name))
            item["pair_keys"].append(pair_key)
            delay_gap = _float_or_none(row.get("post_lst_delay_ms_gap"))
            hop_gap = _float_or_none(row.get("post_lst_hops_gap"))
            if delay_gap is not None:
                item["delay_gaps"].append(delay_gap)
            if hop_gap is not None:
                item["hop_gaps"].append(hop_gap)
            item["total_setup_commands"] = max(
                float(item["total_setup_commands"]),
                _float_or_none(row.get("total_setup_commands")) or 0.0,
            )
            item["max_setup_commands_per_step"] = max(
                float(item["max_setup_commands_per_step"]),
                _float_or_none(row.get("max_setup_commands_per_step")) or 0.0,
            )
            item["mean_setup_commands_per_step"] = max(
                float(item["mean_setup_commands_per_step"]),
                _float_or_none(row.get("mean_setup_commands_per_step")) or 0.0,
            )
            item["mean_building_edges"] = max(
                float(item["mean_building_edges"]),
                _float_or_none(row.get("mean_building_edges")) or 0.0,
            )
            item["max_building_edges"] = max(
                float(item["max_building_edges"]),
                _float_or_none(row.get("max_building_edges")) or 0.0,
            )
            item["building_edge_seconds"] = max(
                float(item["building_edge_seconds"]),
                _float_or_none(row.get("building_edge_seconds")) or 0.0,
            )
            item["num_switches"] = max(float(item["num_switches"]), _float_or_none(row.get("num_switches")) or 0.0)

    rows: list[dict[str, Any]] = []
    for item in grouped.values():
        delay = np.asarray(item["delay_gaps"], dtype=np.float64)
        hops = np.asarray(item["hop_gaps"], dtype=np.float64)
        combined = np.concatenate([delay, hops]) if delay.size or hops.size else np.asarray([], dtype=np.float64)
        rows.append(
            {
                "schedule": item["schedule"],
                "source_run": item["source_run"],
                "original_schedule": item["original_schedule"],
                "pair_count": int(len(set(str(x) for x in item["pair_keys"]))),
                "pair_keys": ";".join(sorted(set(str(x) for x in item["pair_keys"]))),
                "mean_delay_gap": float(np.mean(delay)) if delay.size else math.nan,
                "mean_hop_gap": float(np.mean(hops)) if hops.size else math.nan,
                "mean_all_gap": float(np.mean(combined)) if combined.size else math.nan,
                "max_all_gap": float(np.max(combined)) if combined.size else math.nan,
                "total_setup_commands": int(round(float(item["total_setup_commands"]))),
                "max_setup_commands_per_step": int(round(float(item["max_setup_commands_per_step"]))),
                "mean_setup_commands_per_step": float(item["mean_setup_commands_per_step"]),
                "mean_building_edges": float(item["mean_building_edges"]),
                "max_building_edges": int(round(float(item["max_building_edges"]))),
                "building_edge_seconds": float(item["building_edge_seconds"]),
                "num_switches": int(round(float(item["num_switches"]))),
            }
        )
    rows.sort(key=lambda row: (float(row["mean_all_gap"]), int(row["total_setup_commands"])))
    for row in rows:
        row["pareto"] = not _is_dominated(row, rows)
    return rows


def choose_recommendations(rows: Sequence[Mapping[str, Any]], *, quality_tolerance: float = 0.01) -> dict[str, Any]:
    if not rows:
        return {}
    finite = [row for row in rows if math.isfinite(float(row["mean_all_gap"]))]
    if not finite:
        return {}
    best_quality = min(finite, key=lambda row: float(row["mean_all_gap"]))
    dynamic = [row for row in finite if int(row["total_setup_commands"]) > 0]
    min_setup_dynamic = min(dynamic, key=lambda row: int(row["total_setup_commands"])) if dynamic else best_quality
    threshold = float(best_quality["mean_all_gap"]) * (1.0 + float(quality_tolerance))
    within = [row for row in finite if float(row["mean_all_gap"]) <= threshold]
    efficient = min(within, key=lambda row: int(row["total_setup_commands"])) if within else best_quality

    gaps = np.asarray([float(row["mean_all_gap"]) for row in finite], dtype=np.float64)
    setups = np.asarray([float(row["total_setup_commands"]) for row in finite], dtype=np.float64)
    gap_span = float(np.max(gaps) - np.min(gaps)) or 1.0
    setup_span = float(np.max(setups) - np.min(setups)) or 1.0
    knee = min(
        finite,
        key=lambda row: ((float(row["mean_all_gap"]) - float(np.min(gaps))) / gap_span) ** 2
        + ((float(row["total_setup_commands"]) - float(np.min(setups))) / setup_span) ** 2,
    )

    def compact(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schedule": row["schedule"],
            "source_run": row.get("source_run", ""),
            "original_schedule": row.get("original_schedule", row["schedule"]),
            "mean_all_gap": float(row["mean_all_gap"]),
            "mean_delay_gap": float(row["mean_delay_gap"]),
            "mean_hop_gap": float(row["mean_hop_gap"]),
            "total_setup_commands": int(row["total_setup_commands"]),
            "max_setup_commands_per_step": int(row["max_setup_commands_per_step"]),
            "mean_setup_commands_per_step": float(row.get("mean_setup_commands_per_step", 0.0)),
            "mean_building_edges": float(row.get("mean_building_edges", 0.0)),
            "max_building_edges": int(row.get("max_building_edges", 0)),
            "building_edge_seconds": float(row.get("building_edge_seconds", 0.0)),
            "num_switches": int(row["num_switches"]),
        }

    return {
        "quality_tolerance": float(quality_tolerance),
        "best_quality": compact(best_quality),
        "min_setup_dynamic": compact(min_setup_dynamic),
        "efficient_within_tolerance": compact(efficient),
        "knee": compact(knee),
    }


def build_decision_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_tolerance: float = 0.01,
) -> list[dict[str, Any]]:
    """Add first-goal/second-goal comparison fields to post-LST rows.

    The first goal is closeness to full_link, represented by ``mean_all_gap``.
    The second goal is dynamic setup reduction, represented by total setup
    commands and the maximum setup commands in a sampled step.
    """

    finite = [row for row in rows if math.isfinite(float(row["mean_all_gap"]))]
    if not finite:
        return []
    best_quality = min(finite, key=lambda row: float(row["mean_all_gap"]))
    best_gap = float(best_quality["mean_all_gap"])
    best_setup = float(best_quality["total_setup_commands"])
    best_max_setup = float(best_quality["max_setup_commands_per_step"])
    threshold = best_gap * (1.0 + float(quality_tolerance))

    quality_rank = {
        str(row["schedule"]): rank
        for rank, row in enumerate(sorted(finite, key=lambda item: float(item["mean_all_gap"])), start=1)
    }
    setup_rank = {
        str(row["schedule"]): rank
        for rank, row in enumerate(sorted(finite, key=lambda item: float(item["total_setup_commands"])), start=1)
    }
    max_setup_rank = {
        str(row["schedule"]): rank
        for rank, row in enumerate(sorted(finite, key=lambda item: float(item["max_setup_commands_per_step"])), start=1)
    }

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        schedule = str(item["schedule"])
        gap = float(item["mean_all_gap"])
        setup = float(item["total_setup_commands"])
        max_setup = float(item["max_setup_commands_per_step"])
        item["quality_rank"] = int(quality_rank.get(schedule, 0))
        item["total_setup_rank"] = int(setup_rank.get(schedule, 0))
        item["max_setup_rank"] = int(max_setup_rank.get(schedule, 0))
        item["within_quality_tolerance"] = bool(gap <= threshold)
        item["quality_delta_vs_best"] = float(gap - best_gap)
        item["quality_delta_pct_vs_best"] = float((gap - best_gap) / best_gap) if abs(best_gap) > 1e-12 else math.nan
        item["total_setup_delta_vs_best_quality"] = int(round(setup - best_setup))
        item["total_setup_reduction_vs_best_quality"] = int(round(best_setup - setup))
        item["total_setup_reduction_pct_vs_best_quality"] = (
            float((best_setup - setup) / best_setup) if abs(best_setup) > 1e-12 else math.nan
        )
        item["max_setup_delta_vs_best_quality"] = int(round(max_setup - best_max_setup))
        item["max_setup_reduction_vs_best_quality"] = int(round(best_max_setup - max_setup))
        item["max_setup_reduction_pct_vs_best_quality"] = (
            float((best_max_setup - max_setup) / best_max_setup) if abs(best_max_setup) > 1e-12 else math.nan
        )
        out.append(item)
    out.sort(
        key=lambda item: (
            not bool(item["within_quality_tolerance"]),
            int(item["total_setup_commands"]),
            float(item["mean_all_gap"]),
        )
    )
    return out


def build_quality_tolerance_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerances: Sequence[float],
) -> list[dict[str, Any]]:
    """Summarise the second-goal optimum inside first-goal tolerance bands.

    The first goal is represented by ``mean_all_gap``: smaller means closer to
    full_link across the selected delay/hop metrics. For each tolerance band
    around the best gap, this table reports which schedule minimises total setup
    commands and which schedule minimises peak setup commands.
    """

    finite = [row for row in rows if math.isfinite(float(row["mean_all_gap"]))]
    if not finite:
        return []
    best = min(finite, key=lambda row: float(row["mean_all_gap"]))
    best_gap = float(best["mean_all_gap"])
    best_setup = float(best["total_setup_commands"])
    best_peak = float(best["max_setup_commands_per_step"])

    unique_tolerances = sorted({max(0.0, float(tol)) for tol in tolerances})
    out: list[dict[str, Any]] = []

    def compact(choice: Mapping[str, Any], *, tolerance: float, rule: str, candidate_count: int) -> dict[str, Any]:
        gap = float(choice["mean_all_gap"])
        setup = float(choice["total_setup_commands"])
        peak = float(choice["max_setup_commands_per_step"])
        return {
            "quality_tolerance": float(tolerance),
            "quality_threshold": float(best_gap * (1.0 + float(tolerance))),
            "selection_rule": str(rule),
            "candidate_count": int(candidate_count),
            "schedule": str(choice["schedule"]),
            "source_run": str(choice.get("source_run", "")),
            "original_schedule": str(choice.get("original_schedule", choice["schedule"])),
            "mean_all_gap": gap,
            "mean_delay_gap": float(choice["mean_delay_gap"]),
            "mean_hop_gap": float(choice["mean_hop_gap"]),
            "quality_delta_vs_best": float(gap - best_gap),
            "quality_delta_pct_vs_best": float((gap - best_gap) / best_gap) if abs(best_gap) > 1e-12 else math.nan,
            "total_setup_commands": int(round(setup)),
            "total_setup_reduction_vs_best_quality": int(round(best_setup - setup)),
            "total_setup_reduction_pct_vs_best_quality": (
                float((best_setup - setup) / best_setup) if abs(best_setup) > 1e-12 else math.nan
            ),
            "max_setup_commands_per_step": int(round(peak)),
            "mean_setup_commands_per_step": float(choice.get("mean_setup_commands_per_step", 0.0)),
            "mean_building_edges": float(choice.get("mean_building_edges", 0.0)),
            "max_building_edges": int(round(float(choice.get("max_building_edges", 0.0)))),
            "building_edge_seconds": float(choice.get("building_edge_seconds", 0.0)),
            "max_setup_reduction_vs_best_quality": int(round(best_peak - peak)),
            "max_setup_reduction_pct_vs_best_quality": (
                float((best_peak - peak) / best_peak) if abs(best_peak) > 1e-12 else math.nan
            ),
            "num_switches": int(round(float(choice["num_switches"]))),
        }

    for tolerance in unique_tolerances:
        threshold = best_gap * (1.0 + float(tolerance))
        candidates = [row for row in finite if float(row["mean_all_gap"]) <= threshold]
        if not candidates:
            continue
        candidate_count = len(candidates)
        out.append(
            compact(
                min(candidates, key=lambda row: float(row["mean_all_gap"])),
                tolerance=tolerance,
                rule="best_quality",
                candidate_count=candidate_count,
            )
        )
        out.append(
            compact(
                min(candidates, key=lambda row: (int(row["total_setup_commands"]), float(row["mean_all_gap"]))),
                tolerance=tolerance,
                rule="min_total_setup",
                candidate_count=candidate_count,
            )
        )
        out.append(
            compact(
                min(
                    candidates,
                    key=lambda row: (
                        int(row["max_setup_commands_per_step"]),
                        int(row["total_setup_commands"]),
                        float(row["mean_all_gap"]),
                    ),
                ),
                tolerance=tolerance,
                rule="min_peak_setup",
                candidate_count=candidate_count,
            )
        )
    return out


def _short_schedule_label(name: str) -> str:
    text = str(name)
    if "::" in text:
        text = text.split("::", 1)[1]
    prefix = "dp_new_edge_penalty_"
    if text.startswith(prefix):
        return "p=" + text[len(prefix) :]
    burst_prefix = "dp_burst2_penalty_"
    if text.startswith(burst_prefix):
        return "b2=" + text[len(burst_prefix) :]
    cap_prefix = "dp_new_edge_cap_"
    if text.startswith(cap_prefix):
        return "cap=" + text[len(cap_prefix) :]
    dwell_token = "_new_edge_penalty_"
    if text.startswith("dp_dwell") and dwell_token in text:
        left, penalty = text.split(dwell_token, 1)
        return "dw" + left[len("dp_dwell") :] + ",p=" + penalty
    dwell_burst_token = "_burst2_penalty_"
    if text.startswith("dp_dwell") and dwell_burst_token in text:
        left, penalty = text.split(dwell_burst_token, 1)
        return "dw" + left[len("dp_dwell") :] + ",b2=" + penalty
    if text == "greedy_no_switch_penalty":
        return "greedy"
    if text == "static_best":
        return "static"
    return text


def plot_pareto_report(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray([float(row["total_setup_commands"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["mean_all_gap"]) for row in rows], dtype=np.float64)
    max_setup = np.asarray([float(row["max_setup_commands_per_step"]) for row in rows], dtype=np.float64)
    pareto = np.asarray([bool(row.get("pareto", False)) for row in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=170)
    sc = ax.scatter(x[~pareto], y[~pareto], c=max_setup[~pareto], cmap="viridis", s=44, alpha=0.55)
    ax.scatter(x[pareto], y[pareto], c=max_setup[pareto], cmap="viridis", s=74, edgecolors="black", linewidths=0.6)
    for row, xi, yi in zip(rows, x, y):
        label = _short_schedule_label(str(row["schedule"]))
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlabel("total setup commands (symlog)")
    ax.set_ylabel("mean relative gap")
    ax.set_title("Post-LST Pareto report")
    ax.grid(True, alpha=0.25, which="both")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("max setup commands per step")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_post_lst_report(
    *,
    summary_paths: Sequence[str | Path],
    out_dir: str | Path,
    quality_tolerance: float = 0.01,
    frontier_tolerances: Sequence[float] | None = None,
    prefix: str = "post_lst",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate_post_lst_summaries(summary_paths)
    summary_csv = out_dir / f"{prefix}_pareto_summary.csv"
    decision_csv = out_dir / f"{prefix}_decision_table.csv"
    frontier_csv = out_dir / f"{prefix}_quality_tolerance_frontier.csv"
    recommendations_json = out_dir / f"{prefix}_recommendations.json"
    plot_png = out_dir / f"{prefix}_pareto_summary.png"
    _write_rows(summary_csv, rows)
    decision_rows = build_decision_table(rows, quality_tolerance=float(quality_tolerance))
    _write_rows(decision_csv, decision_rows)
    recommendations = choose_recommendations(rows, quality_tolerance=float(quality_tolerance))
    if frontier_tolerances is None:
        frontier_tolerances = [0.0, 0.005, float(quality_tolerance), 0.02, 0.05]
    frontier_rows = build_quality_tolerance_frontier(rows, tolerances=frontier_tolerances)
    _write_rows(frontier_csv, frontier_rows)
    payload = {
        "summary_paths": [str(Path(path)) for path in summary_paths],
        "quality_tolerance": float(quality_tolerance),
        "frontier_tolerances": [float(x) for x in sorted({max(0.0, float(tol)) for tol in frontier_tolerances})],
        "recommendations": recommendations,
        "outputs": {
            "summary_csv": str(summary_csv),
            "decision_table_csv": str(decision_csv),
            "quality_tolerance_frontier_csv": str(frontier_csv),
            "recommendations_json": str(recommendations_json),
            "plot_png": str(plot_png),
        },
    }
    recommendations_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_pareto_report(plot_png, rows)
    return payload
