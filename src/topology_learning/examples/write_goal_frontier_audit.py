from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.full_link_gap_selector import _write_rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _best_by(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    reverse: bool = False,
) -> dict[str, Any] | None:
    rows = list(rows)
    if not rows:
        return None
    return dict(sorted(rows, key=lambda row: _float(row, key), reverse=reverse)[0])


def _frontier_rows(frontier_csv: Path) -> list[dict[str, Any]]:
    rows = _read_rows(frontier_csv)
    out = []
    for row in rows:
        out.append(
            {
                "quality_tolerance": _float(row, "quality_tolerance"),
                "quality_threshold": _float(row, "quality_threshold"),
                "selection_rule": row.get("selection_rule", ""),
                "schedule": row.get("schedule", ""),
                "source_run": row.get("source_run", ""),
                "original_schedule": row.get("original_schedule", ""),
                "mean_all_gap": _float(row, "mean_all_gap"),
                "mean_delay_gap": _float(row, "mean_delay_gap"),
                "mean_hop_gap": _float(row, "mean_hop_gap"),
                "total_setup_commands": _int(row, "total_setup_commands"),
                "max_setup_commands_per_step": _int(row, "max_setup_commands_per_step"),
                "num_switches": _int(row, "num_switches"),
            }
        )
    return out


def _boundary_summary(boundary_csv: Path, quality_threshold: float) -> dict[str, Any]:
    rows = _read_rows(boundary_csv)
    parsed = []
    for row in rows:
        parsed.append(
            {
                "source_run": row.get("source_run", ""),
                "schedule": row.get("schedule", ""),
                "mean_all_global_gap": _float(row, "mean_all_global_gap"),
                "mean_delay_global_gap": _float(row, "mean_delay_global_gap"),
                "mean_hop_global_gap": _float(row, "mean_hop_global_gap"),
                "total_setup_commands": _int(row, "total_setup_commands"),
                "max_setup_commands_per_step": _int(row, "max_setup_commands_per_step"),
                "within_quality_threshold": str(row.get("within_quality_threshold", "")).lower() == "true",
            }
        )
    best = _best_by(parsed, key="mean_all_global_gap")
    return {
        "path": str(boundary_csv),
        "num_rows": len(parsed),
        "quality_threshold": float(quality_threshold),
        "best": best,
        "num_within_threshold": sum(1 for row in parsed if bool(row.get("within_quality_threshold"))),
    }


def _splice_summaries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        payload = _read_json(path)
        if not payload:
            continue
        out.append(
            {
                "name": path.parent.name,
                "path": str(path),
                "base_gap": payload.get("base_gap"),
                "base_setup": payload.get("base_setup"),
                "target_setup": payload.get("target_setup"),
                "num_donor_schedules": payload.get("num_donor_schedules"),
                "num_interval_candidates": payload.get("num_interval_candidates"),
                "num_saving_candidates_used": payload.get("num_saving_candidates_used"),
                "num_quality_candidates_used": payload.get("num_quality_candidates_used"),
                "checked_pairs": payload.get("checked_pairs"),
                "num_feasible_splices": payload.get("num_feasible_splices"),
            }
        )
    return out


def _segment_summaries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        payload = _read_json(path)
        if not payload:
            continue
        score = payload.get("score", {}) or {}
        out.append(
            {
                "name": path.parent.name,
                "path": str(path),
                "budget": payload.get("budget"),
                "used_budget": payload.get("result_used_budget", payload.get("used_budget")),
                "same_as_base": payload.get("same_as_base"),
                "mean_all_global_gap": score.get("mean_all_global_gap"),
                "mean_delay_global_gap": score.get("mean_delay_global_gap"),
                "mean_hop_global_gap": score.get("mean_hop_global_gap"),
                "total_setup_commands": score.get("total_setup_commands"),
                "within_quality_threshold": score.get("within_quality_threshold"),
            }
        )
    return out


def _local_window_summaries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        payload = _read_json(path)
        if not payload:
            continue
        score = payload.get("score", {}) or {}
        out.append(
            {
                "name": path.parent.name,
                "path": str(path),
                "left_row": payload.get("left_row"),
                "right_row": payload.get("right_row"),
                "target_total_setup": payload.get("target_total_setup"),
                "local_budget": payload.get("local_budget", payload.get("budget")),
                "local_used_budget": payload.get("local_used_budget", payload.get("used_budget")),
                "outside_setup": payload.get("outside_setup"),
                "same_as_base": score.get("same_as_base"),
                "mean_all_global_gap": score.get("mean_all_global_gap"),
                "mean_delay_global_gap": score.get("mean_delay_global_gap"),
                "mean_hop_global_gap": score.get("mean_hop_global_gap"),
                "total_setup_commands": score.get("total_setup_commands"),
                "within_quality_threshold": score.get("within_quality_threshold"),
            }
        )
    return out


def _budget_curve_summary(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    parsed = []
    for row in rows:
        parsed.append(
            {
                "source_summary": row.get("source_summary", ""),
                "schedule": row.get("schedule", ""),
                "budget_limit": _int(row, "budget_limit"),
                "dp_used_budget": _int(row, "dp_used_budget"),
                "total_setup_commands": _int(row, "total_setup_commands"),
                "max_setup_commands_per_step": _int(row, "max_setup_commands_per_step"),
                "num_switches": _int(row, "num_switches"),
                "mean_all_global_gap": _float(row, "mean_all_global_gap"),
                "mean_delay_global_gap": _float(row, "mean_delay_global_gap"),
                "mean_hop_global_gap": _float(row, "mean_hop_global_gap"),
                "quality_threshold": _float(row, "quality_threshold"),
                "within_quality_threshold": str(row.get("within_quality_threshold", "")).lower() == "true",
                "delta_to_threshold": _float(row, "delta_to_threshold"),
            }
        )
    within = [row for row in parsed if bool(row["within_quality_threshold"])]
    outside = [row for row in parsed if not bool(row["within_quality_threshold"])]
    best_within = (
        sorted(within, key=lambda row: (int(row["total_setup_commands"]), float(row["mean_all_global_gap"])))[0]
        if within
        else None
    )
    closest_outside = (
        sorted(outside, key=lambda row: (abs(float(row["delta_to_threshold"])), int(row["total_setup_commands"])))[0]
        if outside
        else None
    )
    return {
        "path": str(path),
        "num_rows": len(parsed),
        "best_within_threshold": best_within,
        "closest_outside_threshold": closest_outside,
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    frontier = payload["post_lst_frontier"]
    boundary = payload["b2628_boundary"]
    lines = [
        "# G60 topology learning frontier audit",
        "",
        f"Quality threshold used for strict 1% audit: `{payload['quality_threshold']}`.",
        "",
        "## Frontier by tolerance",
        "",
        "| tolerance | rule | schedule | gap | setup | peak | switches |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in frontier:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['quality_tolerance']:.3g}",
                    str(row["selection_rule"]),
                    str(row["schedule"]),
                    f"{row['mean_all_gap']:.10f}",
                    str(row["total_setup_commands"]),
                    str(row["max_setup_commands_per_step"]),
                    str(row["num_switches"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 2628 Boundary Search",
            "",
            f"Rows checked: `{boundary['num_rows']}`.",
            f"Rows within threshold: `{boundary['num_within_threshold']}`.",
        ]
    )
    best = boundary.get("best")
    if best:
        lines.extend(
            [
                "",
                "Best 2628-boundary candidate:",
                "",
                f"- source: `{best['source_run']}`",
                f"- schedule: `{best['schedule']}`",
                f"- mean_all_global_gap: `{best['mean_all_global_gap']}`",
                f"- setup: `{best['total_setup_commands']}`",
                f"- peak: `{best['max_setup_commands_per_step']}`",
            ]
        )
    lines.extend(["", "## Splice Audits", ""])
    for item in payload["splice_audits"]:
        lines.append(
            f"- `{item['name']}`: intervals `{item['num_interval_candidates']}`, "
            f"checked pairs `{item['checked_pairs']}`, feasible `{item['num_feasible_splices']}`."
        )
    lines.extend(["", "## Segment-Boundary All-Action DP", ""])
    for item in payload["segment_boundary_audits"]:
        lines.append(
            f"- `{item['name']}`: budget `{item['budget']}`, used `{item['used_budget']}`, "
            f"gap `{item['mean_all_global_gap']}`, setup `{item['total_setup_commands']}`, "
            f"within `{item['within_quality_threshold']}`."
        )
    if payload.get("local_window_audits"):
        lines.extend(["", "## Local-Window All-Action DP", ""])
        for item in payload["local_window_audits"]:
            lines.append(
                f"- `{item['name']}`: rows `{item['left_row']}..{item['right_row']}`, "
                f"target setup `{item['target_total_setup']}`, local budget `{item['local_budget']}`, "
                f"used `{item['local_used_budget']}`, gap `{item['mean_all_global_gap']}`, "
                f"setup `{item['total_setup_commands']}`, same_as_base `{item['same_as_base']}`, "
                f"within `{item['within_quality_threshold']}`."
            )
    if payload.get("budget_curve_audits"):
        lines.extend(["", "## Budget-Curve Audits", ""])
        for item in payload["budget_curve_audits"]:
            best = item.get("best_within_threshold")
            closest = item.get("closest_outside_threshold")
            lines.append(f"- `{Path(item['path']).name}`: rows `{item['num_rows']}`.")
            if best:
                lines.append(
                    f"  Best within threshold: `{best['schedule']}`, setup `{best['total_setup_commands']}`, "
                    f"gap `{best['mean_all_global_gap']}`."
                )
            if closest:
                lines.append(
                    f"  Closest outside threshold: `{closest['schedule']}`, setup `{closest['total_setup_commands']}`, "
                    f"gap `{closest['mean_all_global_gap']}`, delta `{closest['delta_to_threshold']}`."
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a consolidated audit for the current topology-learning goal.")
    parser.add_argument("--frontier-csv", type=Path, required=True)
    parser.add_argument("--boundary-csv", type=Path, required=True)
    parser.add_argument("--splice-summary", type=Path, action="append", default=None)
    parser.add_argument("--segment-summary", type=Path, action="append", default=None)
    parser.add_argument("--local-window-summary", type=Path, action="append", default=None)
    parser.add_argument("--budget-curve", type=Path, action="append", default=None)
    parser.add_argument("--quality-threshold", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="goal_frontier_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frontier = _frontier_rows(Path(args.frontier_csv))
    boundary = _boundary_summary(Path(args.boundary_csv), float(args.quality_threshold))
    splice = _splice_summaries([Path(path) for path in args.splice_summary or []])
    segment = _segment_summaries([Path(path) for path in args.segment_summary or []])
    local_window = _local_window_summaries([Path(path) for path in args.local_window_summary or []])
    budget_curves = [_budget_curve_summary(Path(path)) for path in args.budget_curve or []]
    payload = {
        "quality_threshold": float(args.quality_threshold),
        "inputs": {
            "frontier_csv": str(Path(args.frontier_csv)),
            "boundary_csv": str(Path(args.boundary_csv)),
            "splice_summary": [str(Path(path)) for path in args.splice_summary or []],
            "segment_summary": [str(Path(path)) for path in args.segment_summary or []],
            "local_window_summary": [str(Path(path)) for path in args.local_window_summary or []],
        },
        "post_lst_frontier": frontier,
        "b2628_boundary": boundary,
        "splice_audits": splice,
        "segment_boundary_audits": segment,
        "local_window_audits": local_window,
        "budget_curve_audits": budget_curves,
    }
    json_path = out_dir / f"{args.prefix}.json"
    md_path = out_dir / f"{args.prefix}.md"
    frontier_path = out_dir / f"{args.prefix}_frontier.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_rows(frontier_path, frontier)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "frontier_csv": str(frontier_path),
                "best_b2628": boundary.get("best"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
