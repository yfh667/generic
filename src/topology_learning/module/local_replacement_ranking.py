from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _float_or_nan(value: Any) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int_or_none(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes", "y"}:
        return True
    if raw in {"false", "0", "no", "n"}:
        return False
    return None


def read_replacement_score_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            path = path / "score.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                item = dict(row)
                item["_source_path"] = str(path)
                rows.append(item)
    return rows


def normalise_replacement_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_threshold: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        gap = _float_or_nan(item.get("mean_all_global_gap", item.get("mean_all_gap")))
        delay_gap = _float_or_nan(item.get("mean_delay_global_gap"))
        hop_gap = _float_or_nan(item.get("mean_hop_global_gap"))
        setup = _int_or_none(item.get("total_setup_commands"))
        peak = _int_or_none(item.get("max_setup_commands_per_step"))
        within = _bool_or_none(item.get("within_quality_threshold"))
        if quality_threshold is not None:
            within = bool(math.isfinite(gap) and gap <= float(quality_threshold))
            item["quality_threshold"] = float(quality_threshold)
        item["_gap"] = gap
        item["_delay_gap"] = delay_gap
        item["_hop_gap"] = hop_gap
        item["_setup"] = setup
        item["_peak"] = peak
        item["_within"] = within
        out.append(item)
    return out


def pareto_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: tuple[str, ...] = ("_gap", "_setup", "_peak"),
) -> list[dict[str, Any]]:
    parsed = [dict(row) for row in rows]

    def value(row: Mapping[str, Any], key: str) -> float:
        raw = row.get(key)
        if raw is None:
            return math.inf
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return math.inf
        if not math.isfinite(val):
            return math.inf
        return val

    frontier: list[dict[str, Any]] = []
    for idx, row in enumerate(parsed):
        vals = tuple(value(row, key) for key in keys)
        dominated = False
        for other_idx, other in enumerate(parsed):
            if other_idx == idx:
                continue
            other_vals = tuple(value(other, key) for key in keys)
            if all(o <= v for o, v in zip(other_vals, vals)) and any(o < v for o, v in zip(other_vals, vals)):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: tuple(value(row, key) for key in keys))
    return frontier


def choose_replacement_recommendations(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_threshold: float | None = None,
) -> dict[str, Any]:
    parsed = normalise_replacement_rows(rows, quality_threshold=quality_threshold)
    finite = [row for row in parsed if math.isfinite(float(row.get("_gap", math.nan))) and row.get("_setup") is not None]
    if not finite:
        return {
            "num_rows": int(len(parsed)),
            "quality_threshold": None if quality_threshold is None else float(quality_threshold),
            "best_quality": None,
            "min_setup_within_threshold": None,
            "min_peak_within_threshold": None,
            "best_quality_replacement": None,
            "min_setup_replacement_within_threshold": None,
            "min_peak_replacement_within_threshold": None,
            "pareto_frontier": [],
        }
    within = [row for row in finite if bool(row.get("_within"))]
    replacement_rows = [
        row
        for row in finite
        if str(row.get("left_step", "")).strip() not in {"", "None", "nan"}
        or "__replace__" in str(row.get("schedule", ""))
    ]
    replacement_within = [row for row in replacement_rows if bool(row.get("_within"))]

    def public(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        keep = [
            "schedule",
            "mean_all_global_gap",
            "mean_delay_global_gap",
            "mean_hop_global_gap",
            "total_setup_commands",
            "max_setup_commands_per_step",
            "within_quality_threshold",
            "quality_threshold",
            "left_step",
            "right_step",
            "entry_setup",
            "exit_setup",
            "_source_path",
        ]
        return {key: row.get(key) for key in keep if key in row}

    best_quality = min(finite, key=lambda row: (float(row["_gap"]), int(row["_setup"]), int(row.get("_peak") or 10**9)))
    best_quality_replacement = (
        min(replacement_rows, key=lambda row: (float(row["_gap"]), int(row["_setup"]), int(row.get("_peak") or 10**9)))
        if replacement_rows
        else None
    )
    min_setup = (
        min(within, key=lambda row: (int(row["_setup"]), float(row["_gap"]), int(row.get("_peak") or 10**9)))
        if within
        else None
    )
    min_peak = (
        min(within, key=lambda row: (int(row.get("_peak") or 10**9), float(row["_gap"]), int(row["_setup"])))
        if within
        else None
    )
    min_setup_replacement = (
        min(
            replacement_within,
            key=lambda row: (int(row["_setup"]), float(row["_gap"]), int(row.get("_peak") or 10**9)),
        )
        if replacement_within
        else None
    )
    min_peak_replacement = (
        min(
            replacement_within,
            key=lambda row: (int(row.get("_peak") or 10**9), float(row["_gap"]), int(row["_setup"])),
        )
        if replacement_within
        else None
    )
    frontier = pareto_frontier(finite)
    return {
        "num_rows": int(len(parsed)),
        "num_finite_rows": int(len(finite)),
        "quality_threshold": None if quality_threshold is None else float(quality_threshold),
        "num_within_threshold": int(len(within)),
        "num_replacement_rows": int(len(replacement_rows)),
        "num_replacement_within_threshold": int(len(replacement_within)),
        "best_quality": public(best_quality),
        "min_setup_within_threshold": public(min_setup),
        "min_peak_within_threshold": public(min_peak),
        "best_quality_replacement": public(best_quality_replacement),
        "min_setup_replacement_within_threshold": public(min_setup_replacement),
        "min_peak_replacement_within_threshold": public(min_peak_replacement),
        "pareto_frontier": [public(row) for row in frontier],
    }


def write_replacement_ranking_outputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    quality_threshold: float | None = None,
    prefix: str = "local_replacement",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = normalise_replacement_rows(rows, quality_threshold=quality_threshold)
    recommendations = choose_replacement_recommendations(parsed, quality_threshold=quality_threshold)
    frontier = [row for row in pareto_frontier(parsed) if row.get("_setup") is not None and math.isfinite(float(row.get("_gap", math.nan)))]

    ranking_path = out_dir / f"{prefix}_ranking.csv"
    public_fields = [
        "schedule",
        "mean_all_global_gap",
        "mean_delay_global_gap",
        "mean_hop_global_gap",
        "total_setup_commands",
        "max_setup_commands_per_step",
        "within_quality_threshold",
        "quality_threshold",
        "left_step",
        "right_step",
        "entry_setup",
        "exit_setup",
        "_source_path",
    ]
    sorted_rows = sorted(
        parsed,
        key=lambda row: (
            not bool(row.get("_within")),
            int(row.get("_setup") or 10**12),
            float(row.get("_gap", math.inf)),
            int(row.get("_peak") or 10**12),
        ),
    )
    with ranking_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=public_fields)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({key: row.get(key, "") for key in public_fields})

    frontier_path = out_dir / f"{prefix}_pareto_frontier.csv"
    with frontier_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=public_fields)
        writer.writeheader()
        for row in frontier:
            writer.writerow({key: row.get(key, "") for key in public_fields})

    json_path = out_dir / f"{prefix}_recommendations.json"
    json_path.write_text(json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / f"{prefix}_recommendations.md"
    lines = [
        "# Local Hybrid Replacement Recommendations",
        "",
        f"quality_threshold: `{recommendations.get('quality_threshold')}`",
        f"num_rows: `{recommendations.get('num_rows')}`",
        f"num_within_threshold: `{recommendations.get('num_within_threshold')}`",
        "",
    ]
    for key, title in [
        ("best_quality", "Best Quality"),
        ("min_setup_within_threshold", "Min Setup Within Threshold"),
        ("min_peak_within_threshold", "Min Peak Within Threshold"),
        ("best_quality_replacement", "Best Quality Replacement"),
        ("min_setup_replacement_within_threshold", "Min Setup Replacement Within Threshold"),
        ("min_peak_replacement_within_threshold", "Min Peak Replacement Within Threshold"),
    ]:
        value = recommendations.get(key)
        lines.extend([f"## {title}", ""])
        if value is None:
            lines.extend(["None", ""])
            continue
        lines.extend(
            [
                f"- schedule: `{value.get('schedule')}`",
                f"- gap: `{value.get('mean_all_global_gap')}`",
                f"- setup: `{value.get('total_setup_commands')}`",
                f"- peak: `{value.get('max_setup_commands_per_step')}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "ranking_csv": str(ranking_path),
        "pareto_frontier_csv": str(frontier_path),
        "recommendations_json": str(json_path),
        "recommendations_md": str(md_path),
        "recommendations": recommendations,
    }
