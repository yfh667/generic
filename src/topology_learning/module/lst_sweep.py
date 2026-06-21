from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .lst_schedule_eval import run_lst_evaluation_from_selector, setup_command_counts_from_command_mask, setup_command_counts_from_target_mask


def format_number_token(value: int | float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def schedule_names_from_selector(selector_dir: str | Path) -> tuple[str, ...]:
    arrays = np.load(Path(selector_dir) / "selector_arrays.npz", allow_pickle=False)
    return tuple(str(name)[len("schedule_") :] for name in arrays.files if str(name).startswith("schedule_"))


def _read_selector_summary(selector_dir: str | Path) -> dict[str, dict[str, str]]:
    path = Path(selector_dir) / "summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row.get("schedule", "")): dict(row) for row in csv.DictReader(f) if row.get("schedule")}


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


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _relative_gap(value: Any, reference: Any) -> float | None:
    value_f = _float_or_none(value)
    reference_f = _float_or_none(reference)
    if value_f is None or reference_f is None or abs(reference_f) <= 1e-12:
        return None
    return abs(value_f - reference_f) / abs(reference_f)


def _metric_prefixes_for_pair(pair_key: str) -> tuple[str, ...]:
    key = str(pair_key).lower()
    aliases = {
        "china_europe": ("ce", "china_europe"),
        "china_africa": ("ca", "china_africa"),
        "china_america": ("cam", "china_america"),
    }
    return aliases.get(key, (key,))


def _base_float(base: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _float_or_none(base.get(str(key)))
        if value is not None:
            return value
    return None


def _setup_command_stats_from_eval_dir(eval_dir: str | Path, meta: Mapping[str, Any]) -> dict[str, int | float]:
    total = _int_or_none(meta.get("total_setup_commands"))
    max_step = _int_or_none(meta.get("max_setup_commands_per_step"))
    mean_step = _float_or_none(meta.get("mean_setup_commands_per_step"))
    if total is not None and max_step is not None and mean_step is not None:
        return {
            "total_setup_commands": int(total),
            "max_setup_commands_per_step": int(max_step),
            "mean_setup_commands_per_step": float(mean_step),
        }
    counts_path = Path(eval_dir) / "setup_command_counts.npy"
    if counts_path.exists():
        counts = np.load(counts_path, allow_pickle=False)
    else:
        command_path = Path(eval_dir) / "setup_command_mask.npy"
        target_path = Path(eval_dir) / "target_mask.npy"
        if command_path.exists():
            command = np.load(command_path, allow_pickle=False)
            counts = setup_command_counts_from_command_mask(command)
            np.save(counts_path, counts.astype(np.int32, copy=False))
        elif target_path.exists():
            target = np.load(target_path, allow_pickle=False)
            counts = setup_command_counts_from_target_mask(target, warm_start=True)
            np.save(counts_path, counts.astype(np.int32, copy=False))
        else:
            return {
                "total_setup_commands": 0,
                "max_setup_commands_per_step": 0,
                "mean_setup_commands_per_step": 0.0,
            }
    if counts.size == 0:
        return {
            "total_setup_commands": 0,
            "max_setup_commands_per_step": 0,
            "mean_setup_commands_per_step": 0.0,
        }
    return {
        "total_setup_commands": int(np.sum(counts, dtype=np.int64)),
        "max_setup_commands_per_step": int(np.max(counts)) if counts.size else 0,
        "mean_setup_commands_per_step": float(np.mean(counts)) if counts.size else 0.0,
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _short_schedule_label(name: str) -> str:
    text = str(name)
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


def plot_lst_sweep_summary(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    labels = [_short_schedule_label(str(row.get("schedule", ""))) for row in rows]
    x = np.asarray([_float_or_none(row.get("total_setup_commands")) or 0.0 for row in rows], dtype=np.float64)
    max_setup = np.asarray([_float_or_none(row.get("max_setup_commands_per_step")) or 0.0 for row in rows], dtype=np.float64)
    delay_gap = np.asarray([_float_or_none(row.get("post_lst_delay_ms_gap")) or np.nan for row in rows], dtype=np.float64)
    hops_gap = np.asarray([_float_or_none(row.get("post_lst_hops_gap")) or np.nan for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=170, sharex=True)
    for ax, y, title, ylabel in [
        (axes[0], delay_gap, "Post-LST delay gap vs setup commands", "relative gap to full_link delay"),
        (axes[1], hops_gap, "Post-LST hop gap vs setup commands", "relative gap to full_link hops"),
    ]:
        sc = ax.scatter(x, y, c=max_setup, cmap="viridis", s=54, edgecolors="black", linewidths=0.4)
        for xi, yi, label in zip(x, y, labels):
            if np.isfinite(yi):
                ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("total setup commands")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("max setup commands per step")
    fig.suptitle(str(rows[0].get("pair_key", "pair")) + " LST sweep", y=0.98)
    fig.savefig(path)
    plt.close(fig)
    return path


def run_lst_sweep_from_selector(
    *,
    selector_dir: str | Path,
    schedules: Sequence[str] | None,
    setup_time_seconds: int | float,
    delay_store_dir: str | Path,
    position_cache_dir: str | Path | None,
    group_xml: str | Path,
    group_cache_dir: str | Path,
    source_group_id: int,
    target_group_id: int,
    pair_key: str,
    out_root: str | Path | None = None,
    setup_mode: str = "break_before_make",
    setup_timing: str = "reactive",
    force: bool = False,
    force_group_cache: bool = False,
) -> Path:
    selector_dir = Path(selector_dir)
    if schedules is None or not schedules:
        schedules = schedule_names_from_selector(selector_dir)
    schedules = tuple(str(x) for x in schedules)
    if not schedules:
        raise ValueError("no schedules were provided or found")

    setup_token = format_number_token(setup_time_seconds)
    mode = str(setup_mode or "break_before_make")
    timing = str(setup_timing or "reactive")
    mode_token = "" if mode == "break_before_make" else "_" + mode
    timing_token = "" if timing == "reactive" else "_" + timing
    out_root_path = Path(out_root) if out_root is not None else selector_dir
    summary_rows: list[dict[str, Any]] = []
    selector_summary = _read_selector_summary(selector_dir)

    for schedule in schedules:
        out_dir = out_root_path / f"lst{setup_token}_{pair_key}{mode_token}{timing_token}_{schedule}_eval"
        meta_path = out_dir / "meta.json"
        if meta_path.exists() and not bool(force):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            run_lst_evaluation_from_selector(
                selector_dir=selector_dir,
                schedule_name=str(schedule),
                setup_time_seconds=float(setup_time_seconds),
                delay_store_dir=delay_store_dir,
                position_cache_dir=position_cache_dir,
                group_xml=group_xml,
                group_cache_dir=group_cache_dir,
                source_group_id=int(source_group_id),
                target_group_id=int(target_group_id),
                out_dir=out_dir,
                setup_mode=mode,
                setup_timing=timing,
                force_group_cache=bool(force_group_cache),
            )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        setup_stats = _setup_command_stats_from_eval_dir(out_dir, meta)
        base = selector_summary.get(str(schedule), {})
        pair_prefixes = _metric_prefixes_for_pair(pair_key)
        delay_full_link = _base_float(
            base,
            ["mean_delay_ms_full_link", *(f"mean_{prefix}_delay_ms_full_link" for prefix in pair_prefixes)],
        )
        hops_full_link = _base_float(
            base,
            ["mean_hops_full_link", *(f"mean_{prefix}_hops_full_link" for prefix in pair_prefixes)],
        )
        selector_delay_gap = _base_float(
            base,
            ["mean_delay_ms_gap", *(f"mean_{prefix}_delay_ms_gap" for prefix in pair_prefixes)],
        )
        selector_hops_gap = _base_float(
            base,
            ["mean_hops_gap", *(f"mean_{prefix}_hops_gap" for prefix in pair_prefixes)],
        )
        row = {
            "pair_key": str(pair_key),
            "setup_time_seconds": float(setup_time_seconds),
            "setup_mode": mode,
            "setup_timing": timing,
            "schedule": str(schedule),
            "eval_dir": str(out_dir),
            "selector_mean_stage_cost": _float_or_none(base.get("mean_stage_cost")),
            "selector_num_switches": _int_or_none(base.get("num_switches")),
            "selector_num_unique_topologies": _int_or_none(base.get("num_unique_topologies")),
            "selector_mean_delay_ms_gap": selector_delay_gap,
            "selector_mean_hops_gap": selector_hops_gap,
            "full_link_mean_delay_ms": delay_full_link,
            "full_link_mean_hops": hops_full_link,
            "post_lst_mean_shortest_delay_ms": _float_or_none(meta.get("mean_shortest_delay_ms")),
            "post_lst_mean_shortest_hops": _float_or_none(meta.get("mean_shortest_hops")),
            "post_lst_delay_ms_gap": _relative_gap(meta.get("mean_shortest_delay_ms"), delay_full_link),
            "post_lst_hops_gap": _relative_gap(meta.get("mean_shortest_hops"), hops_full_link),
            "num_switches": _int_or_none(meta.get("num_switches")),
            "num_union_edges": _int_or_none(meta.get("num_union_edges")),
            "mean_building_edges": _float_or_none(meta.get("mean_building_edges")),
            "max_building_edges": _int_or_none(meta.get("max_building_edges")),
            "building_edge_seconds": _float_or_none(meta.get("building_edge_seconds")),
            "total_setup_commands": setup_stats["total_setup_commands"],
            "max_setup_commands_per_step": setup_stats["max_setup_commands_per_step"],
            "mean_setup_commands_per_step": setup_stats["mean_setup_commands_per_step"],
        }
        summary_rows.append(row)

    summary_path = out_root_path / f"lst{setup_token}{mode_token}{timing_token}_{pair_key}_sweep_summary.csv"
    _write_rows(summary_path, summary_rows)
    plot_lst_sweep_summary(summary_path.with_suffix(".png"), summary_rows)
    return summary_path
