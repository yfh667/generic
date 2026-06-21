from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_m56_local_patch_hybrid_topology import build_hybrid_edge_table  # noqa: E402
from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs, wrap_planes_from_config  # noqa: E402
from run_paper1_region_internal_grid_metrics import (  # noqa: E402
    _apply_constraint_context_fast,
    _build_adjacency,
    _build_pair_step_contexts,
    _summarize_delay,
    _summarize_hops,
)
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402


DEFAULT_REFERENCE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_hybrid_search_all110"
    r"\full_t0_86160_s60"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_oracle"
)


@dataclass(frozen=True)
class Metric:
    hops: float
    delay_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy per-y-row 000040/000056 fusion oracle for hard China-Europe time slices."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--max-passes", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=0, help="If >0, also run sequential row-mask beam search.")
    parser.add_argument("--forced-option", type=int, default=None)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def _parse_rows_from_candidate_name(name: str, n: int) -> tuple[int, ...]:
    import re

    if str(name) == "pure_000040":
        return tuple(range(int(n)))
    if str(name) == "pure_000056":
        return tuple()
    rows: set[int] = set()
    for match in re.finditer(r"y(\d{2})_(\d{2})(?:_l(\d{2}))?", str(name)):
        start = int(match.group(1)) % int(n)
        if match.group(3):
            length = int(match.group(3))
            rows.update((start + idx) % int(n) for idx in range(length))
        else:
            end = int(match.group(2)) % int(n)
            if start <= end:
                rows.update(range(start, end + 1))
            else:
                rows.update(list(range(start, int(n))) + list(range(0, end + 1)))
    return tuple(sorted(rows))


def _mask_key(rows: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(int(x) for x in rows)))


def _mask_token(rows: Sequence[int], n: int) -> str:
    values = sorted(set(int(x) % int(n) for x in rows))
    return "{" + ",".join(f"{x:02d}" for x in values) + "}"


def _read_reference(reference_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    def read_wide_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"empty CSV: {path}")
        names = [name for name in rows[0].keys() if name != "step"]
        steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
        matrix = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
        return steps, names, matrix

    steps_hop, names_hop, hop = read_wide_csv(Path(reference_dir) / "compare_mean_shortest_hops.csv")
    steps_delay, names_delay, delay = read_wide_csv(Path(reference_dir) / "compare_mean_shortest_delay_ms.csv")
    if not np.array_equal(steps_hop, steps_delay):
        raise ValueError("reference hop/delay steps differ")
    if names_hop != names_delay:
        raise ValueError("reference hop/delay candidate names differ")
    return steps_hop, names_hop, hop, delay


def _select_hard_steps(
    *,
    steps: np.ndarray,
    names: Sequence[str],
    hop: np.ndarray,
    delay: np.ndarray,
    max_steps: int,
) -> list[int]:
    hop_env = np.nanmin(hop, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hop) - np.nanmin(hop)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    gap = np.maximum((hop - hop_env[:, None]) / hop_scale, (delay - delay_env[:, None]) / delay_scale)
    best_gap = np.nanmin(gap, axis=1)
    order = np.argsort(-best_gap)
    selected = [int(steps[idx]) for idx in order[: int(max_steps)]]
    selected.sort()
    return selected


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    args = parse_args()
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    paths = raw.get("paths", {})
    motif_library = raw.get("motif_library", {})
    motif_csv = Path(str(motif_library.get("csv_name", "motif0040_0056.csv")))
    if not motif_csv.is_absolute():
        motif_csv = path_from(paths, "motif_library_dir") / motif_csv
    wrap_planes = wrap_planes_from_config(raw)
    forced_option = int(args.forced_option) if args.forced_option is not None else int(
        raw.get("region_internal_constraint", {}).get("forced_option", 0)
    )
    pair_spec = region_pair_specs(raw, subset=[str(args.target_pair)])[0]
    steps_ref, names_ref, hop_ref, delay_ref = _read_reference(Path(args.reference_dir))
    selected_steps = (
        [int(x) for x in args.steps]
        if args.steps
        else _select_hard_steps(
            steps=steps_ref,
            names=names_ref,
            hop=hop_ref,
            delay=delay_ref,
            max_steps=int(args.max_steps),
        )
    )
    step_to_row = {int(step): idx for idx, step in enumerate(steps_ref.tolist())}
    selected_rows = [step_to_row[int(step)] for step in selected_steps]

    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library="selected_motif",
        name_prefix=str(motif_library.get("name_prefix", "selected")),
        add_intra_ring=True,
        wrap_planes=bool(wrap_planes),
    )
    by_id = {int(spec.motif_id): spec for spec in specs if spec.motif_id is not None}
    base = by_id[56]
    patch = by_id[40]

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=selected_steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(raw.get("time", {}).get("stride", 60)),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    delay_store = FullLinkDelayStore(path_from(paths, "delay_store_dir"))
    contexts = _build_pair_step_contexts(
        config=config,
        group_data=group_data,
        steps=selected_steps,
        pair_specs=[pair_spec],
        forced_option=int(forced_option),
        wrap_planes=bool(wrap_planes),
    )[str(pair_spec.key)]

    hop_scale = max(1e-9, float(np.nanmax(hop_ref) - np.nanmin(hop_ref)))
    delay_scale = max(1e-9, float(np.nanmax(delay_ref) - np.nanmin(delay_ref)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "row_mask_oracle_hard_steps.csv"
    rows_out: list[dict[str, Any]] = _read_existing_rows(rows_path)
    completed_steps = {_as_int(row, "step") for row in rows_out}
    if completed_steps:
        print(
            f"[row-mask] resuming {len(completed_steps)} completed step(s) from {rows_path}",
            flush=True,
        )
    started = time.perf_counter()

    for local_idx, step in enumerate(selected_steps):
        if int(step) in completed_steps:
            print(
                f"[row-mask] skip completed {local_idx + 1}/{len(selected_steps)} step={step}",
                flush=True,
            )
            continue
        ref_row = selected_rows[local_idx]
        context = contexts[local_idx]
        hop_env = float(np.nanmin(hop_ref[ref_row]))
        delay_env = float(np.nanmin(delay_ref[ref_row]))
        hop_arg = int(np.nanargmin(hop_ref[ref_row]))
        delay_arg = int(np.nanargmin(delay_ref[ref_row]))
        balanced_gap = np.maximum(
            (hop_ref[ref_row] - hop_env) / hop_scale,
            (delay_ref[ref_row] - delay_env) / delay_scale,
        )
        balanced_arg = int(np.nanargmin(balanced_gap))

        edge_cache: dict[tuple[int, ...], Any] = {}
        metric_cache: dict[tuple[int, ...], Metric] = {}
        eval_count = 0

        def edge_table_for(rows: Sequence[int]):
            key = _mask_key(rows)
            if key not in edge_cache:
                edge_table, _added, _removed, _degree = build_hybrid_edge_table(
                    base_spec=base,
                    patch_spec=patch,
                    p=int(config.P),
                    n=int(config.N),
                    band=key,
                    patch_mode="all",
                )
                edge_cache[key] = edge_table
            return edge_cache[key]

        def evaluate(rows: Sequence[int]) -> Metric:
            nonlocal eval_count
            key = _mask_key(rows)
            if key in metric_cache:
                return metric_cache[key]
            eval_count += 1
            constrained, _forced, _dropped = _apply_constraint_context_fast(
                base_edge_table=edge_table_for(key),
                context=context,
                total_nodes=int(config.total_sats),
                p=int(config.P),
                n=int(config.N),
                forced_option=int(forced_option),
                wrap_planes=bool(wrap_planes),
            )
            adjacency = _build_adjacency(constrained.src, constrained.dst, int(config.total_sats))
            _rh, mean_hops, _min_h, _max_h = _summarize_hops(
                adjacency=adjacency,
                sources=context.sources,
                targets=context.targets,
            )
            _rd, mean_delay, _min_d, _max_d = _summarize_delay(
                edge_table=constrained,
                config=config,
                delay_store=delay_store,
                position_store=None,
                delay_row=delay_store.row_for_time_step(int(step)),
                position_row=None,
                sources=context.sources,
                targets=context.targets,
                engine="auto",
            )
            metric_cache[key] = Metric(float(mean_hops), float(mean_delay))
            return metric_cache[key]

        def cost(rows: Sequence[int]) -> tuple[float, Metric]:
            metric = evaluate(rows)
            return max((metric.hops - hop_env) / hop_scale, (metric.delay_ms - delay_env) / delay_scale), metric

        seed_masks = {
            "empty": tuple(),
            "all": tuple(range(int(config.N))),
            "hop_arg": _parse_rows_from_candidate_name(names_ref[hop_arg], int(config.N)),
            "delay_arg": _parse_rows_from_candidate_name(names_ref[delay_arg], int(config.N)),
            "balanced_arg": _parse_rows_from_candidate_name(names_ref[balanced_arg], int(config.N)),
        }
        best_seed = ""
        best_rows: tuple[int, ...] = tuple()
        best_cost = float("inf")
        best_metric = Metric(float("nan"), float("nan"))

        for seed_name, seed_rows in seed_masks.items():
            current = set(seed_rows)
            current_cost, current_metric = cost(tuple(current))
            improved = True
            passes = 0
            while improved and passes < int(args.max_passes):
                improved = False
                passes += 1
                best_toggle: tuple[float, int, Metric, set[int]] | None = None
                for y in range(int(config.N)):
                    trial = set(current)
                    if y in trial:
                        trial.remove(y)
                    else:
                        trial.add(y)
                    trial_cost, trial_metric = cost(tuple(trial))
                    if trial_cost + 1e-12 < current_cost:
                        if best_toggle is None or trial_cost < best_toggle[0]:
                            best_toggle = (trial_cost, y, trial_metric, trial)
                if best_toggle is not None:
                    current_cost, _y, current_metric, current = best_toggle
                    improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = _mask_key(tuple(current))
                best_metric = current_metric
                best_seed = seed_name

        beam_rows: tuple[int, ...] = tuple()
        beam_cost = float("nan")
        beam_metric = Metric(float("nan"), float("nan"))
        if int(args.beam_width) > 0:
            beam: list[tuple[float, tuple[int, ...], Metric]] = []
            seed_cost, seed_metric = cost(tuple())
            beam.append((seed_cost, tuple(), seed_metric))
            width = int(args.beam_width)
            for y in range(int(config.N)):
                expanded: dict[tuple[int, ...], tuple[float, Metric]] = {}
                for _old_cost, mask, _old_metric in beam:
                    for trial in (mask, _mask_key((*mask, y))):
                        if trial in expanded:
                            continue
                        trial_cost, trial_metric = cost(trial)
                        expanded[trial] = (trial_cost, trial_metric)
                beam = sorted(
                    [(value[0], mask, value[1]) for mask, value in expanded.items()],
                    key=lambda item: (item[0], item[2].hops + item[2].delay_ms),
                )[:width]
            beam_cost, beam_rows, beam_metric = beam[0]

        rows_out.append(
            {
                "step": int(step),
                "reference_hop_env": hop_env,
                "reference_delay_env_ms": delay_env,
                "reference_hop_arg": names_ref[hop_arg],
                "reference_delay_arg": names_ref[delay_arg],
                "reference_balanced_arg": names_ref[balanced_arg],
                "reference_balanced_hops": float(hop_ref[ref_row, balanced_arg]),
                "reference_balanced_delay_ms": float(delay_ref[ref_row, balanced_arg]),
                "row_oracle_hops": best_metric.hops,
                "row_oracle_delay_ms": best_metric.delay_ms,
                "row_oracle_gap_to_hop_env": best_metric.hops - hop_env,
                "row_oracle_gap_to_delay_env_ms": best_metric.delay_ms - delay_env,
                "row_oracle_balanced_cost": best_cost,
                "row_oracle_seed": best_seed,
                "row_oracle_rows": _mask_token(best_rows, int(config.N)),
                "row_oracle_row_count": len(best_rows),
                "beam_hops": beam_metric.hops,
                "beam_delay_ms": beam_metric.delay_ms,
                "beam_gap_to_hop_env": beam_metric.hops - hop_env,
                "beam_gap_to_delay_env_ms": beam_metric.delay_ms - delay_env,
                "beam_balanced_cost": beam_cost,
                "beam_rows": _mask_token(beam_rows, int(config.N)) if int(args.beam_width) > 0 else "",
                "beam_row_count": len(beam_rows) if int(args.beam_width) > 0 else 0,
                "eval_count": eval_count,
                "elapsed_s": time.perf_counter() - started,
            }
        )
        _write_rows(rows_path, rows_out)
        print(
            f"[row-mask] {local_idx + 1}/{len(selected_steps)} step={step} "
            f"ref=({hop_env:.3f},{delay_env:.3f}) row=({best_metric.hops:.3f},{best_metric.delay_ms:.3f}) "
            f"rows={len(best_rows)} eval={eval_count} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    _write_rows(rows_path, rows_out)
    meta = {
        "config": str(Path(args.config)),
        "reference_dir": str(Path(args.reference_dir)),
        "out_dir": str(out_dir),
        "selected_steps": selected_steps,
        "max_passes": int(args.max_passes),
        "beam_width": int(args.beam_width),
        "rows_csv": str(rows_path),
        "completed_steps": len(rows_out),
        "mean_reference_balanced_hops": float(np.nanmean([_as_float(r, "reference_balanced_hops") for r in rows_out])),
        "mean_reference_balanced_delay_ms": float(np.nanmean([_as_float(r, "reference_balanced_delay_ms") for r in rows_out])),
        "mean_row_oracle_hops": float(np.nanmean([_as_float(r, "row_oracle_hops") for r in rows_out])),
        "mean_row_oracle_delay_ms": float(np.nanmean([_as_float(r, "row_oracle_delay_ms") for r in rows_out])),
        "mean_beam_hops": float(np.nanmean([_as_float(r, "beam_hops") for r in rows_out])),
        "mean_beam_delay_ms": float(np.nanmean([_as_float(r, "beam_delay_ms") for r in rows_out])),
        "row_oracle_better_hop_steps": int(
            sum(_as_float(r, "row_oracle_hops") < _as_float(r, "reference_hop_env") - 1e-9 for r in rows_out)
        ),
        "row_oracle_better_delay_steps": int(
            sum(_as_float(r, "row_oracle_delay_ms") < _as_float(r, "reference_delay_env_ms") - 1e-9 for r in rows_out)
        ),
        "beam_better_hop_steps": int(
            sum(_as_float(r, "beam_hops") < _as_float(r, "reference_hop_env") - 1e-9 for r in rows_out)
        ),
        "beam_better_delay_steps": int(
            sum(_as_float(r, "beam_delay_ms") < _as_float(r, "reference_delay_env_ms") - 1e-9 for r in rows_out)
        ),
    }
    (out_dir / "row_mask_oracle_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
