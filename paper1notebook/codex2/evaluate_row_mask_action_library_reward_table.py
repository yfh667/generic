from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
from search_row_mask_oracle_0040_0056 import DEFAULT_REFERENCE_DIR, _mask_key, _mask_token, _read_reference  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402


DEFAULT_ACTION_LIBRARY = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_action_scorer_hard96"
    r"\row_mask_action_library.npy"
)
DEFAULT_OUT_DIR = DEFAULT_ACTION_LIBRARY.parent / "action_library_reward_table"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every mask in the 000040/000056 row-mask action library with "
            "region-internal +grid constraints."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--action-library", type=Path, default=DEFAULT_ACTION_LIBRARY)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", nargs="*", default=None, help="Time steps to evaluate; accepts spaces or commas.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit selected reference rows after --steps filtering.")
    parser.add_argument("--row-start", type=int, default=0, help="Start index in the selected reference step list.")
    parser.add_argument("--row-end", type=int, default=0, help="Exclusive end index in the selected reference step list; 0 means no limit.")
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--forced-option", type=int, default=None)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _parse_steps(values: Sequence[str] | None) -> list[int] | None:
    if not values:
        return None
    steps: list[int] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            steps.append(int(float(token)))
    return steps


def _load_action_library(path: Path, *, n: int) -> np.ndarray:
    actions = np.load(Path(path))
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        raise ValueError(f"action library must be a 2-D array, got shape={actions.shape}")
    if int(actions.shape[1]) != int(n):
        raise ValueError(f"action library width {actions.shape[1]} does not match config.N={n}")
    actions = (actions != 0).astype(np.int8, copy=False)
    return actions


def _rows_from_action(action: np.ndarray) -> tuple[int, ...]:
    return tuple(int(idx) for idx, value in enumerate(action.tolist()) if int(value) != 0)


def _select_steps(
    *,
    reference_steps: np.ndarray,
    requested_steps: Sequence[int] | None,
    max_rows: int,
    row_start: int,
    row_end: int,
) -> list[int]:
    available = {int(step) for step in reference_steps.tolist()}
    if requested_steps is None:
        steps = [int(step) for step in reference_steps.tolist()]
    else:
        steps = [int(step) for step in requested_steps]
        missing = [step for step in steps if step not in available]
        if missing:
            raise ValueError(f"{len(missing)} requested steps not found in reference, first={missing[:5]}")
    start = max(0, int(row_start))
    end = int(row_end)
    if start or end > 0:
        steps = steps[start : (end if end > 0 else None)]
    if int(max_rows) > 0:
        steps = steps[: int(max_rows)]
    if not steps:
        raise ValueError("no steps selected")
    return steps


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


def _write_wide_csv(path: Path, *, steps: Sequence[int], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step"] + [f"action_{idx:03d}" for idx in range(int(matrix.shape[1]))]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row_idx, step in enumerate(steps):
            writer.writerow([int(step)] + [f"{float(value):.12g}" for value in matrix[row_idx].tolist()])


def _finite_nanmin(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmin(values, axis=axis)


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
    selected_steps = _select_steps(
        reference_steps=steps_ref,
        requested_steps=_parse_steps(args.steps),
        max_rows=int(args.max_rows),
        row_start=int(args.row_start),
        row_end=int(args.row_end),
    )
    step_to_ref = {int(step): idx for idx, step in enumerate(steps_ref.tolist())}
    ref_rows = [step_to_ref[int(step)] for step in selected_steps]

    actions = _load_action_library(Path(args.action_library), n=int(config.N))
    action_masks = [_rows_from_action(actions[idx]) for idx in range(int(actions.shape[0]))]

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

    edge_cache: dict[tuple[int, ...], Any] = {}

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_steps = len(selected_steps)
    num_actions = int(actions.shape[0])
    hops = np.full((num_steps, num_actions), np.nan, dtype=np.float64)
    delay_ms = np.full((num_steps, num_actions), np.nan, dtype=np.float64)
    forced_counts = np.zeros((num_steps, num_actions), dtype=np.int32)
    dropped_counts = np.zeros((num_steps, num_actions), dtype=np.int32)
    started = time.perf_counter()

    for step_idx, step in enumerate(selected_steps):
        context = contexts[step_idx]
        delay_row = delay_store.row_for_time_step(int(step))
        for action_idx, mask in enumerate(action_masks):
            constrained, forced_count, dropped_count = _apply_constraint_context_fast(
                base_edge_table=edge_table_for(mask),
                context=context,
                total_nodes=int(config.total_sats),
                p=int(config.P),
                n=int(config.N),
                forced_option=int(forced_option),
                wrap_planes=bool(wrap_planes),
            )
            adjacency = _build_adjacency(constrained.src, constrained.dst, int(config.total_sats))
            _rh, mean_hops, _min_hops, _max_hops = _summarize_hops(
                adjacency=adjacency,
                sources=context.sources,
                targets=context.targets,
            )
            _rd, mean_delay, _min_delay, _max_delay = _summarize_delay(
                edge_table=constrained,
                config=config,
                delay_store=delay_store,
                position_store=None,
                delay_row=delay_row,
                position_row=None,
                sources=context.sources,
                targets=context.targets,
                engine="auto",
            )
            hops[step_idx, action_idx] = float(mean_hops)
            delay_ms[step_idx, action_idx] = float(mean_delay)
            forced_counts[step_idx, action_idx] = int(forced_count)
            dropped_counts[step_idx, action_idx] = int(dropped_count)

        every = max(1, int(args.progress_every))
        if (step_idx + 1) % every == 0 or step_idx + 1 == num_steps:
            print(
                f"[action-table] {step_idx + 1}/{num_steps} steps "
                f"actions={num_actions} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    hops_csv = out_dir / "row_mask_action_library_mean_hops_wide.csv"
    delay_csv = out_dir / "row_mask_action_library_mean_delay_ms_wide.csv"
    action_csv = out_dir / "row_mask_action_library_actions.csv"
    summary_path = out_dir / "row_mask_action_library_reward_table_summary.json"
    _write_wide_csv(hops_csv, steps=selected_steps, matrix=hops)
    _write_wide_csv(delay_csv, steps=selected_steps, matrix=delay_ms)
    _write_rows(
        action_csv,
        [
            {
                "action": f"action_{idx:03d}",
                "action_index": int(idx),
                "mask": _mask_token(action_masks[idx], int(config.N)),
                "row_count": int(len(action_masks[idx])),
                **{f"y{y:02d}": int(actions[idx, y]) for y in range(int(config.N))},
            }
            for idx in range(num_actions)
        ],
    )

    ref_hop_env = _finite_nanmin(hop_ref[ref_rows], axis=1)
    ref_delay_env = _finite_nanmin(delay_ref[ref_rows], axis=1)
    action_hop_env = _finite_nanmin(hops, axis=1)
    action_delay_env = _finite_nanmin(delay_ms, axis=1)
    best_hop_idx = np.nanargmin(hops, axis=1).astype(np.int64)
    best_delay_idx = np.nanargmin(delay_ms, axis=1).astype(np.int64)
    summary = {
        "config": str(Path(args.config)),
        "action_library": str(Path(args.action_library)),
        "reference_dir": str(Path(args.reference_dir)),
        "out_dir": str(out_dir),
        "target_pair": str(args.target_pair),
        "forced_option": int(forced_option),
        "steps": [int(step) for step in selected_steps],
        "num_steps": int(num_steps),
        "num_actions": int(num_actions),
        "action_width": int(actions.shape[1]),
        "unique_masks": int(len({_mask_key(mask) for mask in action_masks})),
        "reference_candidate_count": int(len(names_ref)),
        "hops_wide_csv": str(hops_csv),
        "delay_ms_wide_csv": str(delay_csv),
        "action_metadata_csv": str(action_csv),
        "mean_reference_hop_envelope": float(np.nanmean(ref_hop_env)),
        "mean_reference_delay_envelope_ms": float(np.nanmean(ref_delay_env)),
        "mean_action_library_hop_envelope": float(np.nanmean(action_hop_env)),
        "mean_action_library_delay_envelope_ms": float(np.nanmean(action_delay_env)),
        "mean_gap_action_to_reference_hop_envelope": float(np.nanmean(action_hop_env - ref_hop_env)),
        "mean_gap_action_to_reference_delay_envelope_ms": float(np.nanmean(action_delay_env - ref_delay_env)),
        "best_hop_actions": [f"action_{idx:03d}" for idx in best_hop_idx.tolist()],
        "best_delay_actions": [f"action_{idx:03d}" for idx in best_delay_idx.tolist()],
        "mean_forced_region_edges": float(np.mean(forced_counts)),
        "mean_dropped_edges": float(np.mean(dropped_counts)),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
