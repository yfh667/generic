from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


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
    _build_pair_step_contexts,
)
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402
from src.topology_workflow.module.edge_tables import build_full_option_plus_intra_edge_table  # noqa: E402


DEFAULT_REWARD_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_action_scorer_hard96"
    r"\action_library_reward_table_full1437_merged"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_dp_edge_transition_penalty"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic-programming action selector over row-mask 000040/000056 actions, "
            "using the real constrained edge-set transition size as the switch penalty."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--base-motif", type=int, default=56)
    parser.add_argument("--patch-motif", type=int, default=40)
    parser.add_argument("--patch-mode", choices=("c", "cb", "all"), default="all")
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    parser.add_argument(
        "--switch-penalties",
        nargs="+",
        type=float,
        default=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
        help="Penalty multiplier applied to added-edge transition count / edge-scale.",
    )
    parser.add_argument("--edge-scale", type=float, default=100.0)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--reuse-mask-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def edge_key(src: int, dst: int) -> tuple[int, int]:
    a, b = (int(src), int(dst)) if int(src) <= int(dst) else (int(dst), int(src))
    return a, b


def edge_key_from_table(edge_table: EdgeTable, idx: int) -> tuple[int, int]:
    return edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))


def read_action_table(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(path)
    action_names = [str(x) for x in df["action"].tolist()]
    y_cols = [f"y{i:02d}" for i in range(36)]
    masks = df[y_cols].to_numpy(dtype=np.int8)
    mask_tokens = [str(x) for x in df["mask"].tolist()]
    return action_names, masks, mask_tokens


def read_wide(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(path)
    steps = df["step"].to_numpy(dtype=np.int64)
    names = [str(name) for name in df.columns if name != "step"]
    values = df[names].to_numpy(dtype=np.float64)
    return steps, names, values


def ensure_same_actions(expected: Sequence[str], actual: Sequence[str], label: str) -> None:
    if list(expected) != list(actual):
        raise ValueError(f"{label} action columns differ from action table")


def topology_specs_by_id(*, motif_csv: Path, config, name_prefix: str, wrap_planes: bool) -> dict[int, Any]:
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library=name_prefix,
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=bool(wrap_planes),
    )
    return {int(spec.motif_id): spec for spec in specs if spec.motif_id is not None}


def table_to_bitset(
    edge_table: EdgeTable,
    key_to_col_lookup: np.ndarray,
    *,
    total_nodes: int,
    words: int,
) -> np.ndarray:
    out = np.zeros(int(words), dtype=np.uint64)
    src = np.asarray(edge_table.src, dtype=np.int64)
    dst = np.asarray(edge_table.dst, dtype=np.int64)
    keys = np.minimum(src, dst) * int(total_nodes) + np.maximum(src, dst)
    cols = np.asarray(key_to_col_lookup[keys], dtype=np.int32)
    if np.any(cols < 0):
        bad = int(np.flatnonzero(cols < 0)[0])
        raise ValueError(f"edge is missing from full-option-plus-intra universe: src={int(src[bad])} dst={int(dst[bad])}")
    word_idx = cols >> 6
    bit_idx = (cols & 63).astype(np.uint64)
    values = np.left_shift(np.uint64(1), bit_idx)
    np.bitwise_or.at(out, word_idx, values)
    return out


def build_or_load_action_edge_masks(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    raw: dict[str, Any],
    config,
    steps: np.ndarray,
    action_masks: np.ndarray,
) -> tuple[np.ndarray, int]:
    cache_path = out_dir / "action_edge_masks_uint64.npy"
    meta_path = out_dir / "action_edge_masks_meta.json"
    if bool(args.reuse_mask_cache) and cache_path.exists() and meta_path.exists():
        masks = np.load(cache_path, mmap_mode=None)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return masks, int(meta["edge_universe_size"])

    paths = raw.get("paths", {})
    motif_library = raw.get("motif_library", {})
    motif_csv = Path(str(motif_library.get("csv_name", "motif0040_0056.csv")))
    if not motif_csv.is_absolute():
        motif_csv = path_from(paths, "motif_library_dir") / motif_csv
    name_prefix = str(motif_library.get("name_prefix", "selected"))
    wrap_planes = bool(wrap_planes_from_config(raw))
    forced_option = int(raw.get("region_internal_constraint", {}).get("forced_option", 0))
    pair_spec = region_pair_specs(raw, subset=[str(args.target_pair)])[0]

    universe = build_full_option_plus_intra_edge_table(
        config=config,
        options=(0, 1, 2, 4),
        add_intra_ring=True,
        wrap_planes=wrap_planes,
    )
    total_nodes = int(config.total_sats)
    key_to_col_lookup = np.full(total_nodes * total_nodes, -1, dtype=np.int32)
    for idx in range(universe.num_edges):
        src = int(universe.src[idx])
        dst = int(universe.dst[idx])
        a, b = (src, dst) if src <= dst else (dst, src)
        key_to_col_lookup[a * total_nodes + b] = int(idx)
    words = int((int(universe.num_edges) + 63) // 64)

    specs_by_id = topology_specs_by_id(
        motif_csv=motif_csv,
        config=config,
        name_prefix=name_prefix,
        wrap_planes=wrap_planes,
    )
    base_spec = specs_by_id[int(args.base_motif)]
    patch_spec = specs_by_id[int(args.patch_motif)]
    raw_tables: list[EdgeTable] = []
    for action_idx in range(action_masks.shape[0]):
        rows = tuple(int(y) for y, value in enumerate(action_masks[action_idx].tolist()) if int(value) != 0)
        edge_table, _added, _removed, _degree = build_hybrid_edge_table(
            base_spec=base_spec,
            patch_spec=patch_spec,
            p=int(config.P),
            n=int(config.N),
            band=rows,
            patch_mode=str(args.patch_mode),
        )
        raw_tables.append(edge_table)

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=[int(x) for x in steps.tolist()],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(raw.get("time", {}).get("stride", 60)),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    contexts = _build_pair_step_contexts(
        config=config,
        group_data=group_data,
        steps=[int(x) for x in steps.tolist()],
        pair_specs=[pair_spec],
        forced_option=forced_option,
        wrap_planes=wrap_planes,
    )[str(pair_spec.key)]

    t_count = int(steps.size)
    a_count = int(action_masks.shape[0])
    masks = np.zeros((t_count, a_count, words), dtype=np.uint64)
    partial_path = out_dir / "action_edge_masks_uint64.partial.npy"
    progress_path = out_dir / "action_edge_masks_progress.json"
    start_t = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if list(progress.get("shape", [])) == [t_count, a_count, words]:
            masks = np.load(partial_path, mmap_mode=None)
            start_t = int(progress.get("completed_steps", 0))
            print(f"[edge-transition-dp] resumed partial action edge masks from step row {start_t}", flush=True)
    started = time.perf_counter()
    for t_idx, context in enumerate(contexts):
        if t_idx < start_t:
            continue
        for action_idx, raw_table in enumerate(raw_tables):
            constrained, _forced, _dropped = _apply_constraint_context_fast(
                base_edge_table=raw_table,
                context=context,
                total_nodes=int(config.total_sats),
                p=int(config.P),
                n=int(config.N),
                forced_option=forced_option,
                wrap_planes=wrap_planes,
            )
            masks[t_idx, action_idx] = table_to_bitset(
                constrained,
                key_to_col_lookup,
                total_nodes=int(config.total_sats),
                words=words,
            )
        if (t_idx + 1) % int(args.progress_every) == 0 or t_idx + 1 == t_count:
            np.save(partial_path, masks)
            progress_path.write_text(
                json.dumps(
                    {"completed_steps": int(t_idx + 1), "shape": [t_count, a_count, words]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            elapsed = time.perf_counter() - started
            print(
                f"[edge-transition-dp] built action edge masks {t_idx + 1}/{t_count} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    np.save(cache_path, masks)
    if partial_path.exists():
        partial_path.unlink()
    if progress_path.exists():
        progress_path.unlink()
    meta = {
        "edge_universe_size": int(universe.num_edges),
        "words": int(words),
        "shape": [int(x) for x in masks.shape],
        "config": str(Path(args.config)),
        "target_pair": str(args.target_pair),
        "base_motif": int(args.base_motif),
        "patch_motif": int(args.patch_motif),
        "patch_mode": str(args.patch_mode),
        "wrap_planes": wrap_planes,
        "forced_option": forced_option,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return masks, int(universe.num_edges)


_POPCOUNT_LOOKUP = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def popcount_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint64)
    byte_view = arr.view(np.uint8).reshape(arr.shape[0], -1)
    return _POPCOUNT_LOOKUP[byte_view].sum(axis=1).astype(np.float64)


def popcount_one(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=np.uint64).reshape(1, -1)
    return int(popcount_rows(arr)[0])


def run_edge_transition_dp(
    *,
    stage_cost: np.ndarray,
    edge_masks: np.ndarray,
    switch_penalty: float,
    edge_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_count, a_count = stage_cost.shape
    dp_prev = stage_cost[0].astype(np.float64, copy=True)
    parent = np.full((t_count, a_count), -1, dtype=np.int16)
    selected_add_cost = np.zeros(t_count, dtype=np.float64)
    norm = max(1e-9, float(edge_scale))
    for t in range(1, t_count):
        prev_masks = edge_masks[t - 1]
        curr_masks = edge_masks[t]
        dp_curr = np.empty(a_count, dtype=np.float64)
        for curr_idx in range(a_count):
            added = popcount_rows(np.bitwise_and(curr_masks[curr_idx][None, :], np.bitwise_not(prev_masks)))
            candidate = dp_prev + float(switch_penalty) * (added / norm)
            best_prev = int(np.argmin(candidate))
            parent[t, curr_idx] = best_prev
            dp_curr[curr_idx] = float(stage_cost[t, curr_idx]) + float(candidate[best_prev])
        dp_prev = dp_curr
    selected = np.empty(t_count, dtype=np.int16)
    selected[-1] = int(np.argmin(dp_prev))
    for t in range(t_count - 1, 0, -1):
        selected[t - 1] = int(parent[t, selected[t]])
    for t in range(1, t_count):
        prev_idx = int(selected[t - 1])
        curr_idx = int(selected[t])
        added = np.bitwise_and(edge_masks[t, curr_idx], np.bitwise_not(edge_masks[t - 1, prev_idx]))
        selected_add_cost[t] = float(popcount_one(added))
    return selected, selected_add_cost


def transition_summary(selected: np.ndarray, edge_masks: np.ndarray) -> dict[str, Any]:
    added: list[int] = []
    removed: list[int] = []
    changed: list[int] = []
    for t in range(1, int(selected.size)):
        prev_mask = edge_masks[t - 1, int(selected[t - 1])]
        curr_mask = edge_masks[t, int(selected[t])]
        add_count = popcount_one(np.bitwise_and(curr_mask, np.bitwise_not(prev_mask)))
        rem_count = popcount_one(np.bitwise_and(prev_mask, np.bitwise_not(curr_mask)))
        added.append(add_count)
        removed.append(rem_count)
        changed.append(add_count + rem_count)
    return {
        "mean_added_edges_from_prev": float(np.mean(added)) if added else 0.0,
        "max_added_edges_from_prev": int(np.max(added)) if added else 0,
        "total_added_edges_from_prev": int(np.sum(added)) if added else 0,
        "mean_removed_edges_from_prev": float(np.mean(removed)) if removed else 0.0,
        "max_removed_edges_from_prev": int(np.max(removed)) if removed else 0,
        "total_removed_edges_from_prev": int(np.sum(removed)) if removed else 0,
        "mean_changed_edges_from_prev": float(np.mean(changed)) if changed else 0.0,
        "max_changed_edges_from_prev": int(np.max(changed)) if changed else 0,
        "total_changed_edges_from_prev": int(np.sum(changed)) if changed else 0,
        "nonzero_added_steps": int(sum(1 for x in added if x > 0)),
    }


def schedule_rows(
    *,
    steps: np.ndarray,
    selected: np.ndarray,
    policy: str,
    lambda_hop: float,
    switch_penalty: float,
    edge_scale: float,
    action_names: Sequence[str],
    action_masks: np.ndarray,
    mask_tokens: Sequence[str],
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
    added_from_prev: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(selected.astype(int).tolist()):
        mask = action_masks[action_idx]
        row: dict[str, Any] = {
            "step": int(steps[t]),
            "policy": policy,
            "lambda_hop": float(lambda_hop),
            "switch_penalty": float(switch_penalty),
            "edge_scale": float(edge_scale),
            "action": str(action_names[action_idx]),
            "action_index": int(action_idx),
            "mask": str(mask_tokens[action_idx]),
            "row_count": int(mask.sum()),
            "mean_hops": float(hops[t, action_idx]),
            "mean_delay_ms": float(delay[t, action_idx]),
            "reference_hop_envelope": float(hop_env[t]),
            "reference_delay_envelope_ms": float(delay_env[t]),
            "gap_hop_env": float(hops[t, action_idx] - hop_env[t]),
            "gap_delay_env_ms": float(delay[t, action_idx] - delay_env[t]),
            "added_edges_from_prev": float(added_from_prev[t]),
        }
        for y in range(mask.shape[0]):
            row[f"y{y:02d}"] = int(mask[y])
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_policy(
    *,
    selected: np.ndarray,
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
    edge_masks: np.ndarray,
) -> dict[str, Any]:
    idx = selected.astype(int)
    rows = np.arange(idx.size)
    switches = int(np.count_nonzero(idx[1:] != idx[:-1]))
    out = {
        "mean_hops": float(np.mean(hops[rows, idx])),
        "mean_delay_ms": float(np.mean(delay[rows, idx])),
        "mean_gap_hop_env": float(np.mean(hops[rows, idx] - hop_env)),
        "mean_gap_delay_env_ms": float(np.mean(delay[rows, idx] - delay_env)),
        "switches": switches,
        "num_segments": int(switches + 1),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
    }
    out.update(transition_summary(selected, edge_masks))
    return out


def main() -> int:
    args = parse_args()
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    reward_dir = Path(args.reward_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_names, action_masks, mask_tokens = read_action_table(reward_dir / "row_mask_action_library_actions.csv")
    steps_hop, names_hop, hops = read_wide(reward_dir / "row_mask_action_library_mean_hops_wide.csv")
    steps_delay, names_delay, delay = read_wide(reward_dir / "row_mask_action_library_mean_delay_ms_wide.csv")
    if not np.array_equal(steps_hop, steps_delay):
        raise ValueError("hop/delay tables have different steps")
    ensure_same_actions(action_names, names_hop, "hop table")
    ensure_same_actions(action_names, names_delay, "delay table")

    edge_masks, edge_universe_size = build_or_load_action_edge_masks(
        args=args,
        out_dir=out_dir,
        raw=raw,
        config=config,
        steps=steps_hop,
        action_masks=action_masks,
    )
    if edge_masks.shape[:2] != hops.shape:
        raise ValueError(f"edge mask shape {edge_masks.shape[:2]} does not match reward shape {hops.shape}")

    hop_env = np.nanmin(hops, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hops) - np.nanmin(hops)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    stage_cost = (
        float(args.lambda_hop) * ((hops - hop_env[:, None]) / hop_scale)
        + (1.0 - float(args.lambda_hop)) * ((delay - delay_env[:, None]) / delay_scale)
    )

    summary_rows: list[dict[str, Any]] = []
    for penalty in args.switch_penalties:
        selected, added_from_prev = run_edge_transition_dp(
            stage_cost=stage_cost,
            edge_masks=edge_masks,
            switch_penalty=float(penalty),
            edge_scale=float(args.edge_scale),
        )
        policy = f"edge_dp_lambda{float(args.lambda_hop):.2f}_sw{float(penalty):g}_scale{float(args.edge_scale):g}"
        schedule = schedule_rows(
            steps=steps_hop,
            selected=selected,
            policy=policy,
            lambda_hop=float(args.lambda_hop),
            switch_penalty=float(penalty),
            edge_scale=float(args.edge_scale),
            action_names=action_names,
            action_masks=action_masks,
            mask_tokens=mask_tokens,
            hops=hops,
            delay=delay,
            hop_env=hop_env,
            delay_env=delay_env,
            added_from_prev=added_from_prev,
        )
        schedule_path = out_dir / f"{policy}.csv"
        write_rows(schedule_path, schedule)
        summary = summarize_policy(
            selected=selected,
            hops=hops,
            delay=delay,
            hop_env=hop_env,
            delay_env=delay_env,
            edge_masks=edge_masks,
        )
        summary_rows.append(
            {
                "policy": policy,
                "lambda_hop": float(args.lambda_hop),
                "switch_penalty": float(penalty),
                "edge_scale": float(args.edge_scale),
                "edge_universe_size": int(edge_universe_size),
                "schedule_csv": str(schedule_path),
                **summary,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "edge_transition_dp_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta = {
        "config": str(Path(args.config)),
        "reward_dir": str(reward_dir),
        "out_dir": str(out_dir),
        "target_pair": str(args.target_pair),
        "base_motif": int(args.base_motif),
        "patch_motif": int(args.patch_motif),
        "patch_mode": str(args.patch_mode),
        "lambda_hop": float(args.lambda_hop),
        "edge_scale": float(args.edge_scale),
        "hop_scale": hop_scale,
        "delay_scale": delay_scale,
        "mean_hop_envelope": float(np.mean(hop_env)),
        "mean_delay_envelope_ms": float(np.mean(delay_env)),
        "summary_csv": str(summary_path),
    }
    (out_dir / "edge_transition_dp_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
