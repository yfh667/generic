from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
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

from select_row_mask_dp_edge_transition_penalty import (  # noqa: E402
    DEFAULT_REWARD_DIR,
    build_or_load_action_edge_masks,
    read_action_table,
    read_wide,
)
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402
from src.topology_workflow.module.edge_tables import build_full_option_plus_intra_edge_table  # noqa: E402
from run_paper1_motif_shortest_hops import load_yaml, wrap_planes_from_config  # noqa: E402


DEFAULT_MASK_CACHE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_dp_edge_transition_penalty"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_beam_lst_drop_aware"
)


@dataclass(frozen=True)
class BeamState:
    prev_action: int
    curr_action: int
    cost: float
    path: tuple[int, ...]
    predicted_active_drop: int
    predicted_building: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Beam-search LST-aware row-mask policy. The local LST cost uses the exact "
            "backward-building rule for a 120s setup time on a 60s sampled schedule."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--mask-cache-dir", type=Path, default=DEFAULT_MASK_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--base-motif", type=int, default=56)
    parser.add_argument("--patch-motif", type=int, default=40)
    parser.add_argument("--patch-mode", choices=("c", "cb", "all"), default="all")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--lambda-hop", type=float, default=0.5)
    parser.add_argument("--active-drop-weights", nargs="+", type=float, default=(0.0, 0.5, 1.0, 2.0, 5.0))
    parser.add_argument("--building-weight", type=float, default=0.0)
    parser.add_argument("--edge-scale", type=float, default=100.0)
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--top-k-actions", type=int, default=28)
    parser.add_argument("--include-schedules", nargs="*", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--reuse-mask-cache", action="store_true", default=True)
    return parser.parse_args()


def ensure_same_actions(expected: Sequence[str], actual: Sequence[str], label: str) -> None:
    if list(expected) != list(actual):
        raise ValueError(f"{label} action columns differ from action table")


def words_to_int(words: np.ndarray) -> int:
    return int.from_bytes(np.asarray(words, dtype=np.uint64).tobytes(), "little", signed=False)


def edge_key(src: int, dst: int) -> tuple[int, int]:
    a, b = (int(src), int(dst)) if int(src) <= int(dst) else (int(dst), int(src))
    return a, b


def edge_key_from_table(edge_table: EdgeTable, idx: int) -> tuple[int, int]:
    return edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))


def build_port_maps(config, *, wrap_planes: bool) -> tuple[list[int], list[int]]:
    universe = build_full_option_plus_intra_edge_table(
        config=config,
        options=(0, 1, 2, 4),
        add_intra_ring=True,
        wrap_planes=bool(wrap_planes),
    )
    num_edges = int(universe.num_edges)
    total_ports = int(config.total_sats) * 2
    edge_port_int = [0 for _ in range(num_edges)]
    port_to_edges_int = [0 for _ in range(total_ports)]
    for edge_idx in range(num_edges):
        if int(universe.option[edge_idx]) == -1:
            continue
        src = int(universe.src[edge_idx])
        dst = int(universe.dst[edge_idx])
        src_plane = int(universe.src_plane[edge_idx])
        dst_plane = int(universe.dst_plane[edge_idx])
        if src_plane <= dst_plane:
            ports = (src * 2 + 1, dst * 2 + 0)
        else:
            ports = (src * 2 + 0, dst * 2 + 1)
        port_mask = 0
        edge_bit = 1 << edge_idx
        for port in ports:
            port_mask |= 1 << int(port)
            port_to_edges_int[int(port)] |= edge_bit
        edge_port_int[edge_idx] = port_mask
    return edge_port_int, port_to_edges_int


def iter_set_bits(value: int):
    x = int(value)
    while x:
        low = x & -x
        idx = low.bit_length() - 1
        yield idx
        x ^= low


def local_lst_cost(
    *,
    e0: int,
    e1: int,
    e2: int,
    edge_port_int: list[int],
    port_to_edges_int: list[int],
) -> tuple[int, int, int]:
    # stride=60, LST=120: row r builds edges that first activate at r+1
    # and r+2. Deadline r+1 edges are processed first, then r+2.
    b1 = int(e1) & ~int(e0)
    b2 = int(e2) & ~int(e1) & ~int(e0)
    owner_ports = 0
    building = 0
    building_dropped = 0
    for requested in (b1, b2):
        for edge_idx in iter_set_bits(requested):
            port_mask = int(edge_port_int[edge_idx])
            if port_mask == 0:
                continue
            if owner_ports & port_mask:
                building_dropped += 1
                continue
            owner_ports |= port_mask
            building += 1
    conflict_edges = 0
    for port_idx in iter_set_bits(owner_ports):
        conflict_edges |= int(port_to_edges_int[port_idx])
    active_drop = (int(e0) & conflict_edges).bit_count()
    return active_drop, building, building_dropped


def read_schedule_actions(path: Path, *, num_steps: int) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if "action_index" not in df.columns or len(df) != int(num_steps):
        return None
    return df["action_index"].to_numpy(dtype=np.int16)


def default_include_schedules() -> list[Path]:
    root = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
    return [
        root
        / "motif0040_0056_region_internal_plus_grid_tabular_bandit_oracle_schedules"
        / "row_mask_tabular_bandit_oracle_lambda0.50.csv",
        root
        / "motif0040_0056_region_internal_plus_grid_row_mask_dp_switch_penalty"
        / "dp_lambda0.50_sw0.5_dwell1.csv",
        root
        / "motif0040_0056_region_internal_plus_grid_row_mask_dp_edge_transition_penalty"
        / "edge_dp_lambda0.50_sw0.5_scale100.csv",
        root
        / "motif0040_0056_region_internal_plus_grid_row_mask_dp_edge_transition_penalty"
        / "edge_dp_lambda0.50_sw2_scale100.csv",
    ]


def candidate_sets(
    *,
    stage_cost: np.ndarray,
    include_actions: list[np.ndarray],
    top_k: int,
) -> list[np.ndarray]:
    t_count, a_count = stage_cost.shape
    out: list[np.ndarray] = []
    for t in range(t_count):
        selected = set(int(x) for x in np.argsort(stage_cost[t])[: int(top_k)].tolist())
        selected.add(0)
        for actions in include_actions:
            selected.add(int(actions[t]))
        values = np.asarray(sorted(x for x in selected if 0 <= x < a_count), dtype=np.int16)
        out.append(values)
    return out


def reduce_beam(states: dict[tuple[int, int], BeamState], beam_width: int) -> dict[tuple[int, int], BeamState]:
    values = sorted(states.values(), key=lambda s: s.cost)
    kept = values[: int(beam_width)]
    return {(int(s.prev_action), int(s.curr_action)): s for s in kept}


def run_beam_search(
    *,
    stage_cost: np.ndarray,
    mask_ints: list[list[int]],
    candidates: list[np.ndarray],
    edge_port_int: list[int],
    port_to_edges_int: list[int],
    active_drop_weight: float,
    building_weight: float,
    edge_scale: float,
    beam_width: int,
    progress_every: int,
) -> BeamState:
    t_count = int(stage_cost.shape[0])
    if t_count < 2:
        raise ValueError("need at least two time steps")
    beam: dict[tuple[int, int], BeamState] = {}
    for a0 in candidates[0]:
        for a1 in candidates[1]:
            a0_i = int(a0)
            a1_i = int(a1)
            cost = float(stage_cost[0, a0_i] + stage_cost[1, a1_i])
            state = BeamState(
                prev_action=a0_i,
                curr_action=a1_i,
                cost=cost,
                path=(a0_i, a1_i),
                predicted_active_drop=0,
                predicted_building=0,
            )
            key = (a0_i, a1_i)
            old = beam.get(key)
            if old is None or state.cost < old.cost:
                beam[key] = state
    beam = reduce_beam(beam, int(beam_width))
    norm = max(1e-9, float(edge_scale))
    started = time.perf_counter()
    for t in range(2, t_count):
        next_states: dict[tuple[int, int], BeamState] = {}
        for state in beam.values():
            a = int(state.prev_action)
            b = int(state.curr_action)
            e0 = mask_ints[t - 2][a]
            e1 = mask_ints[t - 1][b]
            for c_raw in candidates[t]:
                c = int(c_raw)
                e2 = mask_ints[t][c]
                active_drop, building, _building_dropped = local_lst_cost(
                    e0=e0,
                    e1=e1,
                    e2=e2,
                    edge_port_int=edge_port_int,
                    port_to_edges_int=port_to_edges_int,
                )
                lst_cost = (
                    float(active_drop_weight) * float(active_drop)
                    + float(building_weight) * float(building)
                ) / norm
                new_cost = float(state.cost) + float(stage_cost[t, c]) + lst_cost
                key = (b, c)
                old = next_states.get(key)
                if old is None or new_cost < old.cost:
                    next_states[key] = BeamState(
                        prev_action=b,
                        curr_action=c,
                        cost=new_cost,
                        path=state.path + (c,),
                        predicted_active_drop=int(state.predicted_active_drop + active_drop),
                        predicted_building=int(state.predicted_building + building),
                    )
        beam = reduce_beam(next_states, int(beam_width))
        if (t + 1) % int(progress_every) == 0 or t + 1 == t_count:
            best = min(beam.values(), key=lambda s: s.cost)
            elapsed = time.perf_counter() - started
            print(
                f"[lst-drop-beam] t={t + 1}/{t_count} beam={len(beam)} "
                f"best_cost={best.cost:.4f} drop={best.predicted_active_drop} elapsed={elapsed:.1f}s",
                flush=True,
            )

    final_states: dict[tuple[int, int], BeamState] = {}
    zero_future = 0
    for state in beam.values():
        a = int(state.prev_action)
        b = int(state.curr_action)
        e0 = mask_ints[t_count - 2][a]
        e1 = mask_ints[t_count - 1][b]
        active_drop, building, _building_dropped = local_lst_cost(
            e0=e0,
            e1=e1,
            e2=zero_future,
            edge_port_int=edge_port_int,
            port_to_edges_int=port_to_edges_int,
        )
        lst_cost = (
            float(active_drop_weight) * float(active_drop)
            + float(building_weight) * float(building)
        ) / norm
        final_states[(a, b)] = BeamState(
            prev_action=a,
            curr_action=b,
            cost=float(state.cost) + lst_cost,
            path=state.path,
            predicted_active_drop=int(state.predicted_active_drop + active_drop),
            predicted_building=int(state.predicted_building + building),
        )
    return min(final_states.values(), key=lambda s: s.cost)


def schedule_rows(
    *,
    steps: np.ndarray,
    selected: np.ndarray,
    policy: str,
    lambda_hop: float,
    active_drop_weight: float,
    building_weight: float,
    edge_scale: float,
    action_names: Sequence[str],
    action_masks: np.ndarray,
    mask_tokens: Sequence[str],
    hops: np.ndarray,
    delay: np.ndarray,
    hop_env: np.ndarray,
    delay_env: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, action_idx in enumerate(selected.astype(int).tolist()):
        mask = action_masks[action_idx]
        row: dict[str, Any] = {
            "step": int(steps[t]),
            "policy": policy,
            "lambda_hop": float(lambda_hop),
            "active_drop_weight": float(active_drop_weight),
            "building_weight": float(building_weight),
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
    final_state: BeamState,
) -> dict[str, Any]:
    rows = np.arange(int(selected.size))
    idx = selected.astype(int)
    switches = int(np.count_nonzero(idx[1:] != idx[:-1]))
    return {
        "mean_hops": float(np.mean(hops[rows, idx])),
        "mean_delay_ms": float(np.mean(delay[rows, idx])),
        "mean_gap_hop_env": float(np.mean(hops[rows, idx] - hop_env)),
        "mean_gap_delay_env_ms": float(np.mean(delay[rows, idx] - delay_env)),
        "switches": switches,
        "num_segments": int(switches + 1),
        "num_used_actions": int(len(set(int(x) for x in idx.tolist()))),
        "beam_objective_cost": float(final_state.cost),
        "predicted_active_drop_sum": int(final_state.predicted_active_drop),
        "predicted_building_sum": int(final_state.predicted_building),
        "predicted_active_drop_mean": float(final_state.predicted_active_drop / max(1, int(selected.size))),
        "predicted_building_mean": float(final_state.predicted_building / max(1, int(selected.size))),
    }


def main() -> int:
    args = parse_args()
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_names, action_masks, mask_tokens = read_action_table(Path(args.reward_dir) / "row_mask_action_library_actions.csv")
    steps_hop, names_hop, hops = read_wide(Path(args.reward_dir) / "row_mask_action_library_mean_hops_wide.csv")
    steps_delay, names_delay, delay = read_wide(Path(args.reward_dir) / "row_mask_action_library_mean_delay_ms_wide.csv")
    if not np.array_equal(steps_hop, steps_delay):
        raise ValueError("hop/delay tables have different steps")
    ensure_same_actions(action_names, names_hop, "hop table")
    ensure_same_actions(action_names, names_delay, "delay table")

    cache_dir = Path(args.mask_cache_dir)
    edge_masks, _edge_universe_size = build_or_load_action_edge_masks(
        args=args,
        out_dir=cache_dir,
        raw=raw,
        config=config,
        steps=steps_hop,
        action_masks=action_masks,
    )
    mask_ints = [
        [words_to_int(edge_masks[t, a]) for a in range(edge_masks.shape[1])]
        for t in range(edge_masks.shape[0])
    ]
    edge_port_int, port_to_edges_int = build_port_maps(config, wrap_planes=bool(wrap_planes_from_config(raw)))

    hop_env = np.nanmin(hops, axis=1)
    delay_env = np.nanmin(delay, axis=1)
    hop_scale = max(1e-9, float(np.nanmax(hops) - np.nanmin(hops)))
    delay_scale = max(1e-9, float(np.nanmax(delay) - np.nanmin(delay)))
    stage_cost = (
        float(args.lambda_hop) * ((hops - hop_env[:, None]) / hop_scale)
        + (1.0 - float(args.lambda_hop)) * ((delay - delay_env[:, None]) / delay_scale)
    )

    include_paths = args.include_schedules if args.include_schedules else default_include_schedules()
    include_actions = []
    for path in include_paths:
        values = read_schedule_actions(Path(path), num_steps=int(steps_hop.size))
        if values is not None:
            include_actions.append(values)
    candidates = candidate_sets(stage_cost=stage_cost, include_actions=include_actions, top_k=int(args.top_k_actions))

    summary_rows: list[dict[str, Any]] = []
    for active_drop_weight in args.active_drop_weights:
        policy = (
            f"beam_lstdrop_lambda{float(args.lambda_hop):.2f}"
            f"_drop{float(active_drop_weight):g}_build{float(args.building_weight):g}"
            f"_k{int(args.top_k_actions)}_b{int(args.beam_width)}"
        )
        final_state = run_beam_search(
            stage_cost=stage_cost,
            mask_ints=mask_ints,
            candidates=candidates,
            edge_port_int=edge_port_int,
            port_to_edges_int=port_to_edges_int,
            active_drop_weight=float(active_drop_weight),
            building_weight=float(args.building_weight),
            edge_scale=float(args.edge_scale),
            beam_width=int(args.beam_width),
            progress_every=int(args.progress_every),
        )
        selected = np.asarray(final_state.path, dtype=np.int16)
        rows = schedule_rows(
            steps=steps_hop,
            selected=selected,
            policy=policy,
            lambda_hop=float(args.lambda_hop),
            active_drop_weight=float(active_drop_weight),
            building_weight=float(args.building_weight),
            edge_scale=float(args.edge_scale),
            action_names=action_names,
            action_masks=action_masks,
            mask_tokens=mask_tokens,
            hops=hops,
            delay=delay,
            hop_env=hop_env,
            delay_env=delay_env,
        )
        schedule_path = out_dir / f"{policy}.csv"
        write_rows(schedule_path, rows)
        summary_rows.append(
            {
                "policy": policy,
                "schedule_csv": str(schedule_path),
                "lambda_hop": float(args.lambda_hop),
                "active_drop_weight": float(active_drop_weight),
                "building_weight": float(args.building_weight),
                "edge_scale": float(args.edge_scale),
                "beam_width": int(args.beam_width),
                "top_k_actions": int(args.top_k_actions),
                **summarize_policy(
                    selected=selected,
                    hops=hops,
                    delay=delay,
                    hop_env=hop_env,
                    delay_env=delay_env,
                    final_state=final_state,
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "beam_lst_drop_aware_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta = {
        "config": str(Path(args.config)),
        "reward_dir": str(Path(args.reward_dir)),
        "mask_cache_dir": str(Path(args.mask_cache_dir)),
        "out_dir": str(out_dir),
        "hop_scale": hop_scale,
        "delay_scale": delay_scale,
        "mean_hop_envelope": float(np.mean(hop_env)),
        "mean_delay_envelope_ms": float(np.mean(delay_env)),
        "summary_csv": str(summary_path),
    }
    (out_dir / "beam_lst_drop_aware_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
