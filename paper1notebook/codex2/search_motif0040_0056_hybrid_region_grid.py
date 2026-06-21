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

from run_m56_local_patch_hybrid_topology import build_hybrid_edge_table, cyclic_band, degree_stats  # noqa: E402
from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs, wrap_planes_from_config  # noqa: E402
from run_paper1_region_internal_grid_metrics import (  # noqa: E402
    _apply_constraint_context_fast,
    _build_adjacency,
    _build_pair_step_contexts,
    _summarize_delay,
    _summarize_hops,
)
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import TopologySpec, topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402


DEFAULT_CONFIG = (
    GENERIC_ROOT
    / "paper1notebook"
    / "codex2"
    / "configs"
    / "g60_motif0040_0056_china_europe_region_internal_plus_grid.yaml"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_hybrid_search"
)
PURE_METRIC_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\static_motif0040_0056_china_europe_region_internal_plus_grid"
    r"\region_internal_grid_metrics_t0_86160_stride60\china_europe"
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    topology: TopologySpec
    mode: str
    band_start: int | None
    band_end: int | None
    band_len: int
    added_edges: int
    removed_edges: int
    degree: dict[str, Any]

    @property
    def changed_edges(self) -> int:
        return int(self.added_edges + self.removed_edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search local 000040/000056 hybrid candidates under China-Europe region-internal +grid constraints."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--screen-sample-stride", type=int, default=1800)
    parser.add_argument("--band-lengths", nargs="+", type=int, default=(6, 12, 18))
    parser.add_argument("--patch-modes", nargs="+", default=("all",), choices=("c", "cb", "all"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--selected-indices",
        nargs="+",
        type=int,
        default=None,
        help="Skip screen evaluation and run full metrics for these candidate indices. Pure 000056/000040 are added automatically.",
    )
    parser.add_argument(
        "--reuse-full-dir",
        type=Path,
        default=None,
        help="Reuse candidate columns from an existing full_t*/ directory and only evaluate missing selected candidates.",
    )
    parser.add_argument("--hop-weight", type=float, default=1.0)
    parser.add_argument("--delay-weight", type=float, default=1.0)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(value)).strip("_")


def edge_key_set(topology: TopologySpec) -> frozenset[tuple[int, int]]:
    table = topology.edge_table
    pairs = []
    for idx in range(table.num_edges):
        a = int(table.src[idx])
        b = int(table.dst[idx])
        pairs.append((a, b) if a <= b else (b, a))
    return frozenset(pairs)


def build_candidate_specs(
    *,
    config: Any,
    motif_csv: Path,
    name_prefix: str,
    band_lengths: Sequence[int],
    patch_modes: Sequence[str],
    wrap_planes: bool,
) -> list[CandidateSpec]:
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library="selected_motif",
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=bool(wrap_planes),
    )
    by_id = {int(spec.motif_id): spec for spec in specs if spec.motif_id is not None}
    base = by_id[56]
    patch = by_id[40]
    candidates: list[CandidateSpec] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    for spec, label in ((base, "pure_000056"), (patch, "pure_000040")):
        key = edge_key_set(spec)
        seen.add(key)
        candidates.append(
            CandidateSpec(
                name=label,
                topology=TopologySpec(
                    name=label,
                    edge_table=spec.edge_table,
                    library="selected_motif",
                    motif_id=spec.motif_id,
                    motif=spec.motif,
                    source_w=spec.source_w,
                    source_h=spec.source_h,
                    edge_count_local=spec.edge_count_local,
                    support=spec.support,
                    baseline=False,
                    meta={"kind": "pure"},
                ),
                mode="pure",
                band_start=None,
                band_end=None,
                band_len=0,
                added_edges=0,
                removed_edges=0,
                degree=degree_stats(spec.edge_table, total_sats=int(config.total_sats)),
            )
        )

    for mode in patch_modes:
        for band_len in band_lengths:
            for band_start in range(int(config.N)):
                band_end = (int(band_start) + int(band_len) - 1) % int(config.N)
                edge_table, added, removed, degree = build_hybrid_edge_table(
                    base_spec=base,
                    patch_spec=patch,
                    p=int(config.P),
                    n=int(config.N),
                    band=cyclic_band(band_start, band_end, n=int(config.N)),
                    patch_mode=str(mode),
                )
                if int(degree["max_out_degree"]) > 1 or int(degree["max_in_degree"]) > 1:
                    continue
                name = f"hybrid_m56_m40_{mode}_y{band_start:02d}_{band_end:02d}_l{int(band_len):02d}"
                topology = TopologySpec(
                    name=name,
                    edge_table=edge_table,
                    library="hybrid_000056_000040",
                    motif_id=None,
                    motif=f"base=000056 patch=000040 mode={mode} y={band_start}-{band_end} len={band_len}",
                    baseline=False,
                    meta={
                        "kind": "hybrid",
                        "base": "000056",
                        "patch": "000040",
                        "mode": str(mode),
                        "band_start": int(band_start),
                        "band_end": int(band_end),
                        "band_len": int(band_len),
                    },
                )
                key = edge_key_set(topology)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    CandidateSpec(
                        name=name,
                        topology=topology,
                        mode=str(mode),
                        band_start=int(band_start),
                        band_end=int(band_end),
                        band_len=int(band_len),
                        added_edges=int(len(added)),
                        removed_edges=int(len(removed)),
                        degree=degree,
                    )
                )
    return candidates


def finite_minmax(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def read_pure_envelopes(path: Path, *, steps: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    def read_csv(csv_path: Path) -> dict[int, tuple[float, float]]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        out: dict[int, tuple[float, float]] = {}
        for row in rows:
            step = int(float(row["step"]))
            out[step] = (float(row["selected_motif_000040"]), float(row["selected_motif_000056"]))
        return out

    hops = read_csv(path / "compare_mean_shortest_hops.csv")
    delay = read_csv(path / "compare_mean_shortest_delay_ms.csv")
    hop_env = []
    delay_env = []
    for step in steps:
        hop_env.append(min(hops[int(step)]))
        delay_env.append(min(delay[int(step)]))
    return np.asarray(hop_env, dtype=np.float64), np.asarray(delay_env, dtype=np.float64)


def evaluate_candidates(
    *,
    candidates: Sequence[CandidateSpec],
    config: Any,
    group_data: dict,
    pair_spec: Any,
    steps: Sequence[int],
    delay_store: FullLinkDelayStore,
    forced_option: int,
    wrap_planes: bool,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    contexts = _build_pair_step_contexts(
        config=config,
        group_data=group_data,
        steps=[int(x) for x in steps],
        pair_specs=[pair_spec],
        forced_option=int(forced_option),
        wrap_planes=bool(wrap_planes),
    )[str(pair_spec.key)]
    hop = np.full((len(steps), len(candidates)), np.nan, dtype=np.float32)
    delay = np.full((len(steps), len(candidates)), np.nan, dtype=np.float32)
    started = time.perf_counter()
    for col, candidate in enumerate(candidates):
        for row, step in enumerate(steps):
            constrained_edge_table, _forced_count, _dropped_count = _apply_constraint_context_fast(
                base_edge_table=candidate.topology.edge_table,
                context=contexts[row],
                total_nodes=int(config.total_sats),
                p=int(config.P),
                n=int(config.N),
                forced_option=int(forced_option),
                wrap_planes=bool(wrap_planes),
            )
            adjacency = _build_adjacency(constrained_edge_table.src, constrained_edge_table.dst, int(config.total_sats))
            _rh, mean_hops, _min_h, _max_h = _summarize_hops(
                adjacency=adjacency,
                sources=contexts[row].sources,
                targets=contexts[row].targets,
            )
            _rd, mean_delay, _min_d, _max_d = _summarize_delay(
                edge_table=constrained_edge_table,
                config=config,
                delay_store=delay_store,
                position_store=None,
                delay_row=delay_store.row_for_time_step(int(step)),
                position_row=None,
                sources=contexts[row].sources,
                targets=contexts[row].targets,
                engine="auto",
            )
            hop[row, col] = float(mean_hops)
            delay[row, col] = float(mean_delay)
        if (col + 1) == len(candidates) or (col + 1) % 5 == 0:
            print(
                f"[hybrid-search] {progress_label} {col + 1}/{len(candidates)} "
                f"elapsed={time.perf_counter() - started:.1f}s candidate={candidate.name}",
                flush=True,
            )
    return hop, delay


def score_candidates(hop: np.ndarray, delay: np.ndarray, *, hop_weight: float, delay_weight: float) -> np.ndarray:
    hop_lo, hop_hi = finite_minmax(hop)
    delay_lo, delay_hi = finite_minmax(delay)
    hop_norm = (hop - hop_lo) / (hop_hi - hop_lo)
    delay_norm = (delay - delay_lo) / (delay_hi - delay_lo)
    score = float(hop_weight) * hop_norm + float(delay_weight) * delay_norm
    return np.nanmean(score, axis=0)


def write_summary_csv(
    path: Path,
    *,
    candidates: Sequence[CandidateSpec],
    hop: np.ndarray,
    delay: np.ndarray,
    score: np.ndarray,
    selected_steps: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_steps = np.zeros(len(candidates), dtype=np.int32) if selected_steps is None else selected_steps
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "candidate_idx",
            "name",
            "mode",
            "band_start",
            "band_end",
            "band_len",
            "changed_edges",
            "mean_hops",
            "mean_delay_ms",
            "score",
            "oracle_selected_steps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, cand in enumerate(candidates):
            writer.writerow(
                {
                    "candidate_idx": idx,
                    "name": cand.name,
                    "mode": cand.mode,
                    "band_start": "" if cand.band_start is None else cand.band_start,
                    "band_end": "" if cand.band_end is None else cand.band_end,
                    "band_len": cand.band_len,
                    "changed_edges": cand.changed_edges,
                    "mean_hops": float(np.nanmean(hop[:, idx])),
                    "mean_delay_ms": float(np.nanmean(delay[:, idx])),
                    "score": float(score[idx]),
                    "oracle_selected_steps": int(selected_steps[idx]),
                }
            )


def write_compare_csv(path: Path, *, steps: Sequence[int], candidates: Sequence[CandidateSpec], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["step", *[cand.name for cand in candidates]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, step in enumerate(steps):
            item = {"step": int(step)}
            for col, cand in enumerate(candidates):
                item[cand.name] = float(values[row, col])
            writer.writerow(item)


def write_selected_indices(path: Path, *, selected_indices: Sequence[int], selected_candidates: Sequence[CandidateSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "selected_indices": [int(x) for x in selected_indices],
                "selected_names": [c.name for c in selected_candidates],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read_compare_csv_by_name(path: Path, *, expected_steps: Sequence[int]) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row_steps = [int(float(row["step"])) for row in rows]
    if list(row_steps) != [int(x) for x in expected_steps]:
        print(f"[hybrid-search] skip reuse; step mismatch: {path}", flush=True)
        return {}
    fieldnames = [name for name in rows[0].keys() if name != "step"]
    out: dict[str, np.ndarray] = {}
    for name in fieldnames:
        out[str(name)] = np.asarray([float(row[name]) for row in rows], dtype=np.float32)
    return out


def evaluate_candidates_with_reuse(
    *,
    candidates: Sequence[CandidateSpec],
    config: Any,
    group_data: dict,
    pair_spec: Any,
    steps: Sequence[int],
    delay_store: FullLinkDelayStore,
    forced_option: int,
    wrap_planes: bool,
    reuse_full_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hop_by_name: dict[str, np.ndarray] = {}
    delay_by_name: dict[str, np.ndarray] = {}
    if reuse_full_dir is not None:
        reuse_dir = Path(reuse_full_dir)
        hop_by_name = read_compare_csv_by_name(reuse_dir / "compare_mean_shortest_hops.csv", expected_steps=steps)
        delay_by_name = read_compare_csv_by_name(reuse_dir / "compare_mean_shortest_delay_ms.csv", expected_steps=steps)
        print(
            f"[hybrid-search] reuse dir={reuse_dir} hop_cols={len(hop_by_name)} delay_cols={len(delay_by_name)}",
            flush=True,
        )

    full_hop = np.full((len(steps), len(candidates)), np.nan, dtype=np.float32)
    full_delay = np.full((len(steps), len(candidates)), np.nan, dtype=np.float32)
    missing_candidates: list[CandidateSpec] = []
    missing_cols: list[int] = []
    reused_names: list[str] = []
    for col, candidate in enumerate(candidates):
        if candidate.name in hop_by_name and candidate.name in delay_by_name:
            full_hop[:, col] = hop_by_name[candidate.name]
            full_delay[:, col] = delay_by_name[candidate.name]
            reused_names.append(candidate.name)
        else:
            missing_candidates.append(candidate)
            missing_cols.append(col)

    if missing_candidates:
        print(
            f"[hybrid-search] full reuse={len(reused_names)} missing={len(missing_candidates)}",
            flush=True,
        )
        missing_hop, missing_delay = evaluate_candidates(
            candidates=missing_candidates,
            config=config,
            group_data=group_data,
            pair_spec=pair_spec,
            steps=steps,
            delay_store=delay_store,
            forced_option=forced_option,
            wrap_planes=bool(wrap_planes),
            progress_label="full-missing",
        )
        for local_col, full_col in enumerate(missing_cols):
            full_hop[:, full_col] = missing_hop[:, local_col]
            full_delay[:, full_col] = missing_delay[:, local_col]
    else:
        print(f"[hybrid-search] all selected candidates reused from {reuse_full_dir}", flush=True)

    meta = {
        "reuse_full_dir": "" if reuse_full_dir is None else str(Path(reuse_full_dir)),
        "reused_candidates": len(reused_names),
        "computed_candidates": len(missing_candidates),
        "computed_names": [c.name for c in missing_candidates],
    }
    return full_hop, full_delay, meta


def plot_full(
    *,
    out_dir: Path,
    steps: np.ndarray,
    candidates: Sequence[CandidateSpec],
    hop: np.ndarray,
    delay: np.ndarray,
    pure_hop_env: np.ndarray,
    pure_delay_env: np.ndarray,
    score: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oracle_idx = np.nanargmin(
        ((hop - finite_minmax(hop)[0]) / (finite_minmax(hop)[1] - finite_minmax(hop)[0]))
        + ((delay - finite_minmax(delay)[0]) / (finite_minmax(delay)[1] - finite_minmax(delay)[0])),
        axis=1,
    )
    oracle_hop = hop[np.arange(hop.shape[0]), oracle_idx]
    oracle_delay = delay[np.arange(delay.shape[0]), oracle_idx]
    best_cols = np.argsort(score)[: min(6, len(candidates))]
    x = steps.astype(np.float64) / 3600.0

    for metric, values, pure_env, oracle_values, ylabel, filename in (
        ("hops", hop, pure_hop_env, oracle_hop, "China-Europe mean shortest hops", "hybrid_search_hops.png"),
        ("delay", delay, pure_delay_env, oracle_delay, "China-Europe mean shortest delay (ms)", "hybrid_search_delay_ms.png"),
    ):
        fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=180)
        ax.plot(x, pure_env, color="#111827", linestyle="--", linewidth=2.0, label=f"pure 040/056 {metric} envelope")
        ax.plot(x, oracle_values, color="#059669", linewidth=2.0, label="best hybrid among searched candidates")
        for col in best_cols:
            ax.plot(x, values[:, col], linewidth=0.9, alpha=0.55, label=candidates[int(col)].name)
        ax.set_title(f"G60 China-Europe region-internal +grid hybrid 040/056 search: {metric}")
        ax.set_xlabel("time (hour)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)


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
    forced_option = int(raw.get("region_internal_constraint", {}).get("forced_option", 0))
    pairs = {pair.key: pair for pair in region_pair_specs(raw, subset=[str(args.target_pair)])}
    pair_spec = pairs[str(args.target_pair)]
    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    sample_steps = np.arange(int(args.start), int(args.end) + 1, int(args.screen_sample_stride), dtype=np.int64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_specs(
        config=config,
        motif_csv=motif_csv,
        name_prefix=str(motif_library.get("name_prefix", "selected")),
        band_lengths=tuple(int(x) for x in args.band_lengths),
        patch_modes=tuple(str(x) for x in args.patch_modes),
        wrap_planes=bool(wrap_planes),
    )
    print(f"[hybrid-search] candidates={len(candidates)} sample_steps={len(sample_steps)} full_steps={len(steps)}")

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps.tolist(),
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    delay_store = FullLinkDelayStore(path_from(paths, "delay_store_dir"))

    screen_dir = out_dir / "screen"
    if args.selected_indices:
        selected_indices = sorted(set([0, 1] + [int(x) for x in args.selected_indices]))
        invalid = [idx for idx in selected_indices if idx < 0 or idx >= len(candidates)]
        if invalid:
            raise ValueError(f"selected candidate indices out of range: {invalid}; candidates={len(candidates)}")
        sample_hop = np.empty((0, len(candidates)), dtype=np.float32)
        sample_delay = np.empty((0, len(candidates)), dtype=np.float32)
        sample_score = np.full(len(candidates), np.nan, dtype=np.float32)
    else:
        sample_hop, sample_delay = evaluate_candidates(
            candidates=candidates,
            config=config,
            group_data=group_data,
            pair_spec=pair_spec,
            steps=sample_steps.tolist(),
            delay_store=delay_store,
            forced_option=forced_option,
            wrap_planes=bool(wrap_planes),
            progress_label="screen",
        )
        sample_score = score_candidates(
            sample_hop,
            sample_delay,
            hop_weight=float(args.hop_weight),
            delay_weight=float(args.delay_weight),
        )
        write_summary_csv(
            screen_dir / "candidate_screen_summary.csv",
            candidates=candidates,
            hop=sample_hop,
            delay=sample_delay,
            score=sample_score,
        )
        selected_indices = sorted(set([0, 1] + [int(x) for x in np.argsort(sample_score)[: int(args.top_k)].tolist()]))
    selected_candidates = [candidates[idx] for idx in selected_indices]
    write_selected_indices(screen_dir / "selected_candidate_indices.json", selected_indices=selected_indices, selected_candidates=selected_candidates)
    print(f"[hybrid-search] selected={selected_indices}")

    if bool(args.screen_only):
        return 0

    full_hop, full_delay, reuse_meta = evaluate_candidates_with_reuse(
        candidates=selected_candidates,
        config=config,
        group_data=group_data,
        pair_spec=pair_spec,
        steps=steps.tolist(),
        delay_store=delay_store,
        forced_option=forced_option,
        wrap_planes=bool(wrap_planes),
        reuse_full_dir=args.reuse_full_dir,
    )
    full_score = score_candidates(
        full_hop,
        full_delay,
        hop_weight=float(args.hop_weight),
        delay_weight=float(args.delay_weight),
    )
    oracle_idx = np.nanargmin(
        ((full_hop - finite_minmax(full_hop)[0]) / (finite_minmax(full_hop)[1] - finite_minmax(full_hop)[0]))
        + ((full_delay - finite_minmax(full_delay)[0]) / (finite_minmax(full_delay)[1] - finite_minmax(full_delay)[0])),
        axis=1,
    )
    selected_counts = np.bincount(oracle_idx, minlength=len(selected_candidates))
    full_dir = out_dir / f"full_t{int(args.start)}_{int(args.end)}_s{int(args.stride)}"
    write_summary_csv(
        full_dir / "candidate_full_summary.csv",
        candidates=selected_candidates,
        hop=full_hop,
        delay=full_delay,
        score=full_score,
        selected_steps=selected_counts,
    )
    write_compare_csv(full_dir / "compare_mean_shortest_hops.csv", steps=steps.tolist(), candidates=selected_candidates, values=full_hop)
    write_compare_csv(full_dir / "compare_mean_shortest_delay_ms.csv", steps=steps.tolist(), candidates=selected_candidates, values=full_delay)
    pure_hop_env, pure_delay_env = read_pure_envelopes(PURE_METRIC_DIR, steps=steps.tolist())
    hybrid_hop_env = np.nanmin(full_hop, axis=1)
    hybrid_delay_env = np.nanmin(full_delay, axis=1)
    hybrid_scalar_idx = oracle_idx
    hybrid_scalar_hop = full_hop[np.arange(full_hop.shape[0]), hybrid_scalar_idx]
    hybrid_scalar_delay = full_delay[np.arange(full_delay.shape[0]), hybrid_scalar_idx]
    meta = {
        "config": str(Path(args.config)),
        "out_dir": str(out_dir),
        "candidate_count": len(candidates),
        "selected_indices": selected_indices,
        "selected_names": [c.name for c in selected_candidates],
        "full_steps": int(len(steps)),
        "sample_steps": int(len(sample_steps)),
        "mean_pure_hop_envelope": float(np.mean(pure_hop_env)),
        "mean_hybrid_hop_envelope": float(np.mean(hybrid_hop_env)),
        "mean_pure_delay_envelope_ms": float(np.mean(pure_delay_env)),
        "mean_hybrid_delay_envelope_ms": float(np.mean(hybrid_delay_env)),
        "mean_hybrid_scalar_hops": float(np.mean(hybrid_scalar_hop)),
        "mean_hybrid_scalar_delay_ms": float(np.mean(hybrid_scalar_delay)),
        "hybrid_hop_better_than_pure_env_steps": int(np.count_nonzero(hybrid_hop_env < pure_hop_env - 1e-9)),
        "hybrid_delay_better_than_pure_env_steps": int(np.count_nonzero(hybrid_delay_env < pure_delay_env - 1e-9)),
        "hybrid_scalar_selected_counts": {
            selected_candidates[idx].name: int(count)
            for idx, count in enumerate(selected_counts.tolist())
            if int(count) > 0
        },
        "reuse_meta": reuse_meta,
    }
    (full_dir / "hybrid_search_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_full(
        out_dir=full_dir,
        steps=steps,
        candidates=selected_candidates,
        hop=full_hop,
        delay=full_delay,
        pure_hop_env=pure_hop_env,
        pure_delay_env=pure_delay_env,
        score=full_score,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
