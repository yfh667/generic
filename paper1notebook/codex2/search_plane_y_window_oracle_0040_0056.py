from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_m56_local_patch_hybrid_topology import (  # noqa: E402
    EdgeRecord,
    edge_records_from_table,
    node_id,
    patch_option_allowed,
)
from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs, wrap_planes_from_config  # noqa: E402
from run_paper1_region_internal_grid_metrics import (  # noqa: E402
    _apply_constraint_context_fast,
    _build_adjacency,
    _build_pair_step_contexts,
    _summarize_delay,
    _summarize_hops,
)
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG, edge_key_set  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import TopologySpec, topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402
from src.topology_workflow.module.edge_tables import make_edge_table_from_records  # noqa: E402


G60_RUN_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
DEFAULT_ROW_ORACLE_CSV = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_row_mask_oracle_hard8_beam16"
    / "row_mask_oracle_hard_steps.csv"
)
DEFAULT_BASELINE_DIR = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_row_mask_action_library_extended_hard8_beam"
    / "reward_table_hard8"
)
DEFAULT_OUT_DIR = G60_RUN_ROOT / "motif0040_0056_region_internal_plus_grid_plane_y_window_search_hard8"


@dataclass(frozen=True)
class Candidate:
    name: str
    topology: TopologySpec
    plane_start: int | None
    plane_len: int
    row_mask: tuple[int, ...]
    row_source: str
    patch_mode: str
    added_edges: int
    removed_edges: int
    degree: dict[str, Any]


@dataclass(frozen=True)
class Metric:
    hops: float
    delay_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a more local 000040/000056 fusion action space: patch motif 000040 "
            "only inside selected x-plane windows and selected y-row masks."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--row-oracle-csv", type=Path, default=DEFAULT_ROW_ORACLE_CSV)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument(
        "--selected-names",
        nargs="*",
        default=None,
        help="Optional candidate names to evaluate after generating the candidate pool.",
    )
    parser.add_argument(
        "--selected-names-file",
        type=Path,
        default=None,
        help="Text file with one candidate name per line. Empty lines and # comments are ignored.",
    )
    parser.add_argument("--plane-lengths", nargs="+", type=int, default=(3, 6, 9, 18))
    parser.add_argument("--extra-row-band-lengths", nargs="+", type=int, default=(3, 6, 12))
    parser.add_argument("--patch-mode", choices=("c", "cb", "all"), default="all")
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional cap after dedupe; 0 means evaluate all.")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def cyclic_values(start: int, length: int, mod: int) -> tuple[int, ...]:
    if int(length) >= int(mod):
        return tuple(range(int(mod)))
    return tuple((int(start) + idx) % int(mod) for idx in range(int(length)))


def mask_token(rows: Sequence[int]) -> str:
    values = tuple(sorted(set(int(x) for x in rows)))
    return "{" + ",".join(f"{x:02d}" for x in values) + "}"


def safe_token(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_{} ,-]+", "_", str(text))
    text = text.replace("{", "").replace("}", "").replace(",", "_").replace(" ", "")
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "empty"


def parse_mask_token(text: str, n: int) -> tuple[int, ...]:
    values = [int(x) % int(n) for x in re.findall(r"\d+", str(text))]
    return tuple(sorted(set(values)))


def read_seed_steps_and_masks(path: Path, *, n: int) -> tuple[list[int], list[tuple[str, tuple[int, ...]]]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty row oracle CSV: {path}")

    steps = [int(float(row["step"])) for row in rows]
    seeds: dict[tuple[int, ...], str] = {
        tuple(): "pure56_rows",
        tuple(range(int(n))): "pure40_rows",
    }
    for row in rows:
        for key in ("row_oracle_rows", "beam_rows"):
            token = str(row.get(key, "")).strip()
            if not token:
                continue
            mask = parse_mask_token(token, int(n))
            seeds.setdefault(mask, key)
    return steps, [(source, mask) for mask, source in seeds.items()]


def selected_names_from_args(args: argparse.Namespace) -> set[str] | None:
    names: set[str] = set()
    if args.selected_names:
        names.update(str(x).strip() for x in args.selected_names if str(x).strip())
    if args.selected_names_file is not None:
        for line in Path(args.selected_names_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(line)
    return names or None


def read_baseline_envelope(path: Path, *, steps: Sequence[int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    def read_wide(csv_path: Path) -> tuple[list[int], list[str], np.ndarray]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        names = [name for name in rows[0] if name != "step"]
        row_steps = [int(float(row["step"])) for row in rows]
        matrix = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
        return row_steps, names, matrix

    hop_steps, hop_names, hop = read_wide(Path(path) / "row_mask_action_library_mean_hops_wide.csv")
    delay_steps, delay_names, delay = read_wide(Path(path) / "row_mask_action_library_mean_delay_ms_wide.csv")
    if hop_steps != delay_steps or hop_names != delay_names:
        raise ValueError("baseline hop/delay tables are not aligned")
    row_by_step = {int(step): idx for idx, step in enumerate(hop_steps)}
    indices = [row_by_step[int(step)] for step in steps]
    hop_sub = hop[indices, :]
    delay_sub = delay[indices, :]
    return (
        np.nanmin(hop_sub, axis=1),
        np.nanmin(delay_sub, axis=1),
        {
            "baseline_dir": str(Path(path)),
            "baseline_actions": len(hop_names),
            "baseline_steps": [int(x) for x in steps],
            "mean_baseline_hop_envelope": float(np.nanmean(np.nanmin(hop_sub, axis=1))),
            "mean_baseline_delay_envelope_ms": float(np.nanmean(np.nanmin(delay_sub, axis=1))),
        },
    )


def degree_stats_from_records(records: Sequence[EdgeRecord], *, n: int, total_sats: int) -> dict[str, Any]:
    out_degree: Counter[int] = Counter()
    in_degree: Counter[int] = Counter()
    for record in records:
        if int(record.option) == -1:
            continue
        out_degree[node_id(record.src_plane, record.src_y, n=n)] += 1
        in_degree[node_id(record.dst_plane, record.dst_y, n=n)] += 1
    return {
        "total_edges": int(len(records)),
        "inter_edges": int(sum(1 for record in records if int(record.option) != -1)),
        "intra_edges": int(sum(1 for record in records if int(record.option) == -1)),
        "max_out_degree": max(out_degree.values(), default=0),
        "max_in_degree": max(in_degree.values(), default=0),
        "out_degree_gt1_nodes": int(sum(1 for value in out_degree.values() if value > 1)),
        "in_degree_gt1_nodes": int(sum(1 for value in in_degree.values() if value > 1)),
        "zero_out_degree_nodes": int(total_sats - len(out_degree)),
        "zero_in_degree_nodes": int(total_sats - len(in_degree)),
    }


def build_plane_y_hybrid_edge_table(
    *,
    base_spec: TopologySpec,
    patch_spec: TopologySpec,
    p: int,
    n: int,
    planes: Iterable[int],
    rows: Iterable[int],
    patch_mode: str,
) -> tuple[EdgeTable, list[EdgeRecord], list[EdgeRecord], dict[str, Any]]:
    plane_set = {int(x) % int(p) for x in planes}
    row_set = {int(y) % int(n) for y in rows}
    base_records = edge_records_from_table(base_spec.edge_table)
    patch_records_all = edge_records_from_table(patch_spec.edge_table)
    patch_records = [
        record
        for record in patch_records_all
        if record.option != -1
        and record.src_plane in plane_set
        and record.src_y in row_set
        and patch_option_allowed(record.option, patch_mode)
    ]
    patch_sources = {node_id(r.src_plane, r.src_y, n=n) for r in patch_records}
    patch_targets = {node_id(r.dst_plane, r.dst_y, n=n) for r in patch_records}

    kept: list[EdgeRecord] = []
    removed: list[EdgeRecord] = []
    for record in base_records:
        if record.option == -1:
            kept.append(record)
            continue
        src = node_id(record.src_plane, record.src_y, n=n)
        dst = node_id(record.dst_plane, record.dst_y, n=n)
        if src in patch_sources or dst in patch_targets:
            removed.append(record)
            continue
        kept.append(record)

    hybrid_records = kept + patch_records
    degree = degree_stats_from_records(hybrid_records, n=int(n), total_sats=int(p) * int(n))
    edge_table = make_edge_table_from_records(
        p=int(p),
        n=int(n),
        records=[record.tuple for record in hybrid_records],
    )
    return edge_table, patch_records, removed, degree


def build_candidates(
    *,
    config: Any,
    base: TopologySpec,
    patch: TopologySpec,
    row_seeds: Sequence[tuple[str, tuple[int, ...]]],
    extra_row_band_lengths: Sequence[int],
    plane_lengths: Sequence[int],
    patch_mode: str,
    max_candidates: int,
) -> list[Candidate]:
    p = int(config.P)
    n = int(config.N)
    row_seed_map: dict[tuple[int, ...], str] = {tuple(): "pure56_rows", tuple(range(n)): "pure40_rows"}
    for source, mask in row_seeds:
        row_seed_map.setdefault(tuple(sorted(set(int(x) % n for x in mask))), str(source))
    for band_len in extra_row_band_lengths:
        for start in range(n):
            mask = cyclic_values(start, int(band_len), n)
            row_seed_map.setdefault(mask, f"band{int(band_len):02d}")

    candidates: list[Candidate] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    for spec, label in ((base, "pure_000056"), (patch, "pure_000040")):
        key = edge_key_set(spec)
        seen.add(key)
        candidates.append(
            Candidate(
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
                plane_start=None,
                plane_len=p,
                row_mask=tuple(range(n)) if int(spec.motif_id or -1) == 40 else tuple(),
                row_source="pure",
                patch_mode="pure",
                added_edges=0,
                removed_edges=0,
                degree={},
            )
        )

    for plane_len in plane_lengths:
        plane_len = min(int(plane_len), p)
        plane_starts = (0,) if plane_len >= p else tuple(range(p))
        for plane_start in plane_starts:
            planes = cyclic_values(plane_start, plane_len, p)
            for row_mask, row_source in row_seed_map.items():
                if not row_mask:
                    continue
                edge_table, added, removed, degree = build_plane_y_hybrid_edge_table(
                    base_spec=base,
                    patch_spec=patch,
                    p=p,
                    n=n,
                    planes=planes,
                    rows=row_mask,
                    patch_mode=str(patch_mode),
                )
                if int(degree["max_out_degree"]) > 1 or int(degree["max_in_degree"]) > 1:
                    continue
                topology = TopologySpec(
                    name="tmp",
                    edge_table=edge_table,
                    library="plane_y_window_000056_000040",
                    motif_id=None,
                    motif=f"base=000056 patch=000040 mode={patch_mode} planes={plane_start}+{plane_len} rows={mask_token(row_mask)}",
                    baseline=False,
                    meta={
                        "kind": "plane_y_window",
                        "plane_start": int(plane_start),
                        "plane_len": int(plane_len),
                        "row_mask": [int(x) for x in row_mask],
                        "row_source": str(row_source),
                        "patch_mode": str(patch_mode),
                    },
                )
                key = edge_key_set(topology)
                if key in seen:
                    continue
                seen.add(key)
                name = (
                    f"plane_y_p{int(plane_start):02d}_l{int(plane_len):02d}_"
                    f"{safe_token(row_source)}_{safe_token(mask_token(row_mask))}"
                )
                candidates.append(
                    Candidate(
                        name=name,
                        topology=TopologySpec(
                            name=name,
                            edge_table=edge_table,
                            library=topology.library,
                            motif_id=None,
                            motif=topology.motif,
                            source_w=None,
                            source_h=None,
                            edge_count_local=None,
                            support=None,
                            baseline=False,
                            meta=topology.meta,
                        ),
                        plane_start=int(plane_start),
                        plane_len=int(plane_len),
                        row_mask=tuple(row_mask),
                        row_source=str(row_source),
                        patch_mode=str(patch_mode),
                        added_edges=int(len(added)),
                        removed_edges=int(len(removed)),
                        degree=degree,
                    )
                )
                if int(max_candidates) > 0 and len(candidates) >= int(max_candidates):
                    return candidates
    return candidates


def evaluate_candidate(
    *,
    candidate: Candidate,
    contexts: Sequence[Any],
    steps: Sequence[int],
    config: Any,
    delay_store: FullLinkDelayStore,
    forced_option: int,
    wrap_planes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hops = np.full(len(steps), np.nan, dtype=np.float64)
    delay = np.full(len(steps), np.nan, dtype=np.float64)
    for idx, step in enumerate(steps):
        constrained, _forced, _dropped = _apply_constraint_context_fast(
            base_edge_table=candidate.topology.edge_table,
            context=contexts[idx],
            total_nodes=int(config.total_sats),
            p=int(config.P),
            n=int(config.N),
            forced_option=int(forced_option),
            wrap_planes=bool(wrap_planes),
        )
        adjacency = _build_adjacency(constrained.src, constrained.dst, int(config.total_sats))
        _rh, mean_hops, _min_h, _max_h = _summarize_hops(
            adjacency=adjacency,
            sources=contexts[idx].sources,
            targets=contexts[idx].targets,
        )
        _rd, mean_delay, _min_d, _max_d = _summarize_delay(
            edge_table=constrained,
            config=config,
            delay_store=delay_store,
            position_store=None,
            delay_row=delay_store.row_for_time_step(int(step)),
            position_row=None,
            sources=contexts[idx].sources,
            targets=contexts[idx].targets,
            engine="auto",
        )
        hops[idx] = float(mean_hops)
        delay[idx] = float(mean_delay)
    return hops, delay


def write_candidate_summary(
    path: Path,
    *,
    candidates: Sequence[Candidate],
    hops: np.ndarray,
    delay: np.ndarray,
    baseline_hop_env: np.ndarray,
    baseline_delay_env: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "candidate_idx",
            "name",
            "plane_start",
            "plane_len",
            "row_source",
            "row_mask",
            "row_count",
            "added_edges",
            "removed_edges",
            "mean_hops",
            "mean_delay_ms",
            "better_hop_steps_vs_rowmask",
            "better_delay_steps_vs_rowmask",
            "mean_hop_gap_vs_rowmask",
            "mean_delay_gap_ms_vs_rowmask",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, candidate in enumerate(candidates):
            writer.writerow(
                {
                    "candidate_idx": int(idx),
                    "name": candidate.name,
                    "plane_start": "" if candidate.plane_start is None else int(candidate.plane_start),
                    "plane_len": int(candidate.plane_len),
                    "row_source": candidate.row_source,
                    "row_mask": mask_token(candidate.row_mask),
                    "row_count": int(len(candidate.row_mask)),
                    "added_edges": int(candidate.added_edges),
                    "removed_edges": int(candidate.removed_edges),
                    "mean_hops": float(np.nanmean(hops[:, idx])),
                    "mean_delay_ms": float(np.nanmean(delay[:, idx])),
                    "better_hop_steps_vs_rowmask": int(np.count_nonzero(hops[:, idx] < baseline_hop_env - 1e-9)),
                    "better_delay_steps_vs_rowmask": int(np.count_nonzero(delay[:, idx] < baseline_delay_env - 1e-9)),
                    "mean_hop_gap_vs_rowmask": float(np.nanmean(hops[:, idx] - baseline_hop_env)),
                    "mean_delay_gap_ms_vs_rowmask": float(np.nanmean(delay[:, idx] - baseline_delay_env)),
                }
            )


def write_wide_csv(path: Path, *, steps: Sequence[int], names: Sequence[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *names])
        writer.writeheader()
        for row, step in enumerate(steps):
            item = {"step": int(step)}
            for col, name in enumerate(names):
                item[str(name)] = float(values[row, col])
            writer.writerow(item)


def plot_envelope(
    *,
    out_dir: Path,
    steps: np.ndarray,
    hops: np.ndarray,
    delay: np.ndarray,
    baseline_hop_env: np.ndarray,
    baseline_delay_env: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(steps, dtype=np.float64) / 3600.0
    local_hop_env = np.nanmin(hops, axis=1)
    local_delay_env = np.nanmin(delay, axis=1)
    for values, baseline, ylabel, filename in (
        (local_hop_env, baseline_hop_env, "mean shortest hops", "plane_y_window_hop_envelope.png"),
        (local_delay_env, baseline_delay_env, "mean shortest delay (ms)", "plane_y_window_delay_envelope.png"),
    ):
        fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=180)
        ax.plot(x, baseline, color="#111827", linewidth=2.2, linestyle="--", label="row-mask 120-action envelope")
        ax.plot(x, values, color="#2563eb", linewidth=2.2, label="plane-y local window envelope")
        ax.set_title("G60 China-Europe 000040/000056 local fusion hard-step diagnostic")
        ax.set_xlabel("time (hour)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.55)
        ax.legend(loc="best")
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
    pair_spec = region_pair_specs(raw, subset=[str(args.target_pair)])[0]

    seed_steps, row_seeds = read_seed_steps_and_masks(Path(args.row_oracle_csv), n=int(config.N))
    if args.steps:
        steps = [int(x) for x in args.steps]
    elif args.start is not None or args.end is not None:
        start = 0 if args.start is None else int(args.start)
        end = int(raw.get("time", {}).get("end", 86160)) if args.end is None else int(args.end)
        steps = np.arange(start, end + 1, int(args.stride), dtype=np.int64).astype(int).tolist()
    else:
        steps = seed_steps
    baseline_hop_env, baseline_delay_env, baseline_meta = read_baseline_envelope(Path(args.baseline_dir), steps=steps)

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
    candidates = build_candidates(
        config=config,
        base=base,
        patch=patch,
        row_seeds=row_seeds,
        extra_row_band_lengths=tuple(int(x) for x in args.extra_row_band_lengths),
        plane_lengths=tuple(int(x) for x in args.plane_lengths),
        patch_mode=str(args.patch_mode),
        max_candidates=int(args.max_candidates),
    )
    selected_names = selected_names_from_args(args)
    if selected_names is not None:
        by_name = {candidate.name: candidate for candidate in candidates}
        missing = sorted(name for name in selected_names if name not in by_name)
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(f"selected candidate names not found: {preview}")
        candidates = [by_name[name] for name in candidates[0:0]] + [by_name[name] for name in sorted(selected_names)]
        print(f"[plane-y] selected candidate names={len(candidates)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps,
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
        steps=steps,
        pair_specs=[pair_spec],
        forced_option=int(forced_option),
        wrap_planes=bool(wrap_planes),
    )[str(pair_spec.key)]
    delay_store = FullLinkDelayStore(path_from(paths, "delay_store_dir"))

    hops = np.full((len(steps), len(candidates)), np.nan, dtype=np.float64)
    delay = np.full((len(steps), len(candidates)), np.nan, dtype=np.float64)
    started = time.perf_counter()
    for col, candidate in enumerate(candidates):
        hops[:, col], delay[:, col] = evaluate_candidate(
            candidate=candidate,
            contexts=contexts,
            steps=steps,
            config=config,
            delay_store=delay_store,
            forced_option=int(forced_option),
            wrap_planes=bool(wrap_planes),
        )
        if (col + 1) % int(args.progress_every) == 0 or (col + 1) == len(candidates):
            print(
                f"[plane-y] {col + 1}/{len(candidates)} elapsed={time.perf_counter() - started:.1f}s "
                f"candidate={candidate.name}",
                flush=True,
            )

    names = [candidate.name for candidate in candidates]
    write_wide_csv(out_dir / "plane_y_window_mean_hops_wide.csv", steps=steps, names=names, values=hops)
    write_wide_csv(out_dir / "plane_y_window_mean_delay_ms_wide.csv", steps=steps, names=names, values=delay)
    write_candidate_summary(
        out_dir / "plane_y_window_candidate_summary.csv",
        candidates=candidates,
        hops=hops,
        delay=delay,
        baseline_hop_env=baseline_hop_env,
        baseline_delay_env=baseline_delay_env,
    )
    local_hop_env = np.nanmin(hops, axis=1)
    local_delay_env = np.nanmin(delay, axis=1)
    best_hop_idx = np.nanargmin(hops, axis=1)
    best_delay_idx = np.nanargmin(delay, axis=1)
    meta = {
        "config": str(Path(args.config)),
        "out_dir": str(out_dir),
        "steps": [int(x) for x in steps],
        "candidate_count": int(len(candidates)),
        "selected_names": [] if selected_names is None else sorted(selected_names),
        "row_seed_count": int(len(row_seeds)),
        "plane_lengths": [int(x) for x in args.plane_lengths],
        "extra_row_band_lengths": [int(x) for x in args.extra_row_band_lengths],
        "patch_mode": str(args.patch_mode),
        **baseline_meta,
        "mean_plane_y_hop_envelope": float(np.nanmean(local_hop_env)),
        "mean_plane_y_delay_envelope_ms": float(np.nanmean(local_delay_env)),
        "mean_gap_plane_y_to_rowmask_hop": float(np.nanmean(local_hop_env - baseline_hop_env)),
        "mean_gap_plane_y_to_rowmask_delay_ms": float(np.nanmean(local_delay_env - baseline_delay_env)),
        "plane_y_better_hop_steps": int(np.count_nonzero(local_hop_env < baseline_hop_env - 1e-9)),
        "plane_y_better_delay_steps": int(np.count_nonzero(local_delay_env < baseline_delay_env - 1e-9)),
        "best_hop_candidates": [names[int(idx)] for idx in best_hop_idx],
        "best_delay_candidates": [names[int(idx)] for idx in best_delay_idx],
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (out_dir / "plane_y_window_search_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_envelope(
        out_dir=out_dir,
        steps=np.asarray(steps, dtype=np.int64),
        hops=hops,
        delay=delay,
        baseline_hop_env=baseline_hop_env,
        baseline_delay_env=baseline_delay_env,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
