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


DEFAULT_PREDICTIONS = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_policy_all1437_pseudo"
    r"\row_mask_policy_predictions.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_row_mask_policy_all1437_pseudo"
    r"\evaluated_metrics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted 36-row 000040/000056 fusion masks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--mask-source", choices=["prob", "pred", "true"], default="prob")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--forced-option", type=int, default=None)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def _read_prediction_rows(path: Path, *, n: int, max_rows: int = 0) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if int(max_rows) > 0:
        rows = rows[: int(max_rows)]
    for row in rows:
        row["step"] = int(float(row["step"]))
    return rows


def _rows_from_prediction(row: dict[str, Any], *, n: int, mask_source: str, threshold: float) -> tuple[int, ...]:
    selected: list[int] = []
    for idx in range(int(n)):
        if mask_source == "true":
            value = int(float(row[f"true_y{idx:02d}"]))
        elif mask_source == "pred":
            value = int(float(row[f"pred_y{idx:02d}"]))
        else:
            value = 1 if float(row[f"prob_y{idx:02d}"]) >= float(threshold) else 0
        if value:
            selected.append(idx)
    return tuple(selected)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _plot_series(
    out_dir: Path,
    *,
    steps: np.ndarray,
    learned: np.ndarray,
    env: np.ndarray,
    pure_56: np.ndarray | None,
    pure_40: np.ndarray | None,
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional for batch metrics.
        print(f"[warn] matplotlib unavailable, skip plot {stem}: {exc}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(13, 4.8), dpi=160)
    ax.plot(steps, env, color="#222222", linewidth=1.4, label="lower envelope")
    if pure_56 is not None:
        ax.plot(steps, pure_56, color="#1f77b4", linewidth=0.9, alpha=0.75, label="motif000056")
    if pure_40 is not None:
        ax.plot(steps, pure_40, color="#ff7f0e", linewidth=0.9, alpha=0.75, label="motif000040")
    ax.plot(steps, learned, color="#d62728", linewidth=1.0, alpha=0.9, label="learned row-mask")
    ax.set_title(title)
    ax.set_xlabel("time step (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png")
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
    forced_option = int(args.forced_option) if args.forced_option is not None else int(
        raw.get("region_internal_constraint", {}).get("forced_option", 0)
    )
    pair_spec = region_pair_specs(raw, subset=[str(args.target_pair)])[0]

    pred_rows = _read_prediction_rows(Path(args.prediction_csv), n=int(config.N), max_rows=int(args.max_rows))
    steps = [int(row["step"]) for row in pred_rows]
    selected_masks = [
        _rows_from_prediction(row, n=int(config.N), mask_source=str(args.mask_source), threshold=float(args.threshold))
        for row in pred_rows
    ]

    steps_ref, names_ref, hop_ref, delay_ref = _read_reference(Path(args.reference_dir))
    step_to_ref = {int(step): idx for idx, step in enumerate(steps_ref.tolist())}
    missing = [step for step in steps if step not in step_to_ref]
    if missing:
        raise ValueError(f"{len(missing)} prediction steps not found in reference, first={missing[:5]}")
    ref_rows = [step_to_ref[int(step)] for step in steps]
    name_to_col = {name: idx for idx, name in enumerate(names_ref)}

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
        steps=steps,
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
        steps=steps,
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
    started = time.perf_counter()
    rows_out: list[dict[str, Any]] = []
    for idx, (step, mask) in enumerate(zip(steps, selected_masks)):
        context = contexts[idx]
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
        _rh, mean_hops, min_hops, max_hops = _summarize_hops(
            adjacency=adjacency,
            sources=context.sources,
            targets=context.targets,
        )
        _rd, mean_delay, min_delay, max_delay = _summarize_delay(
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
        ref_idx = ref_rows[idx]
        ref_hop_env = float(np.nanmin(hop_ref[ref_idx]))
        ref_delay_env = float(np.nanmin(delay_ref[ref_idx]))
        item = {
            "step": int(step),
            "mask_source": str(args.mask_source),
            "threshold": float(args.threshold),
            "mask": _mask_token(mask, int(config.N)),
            "row_count": int(len(mask)),
            "mean_hops": float(mean_hops),
            "mean_delay_ms": float(mean_delay),
            "min_hops": float(min_hops),
            "max_hops": float(max_hops),
            "min_delay_ms": float(min_delay),
            "max_delay_ms": float(max_delay),
            "forced_region_edges": int(forced_count),
            "dropped_edges": int(dropped_count),
            "reference_hop_envelope": ref_hop_env,
            "reference_delay_envelope_ms": ref_delay_env,
            "gap_to_hop_envelope": float(mean_hops) - ref_hop_env,
            "gap_to_delay_envelope_ms": float(mean_delay) - ref_delay_env,
        }
        for ref_name in ("pure_000056", "pure_000040"):
            if ref_name in name_to_col:
                col = name_to_col[ref_name]
                item[f"{ref_name}_hops"] = float(hop_ref[ref_idx, col])
                item[f"{ref_name}_delay_ms"] = float(delay_ref[ref_idx, col])
        rows_out.append(item)
        if (idx + 1) % 100 == 0 or idx + 1 == len(steps):
            print(
                f"[eval] {idx + 1}/{len(steps)} unique_masks={len(edge_cache)} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    csv_path = out_dir / f"row_mask_policy_metrics_{args.mask_source}_thr{float(args.threshold):.2f}.csv"
    _write_csv(csv_path, rows_out)
    mean_hops_arr = np.asarray([row["mean_hops"] for row in rows_out], dtype=np.float64)
    mean_delay_arr = np.asarray([row["mean_delay_ms"] for row in rows_out], dtype=np.float64)
    ref_hop_env_arr = np.asarray([row["reference_hop_envelope"] for row in rows_out], dtype=np.float64)
    ref_delay_env_arr = np.asarray([row["reference_delay_envelope_ms"] for row in rows_out], dtype=np.float64)
    summary = {
        "prediction_csv": str(Path(args.prediction_csv)),
        "reference_dir": str(Path(args.reference_dir)),
        "out_dir": str(out_dir),
        "metrics_csv": str(csv_path),
        "mask_source": str(args.mask_source),
        "threshold": float(args.threshold),
        "target_pair": str(args.target_pair),
        "rows": int(len(rows_out)),
        "unique_masks": int(len(edge_cache)),
        "mean_hops": float(np.nanmean(mean_hops_arr)),
        "mean_delay_ms": float(np.nanmean(mean_delay_arr)),
        "mean_reference_hop_envelope": float(np.nanmean(ref_hop_env_arr)),
        "mean_reference_delay_envelope_ms": float(np.nanmean(ref_delay_env_arr)),
        "mean_gap_to_hop_envelope": float(np.nanmean(mean_hops_arr - ref_hop_env_arr)),
        "mean_gap_to_delay_envelope_ms": float(np.nanmean(mean_delay_arr - ref_delay_env_arr)),
        "steps_on_hop_envelope": int(np.sum(np.abs(mean_hops_arr - ref_hop_env_arr) <= 1e-9)),
        "steps_on_delay_envelope": int(np.sum(np.abs(mean_delay_arr - ref_delay_env_arr) <= 1e-9)),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    summary_path = out_dir / f"row_mask_policy_metrics_{args.mask_source}_thr{float(args.threshold):.2f}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    step_arr = np.asarray(steps, dtype=np.int64)
    pure56_hops = hop_ref[ref_rows, name_to_col["pure_000056"]] if "pure_000056" in name_to_col else None
    pure40_hops = hop_ref[ref_rows, name_to_col["pure_000040"]] if "pure_000040" in name_to_col else None
    pure56_delay = delay_ref[ref_rows, name_to_col["pure_000056"]] if "pure_000056" in name_to_col else None
    pure40_delay = delay_ref[ref_rows, name_to_col["pure_000040"]] if "pure_000040" in name_to_col else None
    _plot_series(
        out_dir,
        steps=step_arr,
        learned=mean_hops_arr,
        env=ref_hop_env_arr,
        pure_56=pure56_hops,
        pure_40=pure40_hops,
        ylabel="mean shortest hops",
        title=f"{args.target_pair}: learned row-mask hops",
        stem=f"row_mask_policy_hops_{args.mask_source}_thr{float(args.threshold):.2f}",
    )
    _plot_series(
        out_dir,
        steps=step_arr,
        learned=mean_delay_arr,
        env=ref_delay_env_arr,
        pure_56=pure56_delay,
        pure_40=pure40_delay,
        ylabel="mean shortest delay (ms)",
        title=f"{args.target_pair}: learned row-mask delay",
        stem=f"row_mask_policy_delay_ms_{args.mask_source}_thr{float(args.threshold):.2f}",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
