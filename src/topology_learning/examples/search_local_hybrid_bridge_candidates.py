from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import write_edges_csv
from src.topology_workflow.module.batch_shortest_hops import TopologySpec
from src.topology_workflow.module.edge_tables import build_motif_text_edge_table
from src.topology_workflow.module.hybrid_edges import (
    build_y_band_hybrid_edge_table,
    cyclic_rows,
    edge_keys_from_table,
)


DEFAULT_SELECTOR_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\g60_w4h3_full_link_gap_three_pairs_hop_delay_budget_frontier_curve_topkscore120_2pct_fine1721_1739"
)
DEFAULT_MOTIF_CSV = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w4_h3"
    r"\shortest_delay_t0_86160_stride60\topology_library.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\local_hybrid_bridge_candidates"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate y-band hybrid topology candidates and measure setup cost "
            "for a local bridge previous -> hybrid -> next."
        )
    )
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--motif-csv", type=Path, default=DEFAULT_MOTIF_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name-prefix", default="w4h3")
    parser.add_argument("--previous-action", required=True, help="Action index or topology name before the local window.")
    parser.add_argument("--next-action", required=True, help="Action index or topology name after the local window.")
    parser.add_argument("--base-actions", nargs="+", required=True, help="Base action indices or topology names.")
    parser.add_argument("--patch-actions", nargs="+", required=True, help="Patch action indices or topology names.")
    parser.add_argument("--patch-options", nargs="*", type=int, default=None, help="Only copy these inter options from patch.")
    parser.add_argument("--min-width", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument("--row-step", type=int, default=1)
    parser.add_argument("--max-total-setup", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--write-top-edges", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def _load_selector_names(selector_dir: Path) -> np.ndarray:
    arrays_path = Path(selector_dir) / "selector_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as data:
        return np.asarray(data["topology_names"])


def _resolve_action(token: str, names: np.ndarray) -> str:
    raw = str(token).strip()
    if raw.isdigit():
        idx = int(raw)
        if idx < 0 or idx >= names.shape[0]:
            raise IndexError(f"action index {idx} outside topology_names length {names.shape[0]}")
        return str(names[idx])
    if raw.startswith("motif_") and raw.removeprefix("motif_").isdigit():
        return f"w4h3_{raw}"
    if raw.startswith("w4h3_motif_"):
        return raw
    raise ValueError(f"unsupported action token: {token!r}")


def _edge_overlap_ratio_keys(keys_a: set[tuple[int, int]], keys_b: set[tuple[int, int]]) -> float:
    if not keys_b:
        return 0.0
    return float(len(keys_a & keys_b) / len(keys_b))


def _added_count_keys(previous: set[tuple[int, int]], next_keys: set[tuple[int, int]]) -> int:
    return int(sum(1 for key in next_keys if key not in previous))


def _iter_bands(*, n: int, min_width: int, max_width: int | None, row_step: int) -> Iterable[tuple[int, int, tuple[int, ...]]]:
    n = int(n)
    max_width = n if max_width is None else int(max_width)
    if int(min_width) <= 0 or max_width <= 0 or int(row_step) <= 0:
        raise ValueError("min_width, max_width, and row_step must be positive")
    for width in range(int(min_width), min(max_width, n) + 1):
        for start in range(0, n, int(row_step)):
            end = start + width - 1
            yield start, end % n, cyclic_rows(start, end, n=n)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_needed_motif_specs(
    path: Path,
    *,
    needed_names: Iterable[str],
    name_prefix: str,
) -> dict[str, TopologySpec]:
    needed = {str(name) for name in needed_names}
    out: dict[str, TopologySpec] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row_idx, row in enumerate(csv.DictReader(f), start=1):
            name = str(row.get("name", ""))
            if name not in needed:
                continue
            motif_text = str(row.get("motif", ""))
            if not motif_text or "full_option" in motif_text or "option0" in motif_text:
                raise ValueError(f"{name!r} is not a motif-text topology: motif={motif_text!r}")
            motif_id_raw = row.get("motif_id") or row_idx
            motif_id = int(motif_id_raw)
            out[name] = TopologySpec(
                name=name,
                edge_table=build_motif_text_edge_table(
                    motif_text=motif_text,
                    config=G60_CONFIG,
                    add_intra_ring=True,
                    wrap_planes=False,
                ),
                library=str(name_prefix),
                motif_id=motif_id,
                motif=motif_text,
                source_w=int(row["source_w"]) if row.get("source_w") else None,
                source_h=int(row["source_h"]) if row.get("source_h") else None,
                edge_count_local=int(row["edge_count"]) if row.get("edge_count") else None,
                support=str(row.get("support", "")),
                baseline=False,
                meta={k: v for k, v in row.items() if k != "motif"},
            )
    return out


def main() -> int:
    args = parse_args()
    names = _load_selector_names(args.selector_dir)
    previous_name = _resolve_action(args.previous_action, names)
    next_name = _resolve_action(args.next_action, names)
    base_names = [_resolve_action(token, names) for token in args.base_actions]
    patch_names = [_resolve_action(token, names) for token in args.patch_actions]

    spec_by_name = _load_needed_motif_specs(
        args.motif_csv,
        needed_names=[previous_name, next_name, *base_names, *patch_names],
        name_prefix=str(args.name_prefix),
    )
    missing = [name for name in [previous_name, next_name, *base_names, *patch_names] if name not in spec_by_name]
    if missing:
        raise KeyError(f"topology names not found in motif csv: {missing}")

    previous_spec = spec_by_name[previous_name]
    next_spec = spec_by_name[next_name]
    keys_by_name = {
        name: edge_keys_from_table(spec.edge_table)
        for name, spec in spec_by_name.items()
    }
    previous_keys = keys_by_name[previous_name]
    next_keys = keys_by_name[next_name]
    previous_to_next = _added_count_keys(previous_keys, next_keys)
    rows: list[dict] = []
    bands = list(
        _iter_bands(
            n=int(G60_CONFIG.N),
            min_width=int(args.min_width),
            max_width=args.max_width,
            row_step=int(args.row_step),
        )
    )
    candidate_idx = 0

    for base_name in base_names:
        base_spec = spec_by_name[base_name]
        base_keys = keys_by_name[base_name]
        for patch_name in patch_names:
            patch_spec = spec_by_name[patch_name]
            patch_keys = keys_by_name[patch_name]
            if base_name == patch_name:
                continue
            for band_start, band_end, band_rows in bands:
                candidate_idx += 1
                result = build_y_band_hybrid_edge_table(
                    base_edge_table=base_spec.edge_table,
                    patch_edge_table=patch_spec.edge_table,
                    p=int(G60_CONFIG.P),
                    n=int(G60_CONFIG.N),
                    rows=band_rows,
                    patch_options=args.patch_options,
                    fail_on_degree_violation=True,
                )
                result_keys = edge_keys_from_table(result.edge_table)
                entry_setup = _added_count_keys(previous_keys, result_keys)
                exit_setup = _added_count_keys(result_keys, next_keys)
                total_setup = int(entry_setup + exit_setup)
                rows.append(
                    {
                        "candidate": f"{base_name}__patch_{patch_name}__y{band_start}_{band_end}",
                        "base": base_name,
                        "patch": patch_name,
                        "band_start": int(band_start),
                        "band_end": int(band_end),
                        "band_width": int(len(band_rows)),
                        "band_rows": " ".join(str(x) for x in band_rows),
                        "entry_setup": int(entry_setup),
                        "exit_setup": int(exit_setup),
                        "total_bridge_setup": int(total_setup),
                        "previous_to_next_setup": int(previous_to_next),
                        "added_patch_records": int(len(result.added_patch_records)),
                        "removed_base_records": int(len(result.removed_base_records)),
                        "inter_edges": int(result.degree_stats.inter_edges),
                        "intra_edges": int(result.degree_stats.intra_edges),
                        "max_out_degree": int(result.degree_stats.max_out_degree),
                        "max_in_degree": int(result.degree_stats.max_in_degree),
                        "overlap_with_base": _edge_overlap_ratio_keys(result_keys, base_keys),
                        "overlap_with_patch": _edge_overlap_ratio_keys(result_keys, patch_keys),
                        "overlap_with_previous": _edge_overlap_ratio_keys(result_keys, previous_keys),
                        "overlap_with_next": _edge_overlap_ratio_keys(result_keys, next_keys),
                    }
                )
                if int(args.progress_every) > 0 and candidate_idx % int(args.progress_every) == 0:
                    print(f"[hybrid-search] candidates={candidate_idx}", flush=True)

    rows.sort(key=lambda item: (int(item["total_bridge_setup"]), -float(item["overlap_with_patch"]), int(item["band_width"])))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(out_dir / "candidate_setup_costs.csv", rows)
    feasible = [
        row
        for row in rows
        if args.max_total_setup is None or int(row["total_bridge_setup"]) <= int(args.max_total_setup)
    ]
    _write_rows(out_dir / "candidate_setup_costs_feasible.csv", feasible)

    top = feasible[: int(args.top_k)] if args.max_total_setup is not None else rows[: int(args.top_k)]
    (out_dir / "top_candidates.json").write_text(json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8")

    for rank, row in enumerate(top[: int(args.write_top_edges)], start=1):
        base_spec = spec_by_name[str(row["base"])]
        patch_spec = spec_by_name[str(row["patch"])]
        band_rows = [int(x) for x in str(row["band_rows"]).split()]
        result = build_y_band_hybrid_edge_table(
            base_edge_table=base_spec.edge_table,
            patch_edge_table=patch_spec.edge_table,
            p=int(G60_CONFIG.P),
            n=int(G60_CONFIG.N),
            rows=band_rows,
            patch_options=args.patch_options,
            fail_on_degree_violation=True,
        )
        safe_name = str(row["candidate"]).replace(":", "_")
        write_edges_csv(result.edge_table, out_dir / f"top_{rank:02d}_{safe_name}_edges.csv")

    meta = {
        "selector_dir": str(args.selector_dir),
        "motif_csv": str(args.motif_csv),
        "previous": previous_name,
        "next": next_name,
        "base_actions": base_names,
        "patch_actions": patch_names,
        "patch_options": args.patch_options,
        "min_width": int(args.min_width),
        "max_width": None if args.max_width is None else int(args.max_width),
        "row_step": int(args.row_step),
        "previous_to_next_setup": int(previous_to_next),
        "max_total_setup": None if args.max_total_setup is None else int(args.max_total_setup),
        "num_candidates": int(len(rows)),
        "num_feasible": int(len(feasible)),
        "top": top,
        "outputs": {
            "candidate_setup_costs": str(out_dir / "candidate_setup_costs.csv"),
            "candidate_setup_costs_feasible": str(out_dir / "candidate_setup_costs_feasible.csv"),
            "top_candidates": str(out_dir / "top_candidates.json"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**meta, "top": top[:5]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
