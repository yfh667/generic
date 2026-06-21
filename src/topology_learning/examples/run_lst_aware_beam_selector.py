from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.lst_aware_beam_selector import parse_pair_spec, run_lst_aware_beam_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a beam-search selector that scores active-after-LST graphs."
    )
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True, help="source_group:target_group:metric_prefix[:label]")
    parser.add_argument("--setup-time", type=float, required=True)
    parser.add_argument("--delay-store-dir", type=Path, required=True)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--group-xml", type=Path, required=True)
    parser.add_argument("--group-cache-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--setup-penalty", type=float, default=0.0)
    parser.add_argument("--burst-penalty", type=float, default=0.0)
    parser.add_argument(
        "--critical-setup-penalty",
        type=float,
        default=0.0,
        help="Penalty for newly requested full-link critical edges, weighted by the supplied usage-share vector.",
    )
    parser.add_argument("--critical-edges-csv", type=Path, default=None)
    parser.add_argument("--critical-weights-npy", type=Path, default=None)
    parser.add_argument("--critical-normalize-weights", default="max")
    parser.add_argument("--critical-weight-power", type=float, default=1.0)
    parser.add_argument("--critical-weight-scale", type=float, default=1.0)
    parser.add_argument("--critical-base-new-edge-cost", type=float, default=0.0)
    parser.add_argument("--critical-missing-edge-weight", type=float, default=0.0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--include-schedule",
        action="append",
        default=[],
        help="Existing schedule name whose row-wise actions should always be considered.",
    )
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_lst_aware_beam_selector(
        selector_dir=args.selector_dir,
        out_dir=args.out_dir,
        pair_specs=[parse_pair_spec(text) for text in args.pair],
        setup_time_seconds=float(args.setup_time),
        delay_store_dir=args.delay_store_dir,
        position_cache_dir=args.position_cache_dir,
        group_xml=args.group_xml,
        group_cache_dir=args.group_cache_dir,
        top_k=int(args.top_k),
        beam_width=int(args.beam_width),
        setup_penalty=float(args.setup_penalty),
        burst_penalty=float(args.burst_penalty),
        critical_setup_penalty=float(args.critical_setup_penalty),
        critical_edges_csv=args.critical_edges_csv,
        critical_weights_npy=args.critical_weights_npy,
        critical_normalize_weights=str(args.critical_normalize_weights),
        critical_weight_power=float(args.critical_weight_power),
        critical_weight_scale=float(args.critical_weight_scale),
        critical_base_new_edge_cost=float(args.critical_base_new_edge_cost),
        critical_missing_edge_weight=float(args.critical_missing_edge_weight),
        start_row=int(args.start_row),
        max_rows=args.max_rows,
        include_schedule_names=args.include_schedule,
        force_group_cache=bool(args.force_group_cache),
    )
    print(
        json.dumps(
            {
                "selector_dir": str(result.selector_dir),
                "schedule_name": result.schedule_name,
                "total_setup_commands": int(result.total_setup_commands),
                "max_setup_commands_per_step": int(result.max_setup_commands_per_step),
                "mean_stage_cost": float(result.mean_stage_cost),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
