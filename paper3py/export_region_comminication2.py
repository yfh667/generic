# =====================================================================
# export_region_comminication2.py
#
# Read route/probability CSVs and generate region-pair communication
# probability statistics. The file name keeps the original spelling used
# by the project for compatibility.
# =====================================================================

from __future__ import annotations

import argparse
from pathlib import Path

from src.paper3_route.parallel import run_motifs_parallel, print_motif_results
from src.paper3_route.route_policy import load_route_policy, normalize_motif_names
from src.paper3_route.region_communication import export_region_communication_for_motif


DATA_ROOT_DEFAULT = Path(r"D:\paper3\data")
ROUTE_POLICY_DEFAULT = DATA_ROOT_DEFAULT / "route_policy" / "route1.json"
MOTIFS_DEFAULT = ["grid+"]


def _worker(
    motif_name: str,
    *,
    data_root: str,
    route_policy_path: str,
    prefer_probability_timeseries: bool,
) -> dict:
    return export_region_communication_for_motif(
        motif_name,
        data_root=data_root,
        route_policy_path=route_policy_path,
        prefer_probability_timeseries=prefer_probability_timeseries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate region-pair communication reliability CSVs."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--route-policy", type=Path, default=ROUTE_POLICY_DEFAULT)
    parser.add_argument(
        "--motifs",
        nargs="+",
        default=None,
        help="Motif names under data/topology_design. Use 'all' to scan every motif with config/motif.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of motif-level worker processes. Defaults to route JSON parallel.motif_workers.",
    )
    parser.add_argument(
        "--from-path-csv",
        action="store_true",
        help="Ignore probability/pair_timeseries and recompute reliability directly from path CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_route_policy(args.route_policy)
    motifs = normalize_motif_names(
        args.motifs,
        data_root=args.data_root,
        policy=policy,
        fallback=MOTIFS_DEFAULT,
    )
    workers = max(1, int(args.workers or policy.motif_workers))

    print(f"route_policy = {args.route_policy}")
    print(f"route_name   = {policy.route_name}")
    print(f"data_root    = {args.data_root}")
    print(f"motifs       = {motifs}")
    print(f"workers      = {workers}")

    results = run_motifs_parallel(
        motifs,
        _worker,
        max_workers=workers,
        worker_kwargs={
            "data_root": str(args.data_root),
            "route_policy_path": str(args.route_policy),
            "prefer_probability_timeseries": not args.from_path_csv,
        },
    )
    print_motif_results(results)


if __name__ == "__main__":
    main()
