# =====================================================================
# export_static_shortest_paths2.py
#
# Route-policy driven shortest-path export.
# One motif is one independent task, so motifs can be processed in parallel.
# =====================================================================

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


from src.paper3_route.parallel import run_motifs_parallel, print_motif_results
from src.paper3_route.route_policy import load_route_policy, normalize_motif_names
from src.paper3_route.static_paths import export_shortest_paths_for_motif
# 新增 import
import time
from datetime import datetime





DATA_ROOT_DEFAULT = Path(r"D:\paper3\data")
ROUTE_POLICY_DEFAULT = DATA_ROOT_DEFAULT / "route_policy" / "route1.json"
MOTIFS_DEFAULT = ["grid+"]

# 新增日志函数
_T0 = time.time()
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{time.time() - _T0:8.2f}s] {msg}", flush=True)

# def _worker(motif_name: str, *, data_root: str, route_policy_path: str, force: bool) -> dict:
#     return export_shortest_paths_for_motif(
#         motif_name,
#         data_root=data_root,
#         route_policy_path=route_policy_path,
#         force=force,
#     )
def _worker(
    motif_name: str,
    *,
    data_root: str,
    route_policy_path: str,
    force: bool,
    verbose: bool,
) -> dict:
    return export_shortest_paths_for_motif(
        motif_name,
        data_root=data_root,
        route_policy_path=route_policy_path,
        force=force,
        verbose=verbose,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export static shortest route CSVs for selected motifs."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT_DEFAULT,
        help="Data root, e.g. D:\\paper3\\data",
    )
    parser.add_argument(
        "--route-policy",
        type=Path,
        default=ROUTE_POLICY_DEFAULT,
        help="Route policy JSON, e.g. D:\\paper3\\data\\route_policy\\route1.json",
    )
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
        "--no-force",
        action="store_true",
        help="Do not force overwrite semantics. Pair CSVs with the same names are still overwritten by pandas writes.",
    )
    # parse_args 里新增
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable step timing logs inside each motif task.",
    )
    return parser.parse_args()

def main() -> None:
    t_main = time.time()
    args = parse_args()
    policy = load_route_policy(args.route_policy)
    motifs = normalize_motif_names(
        args.motifs,
        data_root=args.data_root,
        policy=policy,
        fallback=MOTIFS_DEFAULT,
    )
    workers = max(1, int(args.workers or policy.motif_workers))
    verbose = not args.quiet

    log(f"route_policy = {args.route_policy}")
    log(f"route_name   = {policy.route_name}")
    log(f"objective    = {policy.objective}")
    log(f"data_root    = {args.data_root}")
    log(f"motifs       = {motifs}")
    log(f"workers      = {workers}")
    log("Start motif tasks")

    results = run_motifs_parallel(
        motifs,
        _worker,
        max_workers=workers,
        worker_kwargs={
            "data_root": str(args.data_root),
            "route_policy_path": str(args.route_policy),
            "force": not args.no_force,
            "verbose": verbose,
        },
    )
    print_motif_results(results)
    log(f"All done. total_elapsed={time.time() - t_main:.3f}s")



if __name__ == "__main__":
    main()
