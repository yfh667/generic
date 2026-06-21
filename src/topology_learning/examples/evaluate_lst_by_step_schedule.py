from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.lst_schedule_eval import run_lst_evaluation_from_by_step_csv


def parse_pair(text: str) -> tuple[int, int, str]:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) == 2:
        return int(parts[0]), int(parts[1]), f"group{parts[0]}_group{parts[1]}"
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), str(parts[2])
    raise argparse.ArgumentTypeError("pair must be source:target or source:target:key")


def load_hybrid_meta(path: Path | None, *, by_step_csv: Path) -> dict[str, dict]:
    meta_path = path
    if meta_path is None:
        candidate = by_step_csv.parent / "meta.json"
        meta_path = candidate if candidate.exists() else None
    if meta_path is None:
        return {}
    raw = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    topology = raw.get("replacement_topology")
    candidate_meta = raw.get("candidate_meta")
    if topology and isinstance(candidate_meta, dict):
        return {str(topology): candidate_meta}
    if isinstance(raw.get("hybrid_meta_by_name"), dict):
        return {str(k): dict(v) for k, v in raw["hybrid_meta_by_name"].items()}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a by-step topology schedule after generic LST active/building simulation. "
            "The by-step CSV must contain step and topology columns."
        )
    )
    parser.add_argument("--by-step-csv", type=Path, required=True)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--schedule-name", type=str, default=None)
    parser.add_argument("--setup-time", type=float, required=True)
    parser.add_argument("--delay-store-dir", type=Path, required=True)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--group-xml", type=Path, required=True)
    parser.add_argument("--group-cache-dir", type=Path, required=True)
    parser.add_argument("--pair", type=parse_pair, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hybrid-meta-json", type=Path, default=None)
    parser.add_argument(
        "--setup-mode",
        choices=("break_before_make", "make_before_break"),
        default="break_before_make",
    )
    parser.add_argument(
        "--setup-timing",
        choices=("reactive", "advance"),
        default="reactive",
        help="Default reactive means new edges are building for LST seconds after the command row.",
    )
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, target, key = args.pair
    schedule_name = str(args.schedule_name or f"{args.by_step_csv.parent.name}_{key}")
    hybrid_meta = load_hybrid_meta(args.hybrid_meta_json, by_step_csv=args.by_step_csv)
    run_lst_evaluation_from_by_step_csv(
        by_step_csv=args.by_step_csv,
        selector_dir=args.selector_dir,
        schedule_name=schedule_name,
        setup_time_seconds=float(args.setup_time),
        delay_store_dir=args.delay_store_dir,
        position_cache_dir=args.position_cache_dir,
        group_xml=args.group_xml,
        group_cache_dir=args.group_cache_dir,
        source_group_id=int(source),
        target_group_id=int(target),
        out_dir=args.out_dir,
        setup_mode=str(args.setup_mode),
        setup_timing=str(args.setup_timing),
        hybrid_meta_by_name=hybrid_meta,
        force_group_cache=bool(args.force_group_cache),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
