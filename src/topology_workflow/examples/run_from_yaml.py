from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_workflow.module.runner import run_workflow_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a topology workflow from one YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start", type=int, default=None, help="Override YAML time.start.")
    parser.add_argument("--end", type=int, default=None, help="Override YAML time.end.")
    parser.add_argument("--stride", type=int, default=None, help="Override YAML time.stride.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override YAML outputs.out_dir.")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_workflow_from_yaml(
        args.config,
        start=args.start,
        end=args.end,
        stride=args.stride,
        out_dir=args.out_dir,
        force_group_cache=bool(args.force_group_cache),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
