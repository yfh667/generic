from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.full_link_gap_selector import run_full_link_gap_selector_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select dynamic topologies by closeness to full_link plus new-edge penalty.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_full_link_gap_selector_from_yaml(args.config, out_dir=args.out_dir, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

