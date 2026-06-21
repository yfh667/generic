from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.schedule_segments import write_schedule_segment_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write compressed segment/switch reports for selector schedules.")
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--schedule", action="append", default=None, help="Schedule name. Repeat for several schedules.")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = write_schedule_segment_report(
        selector_dir=args.selector_dir,
        schedules=args.schedule or [],
        out_dir=args.out_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
