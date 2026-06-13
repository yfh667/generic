from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.plus_intra_delay_store import (
    build_plus_intra_delay_store,
    default_plus_intra_store_dir,
)
from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval


DEFAULT_INTER_STORE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "full_option_edge_delay"
    / "G60_full_options_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "_position_cache"
    / "cache_0_86164_1s"
)
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "full_options_plus_intra_delay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a G60 full-option plus intra-link delay store.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--inter-store-dir", type=Path, default=DEFAULT_INTER_STORE)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunk-steps", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")

    inter_store = open_delay_store_for_interval(
        int(args.start),
        int(args.end),
        stride=int(args.stride),
        store_dir=args.inter_store_dir,
    )
    position_store = open_position_cache_for_interval(
        int(args.start),
        int(args.end),
        stride=int(args.stride),
        full_cache_dir=args.position_cache_dir,
    )
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_plus_intra_store_dir(
        args.output_base,
        constellation_name=G60_CONFIG.name,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    build_plus_intra_delay_store(
        config=G60_CONFIG,
        inter_store=inter_store,
        position_store=position_store,
        out_dir=out_dir,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
        chunk_steps=int(args.chunk_steps),
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
