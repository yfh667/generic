from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


GENERIC_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "g60_full_link_delay_store.yaml"
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_delay.module.build_full_link_delay_store import default_delay_out_dir, load_yaml_dict, raw_config_to_dataclass
from src.link_delay.module.query import open_delay_store_for_interval


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query one edge delay using an example constellation config.")
    parser.add_argument("store_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--time-step", type=int, required=True)
    parser.add_argument("--src", type=int, required=True)
    parser.add_argument("--dst", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = raw_config_to_dataclass(load_yaml_dict(args.config))
    store_dir = args.store_dir or cfg.paths.out_dir or default_delay_out_dir(cfg)
    store = open_delay_store_for_interval(
        args.time_step,
        args.time_step,
        store_dir=store_dir,
        output_base=cfg.paths.delay_output_base,
        constellation_name=cfg.constellation.name,
    )
    record = store.edge_record(args.src, args.dst)
    payload = {
        "config": str(args.config),
        "store_dir": str(store.store_dir),
        "time_step": int(args.time_step),
        "src_node": int(args.src),
        "dst_node": int(args.dst),
        "edge_idx": int(record.edge_idx),
        "option": int(record.option),
        "delay_ms": store.delay_ms(args.time_step, args.src, args.dst),
        "distance_km": store.distance_km(args.time_step, args.src, args.dst),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
