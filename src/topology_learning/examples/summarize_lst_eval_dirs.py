from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))


PAIR_PREFIXES = {
    "china_europe": "ce",
    "china_africa": "ca",
    "china_america": "cam",
}


def parse_eval(text: str) -> tuple[str, Path]:
    pair, sep, path = str(text).partition(":")
    if not sep or not pair or not path:
        raise argparse.ArgumentTypeError("--eval must look like pair_key:path")
    return pair, Path(path)


def read_selector_reference(summary_csv: Path, *, schedule: str | None = None) -> dict[str, tuple[float, float]]:
    with Path(summary_csv).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty selector summary: {summary_csv}")
    row = None
    if schedule is not None:
        for item in rows:
            if str(item.get("schedule", "")) == str(schedule):
                row = item
                break
    if row is None:
        row = rows[0]
    refs: dict[str, tuple[float, float]] = {}
    for pair, prefix in PAIR_PREFIXES.items():
        refs[pair] = (
            float(row[f"mean_{prefix}_delay_ms_full_link"]),
            float(row[f"mean_{prefix}_hops_full_link"]),
        )
    return refs


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert direct LST eval directories into a post-LST sweep summary CSV."
    )
    parser.add_argument("--selector-summary", type=Path, required=True)
    parser.add_argument("--reference-schedule", type=str, default=None)
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--eval", type=parse_eval, action="append", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refs = read_selector_reference(args.selector_summary, schedule=args.reference_schedule)
    rows: list[dict] = []
    for pair_key, eval_dir in args.eval:
        meta_path = Path(eval_dir) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        delay_ref, hops_ref = refs[str(pair_key)]
        delay_value = float(meta["mean_shortest_delay_ms"])
        hops_value = float(meta["mean_shortest_hops"])
        rows.append(
            {
                "pair_key": str(pair_key),
                "setup_time_seconds": float(meta["setup_time_seconds"]),
                "setup_mode": str(meta.get("setup_mode", "")),
                "setup_timing": str(meta.get("setup_timing", "")),
                "schedule": str(args.schedule),
                "eval_dir": str(eval_dir),
                "full_link_mean_delay_ms": delay_ref,
                "full_link_mean_hops": hops_ref,
                "post_lst_mean_shortest_delay_ms": delay_value,
                "post_lst_mean_shortest_hops": hops_value,
                "post_lst_delay_ms_gap": (delay_value - delay_ref) / delay_ref,
                "post_lst_hops_gap": (hops_value - hops_ref) / hops_ref,
                "num_switches": int(meta["num_switches"]),
                "num_union_edges": int(meta["num_union_edges"]),
                "mean_building_edges": float(meta["mean_building_edges"]),
                "max_building_edges": int(meta["max_building_edges"]),
                "building_edge_seconds": float(meta["building_edge_seconds"]),
                "total_setup_commands": int(meta["total_setup_commands"]),
                "max_setup_commands_per_step": int(meta["max_setup_commands_per_step"]),
                "mean_setup_commands_per_step": float(meta["mean_setup_commands_per_step"]),
            }
        )
    write_rows(args.out_csv, rows)
    print(json.dumps({"out_csv": str(args.out_csv), "rows": len(rows)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
