from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w4_h3_shortest_delay.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper1 G60 w=4,h=3 motif shortest-delay experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit-motifs", type=int, default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--engine", choices=("auto", "scipy", "heapq"), default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--regenerate-library", action="store_true")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def add_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def main() -> int:
    args = parse_args()
    compute_script = THIS_DIR / "run_paper1_motif_shortest_delay.py"
    plot_script = THIS_DIR / "plot_paper1_motif_shortest_delay_paper_style.py"

    if not args.plot_only:
        compute_cmd = [sys.executable, str(compute_script), "--config", str(args.config)]
        add_optional(compute_cmd, "--start", args.start)
        add_optional(compute_cmd, "--end", args.end)
        add_optional(compute_cmd, "--stride", args.stride)
        add_optional(compute_cmd, "--out-dir", args.out_dir)
        add_optional(compute_cmd, "--limit-motifs", args.limit_motifs)
        add_optional(compute_cmd, "--engine", args.engine)
        add_optional(compute_cmd, "--max-workers", args.max_workers)
        if args.pairs:
            compute_cmd.append("--pairs")
            compute_cmd.extend(str(x) for x in args.pairs)
        if args.force:
            compute_cmd.append("--force")
        if args.regenerate_library:
            compute_cmd.append("--regenerate-library")
        subprocess.run(compute_cmd, check=True)

    if not args.compute_only:
        plot_cmd = [sys.executable, str(plot_script), "--config", str(args.config)]
        add_optional(plot_cmd, "--run-dir", args.out_dir)
        if args.pairs:
            plot_cmd.append("--pairs")
            plot_cmd.extend(str(x) for x in args.pairs)
        subprocess.run(plot_cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
