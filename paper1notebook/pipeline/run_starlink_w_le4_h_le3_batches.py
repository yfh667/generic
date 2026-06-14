from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\Starlink_72_22_1_550")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Starlink w<=4 h<=3 paper1 experiments in resumable batches.")
    parser.add_argument(
        "--stage",
        choices=("unconstrained", "region-grid"),
        required=True,
        help="Which Starlink experiment stage to run.",
    )
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--total-motifs", type=int, default=808)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--stop-after-batches", type=int, default=0)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def stage_paths(stage: str) -> tuple[Path, Path, str]:
    if stage == "unconstrained":
        return (
            THIS_DIR / "run_paper1_motif_shortest_delay.py",
            THIS_DIR / "configs" / "starlink_72_22_w_le4_h_le3_shortest_delay.yaml",
            "unconstrained",
        )
    return (
        THIS_DIR / "run_paper1_region_internal_grid_metrics.py",
        THIS_DIR / "configs" / "starlink_72_22_w_le4_h_le3_region_internal_grid_metrics.yaml",
        "region_grid",
    )


def main() -> int:
    args = parse_args()
    script, config, log_prefix = stage_paths(str(args.stage))
    log_dir = Path(args.run_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    offsets = list(range(int(args.start_offset), int(args.total_motifs), int(args.batch_size)))
    if int(args.stop_after_batches) > 0:
        offsets = offsets[: int(args.stop_after_batches)]

    started = time.perf_counter()
    for batch_idx, offset in enumerate(offsets, start=1):
        limit = min(int(args.batch_size), int(args.total_motifs) - int(offset))
        end_idx = int(offset) + int(limit) - 1
        log_path = log_dir / f"{log_prefix}_batch_{offset:03d}_{end_idx:03d}_w{int(args.max_workers)}.log"
        command = [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--motif-offset",
            str(offset),
            "--limit-motifs",
            str(limit),
            "--max-workers",
            str(args.max_workers),
        ]
        print(
            f"[starlink-batches] start {args.stage} batch={batch_idx}/{len(offsets)} "
            f"offset={offset} limit={limit} log={log_path}",
            flush=True,
        )
        batch_started = time.perf_counter()
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(Path(r"E:\paper11")),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            last_heartbeat = batch_started
            while True:
                returncode = process.poll()
                now = time.perf_counter()
                if returncode is not None:
                    break
                if now - last_heartbeat >= 30.0:
                    last_heartbeat = now
                    try:
                        log_size = log_path.stat().st_size
                    except OSError:
                        log_size = -1
                    print(
                        f"[starlink-batches] heartbeat {args.stage} offset={offset} "
                        f"batch_elapsed={now - batch_started:.1f}s log_bytes={log_size}",
                        flush=True,
                    )
                time.sleep(2.0)
        print(
            f"[starlink-batches] done  {args.stage} offset={offset} "
            f"returncode={returncode} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if returncode != 0:
            print(f"[starlink-batches] failed log={log_path}", flush=True)
            return int(returncode)
    print(f"[starlink-batches] all requested batches completed elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
