from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent

DEFAULT_MOTIF_CSV = PROJECT_ROOT / "data" / "linshi" / "motif_w5_h4_exact_box" / "exact_box_w5_h4_primitive.csv"
DEFAULT_DELAY_STORE = PROJECT_ROOT / "data" / "linshi" / "G60_full_options_plus_intra_t0_86164_stride1"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_w5h4_motif_shortest_delay_batches"
DEFAULT_RUNNER = THIS_DIR / "build_g60_motif_gridplus_shortest_delay_timeseries_parallel.py"
DEFAULT_PYTHON = Path(r"C:\ProgramData\miniconda3\envs\paper11\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weighted shortest-delay batches for exact-box motif CSV rows.")
    parser.add_argument("--motif-csv", type=Path, default=DEFAULT_MOTIF_CSV)
    parser.add_argument(
        "--selected-motif-csv",
        type=Path,
        default=None,
        help="Optional CSV containing the exact motif rows to run. When set, motif-id range is ignored.",
    )
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--motif-id-start", type=int, default=None)
    parser.add_argument("--motif-id-end", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=99)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=1)
    parser.add_argument("--sample-pairs-per-step", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def motif_to_support(motif_text: str) -> tuple[int, int, list[tuple[int, int, str]]]:
    columns = [part.strip() for part in str(motif_text).split("|")]
    if not columns or not columns[0]:
        raise ValueError(f"empty motif text: {motif_text!r}")
    h = len(columns[0])
    if any(len(col) != h for col in columns):
        raise ValueError(f"motif columns have inconsistent heights: {motif_text!r}")
    support: list[tuple[int, int, str]] = []
    for x, col in enumerate(columns):
        for y, symbol in enumerate(col):
            if symbol == "-":
                continue
            if symbol not in {"A", "B", "C", "D"}:
                raise ValueError(f"unsupported symbol {symbol!r} in motif {motif_text!r}")
            support.append((x, y, symbol))
    return len(columns) + 1, h, support


def write_motif_yaml(path: Path, *, motif_id: int, motif_text: str) -> None:
    w, h, support = motif_to_support(motif_text)
    lines = [
        f"name: exact_box_w{w}_h{h}_motif_{int(motif_id)}",
        "",
        "grid:",
        "  p: 18",
        "  n: 36",
        "",
        "motif:",
        f"  name: exact_box_w{w}_h{h}_motif_{int(motif_id)}",
        f"  w: {w}",
        f"  h: {h}",
        "  offsets:",
        "    A: [1, 0]",
        "    B: [1, -1]",
        "    C: [1, 1]",
        "    D: [2, 0]",
        "  support:",
    ]
    for x, y, symbol in support:
        lines.append(f"    - [{x}, {y}, {symbol}]")
    lines.extend(
        [
            "",
            "tiling:",
            "  horizontal_step: null",
            "  allow_vertical_overlap: true",
            "  allow_clipped_right: true",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_motif_rows(path: Path, motif_id_start: int, motif_id_end: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            motif_id = int(row["motif_id"])
            if motif_id < int(motif_id_start):
                continue
            if motif_id > int(motif_id_end):
                break
            rows.append(row)
    return rows


def read_selected_motif_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize_output(output_dir: Path) -> dict[str, float | int | str]:
    topology_dir = output_dir / "support_motif_DAD_Cxx"
    values = np.load(topology_dir / "mean_shortest_delay_ms.npy")
    finite = values[np.isfinite(values)]
    return {
        "num_steps": int(values.size),
        "mean_delay_ms": float(np.mean(finite)),
        "min_delay_ms": float(np.min(finite)),
        "max_delay_ms": float(np.max(finite)),
        "output_dir": str(output_dir),
    }


def main() -> int:
    args = parse_args()
    if args.selected_motif_csv is not None:
        rows = read_selected_motif_rows(Path(args.selected_motif_csv))
        range_label = f"selected_{len(rows)}"
    else:
        if args.motif_id_start is None or args.motif_id_end is None:
            raise ValueError("Pass --selected-motif-csv, or pass both --motif-id-start and --motif-id-end.")
        if int(args.motif_id_end) < int(args.motif_id_start):
            raise ValueError("--motif-id-end must be >= --motif-id-start")
        rows = read_motif_rows(Path(args.motif_csv), int(args.motif_id_start), int(args.motif_id_end))
        range_label = f"{int(args.motif_id_start)}_{int(args.motif_id_end)}"
    if not rows:
        raise ValueError("No motif rows selected.")

    out_dir = Path(args.out_dir)
    config_dir = out_dir / "configs"
    result_root = out_dir / f"motifs_{range_label}_t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "batch_summary.csv"

    summary_rows: list[dict] = []
    for row in rows:
        motif_id = int(row["motif_id"])
        motif_text = str(row["motif"])
        motif_out = result_root / f"motif_{motif_id:06d}"
        config_path = config_dir / f"motif_{motif_id:06d}.yaml"
        write_motif_yaml(config_path, motif_id=motif_id, motif_text=motif_text)

        if not args.force and (motif_out / "support_motif_DAD_Cxx" / "mean_shortest_delay_ms.npy").exists():
            print(f"[w5h4-batch] reuse motif_id={motif_id} out={motif_out}", flush=True)
        else:
            command = [
                str(Path(args.python)),
                str(Path(args.runner)),
                "--start",
                str(int(args.start)),
                "--end",
                str(int(args.end)),
                "--stride",
                str(int(args.stride)),
                "--topologies",
                "support_motif_DAD_Cxx",
                "--motif-config",
                str(config_path),
                "--delay-store-dir",
                str(Path(args.delay_store_dir)),
                "--out-dir",
                str(motif_out),
                "--max-workers",
                str(int(args.max_workers)),
                "--chunk-size",
                str(int(args.chunk_size)),
                "--sample-steps",
                str(int(args.sample_steps)),
                "--sample-pairs-per-step",
                str(int(args.sample_pairs_per_step)),
            ]
            print(f"[w5h4-batch] run motif_id={motif_id} motif={motif_text}", flush=True)
            subprocess.run(command, check=True)

        summary = summarize_output(motif_out)
        summary_rows.append(
            {
                "motif_id": motif_id,
                "motif": motif_text,
                "edges": row.get("edges", ""),
                **summary,
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "motif_id",
            "motif",
            "edges",
            "num_steps",
            "mean_delay_ms",
            "min_delay_ms",
            "max_delay_ms",
            "output_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "motif_csv": str(Path(args.motif_csv)),
        "selected_motif_csv": None if args.selected_motif_csv is None else str(Path(args.selected_motif_csv)),
        "delay_store_dir": str(Path(args.delay_store_dir)),
        "motif_id_start": int(args.motif_id_start),
        "motif_id_end": int(args.motif_id_end),
        "selected_motif_count": int(len(rows)),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "runner": str(Path(args.runner)),
        "summary_path": str(summary_path),
    }
    (result_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[w5h4-batch] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
