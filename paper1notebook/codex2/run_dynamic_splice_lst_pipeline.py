from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_m56_local_patch_hybrid_topology import DEFAULT_CONFIG, normalize_motif_name, path_from  # noqa: E402
from run_m56_m40_dynamic_splice_86100 import default_out_dir as dynamic_default_out_dir  # noqa: E402
from run_m56_m40_dynamic_splice_86100 import short_motif_token, short_pair_token  # noqa: E402
from src.topology_workflow.module.config import load_workflow_yaml, time_axis_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command pipeline: build/reuse dynamic motif splice, then apply a user-provided "
            "link setup time (LST) to produce active/building topology masks."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--link-setup-time", "--lst", type=float, required=True, help="LST in seconds.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86100)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--base-motif", default="56")
    parser.add_argument("--patch-motif", default="40")
    parser.add_argument("--target-pair", default="china_america")
    parser.add_argument("--preserve-pairs", nargs="+", default=("china_europe", "china_africa"))
    parser.add_argument("--metric-pairs", nargs="+", default=("china_europe", "china_america", "china_africa"))
    parser.add_argument("--usage-pairs", nargs="+", default=("china_america",))
    parser.add_argument("--patch-modes", nargs="+", default=("c", "cb", "all"), choices=("c", "cb", "all"))
    parser.add_argument("--band-lengths", nargs="+", type=int, default=(6, 12, 18))
    parser.add_argument("--selection-mode", choices=("instant", "transition-dp"), default="instant")
    parser.add_argument("--transition-setup-penalty-per-edge", type=float, default=0.0)
    parser.add_argument("--transition-max-new-edges", type=int, default=0)
    parser.add_argument("--dynamic-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="Final LST output directory.")
    parser.add_argument("--force-dynamic", action="store_true", help="Recompute the dynamic splice before LST.")
    parser.add_argument("--force-lst", action="store_true", help="Recompute the LST-constrained outputs.")
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Skip per-step edge-usage and shortest-path metrics; recommended for stride=1 full-day runs.",
    )
    parser.add_argument(
        "--expand-from-stride",
        type=int,
        default=60,
        help=(
            "For topology-only fine-grained runs, expand an existing coarser dynamic splice "
            "instead of recomputing every fine step. Set 0 to disable."
        ),
    )
    parser.add_argument("--open-gui", action="store_true", help="Open the final LST-constrained 2D viewer.")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def required_dynamic_files(dynamic_dir: Path) -> list[Path]:
    return [
        dynamic_dir / "dynamic_schedule.csv",
        dynamic_dir / "union_edges.csv",
        dynamic_dir / "edge_active_mask.npy",
        dynamic_dir / "edge_usage_values.npy",
        dynamic_dir / "dynamic_splice_meta.json",
    ]


def required_lst_files(out_dir: Path) -> list[Path]:
    return [
        out_dir / "steps.npy",
        out_dir / "union_edges.csv",
        out_dir / "target_edge_active_mask.npy",
        out_dir / "edge_active_mask.npy",
        out_dir / "edge_building_mask.npy",
        out_dir / "edge_usage_values.npy",
        out_dir / "lst_step_stats.csv",
        out_dir / "lst_metrics.csv",
        out_dir / "lst_meta.json",
    ]


def files_exist(paths: Iterable[Path]) -> bool:
    return all(Path(path).exists() for path in paths)


def run_command(cmd: list[str]) -> None:
    print("[lst-pipeline] run:", " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def list_args(flag: str, values: Iterable[Any]) -> list[str]:
    out = [str(flag)]
    out.extend(str(value) for value in values)
    return out


def infer_dynamic_dir(args: argparse.Namespace, *, workflow: dict[str, Any], stride: int) -> Path:
    if args.dynamic_dir is not None:
        return Path(args.dynamic_dir)
    paths_raw = workflow.get("paths", {})
    motif_raw = workflow.get("motif_library", {})
    name_prefix = str(motif_raw.get("name_prefix", "combined"))
    base_name = normalize_motif_name(str(args.base_motif), name_prefix=name_prefix)
    patch_name = normalize_motif_name(str(args.patch_motif), name_prefix=name_prefix)
    return dynamic_default_out_dir(
        run_out_dir=path_from(paths_raw, "out_dir"),
        base_name=base_name,
        patch_name=patch_name,
        start=int(args.start),
        end=int(args.end),
        stride=int(stride),
        target_pair=str(args.target_pair),
        band_lengths=args.band_lengths,
        patch_modes=args.patch_modes,
        selection_mode=str(args.selection_mode),
        transition_setup_penalty_per_edge=float(args.transition_setup_penalty_per_edge),
        transition_max_new_edges=int(args.transition_max_new_edges),
    )


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def expand_dynamic_time_axis(
    *,
    source_dir: Path,
    target_dir: Path,
    start: int,
    end: int,
    stride: int,
) -> None:
    fieldnames, source_rows = read_csv_rows(source_dir / "dynamic_schedule.csv")
    if not source_rows:
        raise ValueError(f"empty dynamic schedule: {source_dir / 'dynamic_schedule.csv'}")
    source_steps = np.asarray([int(row["step"]) for row in source_rows], dtype=np.int64)
    target_steps = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
    if target_steps.size == 0:
        raise ValueError("empty expanded target time axis")
    source_idx = np.searchsorted(source_steps, target_steps, side="right") - 1
    source_idx = np.clip(source_idx, 0, len(source_steps) - 1).astype(np.int64)

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "union_edges.csv", target_dir / "union_edges.csv")
    source_active = np.load(source_dir / "edge_active_mask.npy", mmap_mode="r")
    target_active = np.asarray(source_active[source_idx], dtype=bool)
    np.save(target_dir / "edge_active_mask.npy", target_active)
    np.save(target_dir / "edge_usage_values.npy", np.zeros((1, int(target_active.shape[1])), dtype=np.float32))

    expanded_fieldnames = list(fieldnames)
    if "expanded_from_step" not in expanded_fieldnames:
        expanded_fieldnames.append("expanded_from_step")
    expanded_rows: list[dict[str, Any]] = []
    for step, src_pos in zip(target_steps.tolist(), source_idx.tolist()):
        row = dict(source_rows[int(src_pos)])
        row["expanded_from_step"] = row.get("step", "")
        row["step"] = int(step)
        expanded_rows.append(row)
    write_csv_rows(target_dir / "dynamic_schedule.csv", fieldnames=expanded_fieldnames, rows=expanded_rows)

    source_meta = {}
    meta_path = source_dir / "dynamic_splice_meta.json"
    if meta_path.exists():
        source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = {
        **source_meta,
        "expanded_time_axis": True,
        "expanded_from_dir": str(source_dir),
        "source_steps": [int(source_steps[0]), int(source_steps[-1]), int(source_steps[1] - source_steps[0]) if len(source_steps) > 1 else None],
        "start": int(target_steps[0]),
        "end": int(target_steps[-1]),
        "stride": int(stride),
        "num_steps": int(target_steps.size),
        "skip_usage": True,
        "outputs": {
            "dynamic_schedule": str(target_dir / "dynamic_schedule.csv"),
            "union_edges": str(target_dir / "union_edges.csv"),
            "edge_active_mask": str(target_dir / "edge_active_mask.npy"),
            "edge_usage_values": str(target_dir / "edge_usage_values.npy"),
        },
    }
    (target_dir / "dynamic_splice_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[lst-pipeline] expanded dynamic time axis {source_dir.name} -> {target_dir.name} "
        f"steps={target_steps.size}",
        flush=True,
    )


def infer_lst_dir(args: argparse.Namespace, *, dynamic_dir: Path) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    return Path(dynamic_dir) / f"lst{float(args.link_setup_time):g}s_backward"


def write_pipeline_meta(
    *,
    path: Path,
    args: argparse.Namespace,
    dynamic_dir: Path,
    lst_dir: Path,
    stride: int,
) -> None:
    payload = {
        "pipeline": "dynamic_splice_then_backward_lst",
        "link_setup_time_seconds": float(args.link_setup_time),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(stride),
        "base_motif": str(args.base_motif),
        "patch_motif": str(args.patch_motif),
        "target_pair": str(args.target_pair),
        "preserve_pairs": list(args.preserve_pairs),
        "metric_pairs": list(args.metric_pairs),
        "usage_pairs": list(args.usage_pairs),
        "patch_modes": list(args.patch_modes),
        "band_lengths": [int(x) for x in args.band_lengths],
        "selection_mode": str(args.selection_mode),
        "transition_setup_penalty_per_edge": float(args.transition_setup_penalty_per_edge),
        "transition_max_new_edges": int(args.transition_max_new_edges),
        "topology_only": bool(args.topology_only),
        "dynamic_dir": str(dynamic_dir),
        "lst_dir": str(lst_dir),
        "outputs": {
            "dynamic_schedule": str(dynamic_dir / "dynamic_schedule.csv"),
            "dynamic_meta": str(dynamic_dir / "dynamic_splice_meta.json"),
            "lst_meta": str(lst_dir / "lst_meta.json"),
            "lst_step_stats": str(lst_dir / "lst_step_stats.csv"),
            "lst_metrics": str(lst_dir / "lst_metrics.csv"),
            "edge_active_mask": str(lst_dir / "edge_active_mask.npy"),
            "edge_building_mask": str(lst_dir / "edge_building_mask.npy"),
            "viewer_usage_values": str(lst_dir / "edge_usage_values.npy"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dynamic_splice_builder(args: argparse.Namespace, *, out_dir: Path, stride: int) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "run_m56_m40_dynamic_splice_86100.py"),
        "--config",
        str(args.config),
        "--base-motif",
        str(args.base_motif),
        "--patch-motif",
        str(args.patch_motif),
        "--start",
        str(int(args.start)),
        "--end",
        str(int(args.end)),
        "--stride",
        str(int(stride)),
        "--target-pair",
        str(args.target_pair),
        "--selection-mode",
        str(args.selection_mode),
        "--transition-setup-penalty-per-edge",
        str(float(args.transition_setup_penalty_per_edge)),
        "--transition-max-new-edges",
        str(int(args.transition_max_new_edges)),
        "--out-dir",
        str(out_dir),
        "--check-only",
    ]
    cmd.extend(list_args("--preserve-pairs", args.preserve_pairs))
    cmd.extend(list_args("--metric-pairs", args.metric_pairs))
    cmd.extend(list_args("--usage-pairs", args.usage_pairs))
    cmd.extend(list_args("--patch-modes", args.patch_modes))
    cmd.extend(list_args("--band-lengths", args.band_lengths))
    if bool(args.topology_only):
        cmd.append("--skip-usage")
    run_command(cmd)


def main() -> int:
    args = parse_args()
    workflow = load_workflow_yaml(args.config)
    _cfg_start, _cfg_end, cfg_stride = time_axis_from_config(workflow)
    stride = int(args.stride if args.stride is not None else cfg_stride)
    dynamic_dir = infer_dynamic_dir(args, workflow=workflow, stride=stride)
    lst_dir = infer_lst_dir(args, dynamic_dir=dynamic_dir)

    dynamic_ready = files_exist(required_dynamic_files(dynamic_dir))
    if args.force_dynamic or not dynamic_ready:
        expanded = False
        coarse_stride = int(args.expand_from_stride)
        if (
            bool(args.topology_only)
            and coarse_stride > int(stride)
            and int(stride) > 0
        ):
            coarse_dir = infer_dynamic_dir(args, workflow=workflow, stride=coarse_stride)
            if args.force_dynamic or not files_exist(required_dynamic_files(coarse_dir)):
                print(
                    f"[lst-pipeline] build coarse dynamic splice before expansion: {coarse_dir}",
                    flush=True,
                )
                run_dynamic_splice_builder(args, out_dir=coarse_dir, stride=coarse_stride)
            if files_exist(required_dynamic_files(coarse_dir)):
                expand_dynamic_time_axis(
                    source_dir=coarse_dir,
                    target_dir=dynamic_dir,
                    start=int(args.start),
                    end=int(args.end),
                    stride=int(stride),
                )
                expanded = True
            else:
                print(f"[lst-pipeline] no coarse dynamic splice found for expansion: {coarse_dir}", flush=True)

        if not expanded:
            run_dynamic_splice_builder(args, out_dir=dynamic_dir, stride=stride)
    else:
        print(f"[lst-pipeline] reuse dynamic splice: {dynamic_dir}", flush=True)

    lst_ready = files_exist(required_lst_files(lst_dir))
    if args.force_lst or not lst_ready:
        cmd = [
            sys.executable,
            str(THIS_DIR / "apply_lst_to_dynamic_splice.py"),
            "--config",
            str(args.config),
            "--input-dir",
            str(dynamic_dir),
            "--out-dir",
            str(lst_dir),
            "--link-setup-time",
            str(float(args.link_setup_time)),
            "--check-only",
        ]
        cmd.extend(list_args("--metric-pairs", args.metric_pairs))
        cmd.extend(list_args("--usage-pairs", args.usage_pairs))
        if bool(args.topology_only):
            cmd.append("--topology-only")
        run_command(cmd)
    else:
        print(f"[lst-pipeline] reuse LST topology: {lst_dir}", flush=True)

    write_pipeline_meta(
        path=lst_dir / "pipeline_meta.json",
        args=args,
        dynamic_dir=dynamic_dir,
        lst_dir=lst_dir,
        stride=stride,
    )

    if args.open_gui or args.screenshot is not None:
        cmd = [
            sys.executable,
            str(THIS_DIR / "apply_lst_to_dynamic_splice.py"),
            "--config",
            str(args.config),
            "--input-dir",
            str(dynamic_dir),
            "--out-dir",
            str(lst_dir),
            "--link-setup-time",
            str(float(args.link_setup_time)),
            "--reuse",
            "--width",
            str(int(args.width)),
            "--height",
            str(int(args.height)),
        ]
        cmd.extend(list_args("--usage-pairs", args.usage_pairs))
        if args.offscreen:
            cmd.append("--offscreen")
        if args.screenshot is not None:
            cmd.extend(["--screenshot", str(args.screenshot)])
        run_command(cmd)

    print("[lst-pipeline] done", flush=True)
    print(f"[lst-pipeline] dynamic_dir={dynamic_dir}", flush=True)
    print(f"[lst-pipeline] lst_dir={lst_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
