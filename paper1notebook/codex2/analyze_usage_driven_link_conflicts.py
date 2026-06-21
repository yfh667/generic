from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe\t0_86160_stride60"
    r"\switch36000_54000\setup600_delay"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_conflict_analysis_056_061_056_china_europe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-link conflicts for usage-driven motif000056/motif000061 switching."
    )
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lst", type=int, default=140)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--p", type=int, default=18)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--working-mode", type=str, default="delay")
    return parser.parse_args()


def read_required(plan_dir: Path) -> pd.DataFrame:
    path = Path(plan_dir) / "usage_driven_right_link_switch_plan_required.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["required_by_working"].astype(bool)].copy()
    for col in ("target_first_work_step", "old_last_work_step_before_deadline"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_conflict_columns(df: pd.DataFrame, *, lst: int, stride: int) -> pd.DataFrame:
    out = df.copy()
    old_last = out["old_last_work_step_before_deadline"]
    no_old_work = old_last.isna()
    out["latest_start"] = out["target_first_work_step"].astype(int) - int(lst)
    out["old_release_after"] = np.where(no_old_work, 0, old_last + int(stride))
    out["available_gap_s"] = out["target_first_work_step"].astype(float) - out["old_release_after"].astype(float)
    out["required_lst_s"] = int(lst)
    out["conflict"] = out["available_gap_s"] < int(lst)
    out["gap_deficit_s"] = np.maximum(0.0, float(lst) - out["available_gap_s"].astype(float))
    out["old_last_work_hour"] = out["old_last_work_step_before_deadline"] / 3600.0
    out["target_first_work_hour"] = out["target_first_work_step"] / 3600.0
    return out


def write_summary(df: pd.DataFrame, conflicts: pd.DataFrame, out_dir: Path, *, lst: int, mode: str) -> None:
    by_transition = (
        conflicts.groupby("transition")
        .agg(
            conflicts=("owner", "count"),
            mean_gap_deficit_s=("gap_deficit_s", "mean"),
            max_gap_deficit_s=("gap_deficit_s", "max"),
            mean_available_gap_s=("available_gap_s", "mean"),
        )
        .reset_index()
    )
    by_plane = (
        conflicts.groupby(["owner_p", "transition"])
        .size()
        .reset_index(name="conflicts")
        .sort_values(["transition", "owner_p"])
    )
    by_transition.to_csv(out_dir / "conflicts_by_transition.csv", index=False, encoding="utf-8-sig")
    by_plane.to_csv(out_dir / "conflicts_by_plane.csv", index=False, encoding="utf-8-sig")
    payload = {
        "working_mode": str(mode),
        "lst": int(lst),
        "required_jobs": int(len(df)),
        "conflicts": int(len(conflicts)),
        "non_conflicts": int(len(df) - len(conflicts)),
        "mean_gap_deficit_s": None if conflicts.empty else float(conflicts["gap_deficit_s"].mean()),
        "max_gap_deficit_s": None if conflicts.empty else float(conflicts["gap_deficit_s"].max()),
        "min_available_gap_s": None if df.empty else float(df["available_gap_s"].min()),
        "outputs": {
            "conflicts": "conflict_edges.csv",
            "by_transition": "conflicts_by_transition.csv",
            "by_plane": "conflicts_by_plane.csv",
            "owner_heatmap": "conflict_owner_heatmap.png",
            "gap_histogram": "conflict_gap_histogram.png",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_owner_heatmap(conflicts: pd.DataFrame, out_path: Path, *, p: int, n: int, title: str) -> None:
    grid = np.zeros((int(n), int(p)), dtype=np.int32)
    for row in conflicts.itertuples(index=False):
        grid[int(row.owner_y), int(row.owner_p)] += 1
    fig, ax = plt.subplots(figsize=(13.2, 8.0), dpi=170)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="Reds")
    ax.set_xlabel("plane p")
    ax.set_ylabel("phase y")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, int(p), 1))
    ax.set_yticks(np.arange(0, int(n), 2))
    ax.grid(color="#e5e7eb", linewidth=0.35, alpha=0.75)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("conflict count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gap_histogram(df: pd.DataFrame, conflicts: pd.DataFrame, out_path: Path, *, lst: int) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=170)
    ax.hist(df["available_gap_s"].to_numpy(dtype=float), bins=60, color="#94a3b8", alpha=0.75, label="all required jobs")
    if not conflicts.empty:
        ax.hist(conflicts["available_gap_s"].to_numpy(dtype=float), bins=40, color="#dc2626", alpha=0.65, label="conflicts")
    ax.axvline(float(lst), color="#111827", linestyle="--", linewidth=1.15, label=f"LST={int(lst)}s")
    ax.set_xlabel("available gap between old last work and target first work (s)")
    ax.set_ylabel("jobs")
    ax.set_title("Conflict criterion: available_gap < LST")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    df = add_conflict_columns(read_required(Path(args.plan_dir)), lst=int(args.lst), stride=int(args.stride))
    conflicts = df[df["conflict"]].copy().sort_values(["transition", "target_first_work_step", "owner"])
    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"{str(args.working_mode)}_lst{int(args.lst):03d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "required_edges_with_gap.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(out_dir / "conflict_edges.csv", index=False, encoding="utf-8-sig")
    write_summary(df, conflicts, out_dir, lst=int(args.lst), mode=str(args.working_mode))
    plot_owner_heatmap(
        conflicts,
        out_dir / "conflict_owner_heatmap.png",
        p=int(args.p),
        n=int(args.n),
        title=f"Conflict owners, mode={args.working_mode}, LST={int(args.lst)}s",
    )
    plot_gap_histogram(df, conflicts, out_dir / "conflict_gap_histogram.png", lst=int(args.lst))
    print(f"out_dir={out_dir}")
    print(f"required={len(df)} conflicts={len(conflicts)} non_conflicts={len(df)-len(conflicts)}")
    if not conflicts.empty:
        print(
            f"gap_deficit_mean={conflicts['gap_deficit_s'].mean():.2f}s "
            f"gap_deficit_max={conflicts['gap_deficit_s'].max():.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
