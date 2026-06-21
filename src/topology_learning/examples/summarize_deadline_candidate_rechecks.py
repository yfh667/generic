from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))


PAIR_PREFIXES = {
    "china_europe": "ce",
    "china_africa": "ca",
    "china_america": "cam",
}


def parse_named_path(text: str) -> tuple[str, Path]:
    name, sep, path = str(text).partition(":")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("expected name:path")
    return name, Path(path)


def write_csv(path: Path, rows: list[dict]) -> None:
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


def find_candidate_dirs(roots: list[Path]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if all((root / f"eval_{pair}" / "meta.json").exists() for pair in PAIR_PREFIXES):
            found.append((root.name, root))
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if all((child / f"eval_{pair}" / "meta.json").exists() for pair in PAIR_PREFIXES):
                found.append((child.name, child))
    return found


def load_full_refs(selector_arrays: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    arrays = np.load(selector_arrays, allow_pickle=True)
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    refs: dict[str, dict[str, np.ndarray]] = {}
    for pair, prefix in PAIR_PREFIXES.items():
        refs[pair] = {
            "delay": np.asarray(arrays[f"metric_{prefix}_delay_ms_full_link"], dtype=np.float64),
            "hops": np.asarray(arrays[f"metric_{prefix}_hops_full_link"], dtype=np.float64),
        }
    return steps, refs


def summarize_candidate(name: str, base: Path, refs: dict[str, dict[str, np.ndarray]]) -> tuple[dict, list[dict]]:
    detail_rows: list[dict] = []
    delay_gaps: list[float] = []
    hop_gaps: list[float] = []
    setup_totals: list[int] = []
    setup_maxes: list[int] = []
    building_maxes: list[int] = []
    switches: list[int] = []
    union_edges: list[int] = []
    for pair in PAIR_PREFIXES:
        eval_dir = base / f"eval_{pair}"
        meta = json.loads((eval_dir / "meta.json").read_text(encoding="utf-8"))
        delay = float(meta["mean_shortest_delay_ms_strict_all_pairs"])
        hops = float(meta["mean_shortest_hops_strict_all_pairs"])
        full_delay = float(np.mean(refs[pair]["delay"]))
        full_hops = float(np.mean(refs[pair]["hops"]))
        delay_gap = delay / full_delay - 1.0
        hop_gap = hops / full_hops - 1.0
        delay_gaps.append(delay_gap)
        hop_gaps.append(hop_gap)
        setup_totals.append(int(meta["total_setup_commands"]))
        setup_maxes.append(int(meta["max_setup_commands_per_step"]))
        building_maxes.append(int(meta["max_building_edges"]))
        switches.append(int(meta["num_switches"]))
        union_edges.append(int(meta["num_union_edges"]))
        detail_rows.append(
            {
                "schedule": name,
                "pair": pair,
                "delay_ms": delay,
                "delay_gap_to_full_link": delay_gap,
                "hops": hops,
                "hops_gap_to_full_link": hop_gap,
                "full_link_delay_ms": full_delay,
                "full_link_hops": full_hops,
                "total_setup_commands": int(meta["total_setup_commands"]),
                "max_setup_commands_per_step": int(meta["max_setup_commands_per_step"]),
                "max_building_edges": int(meta["max_building_edges"]),
                "num_switches": int(meta["num_switches"]),
                "num_union_edges": int(meta["num_union_edges"]),
                "disconnected_rows": int(meta["disconnected_rows"]),
                "path": str(base),
            }
        )
    summary = {
        "schedule": name,
        "mean_all_gap": float((np.mean(delay_gaps) + np.mean(hop_gaps)) / 2.0),
        "mean_delay_gap": float(np.mean(delay_gaps)),
        "mean_hop_gap": float(np.mean(hop_gaps)),
        "total_setup_commands": int(max(setup_totals)),
        "max_setup_commands_per_step": int(max(setup_maxes)),
        "max_building_edges": int(max(building_maxes)),
        "num_switches": int(max(switches)),
        "num_union_edges": int(max(union_edges)),
        "path": str(base),
    }
    return summary, detail_rows


def mark_pareto(rows: list[dict]) -> None:
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["total_setup_commands"] <= row["total_setup_commands"]
                and other["mean_all_gap"] <= row["mean_all_gap"]
                and (
                    other["total_setup_commands"] < row["total_setup_commands"]
                    or other["mean_all_gap"] < row["mean_all_gap"]
                )
            ):
                dominated = True
                break
        row["pareto"] = not dominated


def plot_pareto(rows: list[dict], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10.5, 6.2), dpi=170)
    for row in rows:
        is_pareto = bool(row["pareto"])
        cap = int(row["max_setup_commands_per_step"])
        color = "#dc2626" if is_pareto else ("#94a3b8" if cap <= 2 else "#c4b5fd")
        size = 78 if is_pareto else 42
        plt.scatter(
            row["total_setup_commands"],
            row["mean_all_gap"],
            s=size,
            c=color,
            edgecolors="black" if is_pareto else "none",
            linewidths=0.6,
            alpha=0.9,
        )
    for row in rows[:12]:
        plt.annotate(
            str(row["schedule"]),
            (row["total_setup_commands"], row["mean_all_gap"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.2,
        )
    for row in [item for item in rows if item["pareto"]]:
        if row not in rows[:12]:
            plt.annotate(
                str(row["schedule"]),
                (row["total_setup_commands"], row["mean_all_gap"]),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=7.0,
            )
    plt.xlabel("total setup commands")
    plt.ylabel("mean gap to full-link (delay/hops, 3 pairs)")
    plt.title("Deadline-aware LST candidate frontier")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_timeseries(
    *,
    steps: np.ndarray,
    refs: dict[str, dict[str, np.ndarray]],
    selected: list[tuple[str, Path]],
    out_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#64748b", "#059669", "#2563eb", "#dc2626", "#7c3aed", "#ea580c"]
    outputs: list[str] = []
    for pair in PAIR_PREFIXES:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), dpi=170, sharex=True)
        axes[0].plot(steps / 3600.0, refs[pair]["delay"], label="full_link", color="#111827", linewidth=1.6)
        axes[1].plot(steps / 3600.0, refs[pair]["hops"], label="full_link", color="#111827", linewidth=1.6)
        for idx, (name, base) in enumerate(selected):
            eval_dir = base / f"eval_{pair}"
            delay = np.load(eval_dir / "mean_shortest_delay_ms_strict_all_pairs.npy")
            hops = np.load(eval_dir / "mean_shortest_hops_strict_all_pairs.npy")
            color = colors[idx % len(colors)]
            axes[0].plot(steps / 3600.0, delay, label=name, color=color, linewidth=1.15)
            axes[1].plot(steps / 3600.0, hops, label=name, color=color, linewidth=1.15)
        axes[0].set_ylabel("delay (ms)")
        axes[1].set_ylabel("hops")
        axes[1].set_xlabel("time (hour)")
        axes[0].set_title(f"{pair} shortest-delay and shortest-hop time series")
        for ax in axes:
            ax.grid(True, alpha=0.23)
        axes[0].legend(ncol=3, fontsize=8, loc="upper right")
        fig.tight_layout()
        out_png = out_dir / f"{pair}_timeseries.png"
        fig.savefig(out_png)
        plt.close(fig)
        outputs.append(str(out_png))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize deadline-aware LST candidate rechecks and optionally plot selected time series."
    )
    parser.add_argument("--selector-arrays", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_named_path, action="append", default=[])
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--timeseries-candidate", type=parse_named_path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="deadline_candidates")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps, refs = load_full_refs(args.selector_arrays)
    candidates = list(args.candidate) + find_candidate_dirs(args.candidate_root)
    if not candidates:
        raise ValueError("no candidate dirs provided")
    seen: set[Path] = set()
    unique_candidates: list[tuple[str, Path]] = []
    for name, path in candidates:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append((str(name), Path(path)))

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    for name, path in unique_candidates:
        summary, detail = summarize_candidate(name, Path(path), refs)
        summary_rows.append(summary)
        detail_rows.extend(detail)
    mark_pareto(summary_rows)
    summary_rows.sort(key=lambda row: (row["mean_all_gap"], row["total_setup_commands"]))

    out_dir = Path(args.out_dir)
    summary_csv = out_dir / f"{args.prefix}_summary.csv"
    detail_csv = out_dir / f"{args.prefix}_detail.csv"
    pareto_png = out_dir / f"{args.prefix}_pareto.png"
    write_csv(summary_csv, summary_rows)
    write_csv(detail_csv, detail_rows)
    plot_pareto(summary_rows, pareto_png)

    time_outputs: list[str] = []
    if args.timeseries_candidate:
        time_outputs = plot_timeseries(
            steps=steps,
            refs=refs,
            selected=[(str(name), Path(path)) for name, path in args.timeseries_candidate],
            out_dir=out_dir / "timeseries",
        )

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "detail_csv": str(detail_csv),
                "pareto_png": str(pareto_png),
                "timeseries_png": time_outputs,
                "num_candidates": len(unique_candidates),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
