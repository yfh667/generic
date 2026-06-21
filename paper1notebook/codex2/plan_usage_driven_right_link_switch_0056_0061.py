from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402


DEFAULT_DELAY_USAGE_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\delay_edge_usage_056_061_china_europe"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe"
)
PAIR_KEY = "china_europe"


@dataclass(frozen=True)
class TopologyUsage:
    topo: one_way.TopologyData
    hop: np.ndarray
    delay: np.ndarray
    usage: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan no-LST-splice right-link switch times from edge-betweenness first-use deadlines. "
            "Only right-neighbor links are considered."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--setup-duration", type=int, default=600)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--middle-motif-id", type=int, default=61)
    parser.add_argument("--working-mode", choices=("hop", "delay", "any"), default="any")
    parser.add_argument("--working-threshold", type=float, default=0.0)
    parser.add_argument("--delay-usage-root", type=Path, default=DEFAULT_DELAY_USAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def build_topology(motif_id: int, *, start: int, end: int, stride: int) -> one_way.TopologyData:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    spec = one_way.build_topology_spec(int(motif_id), rows[int(motif_id)])
    return one_way.load_topology_data(spec, start=int(start), end=int(end), stride=int(stride))


def load_delay_usage(
    topo: one_way.TopologyData,
    *,
    delay_usage_root: Path,
    start: int,
    end: int,
    stride: int,
) -> np.ndarray:
    path = (
        Path(delay_usage_root)
        / f"t{int(start)}_{int(end)}_stride{int(stride)}"
        / topo.spec.name
        / PAIR_KEY
        / "edge_betweenness.npy"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    if arr.shape != topo.values.shape:
        raise ValueError(f"delay usage shape mismatch for {topo.spec.name}: {arr.shape} != {topo.values.shape}")
    return np.asarray(arr, dtype=np.float32)


def topology_usage(
    motif_id: int,
    *,
    mode: str,
    delay_usage_root: Path,
    start: int,
    end: int,
    stride: int,
) -> TopologyUsage:
    topo = build_topology(int(motif_id), start=start, end=end, stride=stride)
    hop = np.asarray(topo.values, dtype=np.float32)
    delay = load_delay_usage(topo, delay_usage_root=delay_usage_root, start=start, end=end, stride=stride)
    if mode == "hop":
        usage = hop
    elif mode == "delay":
        usage = delay
    else:
        usage = np.maximum(hop, delay)
    return TopologyUsage(topo=topo, hop=hop, delay=delay, usage=np.asarray(usage, dtype=np.float32))


def rows_between(steps_np: np.ndarray, start: int, end: int, *, include_end: bool = False) -> np.ndarray:
    if include_end:
        return np.flatnonzero((steps_np >= int(start)) & (steps_np <= int(end)))
    return np.flatnonzero((steps_np >= int(start)) & (steps_np < int(end)))


def first_work_step(
    *,
    steps_np: np.ndarray,
    usage: np.ndarray,
    edge_idx: int,
    start: int,
    end: int,
    threshold: float,
) -> tuple[int | None, float]:
    rows = rows_between(steps_np, int(start), int(end), include_end=False)
    if rows.size == 0:
        return None, 0.0
    values = np.asarray(usage[rows, int(edge_idx)], dtype=np.float32)
    working = np.flatnonzero(values > float(threshold))
    if working.size == 0:
        return None, float(np.nanmax(values)) if values.size else 0.0
    local = int(working[0])
    return int(steps_np[int(rows[local])]), float(values[local])


def last_work_step_before(
    *,
    steps_np: np.ndarray,
    usage: np.ndarray,
    edge_idx: int,
    start: int,
    deadline: int,
    threshold: float,
) -> tuple[int | None, float]:
    rows = rows_between(steps_np, int(start), int(deadline), include_end=False)
    if rows.size == 0:
        return None, 0.0
    values = np.asarray(usage[rows, int(edge_idx)], dtype=np.float32)
    working = np.flatnonzero(values > float(threshold))
    if working.size == 0:
        return None, float(np.nanmax(values)) if values.size else 0.0
    local = int(working[-1])
    return int(steps_np[int(rows[local])]), float(values[local])


def value_at_step(steps: list[int], values: np.ndarray, step: int, edge_idx: int) -> float:
    row = steps.index(int(step))
    return float(values[row, int(edge_idx)])


def plan_transition(
    *,
    label: str,
    source: TopologyUsage,
    target: TopologyUsage,
    steps: list[int],
    segment_start: int,
    target_segment_start: int,
    target_segment_end: int,
    setup_duration: int,
    threshold: float,
) -> list[dict[str, object]]:
    steps_np = np.asarray(steps, dtype=np.int64)
    switch_idx = steps.index(int(target_segment_start))
    switch_idx_end = min(len(steps), switch_idx + max(2, int(3600 / int(steps[1] - steps[0]))))
    changes = one_way.find_right_link_changes(
        source.topo,
        target.topo,
        switch_idx=switch_idx,
        switch_idx_end=switch_idx_end,
    )
    rows: list[dict[str, object]] = []
    stride = int(steps[1] - steps[0]) if len(steps) > 1 else int(setup_duration)
    for change in changes:
        old = change.old
        new = change.new
        owner = int(change.owner)
        if new is None:
            rows.append(
                {
                    "transition": label,
                    "owner": owner,
                    "owner_p": int(owner // G60_CONFIG.N),
                    "owner_y": int(owner % G60_CONFIG.N),
                    "old_edge": "" if old is None else old.edge_key_text,
                    "old_symbol": "" if old is None else old.symbol,
                    "new_edge": "",
                    "new_symbol": "",
                    "target_first_work_step": "",
                    "target_first_work_hour": "",
                    "target_first_work_value": 0.0,
                    "old_last_work_step_before_deadline": "",
                    "old_last_work_hour_before_deadline": "",
                    "old_last_work_value": 0.0,
                    "plan_start": "",
                    "plan_end": "",
                    "feasible": True,
                    "required_by_working": False,
                    "reason": "target_has_no_right_link",
                }
            )
            continue

        first_step, first_value = first_work_step(
            steps_np=steps_np,
            usage=target.usage,
            edge_idx=int(new.edge_idx),
            start=int(target_segment_start),
            end=int(target_segment_end),
            threshold=float(threshold),
        )
        if first_step is None:
            rows.append(
                {
                    "transition": label,
                    "owner": owner,
                    "owner_p": int(owner // G60_CONFIG.N),
                    "owner_y": int(owner % G60_CONFIG.N),
                    "old_edge": "" if old is None else old.edge_key_text,
                    "old_symbol": "" if old is None else old.symbol,
                    "new_edge": new.edge_key_text,
                    "new_symbol": new.symbol,
                    "target_first_work_step": "",
                    "target_first_work_hour": "",
                    "target_first_work_value": float(first_value),
                    "old_last_work_step_before_deadline": "",
                    "old_last_work_hour_before_deadline": "",
                    "old_last_work_value": 0.0,
                    "plan_start": "",
                    "plan_end": "",
                    "feasible": True,
                    "required_by_working": False,
                    "reason": "target_edge_never_working_in_segment",
                }
            )
            continue

        old_last_step = None
        old_last_value = 0.0
        if old is not None:
            old_last_step, old_last_value = last_work_step_before(
                steps_np=steps_np,
                usage=source.usage,
                edge_idx=int(old.edge_idx),
                start=int(segment_start),
                deadline=int(first_step),
                threshold=float(threshold),
            )
        old_release_after = int(segment_start) if old_last_step is None else int(old_last_step) + int(stride)
        latest_start = int(first_step) - int(setup_duration)
        feasible = latest_start >= old_release_after and latest_start >= int(steps[0])
        if feasible:
            plan_start = int(latest_start)
            reason = "switch_before_target_first_work"
        else:
            # This keeps the target deadline and exposes the old-link conflict explicitly.
            plan_start = int(latest_start)
            reason = "conflict_old_edge_working_inside_setup_window"
        plan_end = int(plan_start) + int(setup_duration)
        rows.append(
            {
                "transition": label,
                "owner": owner,
                "owner_p": int(owner // G60_CONFIG.N),
                "owner_y": int(owner % G60_CONFIG.N),
                "old_edge": "" if old is None else old.edge_key_text,
                "old_symbol": "" if old is None else old.symbol,
                "new_edge": new.edge_key_text,
                "new_symbol": new.symbol,
                "target_first_work_step": int(first_step),
                "target_first_work_hour": float(first_step) / 3600.0,
                "target_first_work_value": float(first_value),
                "old_last_work_step_before_deadline": "" if old_last_step is None else int(old_last_step),
                "old_last_work_hour_before_deadline": "" if old_last_step is None else float(old_last_step) / 3600.0,
                "old_last_work_value": float(old_last_value),
                "plan_start": int(plan_start),
                "plan_end": int(plan_end),
                "feasible": bool(feasible),
                "required_by_working": True,
                "reason": reason,
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def concurrency_by_step(steps: list[int], rows: list[dict[str, object]]) -> np.ndarray:
    counts = np.zeros(len(steps), dtype=np.int32)
    steps_np = np.asarray(steps, dtype=np.int64)
    for row in rows:
        if not row.get("required_by_working"):
            continue
        start = row.get("plan_start", "")
        end = row.get("plan_end", "")
        if start == "" or end == "":
            continue
        idx = rows_between(steps_np, int(start), int(end), include_end=False)
        counts[idx] += 1
    return counts


def plot_plan(path: Path, *, steps: list[int], rows: list[dict[str, object]], concurrency: np.ndarray, switch_step: int, return_switch_step: int) -> None:
    required = [row for row in rows if row.get("required_by_working")]
    starts = np.asarray([int(row["plan_start"]) for row in required], dtype=np.int64) if required else np.asarray([], dtype=np.int64)
    ends = np.asarray([int(row["plan_end"]) for row in required], dtype=np.int64) if required else np.asarray([], dtype=np.int64)
    feasible = np.asarray([bool(row["feasible"]) for row in required], dtype=bool) if required else np.asarray([], dtype=bool)
    owners = np.asarray([int(row["owner"]) for row in required], dtype=np.int32) if required else np.asarray([], dtype=np.int32)

    fig, axes = plt.subplots(2, 1, figsize=(15.8, 8.0), dpi=170, sharex=False)
    hours = np.asarray(steps, dtype=np.float64) / 3600.0
    axes[0].plot(hours, concurrency, color="#b91c1c", linewidth=1.45)
    axes[0].axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[0].axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("building links")
    axes[0].set_title(f"Usage-driven right-link switch schedule, max concurrency={int(np.max(concurrency)) if concurrency.size else 0}")
    axes[0].grid(True, alpha=0.25, linestyle="--", linewidth=0.55)

    if required:
        colors = np.where(feasible, "#2563eb", "#dc2626")
        axes[1].scatter(starts / 3600.0, owners, s=12, c=colors, alpha=0.68, label="plan_start")
        axes[1].scatter(ends / 3600.0, owners, s=8, c="#111827", alpha=0.35, label="plan_end")
    axes[1].axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[1].axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("time (hour)")
    axes[1].set_ylabel("owner node")
    axes[1].grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if int(args.switch_step) not in set(steps) or int(args.return_switch_step) not in set(steps):
        raise ValueError("switch steps must be on the requested time axis")

    usage56 = topology_usage(
        int(args.source_motif_id),
        mode=str(args.working_mode),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    usage61 = topology_usage(
        int(args.middle_motif_id),
        mode=str(args.working_mode),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )

    first_rows = plan_transition(
        label="switch_056_to_061",
        source=usage56,
        target=usage61,
        steps=steps,
        segment_start=int(args.start),
        target_segment_start=int(args.switch_step),
        target_segment_end=int(args.return_switch_step),
        setup_duration=int(args.setup_duration),
        threshold=float(args.working_threshold),
    )
    second_rows = plan_transition(
        label="switch_061_to_056",
        source=usage61,
        target=usage56,
        steps=steps,
        segment_start=int(args.switch_step),
        target_segment_start=int(args.return_switch_step),
        target_segment_end=int(args.end) + int(args.stride),
        setup_duration=int(args.setup_duration),
        threshold=float(args.working_threshold),
    )
    rows = first_rows + second_rows

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"
        / f"setup{int(args.setup_duration)}_{args.working_mode}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "usage_driven_right_link_switch_plan.csv", rows)
    write_rows(out_dir / "usage_driven_right_link_switch_plan_required.csv", [row for row in rows if row.get("required_by_working")])
    write_rows(out_dir / "usage_driven_right_link_switch_conflicts.csv", [row for row in rows if row.get("required_by_working") and not row.get("feasible")])

    concurrency = concurrency_by_step(steps, rows)
    np.save(out_dir / "steps.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "building_concurrency.npy", concurrency)
    plot_plan(
        out_dir / "usage_driven_right_link_switch_plan.png",
        steps=steps,
        rows=rows,
        concurrency=concurrency,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )

    required = [row for row in rows if row.get("required_by_working")]
    conflicts = [row for row in required if not row.get("feasible")]
    summary = {
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "setup_duration": int(args.setup_duration),
        "working_mode": str(args.working_mode),
        "working_threshold": float(args.working_threshold),
        "total_changed_right_ports": int(len(rows)),
        "required_by_working": int(len(required)),
        "not_required_by_working": int(len(rows) - len(required)),
        "conflicts": int(len(conflicts)),
        "feasible_required": int(len(required) - len(conflicts)),
        "max_concurrency_raw": int(np.max(concurrency)) if concurrency.size else 0,
        "outputs": {
            "plan": "usage_driven_right_link_switch_plan.csv",
            "required_plan": "usage_driven_right_link_switch_plan_required.csv",
            "conflicts": "usage_driven_right_link_switch_conflicts.csv",
            "plot": "usage_driven_right_link_switch_plan.png",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"out_dir={out_dir}")
    print(f"changed_right_ports={len(rows)} required_by_working={len(required)} not_required={len(rows)-len(required)}")
    print(f"conflicts={len(conflicts)} feasible_required={len(required)-len(conflicts)}")
    print(f"max_concurrency_raw={int(np.max(concurrency)) if concurrency.size else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
