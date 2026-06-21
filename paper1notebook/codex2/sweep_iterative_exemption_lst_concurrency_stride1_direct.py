from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
import plan_usage_driven_switch_with_36000_exemption as base  # noqa: E402
import sweep_iterative_exemption_lst_concurrency as sweep60  # noqa: E402
from src.topology_metrics.module.stores import MetricStoreLayout  # noqa: E402


DEFAULT_HOP_CACHE_ROOT = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3")
    / "dual_edge_usage_056_061_china_europe"
    / "edge_betweenness_cache"
)
DEFAULT_DELAY_TRACKED_ROOT = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup")
    / "usage_driven_switch_056_061_056_china_europe_1s_delay"
)
DEFAULT_OUT_ROOT = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup")
    / "usage_driven_iterative_exemption_lst_sweep_056_061_china_europe_1s_any_direct"
)
PAIR_KEY = "china_europe"


class CompactHopAccessor:
    def __init__(self, store_dir: Path):
        layout = MetricStoreLayout(Path(store_dir))
        if not layout.unique_state_values_npy.exists() or not layout.state_ids_npy.exists():
            raise FileNotFoundError(f"missing compact hop edge-usage store under {store_dir}")
        self.unique_values = np.load(layout.unique_state_values_npy, mmap_mode="r")
        self.state_ids = np.load(layout.state_ids_npy, mmap_mode="r")
        self.shape = (int(self.state_ids.shape[0]), int(self.unique_values.shape[1]))

    def __getitem__(self, key: Any) -> Any:
        rows, cols = key
        states = self.state_ids[rows]
        return self.unique_values[states, cols]


class TrackedDelayAccessor:
    def __init__(self, tracked_dir: Path, *, num_steps: int, num_edges: int):
        counts_path = Path(tracked_dir) / "tracked_edge_usage_counts.npy"
        tracked_edges_path = Path(tracked_dir) / "tracked_edges.csv"
        time_path = Path(tracked_dir) / "time_indices.npy"
        if not counts_path.exists() or not tracked_edges_path.exists() or not time_path.exists():
            raise FileNotFoundError(f"missing tracked delay usage store under {tracked_dir}")
        self.counts = np.load(counts_path, mmap_mode="r")
        self.times = np.load(time_path, mmap_mode="r")
        if int(self.counts.shape[0]) != int(num_steps):
            raise ValueError(f"delay counts rows {self.counts.shape[0]} != {num_steps} for {tracked_dir}")
        if int(self.times.shape[0]) != int(num_steps):
            raise ValueError(f"time rows {self.times.shape[0]} != {num_steps} for {tracked_dir}")
        self.edge_to_col: dict[int, int] = {}
        with tracked_edges_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                self.edge_to_col[int(row["edge_idx"])] = int(row["tracked_col"])
        self.shape = (int(num_steps), int(num_edges))

    def _zeros_for_rows(self, rows: Any) -> Any:
        if isinstance(rows, (int, np.integer)):
            return np.float32(0.0)
        return np.zeros(np.asarray(self.times[rows]).shape, dtype=np.float32)

    def __getitem__(self, key: Any) -> Any:
        rows, cols = key
        if isinstance(cols, (int, np.integer)):
            col = self.edge_to_col.get(int(cols))
            if col is None:
                return self._zeros_for_rows(rows)
            return self.counts[rows, int(col)]
        cols_arr = np.asarray(cols)
        out_shape = np.broadcast_shapes(np.asarray(self.times[rows]).shape, cols_arr.shape)
        out = np.zeros(out_shape, dtype=np.float32)
        for out_col, edge_idx in np.ndenumerate(cols_arr):
            tracked_col = self.edge_to_col.get(int(edge_idx))
            if tracked_col is not None:
                out[..., out_col[0] if out_col else 0] = self.counts[rows, int(tracked_col)]
        return out


class MaxUsageAccessor:
    def __init__(self, left: Any, right: Any):
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
        self.left = left
        self.right = right
        self.shape = tuple(left.shape)

    def __getitem__(self, key: Any) -> Any:
        return np.maximum(self.left[key], self.right[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the iterative-exemption LST sweep at true 1s sampling. "
            "Hop usage is read from compact 1s stores; delay usage is read from direct tracked 1s counts."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--target-motif-id", type=int, default=61)
    parser.add_argument("--working-threshold", type=float, default=0.0)
    parser.add_argument("--release-guard-seconds", type=int, default=1)
    parser.add_argument("--hop-cache-root", type=Path, default=DEFAULT_HOP_CACHE_ROOT)
    parser.add_argument("--delay-tracked-root", type=Path, default=DEFAULT_DELAY_TRACKED_ROOT)
    parser.add_argument("--delay-tracked-switch-step", type=int, default=None)
    parser.add_argument("--delay-tracked-return-switch-step", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser.parse_args()


def read_motif_spec(motif_id: int) -> one_way.TopologySpec:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    return one_way.build_topology_spec(int(motif_id), rows[int(motif_id)])


def hop_store_dir(spec: one_way.TopologySpec, *, root: Path, start: int, end: int, stride: int) -> Path:
    return Path(root) / f"t{int(start)}_{int(end)}_stride{int(stride)}" / PAIR_KEY / spec.name


def delay_tracked_dir(
    spec: one_way.TopologySpec,
    *,
    root: Path,
    start: int,
    end: int,
    stride: int,
    switch_step: int,
    return_switch_step: int,
) -> Path:
    return (
        Path(root)
        / f"t{int(start)}_{int(end)}_stride{int(stride)}"
        / f"switch{int(switch_step)}_{int(return_switch_step)}"
        / "setup600_delay"
        / "tracked_usage"
        / spec.name
    )


def build_usage_bundle(
    motif_id: int,
    *,
    args: argparse.Namespace,
    num_steps: int,
) -> base.UsageBundle:
    spec = read_motif_spec(int(motif_id))
    right_by_owner, left_by_right = one_way.build_right_links(spec.edge_table)
    hop = CompactHopAccessor(
        hop_store_dir(
            spec,
            root=Path(args.hop_cache_root),
            start=int(args.start),
            end=int(args.end),
            stride=int(args.stride),
        )
    )
    delay = TrackedDelayAccessor(
        delay_tracked_dir(
            spec,
            root=Path(args.delay_tracked_root),
            start=int(args.start),
            end=int(args.end),
            stride=int(args.stride),
            switch_step=int(args.delay_tracked_switch_step)
            if args.delay_tracked_switch_step is not None
            else int(args.switch_step),
            return_switch_step=int(args.delay_tracked_return_switch_step)
            if args.delay_tracked_return_switch_step is not None
            else int(args.return_switch_step),
        ),
        num_steps=int(num_steps),
        num_edges=int(spec.edge_table.num_edges),
    )
    if hop.shape != delay.shape:
        raise ValueError(f"{spec.name}: hop shape {hop.shape} != delay shape {delay.shape}")
    any_usage = MaxUsageAccessor(hop, delay)
    topo = one_way.TopologyData(
        spec=spec,
        values=any_usage,
        right_by_owner=right_by_owner,
        left_by_right=left_by_right,
        edge_key_to_idx=one_way.build_edge_key_index(spec.edge_table),
    )
    return base.UsageBundle(topo=topo, hop=hop, delay=delay, any_usage=any_usage)


def main() -> int:
    args = parse_args()
    if int(args.stride) != 1:
        raise ValueError("this script is for true 1s sampling; use --stride 1")
    steps = np.asarray(list(range(int(args.start), int(args.end) + 1, int(args.stride))), dtype=np.int64)
    source = build_usage_bundle(int(args.source_motif_id), args=args, num_steps=len(steps))
    target = build_usage_bundle(int(args.target_motif_id), args=args, num_steps=len(steps))

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"
        / f"releaseguard{int(args.release_guard_seconds)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for lst in range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)):
        result = sweep60.run_one_lst(
            lst=int(lst),
            source=source,
            target=target,
            steps=steps,
            args=args,
            out_dir=out_dir,
        )
        rows.append(result)
        print(result, flush=True)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "lst_sweep_concurrency_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    sweep60.plot_summary(summary, out_dir)
    meta = {
        "description": (
            "True 1s-sampled iterative exemption LST sweep. "
            "Hop usage comes from compact 1s edge-betweenness stores; delay usage comes from direct tracked 1s shortest-delay counts."
        ),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "lst_values": [int(x) for x in summary["lst"].tolist()],
        "release_guard_seconds": int(args.release_guard_seconds),
        "source_motif_id": int(args.source_motif_id),
        "target_motif_id": int(args.target_motif_id),
        "hop_cache_root": str(args.hop_cache_root),
        "delay_tracked_root": str(args.delay_tracked_root),
        "delay_tracked_switch_step": int(args.delay_tracked_switch_step)
        if args.delay_tracked_switch_step is not None
        else int(args.switch_step),
        "delay_tracked_return_switch_step": int(args.delay_tracked_return_switch_step)
        if args.delay_tracked_return_switch_step is not None
        else int(args.return_switch_step),
        "outputs": {
            "summary_csv": str(summary_path),
            "summary_plot": str(out_dir / "lst_sweep_concurrency_summary.png"),
            "per_lst_dirs": "lstXXX/",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
