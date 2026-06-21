from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_STORE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\ideal_splice_056_061_056_china_europe\t0_86160_stride60"
    r"\switch36000_54000\right_link_state"
)


@dataclass(frozen=True)
class RightLinkStateStore:
    store_dir: Path
    meta: dict
    steps: np.ndarray
    topology_motif_id: np.ndarray
    right_neighbor: np.ndarray
    right_option: np.ndarray
    right_symbol_id: np.ndarray
    right_edge_idx: np.ndarray
    hop_betweenness: np.ndarray
    delay_betweenness: np.ndarray
    working_hop: np.ndarray
    working_delay: np.ndarray
    working_any: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query and plot one node's right-link working state in the ideal 056/061/056 splice."
    )
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--working-mode", choices=("hop", "delay", "any"), default="any")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--show-zero-betweenness", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open the generated PNG after saving it.")
    return parser.parse_args()


def load_store(store_dir: str | Path) -> RightLinkStateStore:
    store_dir = Path(store_dir)
    with (store_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return RightLinkStateStore(
        store_dir=store_dir,
        meta=meta,
        steps=np.load(store_dir / "steps.npy"),
        topology_motif_id=np.load(store_dir / "topology_motif_id.npy"),
        right_neighbor=np.load(store_dir / "right_neighbor.npy", mmap_mode="r"),
        right_option=np.load(store_dir / "right_option.npy", mmap_mode="r"),
        right_symbol_id=np.load(store_dir / "right_symbol_id.npy", mmap_mode="r"),
        right_edge_idx=np.load(store_dir / "right_edge_idx_by_node.npy", mmap_mode="r"),
        hop_betweenness=np.load(store_dir / "node_right_hop_betweenness.npy", mmap_mode="r"),
        delay_betweenness=np.load(store_dir / "node_right_delay_betweenness.npy", mmap_mode="r"),
        working_hop=np.load(store_dir / "node_right_working_hop.npy", mmap_mode="r"),
        working_delay=np.load(store_dir / "node_right_working_delay.npy", mmap_mode="r"),
        working_any=np.load(store_dir / "node_right_working_any.npy", mmap_mode="r"),
    )


def node_xy(node: int, *, n: int) -> tuple[int, int]:
    return int(node // n), int(node % n)


def node_state_dataframe(store: RightLinkStateStore, node: int) -> pd.DataFrame:
    node = int(node)
    num_nodes = int(store.right_neighbor.shape[1])
    if not (0 <= node < num_nodes):
        raise ValueError(f"node must be in [0, {num_nodes - 1}], got {node}")
    n = int(store.meta.get("constellation_n") or 36)
    owner_p, owner_y = node_xy(node, n=n)
    right = np.asarray(store.right_neighbor[:, node], dtype=np.int32)
    option = np.asarray(store.right_option[:, node], dtype=np.int16)
    symbol_ids = np.asarray(store.right_symbol_id[:, node], dtype=np.int8)
    symbols = np.asarray(["", "A", "B", "C", "D"], dtype=object)
    right_p = np.where(right >= 0, right // n, -1)
    right_y = np.where(right >= 0, right % n, -1)
    df = pd.DataFrame(
        {
            "step": np.asarray(store.steps, dtype=np.int64),
            "hour": np.asarray(store.steps, dtype=np.float64) / 3600.0,
            "topology_motif_id": np.asarray(store.topology_motif_id, dtype=np.int16),
            "node": node,
            "node_p": owner_p,
            "node_y": owner_y,
            "right_node": right,
            "right_p": right_p,
            "right_y": right_y,
            "option": option,
            "symbol": symbols[symbol_ids],
            "right_edge_idx": np.asarray(store.right_edge_idx[:, node], dtype=np.int32),
            "hop_betweenness": np.asarray(store.hop_betweenness[:, node], dtype=np.float32),
            "delay_betweenness": np.asarray(store.delay_betweenness[:, node], dtype=np.float32),
            "working_hop": np.asarray(store.working_hop[:, node], dtype=bool),
            "working_delay": np.asarray(store.working_delay[:, node], dtype=bool),
            "working_any": np.asarray(store.working_any[:, node], dtype=bool),
        }
    )
    df.loc[df["right_node"] < 0, ["right_node", "right_p", "right_y", "right_edge_idx"]] = pd.NA
    return df


def contiguous_intervals(df: pd.DataFrame, mask_col: str) -> list[dict[str, object]]:
    rows = df[df[mask_col]].reset_index(drop=True)
    if rows.empty:
        return []
    intervals: list[dict[str, object]] = []
    start_row = rows.iloc[0]
    prev_row = rows.iloc[0]
    stride = int(df["step"].iloc[1] - df["step"].iloc[0]) if len(df) > 1 else 0
    for _, row in rows.iloc[1:].iterrows():
        same_run = (
            int(row["step"]) == int(prev_row["step"]) + stride
            and row["right_node"] == prev_row["right_node"]
            and row["topology_motif_id"] == prev_row["topology_motif_id"]
        )
        if not bool(same_run):
            intervals.append(
                {
                    "start_step": int(start_row["step"]),
                    "end_step": int(prev_row["step"]),
                    "start_hour": float(start_row["hour"]),
                    "end_hour": float(prev_row["hour"]),
                    "topology_motif_id": int(start_row["topology_motif_id"]),
                    "right_node": "" if pd.isna(start_row["right_node"]) else int(start_row["right_node"]),
                    "symbol": str(start_row["symbol"]),
                    "steps": int((int(prev_row["step"]) - int(start_row["step"])) / stride + 1) if stride else 1,
                }
            )
            start_row = row
        prev_row = row
    intervals.append(
        {
            "start_step": int(start_row["step"]),
            "end_step": int(prev_row["step"]),
            "start_hour": float(start_row["hour"]),
            "end_hour": float(prev_row["hour"]),
            "topology_motif_id": int(start_row["topology_motif_id"]),
            "right_node": "" if pd.isna(start_row["right_node"]) else int(start_row["right_node"]),
            "symbol": str(start_row["symbol"]),
            "steps": int((int(prev_row["step"]) - int(start_row["step"])) / stride + 1) if stride else 1,
        }
    )
    return intervals


def write_intervals(path: Path, intervals: list[dict[str, object]]) -> None:
    fieldnames = ["start_step", "end_step", "start_hour", "end_hour", "topology_motif_id", "right_node", "symbol", "steps"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(intervals)


def plot_node(df: pd.DataFrame, out_path: Path, *, node: int, working_mode: str, show_zero_betweenness: bool) -> None:
    x = df["hour"].to_numpy(dtype=np.float64)
    hop = df["hop_betweenness"].to_numpy(dtype=np.float64)
    delay = df["delay_betweenness"].to_numpy(dtype=np.float64)
    working = df[f"working_{working_mode}"].to_numpy(dtype=bool)
    right = df["right_node"].to_numpy(dtype=np.float64)
    topology = df["topology_motif_id"].to_numpy(dtype=np.int16)

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 8.8), dpi=170, sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(x, topology, color="#334155", linewidth=1.25, drawstyle="steps-post")
    ax0.set_ylabel("motif")
    ax0.set_yticks(sorted(set(int(v) for v in topology)))
    ax0.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax0.set_title(f"Node {int(node)} ideal splice right-link state, working={working_mode}")

    valid_right = np.isfinite(right)
    ax1.plot(x[valid_right], right[valid_right], color="#1f2937", linewidth=1.2, drawstyle="steps-post")
    ax1.scatter(x[working], right[working], s=15, color="#dc2626", alpha=0.88, label="working")
    ax1.set_ylabel("right node")
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax1.legend(loc="upper right")

    if bool(show_zero_betweenness):
        ax2.plot(x, hop, color="#2563eb", linewidth=1.05, alpha=0.75, label="hop betweenness")
        ax2.plot(x, delay, color="#dc2626", linewidth=1.05, alpha=0.75, label="delay betweenness")
    else:
        hop_plot = np.where(hop > 0, hop, np.nan)
        delay_plot = np.where(delay > 0, delay, np.nan)
        ax2.plot(x, hop_plot, color="#2563eb", linewidth=1.05, alpha=0.75, label="hop betweenness > 0")
        ax2.plot(x, delay_plot, color="#dc2626", linewidth=1.05, alpha=0.75, label="delay betweenness > 0")
    ax2.fill_between(x, 0, np.nanmax(np.r_[hop, delay, 1.0]), where=working, color="#fca5a5", alpha=0.18, step="post")
    ax2.set_ylabel("edge usage")
    ax2.set_xlabel("time (hour)")
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax2.legend(loc="upper right")

    switch_step = int(df.attrs.get("switch_step", 36000))
    return_switch_step = int(df.attrs.get("return_switch_step", 54000))
    for ax in axes:
        ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    store = load_store(args.store_dir)
    df = node_state_dataframe(store, int(args.node))
    df.attrs["switch_step"] = int(store.meta.get("switch_step", 36000))
    df.attrs["return_switch_step"] = int(store.meta.get("return_switch_step", 54000))

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.store_dir) / "node_queries"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"node_{int(args.node):04d}_right_state.csv"
    intervals_path = out_dir / f"node_{int(args.node):04d}_working_{args.working_mode}_intervals.csv"
    png_path = out_dir / f"node_{int(args.node):04d}_working_{args.working_mode}.png"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    intervals = contiguous_intervals(df, f"working_{args.working_mode}")
    write_intervals(intervals_path, intervals)
    plot_node(
        df,
        png_path,
        node=int(args.node),
        working_mode=str(args.working_mode),
        show_zero_betweenness=bool(args.show_zero_betweenness),
    )
    if bool(args.open):
        os.startfile(str(png_path))

    print(f"store_dir={Path(args.store_dir)}")
    print(f"node={int(args.node)} working_mode={args.working_mode}")
    print(f"csv={csv_path}")
    print(f"intervals={intervals_path}")
    print(f"plot={png_path}")
    print(f"working_steps={int(df[f'working_{args.working_mode}'].sum())}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
