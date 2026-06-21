from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CACHE_DIR = Path(r"E:\paper11\data\linshi\g60_full_link_multi_region_weighted_betweenness_t0_86164_stride1")
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\full_link_weighted_betweenness_three_pairs"
)

PAIRS = {
    "china_europe": "China-Europe",
    "china_america": "China-America",
    "china_africa": "China-Africa",
}

PAIR_COLORS = {
    "china_europe": "#2563EB",
    "china_america": "#DC2626",
    "china_africa": "#16A34A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot full-link weighted shortest-delay edge betweenness time series "
            "for China-Europe, China-America, and China-Africa."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--heatmap-stride", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_edges(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in (
            "edge_idx",
            "src_node",
            "dst_node",
            "src_sat_id",
            "dst_sat_id",
            "src_plane",
            "src_y",
            "dst_plane",
            "dst_y",
            "option",
        ):
            row[key] = int(row[key])
    return rows


def compute_or_load_summary(
    *,
    arr: np.ndarray,
    steps: np.ndarray,
    pair_key: str,
    pair_label: str,
    out_dir: Path,
    chunk_size: int,
    force: bool,
) -> tuple[Path, dict[str, np.ndarray]]:
    summary_path = out_dir / f"{pair_key}_step_summary.csv"
    stats_npz = out_dir / f"{pair_key}_step_summary_arrays.npz"
    if summary_path.exists() and stats_npz.exists() and not force:
        data = np.load(stats_npz)
        return summary_path, {key: np.asarray(data[key]) for key in data.files}

    n_steps = int(arr.shape[0])
    max_edge = np.zeros(n_steps, dtype=np.float32)
    nonzero_edges = np.zeros(n_steps, dtype=np.int16)
    edge_value_sum = np.zeros(n_steps, dtype=np.float32)
    p95_nonzero = np.zeros(n_steps, dtype=np.float32)
    p99_nonzero = np.zeros(n_steps, dtype=np.float32)

    for start in range(0, n_steps, int(chunk_size)):
        end = min(start + int(chunk_size), n_steps)
        block = np.asarray(arr[start:end], dtype=np.float32)
        max_edge[start:end] = np.max(block, axis=1)
        nonzero_edges[start:end] = np.count_nonzero(block > 0.0, axis=1).astype(np.int16)
        edge_value_sum[start:end] = np.sum(block, axis=1, dtype=np.float64).astype(np.float32)
        for offset, row in enumerate(block):
            nz = row[row > 0.0]
            if nz.size:
                p95_nonzero[start + offset] = float(np.percentile(nz, 95))
                p99_nonzero[start + offset] = float(np.percentile(nz, 99))
        print(f"[{pair_key}] summary rows {end}/{n_steps}", flush=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "hour",
                "pair",
                "pair_label",
                "max_edge_betweenness",
                "nonzero_edges",
                "edge_value_sum",
                "p95_nonzero_edge_betweenness",
                "p99_nonzero_edge_betweenness",
            ],
        )
        writer.writeheader()
        for idx, step in enumerate(steps):
            writer.writerow(
                {
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "pair": pair_key,
                    "pair_label": pair_label,
                    "max_edge_betweenness": float(max_edge[idx]),
                    "nonzero_edges": int(nonzero_edges[idx]),
                    "edge_value_sum": float(edge_value_sum[idx]),
                    "p95_nonzero_edge_betweenness": float(p95_nonzero[idx]),
                    "p99_nonzero_edge_betweenness": float(p99_nonzero[idx]),
                }
            )
    np.savez_compressed(
        stats_npz,
        steps=np.asarray(steps, dtype=np.int32),
        max_edge=max_edge,
        nonzero_edges=nonzero_edges,
        edge_value_sum=edge_value_sum,
        p95_nonzero=p95_nonzero,
        p99_nonzero=p99_nonzero,
    )
    return summary_path, {
        "steps": np.asarray(steps, dtype=np.int32),
        "max_edge": max_edge,
        "nonzero_edges": nonzero_edges,
        "edge_value_sum": edge_value_sum,
        "p95_nonzero": p95_nonzero,
        "p99_nonzero": p99_nonzero,
    }


def compute_top_edges(
    *,
    arr: np.ndarray,
    steps: np.ndarray,
    edges: list[dict[str, Any]],
    pair_key: str,
    pair_label: str,
    out_dir: Path,
    chunk_size: int,
    top_k: int,
    force: bool,
) -> tuple[Path, np.ndarray]:
    top_path = out_dir / f"{pair_key}_top_edges.csv"
    idx_path = out_dir / f"{pair_key}_top_edge_indices.npy"
    if top_path.exists() and idx_path.exists() and not force:
        return top_path, np.load(idx_path)

    n_steps, n_edges = int(arr.shape[0]), int(arr.shape[1])
    total = np.zeros(n_edges, dtype=np.float64)
    peak = np.zeros(n_edges, dtype=np.float32)
    peak_step = np.zeros(n_edges, dtype=np.int32)
    active_rows = np.zeros(n_edges, dtype=np.int32)

    for start in range(0, n_steps, int(chunk_size)):
        end = min(start + int(chunk_size), n_steps)
        block = np.asarray(arr[start:end], dtype=np.float32)
        total += np.sum(block, axis=0, dtype=np.float64)
        block_peak = np.max(block, axis=0)
        better = block_peak > peak
        if np.any(better):
            rel = np.argmax(block[:, better], axis=0)
            peak[better] = block_peak[better]
            peak_step[better] = steps[start + rel].astype(np.int32)
        active_rows += np.count_nonzero(block > 0.0, axis=0).astype(np.int32)
        print(f"[{pair_key}] top-edge rows {end}/{n_steps}", flush=True)

    order = np.argsort(-total)
    top_indices = order[: int(top_k)].astype(np.int32)
    np.save(idx_path, top_indices)

    with top_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "rank",
            "pair",
            "pair_label",
            "edge_idx",
            "src_node",
            "dst_node",
            "src_sat_id",
            "dst_sat_id",
            "src_plane",
            "src_y",
            "dst_plane",
            "dst_y",
            "option",
            "total_betweenness",
            "mean_betweenness",
            "peak_betweenness",
            "peak_step",
            "peak_hour",
            "active_fraction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, edge_idx in enumerate(top_indices, 1):
            edge = dict(edges[int(edge_idx)])
            writer.writerow(
                {
                    "rank": int(rank),
                    "pair": pair_key,
                    "pair_label": pair_label,
                    **edge,
                    "total_betweenness": float(total[int(edge_idx)]),
                    "mean_betweenness": float(total[int(edge_idx)] / n_steps),
                    "peak_betweenness": float(peak[int(edge_idx)]),
                    "peak_step": int(peak_step[int(edge_idx)]),
                    "peak_hour": float(peak_step[int(edge_idx)]) / 3600.0,
                    "active_fraction": float(active_rows[int(edge_idx)] / n_steps),
                }
            )
    return top_path, top_indices


def plot_summary(stats_by_pair: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(16.0, 9.0), dpi=180, sharex=True)
    metrics = [
        ("max_edge", "max edge betweenness"),
        ("nonzero_edges", "nonzero edges"),
        ("edge_value_sum", "sum of edge betweenness"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for pair_key, label in PAIRS.items():
            stats = stats_by_pair[pair_key]
            x = np.asarray(stats["steps"], dtype=np.float64) / 3600.0
            y = np.asarray(stats[metric], dtype=np.float64)
            ax.plot(
                x,
                y,
                color=PAIR_COLORS[pair_key],
                linewidth=0.9,
                label=f"{label} | mean={np.nanmean(y):.2f}",
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("time (hour)")
    fig.suptitle("G60 full-link weighted edge betweenness over time", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_edge_heatmaps(
    *,
    cache_dir: Path,
    out_dir: Path,
    steps: np.ndarray,
    top_indices_by_pair: dict[str, np.ndarray],
    heatmap_stride: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(16.0, 9.2), dpi=180, sharex=True)
    for ax, (pair_key, label) in zip(axes, PAIRS.items()):
        arr = np.load(cache_dir / pair_key / "edge_betweenness.npy", mmap_mode="r")
        sample_rows = np.arange(0, int(arr.shape[0]), int(heatmap_stride), dtype=np.int64)
        top_indices = top_indices_by_pair[pair_key]
        heat = np.asarray(arr[np.ix_(sample_rows, top_indices)], dtype=np.float32).T
        im = ax.imshow(
            np.log1p(heat),
            aspect="auto",
            interpolation="nearest",
            cmap="Reds",
            extent=[
                float(steps[int(sample_rows[0])]) / 3600.0,
                float(steps[int(sample_rows[-1])]) / 3600.0,
                int(top_indices.size) + 0.5,
                0.5,
            ],
        )
        ax.set_ylabel(f"{label}\ntop edge rank")
        ax.set_yticks([1, 10, 20, 30, 40, 50])
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
        cbar.set_label("log1p(edge betweenness)")
    axes[-1].set_xlabel("time (hour)")
    fig.suptitle("G60 full-link weighted edge betweenness: top-50 edge heatmaps", y=0.995)
    fig.tight_layout()
    out_path = out_dir / "full_link_weighted_betweenness_top50_heatmaps.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pair_top10(
    *,
    cache_dir: Path,
    out_dir: Path,
    steps: np.ndarray,
    top_indices_by_pair: dict[str, np.ndarray],
    edges: list[dict[str, Any]],
    heatmap_stride: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    sample_rows = np.arange(0, int(steps.size), int(heatmap_stride), dtype=np.int64)
    x = np.asarray(steps[sample_rows], dtype=np.float64) / 3600.0
    for pair_key, label in PAIRS.items():
        arr = np.load(cache_dir / pair_key / "edge_betweenness.npy", mmap_mode="r")
        top10 = top_indices_by_pair[pair_key][:10]
        fig, ax = plt.subplots(figsize=(15.8, 6.4), dpi=180)
        for rank, edge_idx in enumerate(top10, 1):
            edge = edges[int(edge_idx)]
            y = np.asarray(arr[sample_rows, int(edge_idx)], dtype=np.float32)
            edge_label = (
                f"#{rank} e{edge_idx} "
                f"({edge['src_plane']},{edge['src_y']})-({edge['dst_plane']},{edge['dst_y']}) "
                f"opt={edge['option']}"
            )
            ax.plot(x, y, linewidth=0.95, label=edge_label)
        ax.set_title(f"{label}: full-link top-10 weighted edge betweenness")
        ax.set_xlabel("time (hour)")
        ax.set_ylabel("edge betweenness")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(fontsize=7.0, loc="best", ncol=2)
        fig.tight_layout()
        out_path = out_dir / f"{pair_key}_top10_edge_betweenness_timeseries.png"
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def write_readme(out_dir: Path, *, cache_dir: Path, summary_paths: list[Path], top_paths: list[Path], plot_paths: list[Path]) -> None:
    lines = [
        "# G60 Full-Link Weighted Edge Betweenness",
        "",
        "This folder visualizes cached full-link weighted shortest-delay edge betweenness.",
        "",
        f"Source cache: `{cache_dir}`",
        "",
        "Topology: full inter-plane options 0/1/2/4 plus intra-plane y-ring.",
        "Metric: one deterministic shortest-delay path is counted for every source-target satellite pair between groups.",
        "",
        "## Outputs",
        "",
    ]
    for path in summary_paths:
        lines.append(f"- Step summary: `{path.name}`")
    for path in top_paths:
        lines.append(f"- Top edge table: `{path.name}`")
    for path in plot_paths:
        lines.append(f"- Plot: `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = np.load(cache_dir / "time_indices.npy", mmap_mode="r")
    edges = read_edges(cache_dir / "edges.csv")
    summary_paths: list[Path] = []
    top_paths: list[Path] = []
    stats_by_pair: dict[str, dict[str, np.ndarray]] = {}
    top_indices_by_pair: dict[str, np.ndarray] = {}

    for pair_key, label in PAIRS.items():
        arr = np.load(cache_dir / pair_key / "edge_betweenness.npy", mmap_mode="r")
        if arr.shape[0] != steps.shape[0]:
            raise ValueError(f"{pair_key} steps mismatch: {arr.shape[0]} vs {steps.shape[0]}")
        if arr.shape[1] != len(edges):
            raise ValueError(f"{pair_key} edge count mismatch: {arr.shape[1]} vs {len(edges)}")
        summary_path, stats = compute_or_load_summary(
            arr=arr,
            steps=steps,
            pair_key=pair_key,
            pair_label=label,
            out_dir=out_dir,
            chunk_size=int(args.chunk_size),
            force=bool(args.force),
        )
        top_path, top_indices = compute_top_edges(
            arr=arr,
            steps=steps,
            edges=edges,
            pair_key=pair_key,
            pair_label=label,
            out_dir=out_dir,
            chunk_size=int(args.chunk_size),
            top_k=int(args.top_k),
            force=bool(args.force),
        )
        summary_paths.append(summary_path)
        top_paths.append(top_path)
        stats_by_pair[pair_key] = stats
        top_indices_by_pair[pair_key] = top_indices

    plot_paths: list[Path] = []
    summary_plot = out_dir / "full_link_weighted_betweenness_step_summary.png"
    plot_summary(stats_by_pair, summary_plot)
    plot_paths.append(summary_plot)
    plot_paths.append(
        plot_top_edge_heatmaps(
            cache_dir=cache_dir,
            out_dir=out_dir,
            steps=steps,
            top_indices_by_pair=top_indices_by_pair,
            heatmap_stride=int(args.heatmap_stride),
        )
    )
    plot_paths.extend(
        plot_pair_top10(
            cache_dir=cache_dir,
            out_dir=out_dir,
            steps=steps,
            top_indices_by_pair=top_indices_by_pair,
            edges=edges,
            heatmap_stride=int(args.heatmap_stride),
        )
    )
    write_readme(out_dir, cache_dir=cache_dir, summary_paths=summary_paths, top_paths=top_paths, plot_paths=plot_paths)

    print(f"out_dir={out_dir}")
    for path in summary_paths:
        print(f"summary={path}")
    for path in top_paths:
        print(f"top_edges={path}")
    for path in plot_paths:
        print(f"plot={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
