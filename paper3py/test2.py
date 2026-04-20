from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _region_pair_sort_key(s: str):
    m = re.match(r"region(\d+)_to_region(\d+)", str(s))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999, 999)


def _find_best_timeseries_dir(motif_dir: Path):
    analysis = motif_dir / "analysis_link"
    if not analysis.exists():
        return None

    cands = []
    for p in analysis.glob("region_pair_prob_timeseries*"):
        d1 = p / "04_region_pair_timeseries"
        d2 = p / "timeseries"
        if d1.exists():
            cands.append(d1)
        if d2.exists():
            cands.append(d2)

    if not cands:
        return None
    cands = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def plot_cross_motif_regionpair_heatmap(
    topology_root,
    *,
    motifs=None,
    out_dir=None,
    metric_col="mean_reliability",
    annotate=True,
    cmap="YlGnBu",
    figsize=(11, 5),
    dpi=300,
):
    topology_root = Path(topology_root)
    if motifs is None:
        motifs = sorted([p.name for p in topology_root.iterdir() if p.is_dir()])

    if out_dir is None:
        out_dir = topology_root / "_cross_motif_regionpair_heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 00) 发现目录（中间结果）
    discovery = []
    for motif in motifs:
        motif_dir = topology_root / motif
        ts_dir = _find_best_timeseries_dir(motif_dir)
        n_files = len(list(ts_dir.glob("*_timeseries.csv"))) if ts_dir else 0
        discovery.append(
            {"motif": motif, "timeseries_dir": str(ts_dir) if ts_dir else "", "file_count": n_files}
        )
    df_discovery = pd.DataFrame(discovery)
    df_discovery.to_csv(out_dir / "00_discovery.csv", index=False, encoding="utf-8-sig")

    # 01) 长表（中间结果）
    rows = []
    for r in discovery:
        motif = r["motif"]
        ts_dir = Path(r["timeseries_dir"]) if r["timeseries_dir"] else None
        if ts_dir is None or not ts_dir.exists():
            continue

        for f in sorted(ts_dir.glob("*_timeseries.csv")):
            df = pd.read_csv(f)

            if {"region_a", "region_b"}.issubset(df.columns):
                region_pair = f"{df['region_a'].iloc[0]}_to_{df['region_b'].iloc[0]}"
            else:
                region_pair = f.stem.replace("_timeseries", "")

            if metric_col in df.columns:
                v = pd.to_numeric(df[metric_col], errors="coerce")
            elif {"rel_sum", "pair_cnt"}.issubset(df.columns):
                rel_sum = pd.to_numeric(df["rel_sum"], errors="coerce")
                pair_cnt = pd.to_numeric(df["pair_cnt"], errors="coerce").replace(0, np.nan)
                v = rel_sum / pair_cnt
            else:
                continue

            rows.append(
                {
                    "motif": motif,
                    "region_pair": region_pair,
                    "time_points": int(v.notna().sum()),
                    "mean_probability": float(v.mean()),
                    "src_file": str(f),
                }
            )

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        raise ValueError("没读到可用 timeseries。先检查 00_discovery.csv。")
    df_long.to_csv(out_dir / "01_long_mean_probability.csv", index=False, encoding="utf-8-sig")

    # 02) 矩阵（中间结果）
    mat = df_long.pivot(index="region_pair", columns="motif", values="mean_probability")
    mat = mat.reindex(index=sorted(mat.index.tolist(), key=_region_pair_sort_key))
    mat = mat.reindex(columns=motifs)
    mat.to_csv(out_dir / "02_heatmap_matrix.csv", encoding="utf-8-sig")

    # 03) 热力图
    data = mat.to_numpy(dtype=float)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap_obj)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist())

    ax.set_xlabel("motif")
    ax.set_ylabel("region_pair")
    ax.set_title("Mean Reliability Heatmap (Region Pair x Motif)")

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean probability")

    plt.tight_layout()
    fig.savefig(out_dir / "03_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.show(block=False)
    try:
        plt.pause(0.01)
    except Exception:
        pass

    return df_discovery, df_long, mat, fig, ax


# ===== 直接运行 =====
root = Path(r"D:\paper3\data\topology_design")
if not root.exists():
    root = Path(r"C:\user\data\topology_design")

motif_order = ["grid_four", "gridx", "grid_plane_alternating", "grid_x_sparse", "grid+"]
plot_cross_motif_regionpair_heatmap(root, motifs=motif_order)
