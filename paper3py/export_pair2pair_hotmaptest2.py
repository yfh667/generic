from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PAIR_RE = re.compile(
    r"^(?P<region_a>region\d+)--station(?P<station_a>\d+)-(?P<region_b>region\d+)--station(?P<station_b>\d+)\.csv$"
)


def load_all_pair_global_stat(csv_path):
    """
    读取 export_all_pair_global_stat 导出的总表，
    并从 PAIR_CSV_NAME 中解析 region/station 信息。
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "PAIR_CSV_NAME" not in df.columns:
        raise ValueError("缺少列: PAIR_CSV_NAME")

    parsed = df["PAIR_CSV_NAME"].astype(str).str.extract(PAIR_RE)

    bad = parsed.isna().any(axis=1)
    if bad.any():
        bad_names = df.loc[bad, "PAIR_CSV_NAME"].tolist()[:10]
        raise ValueError(f"这些文件名解析失败: {bad_names}")

    out = df.copy()
    out["region_a"] = parsed["region_a"]
    out["station_a"] = parsed["station_a"].astype(int)
    out["region_b"] = parsed["region_b"]
    out["station_b"] = parsed["station_b"].astype(int)

    return out


def build_pair_metric_matrix(
    df,
    *,
    metric="mean",
    region_a=None,
    region_b=None,
    station_a_order=None,
    station_b_order=None,
):
    """
    从 all_pair_global_stat 总表构造矩阵。
    可选按某个 region pair 过滤。
    """
    if metric not in df.columns:
        raise ValueError(f"metric={metric!r} 不存在，可选列有: {df.columns.tolist()}")

    sub = df.copy()

    if region_a is not None:
        sub = sub[sub["region_a"] == region_a]
    if region_b is not None:
        sub = sub[sub["region_b"] == region_b]

    if sub.empty:
        raise ValueError("过滤后没有数据，请检查 region_a / region_b")

    mat = sub.pivot(
        index="station_a",
        columns="station_b",
        values=metric,
    )

    if station_a_order is None:
        station_a_order = sorted(sub["station_a"].unique().tolist())
    if station_b_order is None:
        station_b_order = sorted(sub["station_b"].unique().tolist())

    mat = mat.reindex(index=station_a_order, columns=station_b_order)
    return mat.astype(float)


def plot_pair_metric_heatmap(
    df,
    *,
    metric="mean",
    region_a=None,
    region_b=None,
    station_a_order=None,
    station_b_order=None,
    figsize=(10, 6),
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    annotate=False,
    fmt=".3f",
    show=True,
    save=False,
    save_dir="figs/heatmap",
    basename="pair_metric_heatmap",
    dpi=300,
    return_handles=True,
):
    """
    从 export_all_pair_global_stat 总表直接绘制热力图。
    """
    mat = build_pair_metric_matrix(
        df,
        metric=metric,
        region_a=region_a,
        region_b=region_b,
        station_a_order=station_a_order,
        station_b_order=station_b_order,
    )

    data = mat.to_numpy(dtype=float)

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), rotation=90)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist())

    ax.set_xlabel("station_b")
    ax.set_ylabel("station_a")

    if title is None:
        if region_a is not None and region_b is not None:
            title = f"{region_a} -> {region_b}: {metric}"
        else:
            title = f"All station-pair {metric}"
    ax.set_title(title)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, format(val, fmt),
                        ha="center", va="center",
                        fontsize=7, color="black",
                    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    plt.tight_layout()

    if save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{basename}.png", dpi=dpi, bbox_inches="tight")
        mat.to_csv(save_dir / f"{basename}_matrix.csv", encoding="utf-8-sig")

    if show:
        plt.show(block=False)
        try:
            plt.pause(0.01)
        except Exception:
            pass

    return (fig, ax, mat) if return_handles else None

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_two_metric_mats(
    df,
    *,
    mean_metric="mean",
    hop_metric="hop_mean",   # 改成你真实的跳数列名
    region_a=None,
    region_b=None,
    station_a_order=None,
    station_b_order=None,
):
    mean_mat = build_pair_metric_matrix(
        df,
        metric=mean_metric,
        region_a=region_a,
        region_b=region_b,
        station_a_order=station_a_order,
        station_b_order=station_b_order,
    )
    hop_mat = build_pair_metric_matrix(
        df,
        metric=hop_metric,
        region_a=region_a,
        region_b=region_b,
        station_a_order=station_a_order if station_a_order is not None else mean_mat.index.tolist(),
        station_b_order=station_b_order if station_b_order is not None else mean_mat.columns.tolist(),
    )
    hop_mat = hop_mat.reindex(index=mean_mat.index, columns=mean_mat.columns)
    return mean_mat, hop_mat


def save_intermediate_tables(mean_mat, hop_mat, out_dir, basename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_csv = out_dir / f"{basename}_mean_matrix.csv"
    hop_csv = out_dir / f"{basename}_hop_matrix.csv"
    long_csv = out_dir / f"{basename}_mean_hop_long.csv"

    mean_mat.to_csv(mean_csv, encoding="utf-8-sig")
    hop_mat.to_csv(hop_csv, encoding="utf-8-sig")

    long_mean = mean_mat.stack(dropna=False).rename("mean").reset_index()
    long_hop = hop_mat.stack(dropna=False).rename("hop").reset_index()
    long_df = long_mean.merge(long_hop, on=["station_a", "station_b"], how="outer")
    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")

    return long_df


def plot_mean_with_hop_overlay(
    mean_mat,
    hop_mat,
    *,
    title="mean(颜色) + hop(数字)",
    cmap="viridis",
    vmin=None,
    vmax=None,
    hop_fmt=".1f",
    figsize=(10, 6),
    save=False,
    out_dir=".",
    basename="mean_hop_overlay",
    dpi=300,
):
    mean_data = mean_mat.to_numpy(dtype=float)
    hop_data = hop_mat.to_numpy(dtype=float)

    if vmin is None:
        vmin = np.nanmin(mean_data)
    if vmax is None:
        vmax = np.nanmax(mean_data)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mean_data,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(mean_mat.columns)))
    ax.set_xticklabels(mean_mat.columns.tolist(), rotation=90)
    ax.set_yticks(np.arange(len(mean_mat.index)))
    ax.set_yticklabels(mean_mat.index.tolist())
    ax.set_xlabel("station_b")
    ax.set_ylabel("station_a")
    ax.set_title(title)

    # 格内叠加 hop
    for i in range(mean_data.shape[0]):
        for j in range(mean_data.shape[1]):
            h = hop_data[i, j]
            if np.isnan(h):
                continue
            m = mean_data[i, j]
            if np.isnan(m):
                txt_color = "black"
            else:
                r, g, b, _ = cmap_obj(norm(m))
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = "black" if luma > 0.55 else "white"
            ax.text(j, i, format(h, hop_fmt), ha="center", va="center", fontsize=6, color=txt_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean")
    plt.tight_layout()

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{basename}.png", dpi=dpi, bbox_inches="tight")

    plt.show(block=False)
    try:
        plt.pause(0.01)
    except Exception:
        pass

    return fig, ax


def plot_mean_hop_side_by_side(
    mean_mat,
    hop_mat,
    *,
    title_left="Mean Reliability",
    title_right="Hop Count",
    cmap_left="viridis",
    cmap_right="magma_r",
    figsize=(14, 6),
    save=False,
    out_dir=".",
    basename="mean_hop_side_by_side",
    dpi=300,
):
    mean_data = mean_mat.to_numpy(dtype=float)
    hop_data = hop_mat.to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    im1 = axes[0].imshow(mean_data, aspect="auto", origin="upper", interpolation="nearest", cmap=cmap_left)
    axes[0].set_title(title_left)
    axes[0].set_xlabel("station_b")
    axes[0].set_ylabel("station_a")
    axes[0].set_xticks(np.arange(len(mean_mat.columns)))
    axes[0].set_xticklabels(mean_mat.columns.tolist(), rotation=90)
    axes[0].set_yticks(np.arange(len(mean_mat.index)))
    axes[0].set_yticklabels(mean_mat.index.tolist())
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label("mean")

    im2 = axes[1].imshow(hop_data, aspect="auto", origin="upper", interpolation="nearest", cmap=cmap_right)
    axes[1].set_title(title_right)
    axes[1].set_xlabel("station_b")
    axes[1].set_xticks(np.arange(len(mean_mat.columns)))
    axes[1].set_xticklabels(mean_mat.columns.tolist(), rotation=90)
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label("hop")

    plt.tight_layout()

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{basename}.png", dpi=dpi, bbox_inches="tight")

    plt.show(block=False)
    try:
        plt.pause(0.01)
    except Exception:
        pass

    return fig, axes


Topology_Version = 'grid_plane_alternating'


Topology_DIR = 'topology_design'
P_INTRA = 0.999
P_INTER = 0.99
P = 18
N = 36

# 2) 确保 viewer 有全局引用，避免 GC 回收导致崩溃

DATA_DIR = Path(r"D:\paper3")
BASEDIR =  DATA_DIR / "data"
# data_DIR = Path(FIGURE_DIR) / "region1_to_region2_0_100"   # 你导出的原始 csv 目录
basic_probability_dir = 'analysis_link'

hotmap_dir = BASEDIR/Topology_DIR/Topology_Version/basic_probability_dir




summary_csv = hotmap_dir / f"pairs_stat_intra_{P_INTRA}_inter_{P_INTER}.csv"

df_stat = load_all_pair_global_stat(summary_csv)

print(df_stat.head())
print(df_stat[["PAIR_CSV_NAME", "region_a", "station_a", "region_b", "station_b"]].head())
fig, ax, mat = plot_pair_metric_heatmap(
    df_stat,
    metric="mean",
    region_a="region1",
    region_b="region2",
    title="region1 -> region2: mean reliability",
    vmin=0.90,
    vmax=1.00,
    show=True,
    save=True,
    save_dir=hotmap_dir,
    basename="region1_region2_mean_reliability",
)





#######
# 先确认跳数列名
print(df_stat.columns.tolist())

hop_metric = "hop_mean"  # 改成实际列名，比如 "avg_hop" / "mean_hop" 等
mean_mat, hop_mat = build_two_metric_mats(
    df_stat,
    mean_metric="mean",
    hop_metric=hop_metric,
    region_a="region1",
    region_b="region2",
)

# 持久化中间结果（你要求的中间过程文件）
long_df = save_intermediate_tables(
    mean_mat, hop_mat,
    out_dir=hotmap_dir,
    basename="region1_region2"
)
print(long_df.head())

# 修补版：一张图同时看 mean+hop
plot_mean_with_hop_overlay(
    mean_mat, hop_mat,
    title="region1 -> region2: mean(颜色) + hop(数字)",
    vmin=0.90, vmax=1.00,
    save=True,
    out_dir=hotmap_dir,
    basename="region1_region2_mean_with_hop_overlay",
)

# 新表达：并排双图
plot_mean_hop_side_by_side(
    mean_mat, hop_mat,
    title_left="region1->region2 Mean Reliability",
    title_right="region1->region2 Hop Count",
    save=True,
    out_dir=hotmap_dir,
    basename="region1_region2_mean_hop_side_by_side",
)
