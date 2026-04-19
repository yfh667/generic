from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import src.model.get_intra_inter_link as get_intra_inter_link
import src.paper3_postprocess.read_path_csv as read_path_csv
import src.paper3_postprocess.route_reliable as route_reliable
import src.paper3_postprocess.route_statistic as route_statistic


def enrich_pair_df_with_reliability(
    df,
    *,
    N,
    p_intra=0.999,
    p_inter=0.99,
    rel_col="rel_0999_099",
    path_col="path",
):
    """
    对单个 pair 的时序 df 补齐:
    - intra_links / inter_links
    - intra_hops / inter_hops / total_hops
    - rel_col
    """
    df = df.copy()

    all_intra = []
    all_inter = []

    for path_str in df[path_col]:
        if isinstance(path_str, str) and path_str.strip():
            intra, inter = get_intra_inter_link.parse_path_links(path_str, N=N)
        else:
            intra, inter = [], []

        all_intra.append(intra)
        all_inter.append(inter)

    df["intra_links"] = all_intra
    df["inter_links"] = all_inter
    df["intra_hops"] = df["intra_links"].apply(len)
    df["inter_hops"] = df["inter_links"].apply(len)
    df["total_hops"] = df["intra_hops"] + df["inter_hops"]

    df[rel_col] = route_reliable.compute_route_reliability_series(
        df,
        p_intra=p_intra,
        p_inter=p_inter,
        intra_col="intra_hops",
        inter_col="inter_hops",
        name=rel_col,
    )

    # 空路径默认视为缺失，不参与 mean/p05 统计
    empty_mask = ~df[path_col].fillna("").astype(str).str.strip().astype(bool)
    df.loc[empty_mask, rel_col] = np.nan

    return df


def build_stationpair_reliability_stat_table(
    csv_dir,
    *,
    N,
    p_intra=0.999,
    p_inter=0.99,
    rel_col="rel_0999_099",
    save_dir=None,
    basename="stationpair_reliability_stat",
):
    """
    批量读取一个目录下的 pair CSV，
    复用 route_reliable + route_statistic，
    生成每个 station pair 的全局统计表。
    """
    csv_dir = Path(csv_dir)
    rows = []

    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = read_path_csv.read_pair_csv(csv_path)
        if df.empty:
            continue

        station_a = int(df["station_a"].iloc[0])
        station_b = int(df["station_b"].iloc[0])

        df_enriched = enrich_pair_df_with_reliability(
            df,
            N=N,
            p_intra=p_intra,
            p_inter=p_inter,
            rel_col=rel_col,
        )

        global_stat = route_statistic.reliability_global_stats(
            df_enriched,
            rel_col=rel_col,
        )

        rec = {
            "pair_csv_name": csv_path.name,
            "station_a": station_a,
            "station_b": station_b,
        }
        rec.update(global_stat.to_dict())
        rows.append(rec)

    pair_stat_df = (
        pd.DataFrame(rows)
        .sort_values(["station_a", "station_b"])
        .reset_index(drop=True)
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pair_stat_df.to_csv(
            save_dir / f"{basename}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return pair_stat_df


def build_stationpair_metric_matrix(
    pair_stat_df,
    *,
    metric="mean",
    station_a_order=None,
    station_b_order=None,
):
    """
    把 pair 统计表转成热力图矩阵。
    """
    if metric not in pair_stat_df.columns:
        raise ValueError(f"metric={metric!r} 不存在，可选列有: {pair_stat_df.columns.tolist()}")

    if station_a_order is None:
        station_a_order = sorted(pair_stat_df["station_a"].dropna().unique().tolist())
    if station_b_order is None:
        station_b_order = sorted(pair_stat_df["station_b"].dropna().unique().tolist())

    mat = pd.DataFrame(
        np.nan,
        index=station_a_order,
        columns=station_b_order,
        dtype=float,
    )

    for row in pair_stat_df.itertuples(index=False):
        mat.loc[int(row.station_a), int(row.station_b)] = float(getattr(row, metric))

    return mat


def plot_stationpair_metric_heatmap(
    pair_stat_df,
    *,
    metric="mean",
    station_a_order=None,
    station_b_order=None,
    figsize=(12, 8),
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    annotate=False,
    fmt=".3f",
    x_group_breaks=None,
    y_group_breaks=None,
    show=True,
    save=False,
    save_dir="figs/heatmap",
    basename="stationpair_metric_heatmap",
    dpi=300,
    return_handles=True,
):
    """
    从 pair_stat_df 中取 metric 画热力图。
    同时把矩阵也保存成 csv，方便后续论文复用。
    """
    mat = build_stationpair_metric_matrix(
        pair_stat_df,
        metric=metric,
        station_a_order=station_a_order,
        station_b_order=station_b_order,
    )

    data = mat.to_numpy(dtype=float)

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
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
        title = f"Station-pair {metric} reliability matrix"
    ax.set_title(title)

    # 分块边界，可选；传的是“切分位置的索引”，不是 station id
    if x_group_breaks:
        for xb in x_group_breaks:
            ax.axvline(x=xb - 0.5, color="white", linewidth=1.5)
    if y_group_breaks:
        for yb in y_group_breaks:
            ax.axhline(y=yb - 0.5, color="white", linewidth=1.5)

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
    cbar.set_label(f"{metric} reliability")

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
