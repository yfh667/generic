from __future__ import annotations
from typing import Mapping, Sequence, Iterable, Tuple, Dict, Any, List, Union, Set
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TimeKey = int
Node = int
Snapshot = Mapping[Node, Any]
TimeSeries = Union[Mapping[TimeKey, Snapshot], Sequence[Snapshot]]

# ---------- helpers ----------
def _iter_times_and_snapshots(ts: TimeSeries, start_time: int | None) -> Iterable[Tuple[int, Snapshot]]:
    if isinstance(ts, Mapping):
        for t in sorted(ts.keys()):
            yield int(t), ts[t]
    elif isinstance(ts, Sequence):
        if start_time is None:
            start_time = 0
        for i, snap in enumerate(ts):
            yield start_time + i, snap
    else:
        raise TypeError("Unsupported container for time series.")

def _neighbors_from(obj: Any):
    if obj is None: return []
    if isinstance(obj, Mapping): return obj.keys()
    if isinstance(obj, (set, list, tuple)): return obj
    if isinstance(obj, (int, np.integer)): return [int(obj)]
    try: return list(obj)
    except Exception: return []

def count_unique_edges(snapshot: Snapshot, undirected: bool = True) -> int:
    if not isinstance(snapshot, Mapping):
        raise TypeError("Snapshot must be a mapping {basicSa: neighbors}.")
    seen: Set[Tuple[Node, Node]] = set()
    c = 0
    for u, nbrs in snapshot.items():
        u = int(u)
        for v in _neighbors_from(nbrs):
            v = int(v)
            if u == v:
                continue
            if undirected:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in seen:
                    seen.add((a, b)); c += 1
            else:
                c += 1
    return c

def series_edge_counts(ts: TimeSeries, start_time: int | None = None, undirected: bool = True) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for t, snap in _iter_times_and_snapshots(ts, start_time):
        out[int(t)] = count_unique_edges(snap, undirected=undirected)
    return out

def moving_average(y, w: int | None):
    if not w or w <= 1:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(w) / w
    return np.convolve(ypad, ker, mode="valid")

# ---------- main function ----------
def plot_all_vs_pending_timeseries(
    all_edges: TimeSeries,
    pending_edges: TimeSeries,
    *,
    # 数据选项
    start_time_all: int | None = None,
    start_time_pending: int | None = None,
    undirected: bool = True,
    smooth_window: int | None = None,
    # 画图样式
    figsize=(12, 5),
    line_width=1.8,
    style_all="-",
    style_pending="--",
    color_all=None,            # None = 由 matplotlib 自选
    color_pending=None,
    title="All vs Pending Edges over Time (dual y-axes)",
    rc_override: dict | None = None,
    # 显示/保存
    show=True,                 # Jupyter 非阻塞
    save=None,                 # None/False 不保存；True=默认保存；或传 str/Path/列表
    save_dir="figs",
    basename="all_vs_pending",
    formats=("png","pdf"),
    dpi=300,
    transparent=False,
    # 导出数据
    csv_path: str | Path | None = None,
    # 返回
    return_handles=True        # 返回 (fig, (axL, axR), df, (avg_all, avg_pen))
):
    """
    绘制：已建链(all_edges) vs 正在建链(pending_edges) 的时间序列（双 y 轴）。
    all_edges/pending_edges: dict{t: {basicSa: set(dst)}} 或 list[ snapshot ]
    """
    import pandas as pd
    from matplotlib.ticker import FormatStrFormatter

    # 1) 计数
    c_all = series_edge_counts(all_edges,     start_time=start_time_all,     undirected=undirected)
    c_pen = series_edge_counts(pending_edges, start_time=start_time_pending, undirected=undirected)

    if not c_all and not c_pen:
        raise ValueError("Both series are empty.")

    # 2) 对齐时间轴
    times = sorted(set(c_all) | set(c_pen))
    y_all = np.array([c_all.get(t, 0) for t in times], dtype=float)
    y_pen = np.array([c_pen.get(t, 0) for t in times], dtype=float)

    y_all_s = moving_average(y_all, smooth_window)
    y_pen_s = moving_average(y_pen, smooth_window)

    # 平均值（对当前绘制范围 & 平滑后序列无关）
    avg_all = float(np.mean(y_all)) if len(y_all) else 0.0
    avg_pen = float(np.mean(y_pen)) if len(y_pen) else 0.0

    # 3) 画图（Jupyter 非阻塞）
    base_rc = {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
    }
    if rc_override:
        base_rc.update(rc_override)

    plt.ion()
    with mpl.rc_context(base_rc):
        fig, axL = plt.subplots(figsize=figsize)
        axR = axL.twinx()

        ln1 = axL.plot(times, y_all_s, linewidth=line_width, linestyle=style_all,
                       color=color_all, label="all_edges (built)")
        ln2 = axR.plot(times, y_pen_s, linewidth=line_width, linestyle=style_pending,
                       color=color_pending, label="pending_edges (building)")

        axL.set_title(title)
        axL.set_xlabel("Time")
        axL.set_ylabel("all_edges (unique edges)", color=ln1[0].get_color())
        axR.set_ylabel("pending_edges (unique edges)", color=ln2[0].get_color())

        # y 轴颜色 & 脊线
        axL.tick_params(axis="y", colors=ln1[0].get_color())
        axR.tick_params(axis="y", colors=ln2[0].get_color())
        axL.spines["left"].set_color(ln1[0].get_color())
        axR.spines["right"].set_color(ln2[0].get_color())

        axL.grid(True, linestyle="--", alpha=0.35)
        axL.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axR.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        # 合并图例
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        axL.legend(lines, labels, loc="best")

        # —— 平均值标注（两条水平虚线 + 右上角文本）——
        axL.axhline(avg_all, ls="--", lw=1.0, alpha=0.8, color=ln1[0].get_color())
        axR.axhline(avg_pen, ls="--", lw=1.0, alpha=0.8, color=ln2[0].get_color())
        axL.text(0.98, 0.98, f"avg(all)={avg_all:.2f}", transform=axL.transAxes,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.85))
        axR.text(0.98, 0.90, f"avg(pending)={avg_pen:.2f}", transform=axR.transAxes,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.85))

        plt.tight_layout()
        if show:
            plt.show(block=False)
            try: plt.pause(0.01)
            except Exception: pass

    # 4) 数据导出（可选 CSV）
    df = pd.DataFrame({
        "t": times,
        "all_edges": y_all.astype(int),
        "pending_edges": y_pen.astype(int),
        "all_edges_smooth": y_all_s,
        "pending_edges_smooth": y_pen_s
    })
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    # 5) 保存（可选）
    if save:
        targets = []
        if save is True:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for ext in formats:
                targets.append(Path(save_dir) / f"{basename}.{ext.lstrip('.')}")
        else:
            if isinstance(save, (str, Path)):
                targets = [save]
            else:
                targets = list(save)
        for p in targets:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent)

    print(f"[all_vs_pending] steps={len(times)}  avg(all)={avg_all:.2f}  avg(pending)={avg_pen:.2f}")
    return (fig, (axL, axR), df, (avg_all, avg_pen)) if return_handles else None
