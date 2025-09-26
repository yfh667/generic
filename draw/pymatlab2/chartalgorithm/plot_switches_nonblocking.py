from __future__ import annotations
import math
from pathlib import Path
from pathlib import Path

def plot_pending_edges_timeseries(
    pending_edges: dict,
    *,
    max_t_seconds=None,
    fill_missing=True,
    line_width=2.0,
    figsize=(10, 5),
    title="Edges/sec (pending)",
    rc_override=None,
    show=True,                  # Jupyter 非阻塞显示
    save=None,                  # None/False 不保存；True 用默认（save_dir/basename/formats）；或传 路径/路径列表
    save_dir="figs",
    basename="pending_edges",
    formats=("png", "pdf"),
    dpi=300,
    transparent=False,
    annotate_avg=True,
    avg_line_kw=None,
    avg_text_kw=None,
    # === 新增 ===
    save_pixels=None,           # (width_px, height_px)，如 (3840,2160) 输出 4K；None 则按 figsize*dpi
    keep_display_size=True,     # True: 仅保存时改尺寸，保存后恢复；False: 直接改当前图尺寸
    pad_inches=0.02,            # 保存时边距
    return_handles=True
):
    """
    统计并绘制 pending_edges 的逐秒数量曲线，并（可选）在图上标注平均切换数。
    支持保存时指定确切像素尺寸（如 4K/8K）。
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    if not isinstance(pending_edges, dict) or not pending_edges:
        raise ValueError("pending_edges 为空或类型不是 dict。")

    def _k2i(x):
        try: return int(x)
        except: return None
    keys = sorted(s for s in (_k2i(k) for k in pending_edges.keys()) if s is not None)
    if not keys:
        raise ValueError("pending_edges 的 key 无法转换为整数。")
    t_min, t_max = keys[0], keys[-1]

    def count_edges_at_t(t):
        ed = pending_edges.get(t, {})
        if isinstance(ed, dict):
            s = 0
            for dsts in ed.values():
                if dsts is None: continue
                if isinstance(dsts, (set, list, tuple, dict)):
                    s += len(dsts)
                elif isinstance(dsts, (int, np.integer)):
                    s += int(dsts)
            return int(s)
        if isinstance(ed, (set, list, tuple, dict)): return int(len(ed))
        if isinstance(ed, (int, np.integer)):        return int(ed)
        return 0

    steps = list(range(t_min, t_max + 1)) if fill_missing else keys
    counts = [count_edges_at_t(t) for t in steps]

    if max_t_seconds is not None:
        m = int(max_t_seconds)
        steps, counts = zip(*[(t, c) for t, c in zip(steps, counts) if t <= m])
        steps, counts = list(steps), list(counts)

    import pandas as pd
    df = pd.DataFrame({"step": steps, "pending_edges": counts})
    avg_val = float(sum(counts) / len(counts)) if counts else 0.0

    base_rc = {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
    }
    if rc_override: base_rc.update(rc_override)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.ion()
    with mpl.rc_context(base_rc):
        fig, ax = plt.subplots(figsize=figsize)
        try: fig.canvas.manager.set_window_title(title)
        except Exception: pass

        ax.plot(steps, counts, linewidth=line_width)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of edges")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False, useMathText=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if title: ax.set_title(title)

        if annotate_avg:
            _line_kw = dict(ls="--", lw=1.2, alpha=0.9, color="gray")
            if avg_line_kw: _line_kw.update(avg_line_kw)
            ax.axhline(avg_val, **_line_kw)
            _txt_kw = dict(transform=ax.transAxes, ha="right", va="top",
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.85))
            if avg_text_kw: _txt_kw.update(avg_text_kw)
            ax.text(0.98, 0.98, f"avg = {avg_val:.2f}", **_txt_kw)

        plt.tight_layout()
        if show:
            plt.show(block=False)
            try: plt.pause(0.01)
            except Exception: pass

    # —— 保存（可选，支持 4K/8K 像素尺寸） ——
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

        # 需要临时调整尺寸吗？
        old_size = fig.get_size_inches()
        if save_pixels is not None:
            w_px, h_px = save_pixels
            w_in, h_in = w_px / dpi, h_px / dpi
            fig.set_size_inches(w_in, h_in, forward=True)

        for p in targets:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent, pad_inches=pad_inches)

        # 恢复显示尺寸
        if save_pixels is not None and keep_display_size:
            fig.set_size_inches(old_size, forward=True)

    if steps:
        print(f"[pending] {steps[0]}~{steps[-1]}s | n={len(steps)} | "
              f"max/min={max(counts) if counts else 0}/{min(counts) if counts else 0} | avg={avg_val:.2f}")

    return (fig, ax, df, avg_val) if return_handles else None



from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
# ---------- 基础：把 snapshot 里的“邻居”数出来 ----------
def _count_snapshot_edges(ed: Any) -> int:
    """
    统计某一时刻的 pending 边数量。
    兼容这几种形态：
      - {src: set(dst) / list(dst) / tuple(dst) / dict(...)}
      - 直接是 set/list/tuple/dict
      - 直接是整数
    """
    if ed is None:
        return 0
    # 典型：{src: set(dst)}
    if isinstance(ed, dict):
        s = 0
        for dsts in ed.values():
            if dsts is None:
                continue
            if isinstance(dsts, (set, list, tuple, dict)):
                s += len(dsts)
            elif isinstance(dsts, (int, np.integer)):
                s += int(dsts)
        return int(s)
    # 直接是容器或整数
    if isinstance(ed, (set, list, tuple, dict)):
        return int(len(ed))
    if isinstance(ed, (int, np.integer)):
        return int(ed)
    return 0

def _to_int_sorted_keys(d: Dict[Any, Any]) -> list[int]:
    keys = []
    for k in d.keys():
        try:
            keys.append(int(k))
        except Exception:
            pass
    keys.sort()
    return keys

# ---------- 1) 生成时间序列 DataFrame ----------
def build_pending_series(
    pending_edges: Dict[Any, Any],
    *,
    fill_missing: bool = True,           # True：用 [t_min..t_max] 全部秒；False：只用已有 key
    max_t_seconds: Optional[int] = None, # 只保留 ≤ 该秒
    smooth_window: Optional[int] = None  # 可选：增加一列移动平均
) -> pd.DataFrame:
    """
    返回 DataFrame：两列或三列
      - time            int
      - pending_edges   int
      - pending_ma_w{w} float（可选：移动平均）
    """
    if not isinstance(pending_edges, dict) or not pending_edges:
        raise ValueError("pending_edges 为空或不是 dict。")

    keys = _to_int_sorted_keys(pending_edges)
    if not keys:
        raise ValueError("pending_edges 的 key 无法转换为整数。")

    t_min, t_max = keys[0], keys[-1]
    times = list(range(t_min, t_max + 1)) if fill_missing else keys

    # 截断
    if max_t_seconds is not None:
        m = int(max_t_seconds)
        times = [t for t in times if t <= m]

    counts = [_count_snapshot_edges(pending_edges.get(t, {})) for t in times]
    df = pd.DataFrame({"time": times, "pending_edges": counts})

    # 可选：移动平均
    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        pad = w // 2
        y = np.asarray(counts, dtype=float)
        ypad = np.pad(y, (pad, pad), mode="edge")
        ker = np.ones(w) / w
        ma = np.convolve(ypad, ker, mode="valid")
        df[f"pending_ma_w{w}"] = ma

    return df

# ---------- 2) 导出到文件（Origin 直接导） ----------
def export_pending_series_to_origin(
    pending_edges: Dict[Any, Any],
    *,
    out_dir: Path | str,
    basename: str = "pending_edges",
    to: tuple[str, ...] = ("csv",),       # 可选 ("csv", "xlsx", "parquet")
    fill_missing: bool = True,
    max_t_seconds: Optional[int] = None,
    smooth_window: Optional[int] = None
) -> dict[str, Path]:
    """
    返回 {格式: 路径}，按需写出 CSV/XLSX/Parquet。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_pending_series(
        pending_edges,
        fill_missing=fill_missing,
        max_t_seconds=max_t_seconds,
        smooth_window=smooth_window,
    )

    written: dict[str, Path] = {}

    if "csv" in to:
        p = out_dir / f"{basename}.csv"
        df.to_csv(p, index=False)
        written["csv"] = p

    if "xlsx" in to:
        p = out_dir / f"{basename}.xlsx"
        # 依赖 xlsxwriter；如未安装会抛错，你可以改为 openpyxl
        try:
            with pd.ExcelWriter(p, engine="xlsxwriter") as xw:
                df.to_excel(xw, index=False, sheet_name="pending_edges")
            written["xlsx"] = p
        except Exception as e:
            print(f"[warn] 写 Excel 失败（可能缺少 xlsxwriter）：{e}")

    if "parquet" in to:
        p = out_dir / f"{basename}.parquet"
        try:
            import pyarrow as pa, pyarrow.parquet as pq
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, p, compression="zstd")
            written["parquet"] = p
        except Exception as e:
            print(f"[warn] 写 Parquet 失败（可忽略）：{e}")

    return written


# def plot_pending_edges_timeseries(
#     pending_edges: dict,
#     *,
#     # 统计与裁剪
#     max_t_seconds=None,         # 只画到该最大秒（含）；None 表示全量
#     fill_missing=True,          # 是否用 [t_min..t_max] 连续时轴补齐缺失秒
#     # 画图样式
#     line_width=2.0,
#     figsize=(10, 5),
#     title="Edges/sec (pending)",
#     rc_override=None,           # 可传入 {matplotlib rcParam: value}
#     # 展示/保存
#     show=True,                  # True: 非阻塞展示；False: 不展示（仅保存）
#     save=None,                  # None/False: 不保存；True: 用默认路径与格式；或传 str/Path/列表 指定路径
#     save_dir="figs",            # save=True 时生效：输出目录
#     basename="pending_edges",   # save=True 时生效：文件基础名
#     formats=("png", "pdf"),     # save=True 时生效：保存的格式集合
#     dpi=300,
#     transparent=False,
#     # 返回值
#     return_handles=True         # 返回 (fig, ax, df)；否则返回 None
# ):
#     """
#     统计并绘制 pending_edges 的逐秒数量曲线。
#     pending_edges[t] 的典型结构为: dict {src: set(dst)}，也兼容:
#       - {src: list/tuple/dict-of-dst}
#       - 直接是整数（已统计好的数量）
#     """
#     # ---- 安全导入（避免全局依赖）----
#     import numpy as np
#     import pandas as pd
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     from matplotlib.ticker import FormatStrFormatter
#
#     # ---- 入参校验 ----
#     if not isinstance(pending_edges, dict) or not pending_edges:
#         raise ValueError("pending_edges 为空或类型不是 dict。")
#
#     # ---- 辅助：把可能是字符串的时间键转成 int ----
#     def _to_int_key(x):
#         try:
#             return int(x)
#         except Exception:
#             return x  # 实在不行就原样返回（但后面排序可能失败）
#     keys = sorted(map(_to_int_key, pending_edges.keys()))
#     if not keys or not isinstance(keys[0], (int, np.integer)):
#         raise ValueError("pending_edges 的 key 不是整数时间戳，或无法转换为整数。")
#     t_min, t_max = int(min(keys)), int(max(keys))
#
#     # ---- 与你原函数等价的计数逻辑（更健壮）----
#     def count_edges_at_t(t):
#         ed = pending_edges.get(t, {})
#         # 1) dict: {src: 可迭代的目的集合 或 已计数的整数}
#         if isinstance(ed, dict):
#             total = 0
#             for dsts in ed.values():
#                 if dsts is None:
#                     continue
#                 if isinstance(dsts, (set, list, tuple, dict)):
#                     total += len(dsts)
#                 elif isinstance(dsts, (int, np.integer)):
#                     total += int(dsts)
#                 else:
#                     # 其他类型忽略/当 0
#                     pass
#             return int(total)
#         # 2) 直接给了一个可迭代集合
#         if isinstance(ed, (list, set, tuple, dict)):
#             return int(len(ed))
#         # 3) 直接给了一个数
#         if isinstance(ed, (int, np.integer)):
#             return int(ed)
#         return 0
#
#     # ---- 构造时轴与计数 ----
#     if fill_missing:
#         steps = list(range(t_min, t_max + 1))
#     else:
#         steps = [int(t) for t in keys]
#     counts = [count_edges_at_t(t) for t in steps]
#
#     # ---- 截断到 max_t_seconds ----
#     if max_t_seconds is not None:
#         m = int(max_t_seconds)
#         steps, counts = zip(*[(t, c) for (t, c) in zip(steps, counts) if t <= m])
#         steps, counts = list(steps), list(counts)
#
#     # ---- 打包为数据帧，便于后续保存或复用 ----
#     import pandas as pd
#     df = pd.DataFrame({"step": steps, "pending_edges": counts})
#
#     # ---- 画图（非阻塞）----
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     from matplotlib.ticker import FormatStrFormatter
#
#     base_rc = {
#         "font.family": "Times New Roman",
#         "font.size": 14,
#         "axes.labelsize": 18,
#         "axes.titlesize": 18,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "axes.linewidth": 1.2,
#     }
#     if rc_override:
#         base_rc.update(rc_override)
#
#     # 即使 show=False 也创建图对象（用于保存）
#     plt.ion()  # Jupyter 非阻塞
#     with mpl.rc_context(base_rc):
#         fig, ax = plt.subplots(figsize=figsize)
#         try:
#             fig.canvas.manager.set_window_title(title)
#         except Exception:
#             pass
#         ax.plot(steps, counts, linewidth=line_width)
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Number of edges")
#         ax.grid(True, linestyle="--", alpha=0.35)
#         ax.ticklabel_format(style="plain", axis="y", useOffset=False, useMathText=False)
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
#         if title:
#             ax.set_title(title)
#
#         # 展示（非阻塞）
#         if show:
#             plt.tight_layout()
#             plt.show(block=False)
#             # 让 notebook 事件循环有机会刷新
#             try:
#                 plt.pause(0.01)
#             except Exception:
#                 pass
#
#     # ---- 保存（可选）----
#     if save:
#         # save=True 用默认路径/格式；字符串/Path 或列表按用户给的来
#         targets = []
#         if save is True:
#             Path(save_dir).mkdir(parents=True, exist_ok=True)
#             for ext in formats:
#                 targets.append(Path(save_dir) / f"{basename}.{ext.lstrip('.')}")
#         else:
#             from pathlib import Path as _P
#             if isinstance(save, (str, _P)):
#                 targets = [save]
#             else:
#                 targets = list(save)
#         for p in targets:
#             p = Path(p)
#             p.parent.mkdir(parents=True, exist_ok=True)
#             fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent)
#
#     # 控制台摘要
#     if steps:
#         print(f"[pending] 范围: {steps[0]} ~ {steps[-1]} s | 点数: {len(steps)} | "
#               f"max/min: {max(counts) if counts else 0} / {min(counts) if counts else 0}")
#
#     return (fig, ax, df) if return_handles else None



# # 1) 只在 Jupyter 里“非阻塞显示”，不保存
# plot_pending_edges_timeseries(pending_edges)
#
# # 2) 显示 + 同时保存 PNG 和 PDF（论文友好）
# plot_pending_edges_timeseries(
#     pending_edges,
#     save=True,                    # 使用默认保存规则
#     save_dir="figs",
#     basename="pending_30s",
#     formats=("png", "pdf"),
#     dpi=300,
#     transparent=False
# )
#
# # 3) 不显示、只保存到指定路径（多格式）
# plot_pending_edges_timeseries(
#     pending_edges,
#     show=False,
#     save=["out/pending_edges.png", "out/pending_edges.pdf"]
# )
#
# # 4) 获取返回的 DataFrame，后续自定义处理
# fig, ax, df_counts = plot_pending_edges_timeseries(pending_edges, return_handles=True)
# df_counts.head()
#
