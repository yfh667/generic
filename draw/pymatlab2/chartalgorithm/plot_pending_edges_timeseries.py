def average_switches(pending_edges: dict, *, steps=None) -> float:
    """
    计算平均切换数（每秒 pending 边数的平均）。
    pending_edges[t] 通常是 {src: set(dst)}，也兼容 list/tuple/dict-of-dst 或直接整数。
    steps: 可选，指定参与统计的步列表；默认用 pending_edges 的所有键。
    """
    import numpy as np

    def _count(ed) -> int:
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
        if isinstance(ed, (set, list, tuple, dict)):
            return int(len(ed))
        if isinstance(ed, (int, np.integer)):
            return int(ed)
        return 0

    if not isinstance(pending_edges, dict) or not pending_edges:
        return 0.0

    use_steps = steps if steps is not None else list(pending_edges.keys())
    total_edges = 0
    total_steps = 0
    for t in use_steps:
        total_edges += _count(pending_edges.get(t, {}))
        total_steps += 1
    return (total_edges / total_steps) if total_steps else 0.0

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
    annotate_avg=True,          # 新增：是否在图上标注平均值
    avg_line_kw=None,           # axhline 的样式覆盖
    avg_text_kw=None,           # 平均值文字框的样式覆盖
    return_handles=True         # 返回 (fig, ax, df, avg)
):
    """
    统计并绘制 pending_edges 的逐秒数量曲线，并（可选）在图上标注平均切换数。
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    if not isinstance(pending_edges, dict) or not pending_edges:
        raise ValueError("pending_edges 为空或类型不是 dict。")

    # —— keys → int 并排序 ——
    def _k2i(x):
        try: return int(x)
        except: return None
    keys = sorted(s for s in (_k2i(k) for k in pending_edges.keys()) if s is not None)
    if not keys:
        raise ValueError("pending_edges 的 key 无法转换为整数。")
    t_min, t_max = keys[0], keys[-1]

    # —— 计数：与你原逻辑等价，兼容多形态 ——
    def count_edges_at_t(t):
        ed = pending_edges.get(t, {})
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
        if isinstance(ed, (set, list, tuple, dict)):
            return int(len(ed))
        if isinstance(ed, (int, np.integer)):
            return int(ed)
        return 0

    # —— 构造时轴与计数 ——
    steps = list(range(t_min, t_max + 1)) if fill_missing else keys
    counts = [count_edges_at_t(t) for t in steps]

    # —— 截断可视上限 ——
    if max_t_seconds is not None:
        m = int(max_t_seconds)
        steps, counts = zip(*[(t, c) for t, c in zip(steps, counts) if t <= m])
        steps, counts = list(steps), list(counts)

    # —— 汇成 DF 便于复用 ——
    df = pd.DataFrame({"step": steps, "pending_edges": counts})

    # —— 平均值（对当前绘制范围计算） ——
    avg_val = float(sum(counts) / len(counts)) if counts else 0.0

    # —— 画图（非阻塞） ——
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
        fig, ax = plt.subplots(figsize=figsize)
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        ax.plot(steps, counts, linewidth=line_width)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of edges")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False, useMathText=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if title:
            ax.set_title(title)

        # —— 平均值标注 ——
        if annotate_avg:
            _line_kw = dict(ls="--", lw=1.2, alpha=0.9, color="gray")
            if avg_line_kw: _line_kw.update(avg_line_kw)
            ax.axhline(avg_val, **_line_kw)

            # 右上角文字框
            _txt_kw = dict(
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.85)
            )
            if avg_text_kw: _txt_kw.update(avg_text_kw)
            ax.text(0.98, 0.98, f"avg = {avg_val:.2f}", **_txt_kw)

        plt.tight_layout()
        if show:
            plt.show(block=False)
            try: plt.pause(0.01)
            except Exception:
                pass

    # —— 保存（可选） ——
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

    # 控制台摘要
    if steps:
        print(f"[pending] 范围: {steps[0]} ~ {steps[-1]} s | 点数: {len(steps)} | "
              f"max/min: {max(counts) if counts else 0} / {min(counts) if counts else 0} | "
              f"avg: {avg_val:.2f}")

    return (fig, ax, df, avg_val) if return_handles else None
