def plot_hops_and_reliability_summary(
    df,
    *,
    reliability_col,
    time_col="time",
    intra_col="intra_hops",
    inter_col="inter_hops",
    total_col="total_hops",
    figsize=(10, 7),
    title=None,
    hops_title="Intra-orbit vs Inter-orbit hops",
    reliability_title="End-to-End Route Reliability",
    show=True,
    save=None,
    save_dir="figs",
    basename="hops_reliability_summary",
    formats=("png",),
    dpi=300,
    return_handles=True,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from pathlib import Path

    base_rc = {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "legend.fontsize": 12,
    }

    df_plot = df.copy()

    if total_col not in df_plot.columns:
        df_plot[total_col] = df_plot[intra_col] + df_plot[inter_col]

    reliability = df_plot[reliability_col].astype(float) * 100.0
    t = df_plot[time_col]

    plt.ion()
    with mpl.rc_context(base_rc):
        fig, axes = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1.1, 1.0]}
        )
        ax1, ax2 = axes

        # ===== subplot 1: hops =====
        ax1.plot(
            t, df_plot[intra_col],
            marker="o", markersize=3,
            linewidth=1.5,
            color="#2196F3",
            label="Intra-orbit hops",
        )
        ax1.plot(
            t, df_plot[inter_col],
            marker="s", markersize=3,
            linewidth=1.5,
            color="#F44336",
            label="Inter-orbit hops",
        )
        ax1.plot(
            t, df_plot[total_col],
            marker="^", markersize=3,
            linewidth=1.2,
            color="#888888",
            linestyle="--",
            label="Total hops",
        )
        ax1.set_ylabel("Hops")
        ax1.set_title(hops_title)
        ax1.grid(alpha=0.3, linestyle="--")
        ax1.legend(framealpha=0.9)

        # ===== subplot 2: reliability =====
        ax2.plot(
            t, reliability,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color="#4CAF50",
            label="Route reliability",
        )
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Reliability (%)")
        ax2.set_title(reliability_title)
        ax2.grid(alpha=0.3, linestyle="--")
        ax2.set_ylim(
            bottom=max(0, reliability.min() - 2),
            top=min(100, reliability.max() + 1),
        )

        if title is not None:
            fig.suptitle(title, fontsize=18)

        fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)

        if show:
            plt.show(block=False)
            try:
                plt.pause(0.01)
            except Exception:
                pass

    if save:
        targets = []
        if save is True:
            outdir = Path(save_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            for ext in formats:
                targets.append(outdir / f"{basename}.{ext.lstrip('.')}")
        else:
            targets = [save] if isinstance(save, (str, Path)) else list(save)

        for p in targets:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return (fig, axes, df_plot) if return_handles else None
