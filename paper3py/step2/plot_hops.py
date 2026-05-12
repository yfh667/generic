import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # Windows 本地显示窗口；必须放在 pyplot 前
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams

# ── IEEE double-column style ─────────────────────────────────────────────────
rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        8,
    'axes.titlesize':   8.5,
    'axes.labelsize':   8,
    'xtick.labelsize':  7.5,
    'ytick.labelsize':  7.5,
    'legend.fontsize':  7.5,
    'lines.linewidth':  1.2,
    'axes.linewidth':   0.8,
    'xtick.major.width':0.7,
    'ytick.major.width':0.7,
    'xtick.minor.width':0.5,
    'ytick.minor.width':0.5,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'axes.grid':        False,
    'figure.dpi':       300,
})

# ── Load & compute ───────────────────────────────────────────────────────────
df = pd.read_csv(
    'D:/paper3/data/topology_design/grid_x_sparse/path/'
    'region_pairs_0_86164_route2/region0--station0-region1--station5.csv'
)

df['reliability'] = (0.990 ** df['inter_hops']) * 100

# Downsample: keep every 10th point for plotting speed
df_plot = df.iloc[::10].copy()

t = df_plot['time'].values
ia = df_plot['intra_hops'].values
ie = df_plot['inter_hops'].values
tot = df_plot['total_hops'].values
rel = df_plot['reliability'].values

# ── Routing phase boundaries ─────────────────────────────────────────────────
phase_bounds = [0, 12128, 26768, 55422, 69713, df['time'].max()]
phase_labels = ['Phase Ⅰ', 'Phase Ⅱ', 'Phase Ⅲ', 'Phase Ⅳ', 'Phase Ⅴ']

# ── Color palette ────────────────────────────────────────────────────────────
C_intra = '#4C9BE8'
C_inter = '#F07B5A'
C_total = '#1C2541'
C_rel = '#2ECC8E'
C_phase_odd = '#F8F9FA'
C_phase_even = '#FFFFFF'
C_vline = '#BBBBBB'

# ── Figure layout: upper hops, lower reliability ─────────────────────────────
MM = 1 / 25.4

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(181 * MM, 60 * MM),
    sharex=True,
    gridspec_kw={
        'height_ratios': [35, 22],
        'hspace': 0.0,
    }
)

# ── Phase background ─────────────────────────────────────────────────────────
for ax in (ax1, ax2):
    for i, (lo, hi) in enumerate(zip(phase_bounds[:-1], phase_bounds[1:])):
        fc = C_phase_odd if i % 2 == 0 else C_phase_even
        ax.axvspan(lo, hi, color=fc, zorder=0)

# ── Upper panel: stacked hop areas + total line ──────────────────────────────
ax1.fill_between(
    t, 0, ia,
    step='post',
    color=C_intra,
    alpha=0.50,
    linewidth=0,
    zorder=2,
)

ax1.fill_between(
    t, ia, tot,
    step='post',
    color=C_inter,
    alpha=0.50,
    linewidth=0,
    zorder=2,
)

ax1.step(
    t, tot,
    where='post',
    color=C_total,
    linewidth=1.3,
    zorder=4,
)

ax1.set_ylabel('Hops', color=C_total)
ax1.set_ylim(0, 27)
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.tick_params(axis='y', colors=C_total)
ax1.tick_params(axis='x', which='both',
                bottom=False, top=False, labelbottom=False)


# ── Lower panel: reliability ─────────────────────────────────────────────────
ax2.step(
    t, rel,
    where='post',
    color=C_rel,
    linewidth=1.5,
    zorder=5,
)

ax2.fill_between(
    t, 88, rel,
    step='post',
    color=C_rel,
    alpha=0.18,
    linewidth=0,
    zorder=1,
)

ax2.set_ylabel('Reliability (%)', color=C_rel)
ax2.set_ylim(86, 97.5)
ax2.yaxis.set_major_locator(MultipleLocator(2))
ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
ax2.tick_params(axis='y', colors=C_rel)

# ── Phase boundaries ─────────────────────────────────────────────────────────
for ax in (ax1, ax2):
    for b in phase_bounds[1:-1]:
        ax.axvline(
            b,
            color=C_vline,
            linewidth=0.8,
            linestyle='--',
            zorder=3,
        )

# ── Phase labels only on upper panel ─────────────────────────────────────────
for i, (lo, hi) in enumerate(zip(phase_bounds[:-1], phase_bounds[1:])):
    ax1.text(
        (lo + hi) / 2,
        25.8,
        phase_labels[i],
        ha='center',
        va='top',
        fontsize=6,
        color='#666666',
        style='italic',
        clip_on=True,
    )

# ── X axis ───────────────────────────────────────────────────────────────────
ax2.set_xlim(0, t.max())


ax2.set_xlabel('Time Step')
ax2.xaxis.set_major_locator(MultipleLocator(20000))
ax2.xaxis.set_minor_locator(MultipleLocator(5000))
ax2.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(
        lambda x, _: f'{int(x / 1000)}k' if x > 0 else '0'
    )
)
ax2.tick_params(axis='x', which='both',
                bottom=True, top=False, labelbottom=True)

# ── Title ────────────────────────────────────────────────────────────────────
ax1.set_title(
    'Station 0 → Station 5: Hops and Reliability',
    pad=3,
    fontweight='semibold',
    fontsize=8.5,
)

# ── Legend ───────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=C_intra, alpha=0.7, label='Intra-orbit hops'),
    mpatches.Patch(color=C_inter, alpha=0.7, label='Inter-orbit hops'),
    plt.Line2D([0], [0], color=C_total, linewidth=1.3, label='Total hops'),
    plt.Line2D([0], [0], color=C_rel, linewidth=1.5, label='Reliability'),
]

ax1.legend(
    handles=handles,
    loc='upper right',
    frameon=True,
    framealpha=0.92,
    edgecolor='#CCCCCC',
    ncol=2,
    borderpad=0.5,
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=0.8,
    fontsize=7,
)

# ── Show figure ──────────────────────────────────────────────────────────────
fig.subplots_adjust(
    left=0.075,
    right=0.965,
    bottom=0.18,
    top=0.90,
    hspace=0.0
)

plt.show()


print("Done")
