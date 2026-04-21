from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REGION_NAME_MAP = {
    "region1": "America",
    "region2": "Africa",
    "region3": "China",
    "region4": "Europe",
}

MOTIFS = ["grid_four", "gridx", "grid_plane_alternating", "grid_x_sparse", "grid+"]

root = Path(r"D:\paper3\data\topology_design")
if not root.exists():
    root = Path(r"C:\user\data\topology_design")

out_dir = root / "_cross_motif_regionpair_panels"
out_dir.mkdir(parents=True, exist_ok=True)


def region_pair_sort_key(s: str):
    m = re.match(r"region(\d+)_to_region(\d+)", str(s))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999, 999)


def pretty_region_pair_label(region_pair: str):
    m = re.match(r"^(region\d+)_to_(region\d+)$", str(region_pair))
    if not m:
        return str(region_pair)
    a, b = m.group(1), m.group(2)
    return f"{REGION_NAME_MAP.get(a, a)}-{REGION_NAME_MAP.get(b, b)}"


def find_latest_dir(motif_dir: Path, patterns):
    cands = []
    for pat in patterns:
        cands.extend([p for p in motif_dir.glob(pat) if p.is_dir()])
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def extract_region_pair(df: pd.DataFrame, f: Path):
    if {"region_a", "region_b"}.issubset(df.columns) and len(df) > 0:
        return f"{df['region_a'].iloc[0]}_to_{df['region_b'].iloc[0]}"
    m = re.match(r"^(region\d+_to_region\d+)_timeseries$", f.stem)
    if m:
        return m.group(1)
    return f.stem.replace("_timeseries", "")


def robust_limits(arr, ql=0.02, qh=0.98):
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(v, [ql, qh])
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


rows = []
for motif in MOTIFS:
    motif_dir = root / motif

    rel_dir = find_latest_dir(
        motif_dir,
        [
            "analysis_link/region_pair_prob_timeseries_pi*_pe*/04_region_pair_timeseries",
            "analysis_link/region_pair_prob_timeseries_simple_pi*_pe*/timeseries",
            "analysis_link/region_pair_prob_timeseries_simple*/timeseries",
        ],
    )

    hop_dir = find_latest_dir(
        motif_dir,
        [
            "analysis_link/region_pair_hop_timeseries_simple/timeseries",
            "analysis_link/region_pair_hop_timeseries*/timeseries",
        ],
    )

    if rel_dir is None:
        continue

    for rf in sorted(rel_dir.glob("*_timeseries.csv")):
        dfr = pd.read_csv(rf)
        rp = extract_region_pair(dfr, rf)

        rec = {
            "motif": motif,
            "region_pair": rp,
            "mean_reliability": np.nan,
            "mean_total_hops_on_reachable": np.nan,
            "mean_inter_hops_on_reachable": np.nan,
            "mean_intra_hops_on_reachable": np.nan,
        }

        if "mean_reliability" in dfr.columns:
            rec["mean_reliability"] = float(pd.to_numeric(dfr["mean_reliability"], errors="coerce").mean())

        hop_cols = {
            "mean_total_hops_on_reachable",
            "mean_inter_hops_on_reachable",
            "mean_intra_hops_on_reachable",
        }

        hop_src = None
        if hop_cols.issubset(dfr.columns):
            hop_src = dfr
        elif hop_dir is not None:
            hf = hop_dir / rf.name
            if not hf.exists():
                hf = hop_dir / f"{rp}_timeseries.csv"
            if hf.exists():
                hop_src = pd.read_csv(hf)

        if hop_src is not None:
            for c in hop_cols:
                if c in hop_src.columns:
                    rec[c] = float(pd.to_numeric(hop_src[c], errors="coerce").mean())

        rows.append(rec)

df = pd.DataFrame(rows).drop_duplicates(subset=["motif", "region_pair"], keep="last")
if df.empty:
    raise ValueError("没有读到可用数据。")

df["inter_ratio"] = df["mean_inter_hops_on_reachable"] / df["mean_total_hops_on_reachable"]
df["R_per_hop"] = df["mean_reliability"] / df["mean_total_hops_on_reachable"]

df.to_csv(out_dir / "00_long_metrics.csv", index=False, encoding="utf-8-sig")


def build_mat(col):
    m = df.pivot(index="region_pair", columns="motif", values=col)
    m = m.reindex(index=sorted(m.index.tolist(), key=region_pair_sort_key))
    m = m.reindex(columns=MOTIFS)
    return m


m_rel = build_mat("mean_reliability")
m_ht = build_mat("mean_total_hops_on_reachable")
m_hi = build_mat("mean_inter_hops_on_reachable")
m_ha = build_mat("mean_intra_hops_on_reachable")
m_ir = build_mat("inter_ratio")
m_rh = build_mat("R_per_hop")

m_rel.to_csv(out_dir / "01_mat_reliability.csv", encoding="utf-8-sig")
m_ht.to_csv(out_dir / "02_mat_total_hop.csv", encoding="utf-8-sig")
m_hi.to_csv(out_dir / "03_mat_inter_hop.csv", encoding="utf-8-sig")
m_ha.to_csv(out_dir / "04_mat_intra_hop.csv", encoding="utf-8-sig")
m_ir.to_csv(out_dir / "05_mat_inter_ratio.csv", encoding="utf-8-sig")
m_rh.to_csv(out_dir / "06_mat_R_per_hop.csv", encoding="utf-8-sig")

from matplotlib.colors import LinearSegmentedColormap

def _truncate_cmap(cmap_name="Blues", minval=0.12, maxval=0.95, n=256):
    base = plt.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc",
        base(np.linspace(minval, maxval, n))
    )

COMMON_CMAP = _truncate_cmap("Blues", 0.12, 0.95)
COMMON_CMAP.set_bad("#f5f5f5")

def plot_one(ax, mat, title, fmt, vmin=None, vmax=None):
    data = mat.to_numpy(dtype=float)
    if vmin is None or vmax is None:
        lo, hi = robust_limits(data)
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi

    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=COMMON_CMAP, norm=norm)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([pretty_region_pair_label(x) for x in mat.index.tolist()])
    ax.set_title(title, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = COMMON_CMAP(norm(v))
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            tc = "black" if luma > 0.58 else "white"
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7, color=tc)

    return im


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

im0 = plot_one(axes[0], m_rel, "Mean Reliability", ".4f")
im1 = plot_one(axes[1], m_ht, "Mean Total Hops (reachable)", ".2f")
# im2 = plot_one(axes[2], m_hi, "Mean Inter Hops (reachable)", ".2f")
# im3 = plot_one(axes[3], m_ha, "Mean Intra Hops (reachable)", ".2f")
im4 = plot_one(axes[2], m_ir, "Inter-Hop Ratio", ".3f", vmin=0.0, vmax=1.0)
im5 = plot_one(axes[3], m_rh, "Reliability per Hop", ".4f")

# 删除不用的两个空白子图
# fig.delaxes(axes[2])
# fig.delaxes(axes[3])


for ax, im in zip(axes, [im0, im1,  im4, im5]):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=8)

plt.suptitle("Region-Pair Metrics Across Motifs", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

fig.savefig(out_dir / "07_panels_reliability_hops.png", dpi=300, bbox_inches="tight")
plt.show()
