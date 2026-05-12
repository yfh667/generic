from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ======================
# Config
# ======================

DATA_ROOT = Path(r"D:\paper3\data")
ROUTE_TAG = "route2"

IN_DIR = DATA_ROOT / "postprocess" / f"best_motif_bad95_regionpair_space_{ROUTE_TAG}"
IN_CSV = IN_DIR / f"02_best_pair_metrics_by_region_pair_{ROUTE_TAG}.csv"

OUT_DIR = IN_DIR / "mechanism_type_bar"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLA = 0.95
BAD_TOL = 0.10

# 用于区分 naturally satisfied 和 locally degraded but acceptable
# 如果 bad95 <= 0.5%，认为几乎没有坏窗口
NATURAL_BAD_TOL = 0.005

# 用于定义 stable but SLA-unsatisfied
# 如果 mean < 0.95 且 bad95 >= 90%，说明大多数时间都低于 SLA
STABLE_UNSAT_BAD_MIN = 0.90


REGION_PAIR_ORDER = [
    "America-Africa",
    "America-China",
    "America-Europe",
    "Africa-China",
    "Africa-Europe",
    "China-Europe",
]

TYPE_ORDER = [
    "Naturally satisfied",
    "Locally degraded but acceptable",
    "Stable but SLA-unsatisfied",
    "Large-bad-window intolerable",
]

TYPE_COLORS = {
    "Naturally satisfied": "#2E7D32",
    "Locally degraded but acceptable": "#1976D2",
    "Stable but SLA-unsatisfied": "#F9A825",
    "Large-bad-window intolerable": "#C62828",
}


# ======================
# Classification
# ======================

def classify_mechanism(row):
    mean_v = float(row["best_mean"])
    bad95 = float(row["best_bad95"])

    if mean_v >= SLA:
        if bad95 <= NATURAL_BAD_TOL:
            return "Naturally satisfied"

        if bad95 <= BAD_TOL:
            return "Locally degraded but acceptable"

        return "Large-bad-window intolerable"

    if bad95 >= STABLE_UNSAT_BAD_MIN:
        return "Stable but SLA-unsatisfied"

    return "Large-bad-window intolerable"


# ======================
# Load data
# ======================

df = pd.read_csv(IN_CSV)

need_cols = [
    "PAIR_CSV_NAME",
    "region_pair",
    "best_motif",
    "best_topology",
    "best_mean",
    "best_p05",
    "best_bad95",
]

missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in {IN_CSV}: {missing}")

df["best_mean"] = pd.to_numeric(df["best_mean"], errors="coerce")
df["best_p05"] = pd.to_numeric(df["best_p05"], errors="coerce")
df["best_bad95"] = pd.to_numeric(df["best_bad95"], errors="coerce")

df = df.dropna(subset=["best_mean", "best_bad95"]).copy()
df["mechanism_type"] = df.apply(classify_mechanism, axis=1)


# ======================
# Count by region pair
# ======================

count_mat = (
    df.pivot_table(
        index="region_pair",
        columns="mechanism_type",
        values="PAIR_CSV_NAME",
        aggfunc="count",
        fill_value=0,
    )
    .reindex(index=REGION_PAIR_ORDER)
    .reindex(columns=TYPE_ORDER, fill_value=0)
)

ratio_mat = count_mat.div(count_mat.sum(axis=1).replace(0, np.nan), axis=0)


# ======================
# Save intermediate tables
# ======================

df.to_csv(
    OUT_DIR / f"04_pair_mechanism_type_labeled_{ROUTE_TAG}.csv",
    index=False,
    encoding="utf-8-sig",
)

count_mat.to_csv(
    OUT_DIR / f"05_mechanism_type_counts_by_region_pair_{ROUTE_TAG}.csv",
    encoding="utf-8-sig",
)

ratio_mat.to_csv(
    OUT_DIR / f"06_mechanism_type_ratio_by_region_pair_{ROUTE_TAG}.csv",
    encoding="utf-8-sig",
)


# ======================
# Plot 1: stacked count bar
# ======================

fig, ax = plt.subplots(figsize=(10.5, 5.2))

bottom = np.zeros(len(count_mat))

x = np.arange(len(count_mat.index))

for t in TYPE_ORDER:
    vals = count_mat[t].to_numpy(dtype=float)

    ax.bar(
        x,
        vals,
        bottom=bottom,
        label=t,
        color=TYPE_COLORS[t],
        edgecolor="white",
        linewidth=0.7,
    )

    for i, v in enumerate(vals):
        if v > 0:
            ax.text(
                x[i],
                bottom[i] + v / 2,
                f"{int(v)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if t in ["Naturally satisfied", "Large-bad-window intolerable"] else "black",
            )

    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(count_mat.index.tolist(), rotation=30, ha="right")
ax.set_ylabel("Number of station pairs")
ax.set_xlabel("Region pair")
ax.set_title(f"Mechanism-Type Distribution by Region Pair ({ROUTE_TAG})")
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.legend(frameon=True, fontsize=8.5, ncol=2)

fig.tight_layout()
fig.savefig(
    OUT_DIR / f"fig_mechanism_type_counts_by_region_pair_{ROUTE_TAG}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# ======================
# Plot 2: stacked percentage bar
# ======================

fig, ax = plt.subplots(figsize=(10.5, 5.2))

bottom = np.zeros(len(ratio_mat))

x = np.arange(len(ratio_mat.index))

for t in TYPE_ORDER:
    vals = ratio_mat[t].fillna(0).to_numpy(dtype=float)

    ax.bar(
        x,
        vals,
        bottom=bottom,
        label=t,
        color=TYPE_COLORS[t],
        edgecolor="white",
        linewidth=0.7,
    )

    for i, v in enumerate(vals):
        if v >= 0.05:
            ax.text(
                x[i],
                bottom[i] + v / 2,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if t in ["Naturally satisfied", "Large-bad-window intolerable"] else "black",
            )

    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(ratio_mat.index.tolist(), rotation=30, ha="right")
ax.set_ylabel("Fraction of station pairs")
ax.set_xlabel("Region pair")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0, 1.0)
ax.set_title(f"Mechanism-Type Ratio by Region Pair ({ROUTE_TAG})")
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.legend(frameon=True, fontsize=8.5, ncol=2)

fig.tight_layout()
fig.savefig(
    OUT_DIR / f"fig_mechanism_type_ratio_by_region_pair_{ROUTE_TAG}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


print("[OK] labeled table:", OUT_DIR / f"04_pair_mechanism_type_labeled_{ROUTE_TAG}.csv")
print("[OK] count table  :", OUT_DIR / f"05_mechanism_type_counts_by_region_pair_{ROUTE_TAG}.csv")
print("[OK] ratio table  :", OUT_DIR / f"06_mechanism_type_ratio_by_region_pair_{ROUTE_TAG}.csv")
print("[OK] count figure :", OUT_DIR / f"fig_mechanism_type_counts_by_region_pair_{ROUTE_TAG}.png")
print("[OK] ratio figure :", OUT_DIR / f"fig_mechanism_type_ratio_by_region_pair_{ROUTE_TAG}.png")
