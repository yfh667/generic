from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


DEFAULT_MOTIFS = [
    "grid_plane_alternating",
    "grid_x_sparse",
    "grid+",
    "gridx",
    "grid_four",
]

MOTIF_LABEL = {
    "grid_four": "Grid-Four",
    "gridx": "Grid-X",
    "grid_plane_alternating": "Grid-Plane-Alt",
    "grid_x_sparse": "Grid-X-Sparse",
    "grid+": "Grid+",
}

REGION_NAME = {
    0: "America",
    1: "Africa",
    2: "China",
    3: "Europe",
}

REGION_ORDER = ["America", "Africa", "China", "Europe"]

CLASS_COLORS = {
    "A": "#2ca25f",
    "B": "#3182bd",
    "C": "#fdae6b",
    "D": "#de2d26",
}

CLASS_LABELS = {
    "A": "A: all motifs robust",
    "B": "B: at least one motif robust",
    "C": "C: small bad95 window",
    "D": "D: large / persistent bad95 window",
}


def region_name(v) -> str:
    if pd.isna(v):
        return "Unknown"

    s = str(v).strip()

    if re.fullmatch(r"\d+", s):
        return REGION_NAME.get(int(s), f"region{s}")

    m = re.fullmatch(r"region(\d+)", s, flags=re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        return REGION_NAME.get(idx, f"region{idx}")

    return s


def region_pair_label(a, b) -> str:
    names = [region_name(a), region_name(b)]
    names = sorted(names, key=lambda x: REGION_ORDER.index(x) if x in REGION_ORDER else 999)
    return "-".join(names)


def infer_region_pair(row) -> str:
    if "region_a" in row.index and "region_b" in row.index:
        return region_pair_label(row["region_a"], row["region_b"])

    name = str(row["PAIR_CSV_NAME"])
    m = re.search(r"region(\d+).*region(\d+)", name)
    if m:
        return region_pair_label(m.group(1), m.group(2))

    return "Unknown"


def read_one_motif(data_root: Path, route_tag: str, motif: str) -> pd.DataFrame:
    stat_dir = data_root / "topology_design" / motif / f"station_pair_reliability_{route_tag}"
    csv_path = stat_dir / "all_pair_global_stat.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"missing all_pair_global_stat.csv: {csv_path}")

    df = pd.read_csv(csv_path)

    if "PAIR_CSV_NAME" not in df.columns and "pair_csv_name" in df.columns:
        df = df.rename(columns={"pair_csv_name": "PAIR_CSV_NAME"})

    required = ["PAIR_CSV_NAME", "mean", "p05", "time_ratio_rel_lt_0.95"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df["motif"] = motif
    df["topology"] = MOTIF_LABEL.get(motif, motif)
    df["region_pair"] = df.apply(infer_region_pair, axis=1)

    for c in ["mean", "p05", "time_ratio_rel_lt_0.95"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def classify_pairs(df_long: pd.DataFrame, motif_count: int, require_common: bool) -> pd.DataFrame:
    if require_common:
        counts = df_long.groupby("PAIR_CSV_NAME")["motif"].nunique()
        common_pairs = counts[counts == motif_count].index
        df_long = df_long[df_long["PAIR_CSV_NAME"].isin(common_pairs)].copy()

    rows = []

    for pair_name, g in df_long.groupby("PAIR_CSV_NAME"):
        g = g.dropna(subset=["mean", "p05", "time_ratio_rel_lt_0.95"]).copy()
        if g.empty:
            continue

        best = g.sort_values(
            ["mean", "p05", "time_ratio_rel_lt_0.95"],
            ascending=[False, False, True],
        ).iloc[0]

        good = (g["mean"] >= 0.95) & (g["p05"] >= 0.95)
        all_good = bool(good.all())
        best_good = bool((best["mean"] >= 0.95) and (best["p05"] >= 0.95))

        if all_good:
            cls = "A"
        elif best_good:
            cls = "B"
        elif best["mean"] >= 0.95 and best["time_ratio_rel_lt_0.95"] <= 0.10:
            cls = "C"
        else:
            cls = "D"

        rows.append({
            "PAIR_CSV_NAME": pair_name,
            "region_pair": best["region_pair"],
            "best_topology": best["topology"],
            "best_mean": float(best["mean"]),
            "best_p05": float(best["p05"]),
            "best_bad95": float(best["time_ratio_rel_lt_0.95"]),
            "topology_spread": float(g["mean"].max() - g["mean"].min()),
            "n_motifs_good95": int(good.sum()),
            "n_motifs_seen": int(g["motif"].nunique()),
            "class": cls,
        })

    return pd.DataFrame(rows)


def plot_fig4(pair_class: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    for cls in ["A", "B", "C", "D"]:
        sub = pair_class[pair_class["class"] == cls]
        if sub.empty:
            continue

        ax.scatter(
            sub["best_mean"],
            sub["best_bad95"],
            s=42,
            alpha=0.82,
            c=CLASS_COLORS[cls],
            label=f"{CLASS_LABELS[cls]} (n={len(sub)})",
            edgecolors="white",
            linewidths=0.6,
        )

    ax.axvline(0.95, color="black", linestyle="--", linewidth=1.1)
    ax.axhline(0.10, color="black", linestyle="--", linewidth=1.1)

    ax.set_xlabel("Best motif mean reliability")
    ax.set_ylabel("Best motif bad95 ratio")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.93, 1.001)
    ax.set_ylim(-0.01, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(framealpha=0.95, fontsize=9, loc="upper right")
    ax.set_title(title)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(r"D:\paper3\data"))
    ap.add_argument("--route-tag", default="route2")
    ap.add_argument("--motifs", nargs="+", default=DEFAULT_MOTIFS)
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\paper3\data\postprocess\fig4_best_motif_risk_space"))
    ap.add_argument("--allow-missing-pairs", action="store_true")
    args = ap.parse_args()

    frames = [read_one_motif(args.data_root, args.route_tag, motif) for motif in args.motifs]
    df_long = pd.concat(frames, ignore_index=True)

    pair_class = classify_pairs(
        df_long,
        motif_count=len(args.motifs),
        require_common=not args.allow_missing_pairs,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = args.out_dir / f"all_motif_pair_summary_long_{args.route_tag}.csv"
    class_csv = args.out_dir / f"pair_classification_abcd_{args.route_tag}.csv"
    fig_png = args.out_dir / f"fig4_best_motif_reliability_risk_space_{args.route_tag}.png"

    df_long.to_csv(long_csv, index=False, encoding="utf-8-sig")
    pair_class.to_csv(class_csv, index=False, encoding="utf-8-sig")

    class_summary = pair_class.groupby("class").agg(
        n_pairs=("PAIR_CSV_NAME", "size"),
        mean_best_reliability=("best_mean", "mean"),
        mean_best_bad95=("best_bad95", "mean"),
        mean_spread=("topology_spread", "mean"),
    )
    class_summary.to_csv(args.out_dir / f"class_summary_{args.route_tag}.csv", encoding="utf-8-sig")

    plot_fig4(
        pair_class,
        fig_png,
        title=f"Best-Motif Reliability and bad95 Risk Space ({args.route_tag})",
    )

    print(f"[OK] long table  : {long_csv}")
    print(f"[OK] class table : {class_csv}")
    print(f"[OK] figure      : {fig_png}")


if __name__ == "__main__":
    main()
