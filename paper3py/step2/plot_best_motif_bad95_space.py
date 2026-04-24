from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


DEFAULT_MOTIFS = ["grid_plane_alternating", "grid_x_sparse", "grid+", "gridx", "grid_four"]

MOTIF_LABEL = {
    "grid_plane_alternating": "Grid-Plane-Alt",
    "grid_x_sparse": "Grid-X-Sparse",
    "grid+": "Grid+",
    "gridx": "Grid-X",
    "grid_four": "Grid-Four",
}

CLASS_COLOR = {
    "A": "#d95f5f",  # stable but below SLA
    "B": "#2ca25f",  # fixed motif sufficient
    "C": "#f0b429",  # high mean but bad windows
    "D": "#7b61ff",  # below SLA and bad windows
}

CLASS_LABEL = {
    "A": "A: stable but below SLA",
    "B": "B: fixed motif sufficient",
    "C": "C: high mean with bad windows",
    "D": "D: below SLA with bad windows",
}


def read_motif_stat(data_root: Path, route_tag: str, motif: str) -> pd.DataFrame:
    csv_path = (
        data_root
        / "topology_design"
        / motif
        / f"station_pair_reliability_{route_tag}"
        / "all_pair_global_stat.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"missing: {csv_path}")

    df = pd.read_csv(csv_path)
    if "PAIR_CSV_NAME" not in df.columns and "pair_csv_name" in df.columns:
        df = df.rename(columns={"pair_csv_name": "PAIR_CSV_NAME"})

    need = ["PAIR_CSV_NAME", "mean", "p05", "time_ratio_rel_lt_0.95"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df["motif"] = motif
    df["topology"] = MOTIF_LABEL.get(motif, motif)

    for c in ["mean", "p05", "time_ratio_rel_lt_0.95"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def classify_row(mean_v: float, bad95: float, sla: float, bad_tol: float) -> str:
    if mean_v < sla and bad95 <= bad_tol:
        return "A"
    if mean_v >= sla and bad95 <= bad_tol:
        return "B"
    if mean_v >= sla and bad95 > bad_tol:
        return "C"
    return "D"


def build_best_pair_table(df_long: pd.DataFrame, sla: float, bad_tol: float) -> pd.DataFrame:
    rows = []

    for pair_name, g in df_long.groupby("PAIR_CSV_NAME"):
        g = g.dropna(subset=["mean", "p05", "time_ratio_rel_lt_0.95"]).copy()
        if g.empty:
            continue

        # Best fixed motif for this pair: highest mean, then higher p05, then lower bad95.
        best = g.sort_values(
            ["mean", "p05", "time_ratio_rel_lt_0.95"],
            ascending=[False, False, True],
        ).iloc[0]

        best_mean = float(best["mean"])
        best_p05 = float(best["p05"])
        best_bad95 = float(best["time_ratio_rel_lt_0.95"])

        rows.append({
            "PAIR_CSV_NAME": pair_name,
            "best_motif": best["motif"],
            "best_topology": best["topology"],
            "best_mean": best_mean,
            "best_p05": best_p05,
            "best_bad95": best_bad95,
            "class": classify_row(best_mean, best_bad95, sla, bad_tol),
            "motif_mean_spread": float(g["mean"].max() - g["mean"].min()),
        })

    return pd.DataFrame(rows)


def plot_space(pair_df: pd.DataFrame, out_png: Path, route_tag: str, sla: float, bad_tol: float) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.4))

    for cls in ["A", "B", "C", "D"]:
        sub = pair_df[pair_df["class"] == cls]
        if sub.empty:
            continue
        ax.scatter(
            sub["best_mean"],
            sub["best_bad95"],
            s=42,
            alpha=0.82,
            color=CLASS_COLOR[cls],
            edgecolors="white",
            linewidths=0.55,
            label=f"{CLASS_LABEL[cls]} (n={len(sub)})",
        )

    ax.axvline(sla, color="black", linestyle=":", linewidth=1.4, label=f"SLA mean={sla:.2f}")
    ax.axhline(bad_tol, color="gray", linestyle="--", linewidth=1.4, label=f"bad95 tolerance={bad_tol:.0%}")

    ax.set_xlabel("Best motif mean reliability")
    ax.set_ylabel("Best motif bad95 ratio")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Best-Motif Reliability vs SLA Violation Ratio ({route_tag})")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    ax.set_xlim(max(0.0, pair_df["best_mean"].min() - 0.005), min(1.002, pair_df["best_mean"].max() + 0.005))
    ax.set_ylim(-0.01, min(1.02, max(0.12, pair_df["best_bad95"].max() * 1.08)))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(r"D:\paper3\data"))
    ap.add_argument("--route-tag", default="route2")
    ap.add_argument("--motifs", nargs="+", default=DEFAULT_MOTIFS)
    ap.add_argument("--sla", type=float, default=0.95)
    ap.add_argument("--bad-tol", type=float, default=0.10)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (
        args.data_root / "postprocess" / f"best_motif_bad95_space_{args.route_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_long = pd.concat(
        [read_motif_stat(args.data_root, args.route_tag, m) for m in args.motifs],
        ignore_index=True,
    )
    pair_df = build_best_pair_table(df_long, sla=args.sla, bad_tol=args.bad_tol)

    df_long.to_csv(out_dir / f"01_all_motif_pair_summary_long_{args.route_tag}.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / f"02_best_pair_metrics_{args.route_tag}.csv", index=False, encoding="utf-8-sig")
    pair_df.groupby("class").size().rename("n_pairs").to_csv(out_dir / f"03_class_counts_{args.route_tag}.csv", encoding="utf-8-sig")

    out_png = out_dir / f"fig_best_motif_mean_vs_bad95_{args.route_tag}.png"
    plot_space(pair_df, out_png, route_tag=args.route_tag, sla=args.sla, bad_tol=args.bad_tol)

    print(f"[OK] figure: {out_png}")
    print(f"[OK] table : {out_dir / f'02_best_pair_metrics_{args.route_tag}.csv'}")


if __name__ == "__main__":
    main()
