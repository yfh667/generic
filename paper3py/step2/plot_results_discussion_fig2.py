from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOTIFS = ["grid_plane_alternating", "grid_x_sparse", "grid+", "gridx", "grid_four"]

MOTIF_LABEL = {
    "grid_plane_alternating": "Grid-Plane-Alt",
    "grid_x_sparse": "GridX-Sparse",
    "grid+": "Grid+",
    "gridx": "GridX",
    "grid_four": "Grid-Four",
}

COLORS = {
    "A": "#ff5a5f",
    "B": "#2ecc71",
    "C": "#f2b705",
    "D": "#a66cff",
}


def read_all_motif_stats(data_root: Path, route_tag: str) -> pd.DataFrame:
    rows = []

    for motif in MOTIFS:
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
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def build_best_pair_table(df_long: pd.DataFrame) -> pd.DataFrame:
    out = []

    for pair_name, g in df_long.groupby("PAIR_CSV_NAME"):
        g = g.copy()
        g["mean"] = pd.to_numeric(g["mean"], errors="coerce")
        g["p05"] = pd.to_numeric(g["p05"], errors="coerce")
        g["time_ratio_rel_lt_0.95"] = pd.to_numeric(g["time_ratio_rel_lt_0.95"], errors="coerce")

        g = g.dropna(subset=["mean", "p05"])
        if g.empty:
            continue

        best = g.sort_values(["mean", "p05"], ascending=[False, False]).iloc[0]

        best_mean = float(best["mean"])
        best_p05 = float(best["p05"])
        tail_drop = best_mean - best_p05

        if tail_drop <= 0.02 and best_mean < 0.95:
            cls = "A"
        elif tail_drop <= 0.02 and best_mean >= 0.95:
            cls = "B"
        elif tail_drop > 0.02 and best_mean >= 0.95:
            cls = "C"
        else:
            cls = "D"

        out.append({
            "PAIR_CSV_NAME": pair_name,
            "best_topology": best["topology"],
            "best_mean": best_mean,
            "best_p05": best_p05,
            "best_tail_drop": tail_drop,
            "best_bad95": float(best["time_ratio_rel_lt_0.95"]),
            "class": cls,
        })

    return pd.DataFrame(out)


def plot_decision_space(pair_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))

    for cls in ["A", "B", "C", "D"]:
        sub = pair_df[pair_df["class"] == cls]
        if sub.empty:
            continue

        ax.scatter(
            sub["best_mean"],
            sub["best_tail_drop"],
            s=42,
            alpha=0.8,
            color=COLORS[cls],
            edgecolors="white",
            linewidths=0.5,
            label=f"{cls} (n={len(sub)})",
        )

    ax.axvline(0.95, color="black", linestyle=":", linewidth=1.4, label="SLA mean=0.95")
    ax.axhline(0.02, color="gray", linestyle="--", linewidth=1.4, label="tail drop=0.02")

    ax.set_xlabel("Best motif mean reliability")
    ax.set_ylabel("Best motif tail drop: mean - p05")
    ax.set_title("A/B/C/D decision space")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.95)

    ax.set_xlim(0.938, 1.002)
    ax.set_ylim(-0.005, max(0.105, pair_df["best_tail_drop"].max() * 1.08))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(r"D:\paper3\data"))
    ap.add_argument("--route-tag", default="route2")
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\paper3\data\postprocess\results_discussion_fig2"))
    args = ap.parse_args()

    df_long = read_all_motif_stats(args.data_root, args.route_tag)
    pair_df = build_best_pair_table(df_long)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_long.to_csv(args.out_dir / f"all_motif_pair_summary_long_{args.route_tag}.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(args.out_dir / f"pair_decision_space_{args.route_tag}.csv", index=False, encoding="utf-8-sig")

    out_png = args.out_dir / f"fig2_decision_space_{args.route_tag}.png"
    plot_decision_space(pair_df, out_png)

    print(f"[OK] figure: {out_png}")
    print(f"[OK] table : {args.out_dir / f'pair_decision_space_{args.route_tag}.csv'}")


if __name__ == "__main__":
    main()
