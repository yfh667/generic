from __future__ import annotations

import argparse
import re
from pathlib import Path
import json

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

REGION_NAME = {
    "region0": "America",
    "region1": "Africa",
    "region2": "China",
    "region3": "Europe",
    "0": "America",
    "1": "Africa",
    "2": "China",
    "3": "Europe",
}

REGION_ORDER = ["America", "Africa", "China", "Europe"]

REGION_PAIR_COLORS = {
    "America-Africa": "#0072B2",  # blue
    "America-China": "#D55E00",   # vermillion
    "America-Europe": "#CC79A7",  # reddish purple
    "Africa-China": "#E69F00",    # orange
    "Africa-Europe": "#009E73",   # bluish green
    "China-Europe": "#6A3D9A",    # purple
}


def normalize_region(x) -> str:
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()

    if s in REGION_NAME:
        return REGION_NAME[s]

    m = re.fullmatch(r"region(\d+)", s, flags=re.IGNORECASE)
    if m:
        return REGION_NAME.get(f"region{int(m.group(1))}", s)

    return s


def make_region_pair(a, b) -> str:
    aa = normalize_region(a)
    bb = normalize_region(b)

    names = [aa, bb]
    names = sorted(names, key=lambda x: REGION_ORDER.index(x) if x in REGION_ORDER else 999)
    return f"{names[0]}-{names[1]}"


def infer_region_pair(df: pd.DataFrame) -> pd.Series:
    if {"region_a", "region_b"}.issubset(df.columns):
        return pd.Series(
            [make_region_pair(a, b) for a, b in zip(df["region_a"], df["region_b"])],
            index=df.index,
        )

    if "PAIR_CSV_NAME" not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)

    out = []
    for name in df["PAIR_CSV_NAME"].astype(str):
        m = re.search(r"(region\d+).*?(region\d+)", name)
        if m:
            out.append(make_region_pair(m.group(1), m.group(2)))
        else:
            out.append("Unknown")
    return pd.Series(out, index=df.index)


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
    df["region_pair"] = infer_region_pair(df)

    for c in ["mean", "p05", "time_ratio_rel_lt_0.95"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def classify_zone(mean_v: float, bad95: float, sla: float, bad_tol: float) -> str:
    if mean_v < sla and bad95 <= bad_tol:
        return "stable_below_sla"
    if mean_v >= sla and bad95 <= bad_tol:
        return "fixed_sufficient"
    if mean_v >= sla and bad95 > bad_tol:
        return "high_mean_with_bad_windows"
    return "below_sla_with_bad_windows"


def build_best_pair_table(df_long: pd.DataFrame, sla: float, bad_tol: float) -> pd.DataFrame:
    rows = []

    for pair_name, g in df_long.groupby("PAIR_CSV_NAME"):
        g = g.dropna(subset=["mean", "p05", "time_ratio_rel_lt_0.95"]).copy()
        if g.empty:
            continue

        best = g.sort_values(
            ["mean", "p05", "time_ratio_rel_lt_0.95"],
            ascending=[False, False, True],
        ).iloc[0]

        best_mean = float(best["mean"])
        best_p05 = float(best["p05"])
        best_bad95 = float(best["time_ratio_rel_lt_0.95"])

        rows.append({
            "PAIR_CSV_NAME": pair_name,
            "region_pair": best["region_pair"],
            "best_motif": best["motif"],
            "best_topology": best["topology"],
            "best_mean": best_mean,
            "best_p05": best_p05,
            "best_bad95": best_bad95,
            "zone": classify_zone(best_mean, best_bad95, sla, bad_tol),
            "motif_mean_spread": float(g["mean"].max() - g["mean"].min()),
        })

    return pd.DataFrame(rows)


def plot_space(pair_df: pd.DataFrame, out_png: Path, route_tag: str, sla: float, bad_tol: float) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.5))

    region_pairs = sorted(
        pair_df["region_pair"].dropna().unique().tolist(),
        key=lambda x: (
            REGION_ORDER.index(x.split("-")[0]) if "-" in x and x.split("-")[0] in REGION_ORDER else 999,
            REGION_ORDER.index(x.split("-")[1]) if "-" in x and x.split("-")[1] in REGION_ORDER else 999,
        ),
    )

    for rp in region_pairs:
        sub = pair_df[pair_df["region_pair"] == rp]
        if sub.empty:
            continue

        ax.scatter(
            sub["best_mean"],
            sub["best_bad95"],
            s=44,
            alpha=0.82,
            color=REGION_PAIR_COLORS.get(rp, "#666666"),
            edgecolors="white",
            linewidths=0.55,
            label=f"{rp} (n={len(sub)})",
        )

    ax.axvline(sla, color="black", linestyle=":", linewidth=1.4, label=f"SLA mean={sla:.2f}")
    ax.axhline(bad_tol, color="gray", linestyle="--", linewidth=1.4, label=f"bad95 tolerance={bad_tol:.0%}")

    ax.set_xlabel("Best motif mean reliability")
    ax.set_ylabel("Best motif bad95 ratio")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Region-Pair Distribution in Best-Motif Reliability Risk Space ({route_tag})")
    ax.grid(alpha=0.25, linestyle="--")

    ax.set_xlim(max(0.0, pair_df["best_mean"].min() - 0.005), min(1.002, pair_df["best_mean"].max() + 0.005))
    ax.set_ylim(-0.01, min(1.02, max(0.12, pair_df["best_bad95"].max() * 1.08)))

    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, ncol=1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_interactive_space(pair_df: pd.DataFrame, out_html: Path, route_tag: str, sla: float, bad_tol: float) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive HTML. Install it with: "
            "conda install -c conda-forge plotly"
        ) from exc

    fig = go.Figure()

    region_pairs = sorted(pair_df["region_pair"].dropna().unique().tolist())

    for rp in region_pairs:
        sub = pair_df[pair_df["region_pair"] == rp].copy()
        if sub.empty:
            continue

        custom_cols = [
            "PAIR_CSV_NAME",
            "region_pair",
            "best_topology",
            "best_mean",
            "best_p05",
            "best_bad95",
            "zone",
            "motif_mean_spread",
        ]
        custom = sub[custom_cols].astype(object).to_numpy()

        fig.add_trace(
            go.Scatter(
                x=sub["best_mean"],
                y=sub["best_bad95"],
                mode="markers",
                name=f"{rp} (n={len(sub)})",
                customdata=custom,
                marker=dict(
                    size=9,
                    color=REGION_PAIR_COLORS.get(rp, "#666666"),
                    opacity=0.82,
                    line=dict(width=0.7, color="white"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Region pair: %{customdata[1]}<br>"
                    "Best motif: %{customdata[2]}<br>"
                    "Mean reliability: %{customdata[3]:.5f}<br>"
                    "P05 reliability: %{customdata[4]:.5f}<br>"
                    "bad95 ratio: %{customdata[5]:.2%}<br>"
                    "Zone: %{customdata[6]}<br>"
                    "Motif mean spread: %{customdata[7]:.5f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=sla, line_dash="dot", line_color="black", annotation_text=f"SLA mean={sla:.2f}")
    fig.add_hline(y=bad_tol, line_dash="dash", line_color="gray", annotation_text=f"bad95={bad_tol:.0%}")

    fig.update_layout(
        title=f"Region-Pair Distribution in Best-Motif Reliability Risk Space ({route_tag})",
        xaxis_title="Best motif mean reliability",
        yaxis_title="Best motif bad95 ratio",
        template="plotly_white",
        clickmode="event+select",
        legend_title_text="Region pair",
        width=1100,
        height=720,
    )
    fig.update_yaxes(tickformat=".0%")

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="risk-space-plot")

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Best motif bad95 space - {route_tag}</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #222; }}
.wrap {{ display: grid; grid-template-columns: 1fr 340px; height: 100vh; }}
.plot {{ min-width: 0; }}
.panel {{ border-left: 1px solid #ddd; padding: 18px; background: #fafafa; font-size: 14px; }}
.panel h2 {{ margin: 0 0 12px 0; font-size: 18px; }}
.kv {{ margin: 8px 0; }}
.k {{ color: #666; }}
.v {{ font-weight: 600; word-break: break-all; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="plot">{plot_div}</div>
  <div class="panel" id="detail">
    <h2>Selected pair</h2>
    <div class="kv">Click a point to inspect its station-pair identity and metrics.</div>
  </div>
</div>
<script>
const plot = document.getElementById("risk-space-plot");
const detail = document.getElementById("detail");

function pct(x) {{
  return (Number(x) * 100).toFixed(2) + "%";
}}

plot.on("plotly_click", function(eventData) {{
  const p = eventData.points[0];
  const c = p.customdata;
  detail.innerHTML = `
    <h2>Selected pair</h2>
    <div class="kv"><div class="k">PAIR_CSV_NAME</div><div class="v">${{c[0]}}</div></div>
    <div class="kv"><div class="k">Region pair</div><div class="v">${{c[1]}}</div></div>
    <div class="kv"><div class="k">Best motif</div><div class="v">${{c[2]}}</div></div>
    <div class="kv"><div class="k">Mean reliability</div><div class="v">${{Number(c[3]).toFixed(6)}}</div></div>
    <div class="kv"><div class="k">P05 reliability</div><div class="v">${{Number(c[4]).toFixed(6)}}</div></div>
    <div class="kv"><div class="k">bad95 ratio</div><div class="v">${{pct(c[5])}}</div></div>
    <div class="kv"><div class="k">Zone</div><div class="v">${{c[6]}}</div></div>
    <div class="kv"><div class="k">Motif mean spread</div><div class="v">${{Number(c[7]).toFixed(6)}}</div></div>
  `;
}});
</script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(r"D:\paper3\data"))
    ap.add_argument("--route-tag", default="route2")
    ap.add_argument("--motifs", nargs="+", default=DEFAULT_MOTIFS)
    ap.add_argument("--sla", type=float, default=0.95)
    ap.add_argument("--bad-tol", type=float, default=0.10)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-html", action="store_true", help="Do not export interactive HTML.")

    args = ap.parse_args()

    out_dir = args.out_dir or (
        args.data_root / "postprocess" / f"best_motif_bad95_regionpair_space_{args.route_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_long = pd.concat(
        [read_motif_stat(args.data_root, args.route_tag, m) for m in args.motifs],
        ignore_index=True,
    )
    pair_df = build_best_pair_table(df_long, sla=args.sla, bad_tol=args.bad_tol)

    df_long.to_csv(out_dir / f"01_all_motif_pair_summary_long_{args.route_tag}.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / f"02_best_pair_metrics_by_region_pair_{args.route_tag}.csv", index=False, encoding="utf-8-sig")

    region_summary = pair_df.groupby(["region_pair", "zone"]).size().unstack(fill_value=0)
    region_summary.to_csv(out_dir / f"03_region_pair_zone_counts_{args.route_tag}.csv", encoding="utf-8-sig")

    out_png = out_dir / f"fig_best_motif_mean_vs_bad95_by_region_pair_{args.route_tag}.png"
    plot_space(pair_df, out_png, route_tag=args.route_tag, sla=args.sla, bad_tol=args.bad_tol)

    out_html = out_dir / f"fig_best_motif_mean_vs_bad95_by_region_pair_{args.route_tag}.html"
    if not args.no_html:
        plot_interactive_space(pair_df, out_html, route_tag=args.route_tag, sla=args.sla, bad_tol=args.bad_tol)

    print(f"[OK] figure: {out_png}")
    print(f"[OK] table : {out_dir / f'02_best_pair_metrics_by_region_pair_{args.route_tag}.csv'}")
    print(f"[OK] region summary: {out_dir / f'03_region_pair_zone_counts_{args.route_tag}.csv'}")
    if not args.no_html:
        print(f"[OK] interactive html: {out_html}")


if __name__ == "__main__":
    main()
