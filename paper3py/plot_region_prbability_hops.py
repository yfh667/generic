from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


# =========================
# Helpers
# =========================
def _region_pair_sort_key(s: str):
    m = re.match(r"region(\d+)_to_region(\d+)", str(s))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999, 999)


def _pretty_region_pair_label(region_pair: str, region_name_map: dict[str, str]) -> str:
    m = re.match(r"^(region\d+)_to_(region\d+)$", str(region_pair))
    if not m:
        return str(region_pair)
    a, b = m.group(1), m.group(2)
    return f"{region_name_map.get(a, a)}-{region_name_map.get(b, b)}"


def _extract_region_pair(df: pd.DataFrame, f: Path) -> str:
    if {"region_a", "region_b"}.issubset(df.columns) and len(df) > 0:
        return f"{df['region_a'].iloc[0]}_to_{df['region_b'].iloc[0]}"
    m = re.match(r"^(region\d+_to_region\d+)_timeseries$", f.stem)
    if m:
        return m.group(1)
    return f.stem.replace("_timeseries", "")


def _robust_limits(arr: np.ndarray, ql=0.02, qh=0.98):
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(v, [ql, qh])
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


def _truncate_cmap(cmap_name="Blues", minval=0.12, maxval=0.95, n=256):
    base = plt.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc",
        base(np.linspace(minval, maxval, n)),
    )


def _resolve_timeseries_dir(meta_path: Path, *, kind: str | None = None) -> Path:
    """
    Resolve a directory containing *_timeseries.csv.

    kind:
      "probability" -> prefer region_pair_probability_timeseries
      "hops"        -> prefer region_pair_hops_timeseries
      None          -> accept any compatible timeseries directory

    Supports both old layout:
      analysis_link/region_pair_prob_timeseries.../timeseries

    and new layout:
      region_communication/region_pair_probability_timeseries
      region_communication/region_pair_hops_timeseries
    """
    meta_path = Path(meta_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"meta_path not found: {meta_path}")

    def has_timeseries_csv(p: Path) -> bool:
        return p.is_dir() and any(p.glob("*_timeseries.csv"))

    if has_timeseries_csv(meta_path):
        return meta_path

    preferred_names: list[str] = []
    if kind == "probability":
        preferred_names = [
            "region_pair_probability_timeseries",
            "timeseries",
            "04_region_pair_timeseries",
        ]
    elif kind == "hops":
        preferred_names = [
            "region_pair_hops_timeseries",
            "timeseries",
            "04_region_pair_timeseries",
        ]
    else:
        preferred_names = [
            "region_pair_probability_timeseries",
            "region_pair_hops_timeseries",
            "timeseries",
            "04_region_pair_timeseries",
        ]

    for name in preferred_names:
        p = meta_path / name
        if has_timeseries_csv(p):
            return p

    cands: list[Path] = []

    # New route-policy layout.
    new_patterns = [
        "region_communication/region_pair_probability_timeseries",
        "region_communication/region_pair_hops_timeseries",
        "route_policy/*/region_communication/region_pair_probability_timeseries",
        "route_policy/*/region_communication/region_pair_hops_timeseries",
    ]

    # Old analysis_link layout.
    old_patterns = [
        "analysis_link/region_pair_prob_timeseries*/timeseries",
        "analysis_link/region_pair_prob_timeseries*/04_region_pair_timeseries",
        "analysis_link/region_pair_hop_timeseries*/timeseries",
        "analysis_link/region_pair_hop_timeseries*/04_region_pair_timeseries",
        "analysis_link/region_pair_prob_timeseries_pi*_pe*/04_region_pair_timeseries",
        "analysis_link/region_pair_prob_timeseries_simple*/timeseries",
    ]

    if kind == "probability":
        patterns = [
            "region_communication/region_pair_probability_timeseries",
            "route_policy/*/region_communication/region_pair_probability_timeseries",
            "analysis_link/region_pair_prob_timeseries*/timeseries",
            "analysis_link/region_pair_prob_timeseries*/04_region_pair_timeseries",
            "analysis_link/region_pair_prob_timeseries_pi*_pe*/04_region_pair_timeseries",
            "analysis_link/region_pair_prob_timeseries_simple*/timeseries",
        ]
    elif kind == "hops":
        patterns = [
            "region_communication/region_pair_hops_timeseries",
            "route_policy/*/region_communication/region_pair_hops_timeseries",
            "analysis_link/region_pair_hop_timeseries*/timeseries",
            "analysis_link/region_pair_hop_timeseries*/04_region_pair_timeseries",
        ]
    else:
        patterns = new_patterns + old_patterns

    for pat in patterns:
        for p in meta_path.glob(pat):
            if has_timeseries_csv(p):
                cands.append(p)

    if cands:
        return sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(f"cannot resolve timeseries dir from meta_path: {meta_path}")



def _find_latest_dir(motif_dir: Path, patterns: list[str]) -> Path | None:
    cands = []
    for pat in patterns:
        cands.extend([p for p in motif_dir.glob(pat) if p.is_dir()])
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


# =========================
# Core pipeline
# =========================
def collect_long_metrics(cfg: dict, out_dir: Path):
    items = cfg.get("items", [])
    if not items:
        raise ValueError("config.items is empty")

    hop_cols = {
        "mean_total_hops_on_reachable",
        "mean_inter_hops_on_reachable",
        "mean_intra_hops_on_reachable",
    }

    discovery_rows = []
    rows = []
    motif_order = []

    for it in items:
        motif = str(it["motif"])
        motif_order.append(motif)

        rel_metric_col = str(it.get("reliability_metric_col", "mean_reliability"))
        rel_dir = _resolve_timeseries_dir(Path(it["reliability_meta_path"]), kind="probability")

        hop_dir = None
        if it.get("hop_meta_path"):
            hop_dir = _resolve_timeseries_dir(Path(it["hop_meta_path"]), kind="hops")

        rel_files = sorted(rel_dir.glob("*_timeseries.csv"))

        discovery_rows.append(
            {
                "motif": motif,
                "reliability_meta_path": str(it["reliability_meta_path"]),
                "reliability_timeseries_dir": str(rel_dir),
                "reliability_file_count": len(rel_files),
                "hop_meta_path": str(it.get("hop_meta_path", "")),
                "hop_timeseries_dir": str(hop_dir) if hop_dir is not None else "",
            }
        )

        for rf in rel_files:
            dfr = pd.read_csv(rf)
            rp = _extract_region_pair(dfr, rf)

            rec = {
                "motif": motif,
                "region_pair": rp,
                "mean_reliability": np.nan,
                "mean_total_hops_on_reachable": np.nan,
                "mean_inter_hops_on_reachable": np.nan,
                "mean_intra_hops_on_reachable": np.nan,
                "src_rel_file": str(rf),
            }

            if rel_metric_col in dfr.columns:
                rec["mean_reliability"] = float(pd.to_numeric(dfr[rel_metric_col], errors="coerce").mean())
            elif {"rel_sum", "pair_cnt"}.issubset(dfr.columns):
                rel_sum = pd.to_numeric(dfr["rel_sum"], errors="coerce")
                pair_cnt = pd.to_numeric(dfr["pair_cnt"], errors="coerce").replace(0, np.nan)
                rec["mean_reliability"] = float((rel_sum / pair_cnt).mean())

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

    df_discovery = pd.DataFrame(discovery_rows)
    df_discovery.to_csv(out_dir / "00_discovery.csv", index=False, encoding="utf-8-sig")

    df_long = pd.DataFrame(rows).drop_duplicates(subset=["motif", "region_pair"], keep="last")
    if df_long.empty:
        raise ValueError("no usable motif-region records were read; check 00_discovery.csv")

    df_long["inter_ratio"] = df_long["mean_inter_hops_on_reachable"] / df_long["mean_total_hops_on_reachable"]
    df_long["R_per_hop"] = df_long["mean_reliability"] / df_long["mean_total_hops_on_reachable"]

    df_long.to_csv(out_dir / "01_long_metrics.csv", index=False, encoding="utf-8-sig")
    return df_discovery, df_long, motif_order


def build_matrix(df_long: pd.DataFrame, motif_order: list[str], col: str) -> pd.DataFrame:
    m = df_long.pivot(index="region_pair", columns="motif", values=col)
    m = m.reindex(index=sorted(m.index.tolist(), key=_region_pair_sort_key))
    m = m.reindex(columns=motif_order)
    return m


def plot_one(ax, mat: pd.DataFrame, *, title: str, fmt: str, cmap_obj, vmin=None, vmax=None, ql=0.02, qh=0.98):
    data = mat.to_numpy(dtype=float)

    if vmin is None or vmax is None:
        lo, hi = _robust_limits(data, ql=ql, qh=qh)
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi

    if vmax <= vmin:
        vmax = vmin + 1e-9

    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap_obj, norm=norm)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_title(title, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap_obj(norm(v))
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            tc = "black" if luma > 0.58 else "white"
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7, color=tc)

    return im


def run_panel(cfg: dict):
    out_dir = Path(cfg.get("output_dir", "./_cross_motif_regionpair_panels"))
    out_dir.mkdir(parents=True, exist_ok=True)

    region_name_map = cfg.get(
        "region_name_map",
        {"region1": "America", "region2": "Africa", "region3": "China", "region4": "Europe"},
    )

    panel_cfg = cfg.get("panel", {})
    cmap_name = panel_cfg.get("cmap", "Blues")
    cmap_min = float(panel_cfg.get("cmap_min", 0.12))
    cmap_max = float(panel_cfg.get("cmap_max", 0.95))
    q_low = float(panel_cfg.get("q_low", 0.02))
    q_high = float(panel_cfg.get("q_high", 0.98))
    dpi = int(panel_cfg.get("dpi", 300))
    figsize = tuple(panel_cfg.get("figsize", [16, 12]))
    suptitle = panel_cfg.get("title", "Region-Pair Metrics Across Motifs")

    _, df_long, motif_order = collect_long_metrics(cfg, out_dir)

    m_rel = build_matrix(df_long, motif_order, "mean_reliability")
    m_ht = build_matrix(df_long, motif_order, "mean_total_hops_on_reachable")
    m_ir = build_matrix(df_long, motif_order, "inter_ratio")
    m_rh = build_matrix(df_long, motif_order, "R_per_hop")

    m_rel.to_csv(out_dir / "02_mat_reliability.csv", encoding="utf-8-sig")
    m_ht.to_csv(out_dir / "03_mat_total_hop.csv", encoding="utf-8-sig")
    m_ir.to_csv(out_dir / "04_mat_inter_ratio.csv", encoding="utf-8-sig")
    m_rh.to_csv(out_dir / "05_mat_R_per_hop.csv", encoding="utf-8-sig")

    cmap_obj = _truncate_cmap(cmap_name=cmap_name, minval=cmap_min, maxval=cmap_max)
    cmap_obj.set_bad("#f5f5f5")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    im0 = plot_one(
        axes[0], m_rel,
        title="Mean Reliability", fmt=".4f",
        cmap_obj=cmap_obj, ql=q_low, qh=q_high
    )
    im1 = plot_one(
        axes[1], m_ht,
        title="Mean Total Hops (reachable)", fmt=".2f",
        cmap_obj=cmap_obj, ql=q_low, qh=q_high
    )
    im2 = plot_one(
        axes[2], m_ir,
        title="Inter-Hop Ratio", fmt=".3f",
        cmap_obj=cmap_obj, vmin=0.0, vmax=1.0, ql=q_low, qh=q_high
    )
    im3 = plot_one(
        axes[3], m_rh,
        title="Reliability per Hop", fmt=".4f",
        cmap_obj=cmap_obj, ql=q_low, qh=q_high
    )

    ylabels = [_pretty_region_pair_label(x, region_name_map) for x in m_rel.index.tolist()]
    for ax in axes:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)

    for ax, im in zip(axes, [im0, im1, im2, im3]):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_dir / "06_panels_reliability_hops.png", dpi=dpi, bbox_inches="tight")
    plt.show(block=False)
    try:
        plt.pause(0.01)
    except Exception:
        pass

    return out_dir


# =========================
# Config loading / auto-build
# =========================
def load_config_json(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_config_path() -> Path:
    return Path(__file__).with_name("test.config.json")


def auto_build_config(topology_root: Path, motifs: list[str] | None = None, output_dir: Path | None = None) -> dict:
    """
    Auto-build config from current topology_design layout.

    Supports new layout:
      <topology_root>/<motif>/region_communication/
        region_pair_probability_timeseries/
        region_pair_hops_timeseries/

    Also supports by-route layout:
      <topology_root>/<motif>/route_policy/<route_name>/region_communication/
        region_pair_probability_timeseries/
        region_pair_hops_timeseries/

    Keeps old analysis_link discovery as fallback.
    """
    topology_root = Path(topology_root)

    if motifs is None:
        motifs = sorted(
            [
                p.name
                for p in topology_root.iterdir()
                if p.is_dir() and not p.name.startswith("_")
            ]
        )

    items = []

    for motif in motifs:
        motif_dir = topology_root / motif
        if not motif_dir.exists():
            continue

        rel_dir = _find_latest_dir(
            motif_dir,
            [
                # New by_motif layout.
                "region_communication/region_pair_probability_timeseries",

                # New by_route layout.
                "route_policy/*/region_communication/region_pair_probability_timeseries",

                # Old layout fallback.
                "analysis_link/region_pair_prob_timeseries_policy*/timeseries",
                "analysis_link/region_pair_prob_timeseries_policy*/04_region_pair_timeseries",
                "analysis_link/region_pair_prob_timeseries_simple*/timeseries",
                "analysis_link/region_pair_prob_timeseries_pi*_pe*/04_region_pair_timeseries",
            ],
        )

        hop_dir = _find_latest_dir(
            motif_dir,
            [
                # New by_motif layout.
                "region_communication/region_pair_hops_timeseries",

                # New by_route layout.
                "route_policy/*/region_communication/region_pair_hops_timeseries",

                # Old layout fallback.
                "analysis_link/region_pair_hop_timeseries*/timeseries",
                "analysis_link/region_pair_hop_timeseries*/04_region_pair_timeseries",
            ],
        )

        if rel_dir is None:
            continue

        items.append(
            {
                "motif": motif,
                "reliability_meta_path": str(rel_dir),
                "hop_meta_path": str(hop_dir) if hop_dir is not None else "",
                "reliability_metric_col": "mean_reliability",
            }
        )

    if not items:
        raise ValueError(
            "auto_build_config found no usable motif directories. "
            "Expected region_communication/region_pair_probability_timeseries under each motif."
        )

    if output_dir is None:
        output_dir = topology_root / "_cross_motif_regionpair_panels"

    return {
        "output_dir": str(output_dir),
        "region_name_map": {
            "region1": "America",
            "region2": "Africa",
            "region3": "China",
            "region4": "Europe",
        },
        "items": items,
        "panel": {
            "cmap": "Blues",
            "cmap_min": 0.12,
            "cmap_max": 0.95,
            "q_low": 0.02,
            "q_high": 0.98,
            "dpi": 300,
            "figsize": [16, 10],
            "title": "Region-Pair Reliability and Hops Across Motifs",
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?", help="config.json path OR topology_root path")
    ap.add_argument("--config", default="", help="config json path")
    ap.add_argument("--topology-root", default="", help="topology root path")
    ap.add_argument("--motifs", default="", help="comma-separated motif list")
    ap.add_argument("--output-dir", default="", help="output dir")
    args = ap.parse_args()

    cfg = None

    if args.config:
        cfg = load_config_json(Path(args.config))
    elif args.topology_root:
        motifs = [x.strip() for x in args.motifs.split(",") if x.strip()] if args.motifs else None
        out_dir = Path(args.output_dir) if args.output_dir else None
        cfg = auto_build_config(Path(args.topology_root), motifs=motifs, output_dir=out_dir)
    elif args.entry:
        p = Path(args.entry)
        if p.suffix.lower() == ".json":
            cfg = load_config_json(p)
        else:
            motifs = [x.strip() for x in args.motifs.split(",") if x.strip()] if args.motifs else None
            out_dir = Path(args.output_dir) if args.output_dir else None
            cfg = auto_build_config(p, motifs=motifs, output_dir=out_dir)
    else:
        cfg = load_config_json(_default_config_path())

    run_panel(cfg)


if __name__ == "__main__":
    main()
