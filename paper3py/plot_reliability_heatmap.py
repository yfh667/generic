from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


def _region_pair_sort_key(s: str):
    m = re.match(r"region(\d+)_to_region(\d+)", str(s))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999, 999)


def _pretty_region_pair_label(region_pair: str, region_name_map: dict) -> str:
    m = re.match(r"^(region\d+)_to_(region\d+)$", str(region_pair))
    if not m:
        return str(region_pair)
    a, b = m.group(1), m.group(2)
    a_name = region_name_map.get(a, a)
    b_name = region_name_map.get(b, b)
    return f"{a_name}-{b_name}"


def _truncate_cmap(cmap_name="Blues", minval=0.12, maxval=0.90, n=256):
    base = plt.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc",
        base(np.linspace(minval, maxval, n)),
    )


def _extract_region_pair(df: pd.DataFrame, f: Path) -> str:
    if {"region_a", "region_b"}.issubset(df.columns) and len(df) > 0:
        return f"{df['region_a'].iloc[0]}_to_{df['region_b'].iloc[0]}"
    m = re.match(r"^(region\d+_to_region\d+)_timeseries$", f.stem)
    if m:
        return m.group(1)
    return f.stem.replace("_timeseries", "")


def _resolve_timeseries_dir(meta_path: Path) -> Path:
    # 1) 直接就是 timeseries 目录
    if meta_path.is_dir() and list(meta_path.glob("*_timeseries.csv")):
        return meta_path

    if not meta_path.exists():
        raise FileNotFoundError(f"meta_path not found: {meta_path}")

    # 2) 是上层目录，优先常见子目录
    if meta_path.is_dir():
        c1 = meta_path / "timeseries"
        c2 = meta_path / "04_region_pair_timeseries"
        if c1.exists() and list(c1.glob("*_timeseries.csv")):
            return c1
        if c2.exists() and list(c2.glob("*_timeseries.csv")):
            return c2

        # 3) 再在该目录下找 region_pair_prob_timeseries*/timeseries
        cands = []
        for p in meta_path.glob("region_pair_prob_timeseries*"):
            d1 = p / "timeseries"
            d2 = p / "04_region_pair_timeseries"
            if d1.exists() and list(d1.glob("*_timeseries.csv")):
                cands.append(d1)
            if d2.exists() and list(d2.glob("*_timeseries.csv")):
                cands.append(d2)
        if cands:
            cands = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)
            return cands[0]

    raise FileNotFoundError(f"cannot resolve timeseries dir from meta_path: {meta_path}")


def _read_metric_series(df: pd.DataFrame, metric_col: str) -> pd.Series | None:
    if metric_col in df.columns:
        return pd.to_numeric(df[metric_col], errors="coerce")
    if {"rel_sum", "pair_cnt"}.issubset(df.columns):
        rel_sum = pd.to_numeric(df["rel_sum"], errors="coerce")
        pair_cnt = pd.to_numeric(df["pair_cnt"], errors="coerce").replace(0, np.nan)
        return rel_sum / pair_cnt
    return None


def plot_cross_motif_regionpair_heatmap_from_config(cfg: dict):
    items = cfg.get("items", [])
    if not items:
        raise ValueError("config.items is empty")

    out_dir = Path(cfg.get("output_dir", "./_cross_motif_regionpair_heatmap"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_col = cfg.get("metric_col", "mean_reliability")
    region_name_map = cfg.get("region_name_map", {})

    annotate = bool(cfg.get("annotate", True))
    cmap = str(cfg.get("cmap", "Blues"))
    cmap_min = float(cfg.get("cmap_min", 0.12))
    cmap_max = float(cfg.get("cmap_max", 0.90))
    robust = bool(cfg.get("robust", True))
    q_low = float(cfg.get("q_low", 0.02))
    q_high = float(cfg.get("q_high", 0.98))
    vmin = cfg.get("vmin", None)
    vmax = cfg.get("vmax", None)
    figsize = tuple(cfg.get("figsize", [11, 5]))
    dpi = int(cfg.get("dpi", 300))
    title = str(cfg.get("title", "Mean Reliability Heatmap (Region Pair x Motif)"))

    # 00) discovery
    discovery_rows = []
    motif_order = []
    resolved_items = []
    for it in items:
        motif = str(it["motif"])
        meta_path = Path(it["meta_path"])
        ts_dir = _resolve_timeseries_dir(meta_path)
        n_files = len(list(ts_dir.glob("*_timeseries.csv")))
        motif_order.append(motif)
        resolved_items.append({"motif": motif, "timeseries_dir": ts_dir})
        discovery_rows.append(
            {
                "motif": motif,
                "meta_path": str(meta_path),
                "timeseries_dir": str(ts_dir),
                "file_count": n_files,
            }
        )

    df_discovery = pd.DataFrame(discovery_rows)
    df_discovery.to_csv(out_dir / "00_discovery.csv", index=False, encoding="utf-8-sig")

    # 01) long
    rows = []
    for it in resolved_items:
        motif = it["motif"]
        ts_dir = it["timeseries_dir"]

        for f in sorted(ts_dir.glob("*_timeseries.csv")):
            df = pd.read_csv(f)
            v = _read_metric_series(df, metric_col=metric_col)
            if v is None:
                continue

            region_pair = _extract_region_pair(df, f)
            rows.append(
                {
                    "motif": motif,
                    "region_pair": region_pair,
                    "time_points": int(v.notna().sum()),
                    "mean_metric": float(v.mean()),
                    "src_file": str(f),
                }
            )

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        raise ValueError("no valid timeseries loaded; check 00_discovery.csv")
    df_long.to_csv(out_dir / "01_long_mean_metric.csv", index=False, encoding="utf-8-sig")

    # 02) matrix
    mat = df_long.pivot(index="region_pair", columns="motif", values="mean_metric")
    mat = mat.reindex(index=sorted(mat.index.tolist(), key=_region_pair_sort_key))
    mat = mat.reindex(columns=motif_order)
    mat.to_csv(out_dir / "02_heatmap_matrix.csv", encoding="utf-8-sig")

    # 03) plot
    data = mat.to_numpy(dtype=float)
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        raise ValueError("matrix is all NaN")

    if vmin is None or vmax is None:
        if robust:
            lo, hi = np.quantile(valid, [q_low, q_high])
        else:
            lo, hi = float(valid.min()), float(valid.max())
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)

    if float(vmax) <= float(vmin):
        vmax = float(vmin) + 1e-9

    cmap_obj = _truncate_cmap(cmap_name=cmap, minval=cmap_min, maxval=cmap_max)
    cmap_obj.set_bad("#f5f5f5")
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        norm=norm,
    )

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([_pretty_region_pair_label(x, region_name_map) for x in mat.index.tolist()])
    ax.set_xlabel("motif")
    ax.set_ylabel("region_pair")
    ax.set_title(title)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                r, g, b, _ = cmap_obj(norm(val))
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = "black" if luma > 0.58 else "white"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_col)

    plt.tight_layout()
    fig.savefig(out_dir / "03_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.show(block=False)
    try:
        plt.pause(0.01)
    except Exception:
        pass

    return df_discovery, df_long, mat, fig, ax


def load_config_json(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_config_path() -> Path:
    return Path(__file__).with_name("plot_reliability_heatmap.config.json")
def _find_latest_meta_in_motif(motif_dir: Path) -> Path | None:
    analysis = motif_dir / "analysis_link"
    if not analysis.exists():
        return None

    cands = []
    for p in analysis.glob("region_pair_prob_timeseries*"):
        if (p / "timeseries").exists() or (p / "04_region_pair_timeseries").exists():
            cands.append(p)

    if not cands:
        return None
    return sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def _build_cfg_from_topology_root(
    topology_root: Path,
    *,
    motifs: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict:
    topology_root = Path(topology_root)
    if not topology_root.exists():
        raise FileNotFoundError(f"topology_root not found: {topology_root}")

    if motifs is None:
        motifs = sorted([p.name for p in topology_root.iterdir() if p.is_dir() and not p.name.startswith("_")])

    items = []
    for m in motifs:
        motif_dir = topology_root / m
        meta = _find_latest_meta_in_motif(motif_dir)
        if meta is None:
            continue
        items.append({"motif": m, "meta_path": str(meta)})

    if not items:
        raise ValueError(f"no usable motif meta found under: {topology_root}")

    if output_dir is None:
        output_dir = topology_root / "_cross_motif_regionpair_heatmap"

    return {
        "output_dir": str(output_dir),
        "metric_col": "mean_reliability",
        "title": "Mean Reliability Heatmap (Region Pair x Motif)",
        "annotate": True,
        "cmap": "Blues",
        "cmap_min": 0.12,
        "cmap_max": 0.90,
        "robust": True,
        "q_low": 0.01,
        "q_high": 0.99,
        "dpi": 300,
        "figsize": [12, 5],
        "region_name_map": {
            "region1": "America",
            "region2": "Africa",
            "region3": "China",
            "region4": "Europe"
        },
        "items": items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?", help="config.json path OR topology_root path")
    ap.add_argument("--config", default="", help="config json path")
    ap.add_argument("--topology-root", default="", help="topology root path, e.g. D:/paper3/data/topology_design")
    ap.add_argument("--motifs", default="", help="comma-separated motifs, e.g. grid_four,gridx")
    ap.add_argument("--output-dir", default="", help="output dir path")
    args = ap.parse_args()

    cfg = None

    if args.config:
        cfg = load_config_json(Path(args.config))
    elif args.topology_root:
        motifs = [x.strip() for x in args.motifs.split(",") if x.strip()] if args.motifs else None
        out_dir = Path(args.output_dir) if args.output_dir else None
        cfg = _build_cfg_from_topology_root(Path(args.topology_root), motifs=motifs, output_dir=out_dir)
    elif args.entry:
        p = Path(args.entry)
        if p.suffix.lower() == ".json":
            cfg = load_config_json(p)
        else:
            motifs = [x.strip() for x in args.motifs.split(",") if x.strip()] if args.motifs else None
            out_dir = Path(args.output_dir) if args.output_dir else None
            cfg = _build_cfg_from_topology_root(p, motifs=motifs, output_dir=out_dir)
    else:
        cfg = load_config_json(_default_config_path())


    plot_cross_motif_regionpair_heatmap_from_config(cfg)


if __name__ == "__main__":
    main()


#python D:\paper3\generic\paper3py\plot_reliability_heatmap.py --config D:\paper3\data\postprocess\plot_reliability_heatmap.config.json
