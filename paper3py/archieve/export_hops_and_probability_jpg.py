from __future__ import annotations

import os
import re
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import  json

import numpy as np
import pandas as pd
import matplotlib

from src.model import route_policy_probability

matplotlib.use("Agg")

## 参数区

Topology_Version = 'grid_four'
P_INTRA = 0.999
P_INTER = 0.99
P = 18
N = 36
###



DATA_DIR = Path(r"D:\paper3")
BASEDIR =  DATA_DIR / "data"
Topology_DIR = 'topology_design'
# =========================
# 基本配置
# =========================


# CSV_DIR = Path(r"D:\paper3\data\topology_design\gridx\path\region_pairs_0_86164")
# OUT_DIR = Path(r"D:\paper3\data\topology_design\gridx\path\jpg")


def _load_n_from_motif(topology_version: str) -> int:
    motif_json = BASEDIR / Topology_DIR / topology_version / "config" / "motif.json"
    return int(json.loads(motif_json.read_text(encoding="utf-8"))["N"])


N = _load_n_from_motif(Topology_Version)

ROUTE_POLICY_JSON = BASEDIR / Topology_DIR / Topology_Version / "config" / "route_policy.json"
ROUTE_POLICY = route_policy_probability.load_route_policy(
    ROUTE_POLICY_JSON,
    n=N,
    fallback_p_intra=P_INTRA,
    fallback_p_inter=P_INTER,
)
POLICY_NAME = str(ROUTE_POLICY["policy_name"]).replace(" ", "_")


STRICT_POLICY_CSV_DIR = False

def _resolve_csv_dir() -> Path:
    base = BASEDIR / Topology_DIR / Topology_Version / "path"
    policy_dir = base / f"region_pairs_0_86164_{POLICY_NAME}"
    legacy_dir = base / "region_pairs_0_86164"

    if policy_dir.exists():
        return policy_dir

    if (not STRICT_POLICY_CSV_DIR) and legacy_dir.exists():
        return legacy_dir

    raise FileNotFoundError(
        f"policy-specific path dir not found: {policy_dir}; "
        f"set STRICT_POLICY_CSV_DIR=False to allow fallback to {legacy_dir}"
    )


CSV_DIR = _resolve_csv_dir()
OUT_DIR = BASEDIR / Topology_DIR / Topology_Version / "path" / f"jpg_{POLICY_NAME}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = OUT_DIR / "_metrics"
SUMMARY_CSV = OUT_DIR / "_summary.csv"
FAILED_CSV = OUT_DIR / "_failed.csv"







MAX_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 1))


# =========================
# 工具函数
# =========================
def read_pair_csv(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    encodings = ["utf-8", "utf-8-sig", "gbk"]
    last_err = None

    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"Failed to read CSV: {csv_path}") from last_err


def infer_station_pair(df: pd.DataFrame, csv_path: Path) -> tuple[str, str]:
    if "station_a" in df.columns and "station_b" in df.columns and len(df) > 0:
        try:
            a = int(df["station_a"].iloc[0])
            b = int(df["station_b"].iloc[0])
            return str(a), str(b)
        except Exception:
            pass

    m = re.search(r"station(\d+).*station(\d+)", csv_path.stem)
    if m:
        return m.group(1), m.group(2)

    return "A", "B"


def enrich_hops_and_reliability(
    df: pd.DataFrame,
    *,
    n: int,
    policy: dict,
    path_col: str = "path",
) -> tuple[pd.DataFrame, str]:
    if path_col not in df.columns:
        raise KeyError(f"Missing required column: {path_col}")

    df = df.copy()

    p_intra = float(policy["p_intra"])
    default_p_inter = policy["default_p_inter"]
    option_p_inter = dict(policy["option_p_inter"])
    delta2option = dict(policy["delta2option"])
    policy_name = str(policy["policy_name"]).replace(" ", "_")

    option_keys = sorted(int(k) for k in option_p_inter.keys())
    option_edge_cols = {op: f"option{op}_edges" for op in option_keys}
    option_edge_values = {op: [] for op in option_keys}

    rel_list = []
    intra_hops_list = []
    inter_hops_list = []
    total_hops_list = []
    unknown_edges_list = []
    has_path = []

    path_cache = {}

    for raw_path in df[path_col].to_numpy(dtype=object):
        if pd.isna(raw_path):
            path_str = ""
        else:
            path_str = str(raw_path).strip()

        if (not path_str) or (path_str.lower() in {"nan", "none"}):
            has_path.append(False)
            rel_list.append(0.0)
            intra_hops_list.append(0)
            inter_hops_list.append(0)
            total_hops_list.append(0)
            unknown_edges_list.append(0)
            for op in option_keys:
                option_edge_values[op].append(0)
            continue

        cached = path_cache.get(path_str)
        if cached is None:
            cached = route_policy_probability.compute_path_reliability_and_hops(
                path_str,
                n=n,
                p_intra=p_intra,
                default_p_inter=default_p_inter,
                option_p_inter=option_p_inter,
                delta2option=delta2option,
                unknown_option_action="use_default",
            )
            path_cache[path_str] = cached

        rel, intra_hops, inter_hops, unknown_edges, option_counter = cached

        has_path.append(True)
        rel_list.append(float(rel))
        intra_hops_list.append(int(intra_hops))
        inter_hops_list.append(int(inter_hops))
        total_hops_list.append(int(intra_hops + inter_hops))
        unknown_edges_list.append(int(unknown_edges))

        for op in option_keys:
            option_edge_values[op].append(int(option_counter.get(op, 0)))

    df["has_path"] = has_path
    df["intra_hops"] = intra_hops_list
    df["inter_hops"] = inter_hops_list
    df["total_hops"] = total_hops_list
    df["unknown_option_edges"] = unknown_edges_list

    for op in option_keys:
        df[option_edge_cols[op]] = option_edge_values[op]

    rel_name = f"reliability_{policy_name}"
    df[rel_name] = rel_list

    return df, rel_name


def plot_hops_and_reliability_summary(
    df: pd.DataFrame,
    *,
    reliability_col: str,
    time_col: str = "time",
    intra_col: str = "intra_hops",
    inter_col: str = "inter_hops",
    total_col: str = "total_hops",
    figsize=(10, 7),
    title: str | None = None,
    hops_title: str = "Intra / Inter / Total Hops",
    reliability_title: str = "End-to-End Route Reliability",
    dpi: int = 300,
    save_path: Path | str | None = None,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    df_plot = df.copy()

    required_cols = [time_col, intra_col, inter_col, reliability_col]
    for col in required_cols:
        if col not in df_plot.columns:
            raise KeyError(f"Missing required column for plotting: {col}")

    if total_col not in df_plot.columns:
        df_plot[total_col] = df_plot[intra_col] + df_plot[inter_col]

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

    reliability = df_plot[reliability_col].astype(float) * 100.0
    t = df_plot[time_col]

    with mpl.rc_context(base_rc):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1.1, 1.0]},
        )
        ax1, ax2 = axes

        ax1.plot(
            t,
            df_plot[intra_col],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color="#2196F3",
            label="Intra-orbit hops",
        )
        ax1.plot(
            t,
            df_plot[inter_col],
            marker="s",
            markersize=3,
            linewidth=1.5,
            color="#F44336",
            label="Inter-orbit hops",
        )
        ax1.plot(
            t,
            df_plot[total_col],
            marker="^",
            markersize=3,
            linewidth=1.2,
            color="#888888",
            linestyle="--",
            label="Total hops",
        )
        ax1.set_ylabel("Hops")
        ax1.set_title(hops_title)
        ax1.grid(alpha=0.3, linestyle="--")
        ax1.legend(framealpha=0.9)

        ax2.plot(
            t,
            reliability,
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
            bottom=max(0, float(reliability.min()) - 2),
            top=min(100, float(reliability.max()) + 1),
        )

        if title:
            fig.suptitle(title, fontsize=18)

        fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def build_output_name(csv_path: Path, station_a: str, station_b: str) -> str:
    stem = csv_path.stem
    if station_a != "A" and station_b != "B":
        return f"s{station_a}_s{station_b}_hops_reliability"
    return f"{stem}_hops_reliability"


def process_one_csv(csv_path: str) -> dict:
    csv_path = Path(csv_path)

    try:
        df = read_pair_csv(csv_path)
        station_a, station_b = infer_station_pair(df, csv_path)

        df_plot, rel_name = enrich_hops_and_reliability(
            df,
            n=N,
            policy=ROUTE_POLICY,
            path_col="path",
        )

        basename = build_output_name(csv_path, station_a, station_b)
        jpg_path = OUT_DIR / f"{basename}.jpg"
        metrics_path = METRICS_DIR / f"{basename}.csv"

        plot_hops_and_reliability_summary(
            df_plot,
            reliability_col=rel_name,
            title=f"Station {station_a} ↔ Station {station_b}: Hops and Reliability",
            hops_title="Intra / Inter / Total Hops",
            reliability_title="End-to-End Route Reliability",
            save_path=jpg_path,
        )

        # 中间量持久化保存
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        df_plot.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        return {
            "status": "ok",
            "csv_file": str(csv_path),
            "rows": int(len(df_plot)),
            "station_a": station_a,
            "station_b": station_b,
            "jpg_path": str(jpg_path),
            "metrics_csv": str(metrics_path),
            "mean_intra_hops": float(df_plot["intra_hops"].mean()),
            "mean_inter_hops": float(df_plot["inter_hops"].mean()),
            "mean_total_hops": float(df_plot["total_hops"].mean()),
            "mean_reliability": float(df_plot[rel_name].mean()),
            "policy_name": POLICY_NAME,
            "unknown_option_edge_total": int(df_plot["unknown_option_edges"].sum()),

        }

    except Exception as exc:
        return {
            "status": "failed",
            "csv_file": str(csv_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {CSV_DIR}")

    print(f"[batch] topology={Topology_Version}")
    print(f"[batch] policy={POLICY_NAME}")
    print(f"[batch] csv_dir={CSV_DIR}")
    print(f"[batch] out_dir={OUT_DIR}")
    print(f"[batch] metrics_dir={METRICS_DIR}")
    print(f"[batch] files={len(csv_files)}")
    print(f"[batch] workers={MAX_WORKERS}")

    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one_csv, str(p)): p for p in csv_files}

        done = 0
        total = len(futures)

        for fut in as_completed(futures):
            done += 1
            src = futures[fut]

            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "csv_file": str(src),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

            results.append(result)

            if result["status"] == "ok":
                print(
                    f"[{done}/{total}] ok   {Path(result['csv_file']).name} "
                    f"-> {Path(result['jpg_path']).name}"
                )
            else:
                print(
                    f"[{done}/{total}] fail {Path(result['csv_file']).name} "
                    f"-> {result['error']}"
                )

    ok_rows = [r for r in results if r["status"] == "ok"]
    failed_rows = [r for r in results if r["status"] == "failed"]

    pd.DataFrame(ok_rows).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")

    print(f"[batch] success={len(ok_rows)} failed={len(failed_rows)}")
    print(f"[batch] summary={SUMMARY_CSV}")
    print(f"[batch] failed={FAILED_CSV}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
