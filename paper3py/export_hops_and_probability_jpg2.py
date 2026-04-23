from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model.route_policy_probability import compute_path_reliability_and_hops
from src.paper3_route.probability import _resolve_option_policy
from src.paper3_route.route_policy import (
    load_route_policy,
    motif_config_path,
    path_output_dir,
    probability_pair_dir,
    route_output_root,
)


# =========================
# Default direct-run config
# =========================

ROUTE_POLICY_DEFAULT = Path(r"D:\paper3\data\route_policy\route1.json")
MOTIF_DEFAULT = "grid_four"


# =========================
# Path helpers
# =========================

def infer_data_root_from_route_policy(route_policy_path: str | Path) -> Path:
    p = Path(route_policy_path).expanduser().resolve()

    # Expected:
    #   D:/paper3/data/route_policy/route1.json
    if p.parent.name == "route_policy":
        return p.parent.parent

    # Fallback for unusual layouts.
    return p.parent.parent


def safe_fs_tag(s: str) -> str:
    s = str(s).strip()
    if not s:
        return "route"
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in s)


def route_tag(policy) -> str:
    tag_source = str(policy.get("outputs.route_tag_source", "policy_file")).strip().lower()

    if tag_source == "policy_file" and policy.path is not None:
        raw_tag = policy.path.stem
    elif tag_source == "route_name":
        raw_tag = policy.route_name
    else:
        raw_tag = policy.route_name

    return safe_fs_tag(raw_tag)


def load_n_from_motif(data_root: str | Path, policy, motif_name: str) -> int:
    p = motif_config_path(data_root, policy, motif_name)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        if "N" in raw:
            return int(raw["N"])

    from src.config.viewer_config import G60_CONFIG

    return int(G60_CONFIG.N)


def resolve_source_dir(
    data_root: str | Path,
    policy,
    motif_name: str,
    *,
    source_mode: str,
) -> tuple[Path, str]:
    """
    source_mode:
      auto        -> prefer station_pair_reliability/pair_timeseries, fallback to path CSV
      probability -> force station_pair_reliability/pair_timeseries
      path        -> force route path CSV
    """
    prob_dir = probability_pair_dir(data_root, policy, motif_name)
    path_dir = path_output_dir(data_root, policy, motif_name)

    if source_mode == "probability":
        if not prob_dir.exists():
            raise FileNotFoundError(f"probability pair_timeseries dir not found: {prob_dir}")
        return prob_dir, "probability"

    if source_mode == "path":
        if not path_dir.exists():
            raise FileNotFoundError(f"path dir not found: {path_dir}")
        return path_dir, "path"

    if prob_dir.exists() and any(prob_dir.glob("*.csv")):
        return prob_dir, "probability"

    if path_dir.exists() and any(path_dir.glob("*.csv")):
        return path_dir, "path"

    raise FileNotFoundError(
        "No usable source CSV directory found.\n"
        f"probability dir: {prob_dir}\n"
        f"path dir: {path_dir}"
    )


def resolve_output_dir(data_root: str | Path, policy, motif_name: str, out_dir: str | Path | None = None) -> Path:
    if out_dir:
        return Path(out_dir)

    base_name = policy.output_subdir("onepair_plot_subdir", "onepair_hops_probability")
    append_tag_default = policy.output_layout == "by_motif"
    append_tag = bool(policy.get("outputs.onepair_plot_append_route_tag", append_tag_default))

    if append_tag:
        base_name = f"{base_name}_{route_tag(policy)}"

    return route_output_root(data_root, policy, motif_name) / base_name


# =========================
# CSV / metric helpers
# =========================

def read_pair_csv(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
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
            a = int(pd.to_numeric(df["station_a"], errors="coerce").dropna().iloc[0])
            b = int(pd.to_numeric(df["station_b"], errors="coerce").dropna().iloc[0])
            return str(a), str(b)
        except Exception:
            pass

    m = re.search(r"station(\d+).*station(\d+)", csv_path.stem)
    if m:
        return m.group(1), m.group(2)

    return "A", "B"


def find_existing_reliability_col(df: pd.DataFrame, preferred: str) -> str | None:
    if preferred in df.columns:
        return preferred

    candidates = [
        c for c in df.columns
        if c.startswith("rel_") or c.startswith("reliability_")
    ]
    if len(candidates) == 1:
        return candidates[0]

    return None


def ensure_hops_and_reliability(
    df: pd.DataFrame,
    *,
    n: int,
    option_policy: dict[str, Any],
    rel_col: str,
    path_col: str = "path",
) -> tuple[pd.DataFrame, str]:
    """
    If the CSV already has reliability and hop columns, reuse them.
    Otherwise compute reliability from path by the shared option-aware path function.
    """
    out = df.copy()

    existing_rel_col = find_existing_reliability_col(out, rel_col)
    has_hops = {"intra_hops", "inter_hops"}.issubset(out.columns)

    if existing_rel_col is not None and has_hops:
        if existing_rel_col != rel_col:
            out[rel_col] = pd.to_numeric(out[existing_rel_col], errors="coerce")
        else:
            out[rel_col] = pd.to_numeric(out[rel_col], errors="coerce")

        out["intra_hops"] = pd.to_numeric(out["intra_hops"], errors="coerce")
        out["inter_hops"] = pd.to_numeric(out["inter_hops"], errors="coerce")

        if "total_hops" not in out.columns:
            out["total_hops"] = out["intra_hops"] + out["inter_hops"]
        else:
            out["total_hops"] = pd.to_numeric(out["total_hops"], errors="coerce")

        if "has_path" not in out.columns:
            out["has_path"] = out[rel_col].notna()

        if "unknown_option_edges" not in out.columns:
            out["unknown_option_edges"] = 0

        return out, rel_col

    if path_col not in out.columns:
        raise KeyError(
            f"CSV has neither existing reliability/hop columns nor required path column: {path_col}"
        )

    p_intra = float(option_policy["p_intra"])
    default_p_inter = float(option_policy["default_p_inter"])
    option_p_inter = dict(option_policy["option_p_inter"])
    delta2option = dict(option_policy["delta2option"])
    unknown_option_action = str(option_policy.get("unknown_option_action", "use_default"))

    option_keys = sorted(int(k) for k in option_p_inter.keys())

    rel_list = []
    intra_hops_list = []
    inter_hops_list = []
    total_hops_list = []
    unknown_edges_list = []
    has_path_list = []
    option_hop_values = {op: [] for op in option_keys}

    path_cache: dict[str, tuple] = {}

    for raw_path in out[path_col].to_numpy(dtype=object):
        if pd.isna(raw_path):
            path_str = ""
        else:
            path_str = str(raw_path).strip()

        if (not path_str) or path_str.lower() in {"nan", "none"}:
            has_path_list.append(False)
            rel_list.append(np.nan)
            intra_hops_list.append(np.nan)
            inter_hops_list.append(np.nan)
            total_hops_list.append(np.nan)
            unknown_edges_list.append(0)
            for op in option_keys:
                option_hop_values[op].append(0)
            continue

        cached = path_cache.get(path_str)
        if cached is None:
            cached = compute_path_reliability_and_hops(
                path_str,
                n=int(n),
                p_intra=p_intra,
                default_p_inter=default_p_inter,
                option_p_inter=option_p_inter,
                delta2option=delta2option,
                unknown_option_action=unknown_option_action,
            )
            path_cache[path_str] = cached

        rel, intra_hops, inter_hops, unknown_edges, option_counter = cached

        has_path_list.append(True)
        rel_list.append(float(rel))
        intra_hops_list.append(int(intra_hops))
        inter_hops_list.append(int(inter_hops))
        total_hops_list.append(int(intra_hops + inter_hops))
        unknown_edges_list.append(int(unknown_edges))

        for op in option_keys:
            option_hop_values[op].append(int(option_counter.get(op, 0)))

    out["has_path"] = has_path_list
    out["intra_hops"] = intra_hops_list
    out["inter_hops"] = inter_hops_list
    out["total_hops"] = total_hops_list
    out["unknown_option_edges"] = unknown_edges_list

    for op in option_keys:
        out[f"option{op}_hops"] = option_hop_values[op]

    out[rel_col] = rel_list

    if "min_shortest_path" in out.columns:
        missing = pd.to_numeric(out["min_shortest_path"], errors="coerce").isna()
        out.loc[missing, [rel_col, "intra_hops", "inter_hops", "total_hops"]] = np.nan
        out.loc[missing, "has_path"] = False

    return out, rel_col


def build_output_name(csv_path: Path, station_a: str, station_b: str) -> str:
    if station_a != "A" and station_b != "B":
        return f"s{station_a}_s{station_b}_hops_reliability"
    return f"{csv_path.stem}_hops_reliability"


# =========================
# Plotting
# =========================

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
) -> None:
    import matplotlib as mpl

    df_plot = df.copy()

    required_cols = [time_col, intra_col, inter_col, reliability_col]
    for col in required_cols:
        if col not in df_plot.columns:
            raise KeyError(f"Missing required column for plotting: {col}")

    if total_col not in df_plot.columns:
        df_plot[total_col] = df_plot[intra_col] + df_plot[inter_col]

    t = pd.to_numeric(df_plot[time_col], errors="coerce")
    reliability = pd.to_numeric(df_plot[reliability_col], errors="coerce") * 100.0

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
            pd.to_numeric(df_plot[intra_col], errors="coerce"),
            marker="o",
            markersize=3,
            linewidth=1.5,
            color="#2196F3",
            label="Intra-orbit hops",
        )
        ax1.plot(
            t,
            pd.to_numeric(df_plot[inter_col], errors="coerce"),
            marker="s",
            markersize=3,
            linewidth=1.5,
            color="#F44336",
            label="Inter-orbit hops",
        )
        ax1.plot(
            t,
            pd.to_numeric(df_plot[total_col], errors="coerce"),
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

        valid_rel = reliability.dropna()
        if valid_rel.empty:
            ax2.set_ylim(0, 100)
        else:
            ymin = max(0.0, float(valid_rel.min()) - 2.0)
            ymax = min(100.0, float(valid_rel.max()) + 1.0)
            if ymax <= ymin:
                ymax = min(100.0, ymin + 1.0)
            ax2.set_ylim(ymin, ymax)

        if title:
            fig.suptitle(title, fontsize=18)

        fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        plt.close(fig)


# =========================
# Worker
# =========================

def process_one_csv(task: dict[str, Any]) -> dict[str, Any]:
    csv_path = Path(task["csv_path"])
    out_dir = Path(task["out_dir"])
    metrics_dir = Path(task["metrics_dir"])
    n = int(task["n"])
    option_policy = task["option_policy"]
    rel_col = str(task["rel_col"])
    dpi = int(task["dpi"])

    try:
        df = read_pair_csv(csv_path)
        station_a, station_b = infer_station_pair(df, csv_path)

        df_plot, rel_name = ensure_hops_and_reliability(
            df,
            n=n,
            option_policy=option_policy,
            rel_col=rel_col,
            path_col="path",
        )

        basename = build_output_name(csv_path, station_a, station_b)
        jpg_path = out_dir / f"{basename}.jpg"
        metrics_path = metrics_dir / f"{basename}.csv"

        plot_hops_and_reliability_summary(
            df_plot,
            reliability_col=rel_name,
            title=f"Station {station_a} to Station {station_b}: Hops and Reliability",
            hops_title="Intra / Inter / Total Hops",
            reliability_title="End-to-End Route Reliability",
            save_path=jpg_path,
            dpi=dpi,
        )

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        df_plot.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        rel_s = pd.to_numeric(df_plot[rel_name], errors="coerce")

        rec = {
            "status": "ok",
            "csv_file": str(csv_path),
            "rows": int(len(df_plot)),
            "station_a": station_a,
            "station_b": station_b,
            "jpg_path": str(jpg_path),
            "metrics_csv": str(metrics_path),
            "mean_intra_hops": float(pd.to_numeric(df_plot["intra_hops"], errors="coerce").mean()),
            "mean_inter_hops": float(pd.to_numeric(df_plot["inter_hops"], errors="coerce").mean()),
            "mean_total_hops": float(pd.to_numeric(df_plot["total_hops"], errors="coerce").mean()),
            "mean_reliability": float(rel_s.mean()),
        }

        if "unknown_option_edges" in df_plot.columns:
            rec["unknown_option_edge_total"] = int(
                pd.to_numeric(df_plot["unknown_option_edges"], errors="coerce").fillna(0).sum()
            )

        for c in df_plot.columns:
            if re.match(r"^option\d+_hops$", c):
                rec[f"{c}_total"] = int(pd.to_numeric(df_plot[c], errors="coerce").fillna(0).sum())

        return rec

    except Exception as exc:
        return {
            "status": "failed",
            "csv_file": str(csv_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot per station-pair hops and route reliability for one motif under one route policy."
    )
    ap.add_argument("--route-policy", type=Path, default=ROUTE_POLICY_DEFAULT)
    ap.add_argument("--motif", default=MOTIF_DEFAULT)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional. If omitted, inferred from route-policy path, e.g. D:/paper3/data/route_policy/route1.json -> D:/paper3/data.",
    )
    ap.add_argument(
        "--source",
        choices=["auto", "probability", "path"],
        default="auto",
        help="auto prefers station_pair_reliability/pair_timeseries, then falls back to route path CSV.",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="Optional debug limit for number of pair CSVs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    route_policy_path = Path(args.route_policy).expanduser()
    data_root = Path(args.data_root).expanduser() if args.data_root else infer_data_root_from_route_policy(route_policy_path)
    motif_name = str(args.motif)

    policy = load_route_policy(route_policy_path)
    n = load_n_from_motif(data_root, policy, motif_name)
    option_policy = _resolve_option_policy(policy, N=n)
    rel_col = policy.rel_col

    source_dir, source_kind = resolve_source_dir(
        data_root,
        policy,
        motif_name,
        source_mode=args.source,
    )

    out_dir = resolve_output_dir(data_root, policy, motif_name, out_dir=args.out_dir)
    metrics_dir = out_dir / "_metrics"
    summary_csv = out_dir / "_summary.csv"
    failed_csv = out_dir / "_failed.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(source_dir.glob("*.csv"))
    if args.limit and args.limit > 0:
        csv_files = csv_files[: int(args.limit)]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {source_dir}")

    workers = int(args.workers or min(8, max(1, (os.cpu_count() or 4) - 1)))
    workers = max(1, workers)

    print(f"[batch] motif={motif_name}")
    print(f"[batch] route_policy={route_policy_path}")
    print(f"[batch] route_name={policy.route_name}")
    print(f"[batch] route_tag={route_tag(policy)}")
    print(f"[batch] source_kind={source_kind}")
    print(f"[batch] source_dir={source_dir}")
    print(f"[batch] out_dir={out_dir}")
    print(f"[batch] metrics_dir={metrics_dir}")
    print(f"[batch] rel_col={rel_col}")
    print(f"[batch] N={n}")
    print(f"[batch] files={len(csv_files)}")
    print(f"[batch] workers={workers}")

    common = {
        "out_dir": str(out_dir),
        "metrics_dir": str(metrics_dir),
        "n": int(n),
        "option_policy": option_policy,
        "rel_col": rel_col,
        "dpi": int(args.dpi),
    }

    tasks = []
    for p in csv_files:
        t = dict(common)
        t["csv_path"] = str(p)
        tasks.append(t)

    results = []

    if workers == 1:
        for i, task in enumerate(tasks, start=1):
            result = process_one_csv(task)
            results.append(result)
            _print_progress(i, len(tasks), result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one_csv, task): task for task in tasks}
            total = len(futures)

            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    result = fut.result()
                except Exception as exc:
                    task = futures[fut]
                    result = {
                        "status": "failed",
                        "csv_file": str(task["csv_path"]),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }

                results.append(result)
                _print_progress(i, total, result)

    ok_rows = [r for r in results if r["status"] == "ok"]
    failed_rows = [r for r in results if r["status"] == "failed"]

    pd.DataFrame(ok_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(failed_csv, index=False, encoding="utf-8-sig")

    print(f"[batch] success={len(ok_rows)} failed={len(failed_rows)}")
    print(f"[batch] summary={summary_csv}")
    print(f"[batch] failed={failed_csv}")


def _print_progress(i: int, total: int, result: dict[str, Any]) -> None:
    if result["status"] == "ok":
        print(
            f"[{i}/{total}] ok   {Path(result['csv_file']).name} "
            f"-> {Path(result['jpg_path']).name}"
        )
    else:
        print(
            f"[{i}/{total}] fail {Path(result['csv_file']).name} "
            f"-> {result.get('error', '')}"
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()


#  python -m paper3py.export_hops_and_probability_jpg2 `
# >>   --route-policy D:\paper3\data\route_policy\route1.json `
# >>   --motif "grid+"