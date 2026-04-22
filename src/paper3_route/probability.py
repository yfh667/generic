from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import math
import re
import time

import numpy as np
import pandas as pd

from .route_policy import (
    load_route_policy as load_run_policy,
    path_output_dir,
    probability_pair_dir,
    global_stat_dir,
)
from src.model.route_policy_probability import (
    build_delta2option,
    compute_path_reliability_and_hops,
)

from .static_paths import load_g60_constants


def count_hops_from_path_str(path_str: Any, *, N: int) -> tuple[float, float, float]:
    if not isinstance(path_str, str) or not path_str.strip():
        return math.nan, math.nan, math.nan
    try:
        nodes = [int(x) for x in path_str.split("->") if str(x).strip()]
    except Exception:
        return math.nan, math.nan, math.nan
    if len(nodes) < 1:
        return math.nan, math.nan, math.nan
    intra = 0
    inter = 0
    for u, v in zip(nodes, nodes[1:]):
        if int(u) // int(N) == int(v) // int(N):
            intra += 1
        else:
            inter += 1
    return float(intra), float(inter), float(intra + inter)


def _read_csv_columns(csv_path: str | Path) -> list[str]:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def _parse_station_pair_from_filename(path: str | Path) -> tuple[int | None, int | None]:
    m = re.match(r".*station(\d+)-.*station(\d+)", Path(path).stem)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def read_route_pair_csv_minimal(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    cols = _read_csv_columns(csv_path)
    desired = [
        "time",
        "region_a",
        "region_b",
        "station_a",
        "station_b",
        "path",
        "intra_hops",
        "inter_hops",
        "total_hops",
        "min_shortest_path",
    ]
    usecols = [c for c in desired if c in cols]
    df = pd.read_csv(csv_path, usecols=usecols if usecols else None)

    if "station_a" not in df.columns or "station_b" not in df.columns:
        sa, sb = _parse_station_pair_from_filename(csv_path)
        if sa is None or sb is None:
            raise ValueError(f"Cannot infer station pair from {csv_path.name}")
        df["station_a"] = sa
        df["station_b"] = sb
    return df

def _resolve_option_policy(run_policy, *, N: int) -> dict[str, Any]:
    raw = dict(run_policy.raw or {})
    lp = raw.get("link_probability", {}) if isinstance(raw.get("link_probability", {}), dict) else {}

    p_intra = float(lp.get("p_intra", raw.get("p_intra", run_policy.p_intra)))
    default_p_inter = float(
        lp.get(
            "default_p_inter",
            lp.get("p_inter", raw.get("default_p_inter", raw.get("p_inter", run_policy.p_inter))),
        )
    )
    option_raw = lp.get("option_p_inter", raw.get("option_p_inter", {})) or {}
    option_p_inter = {int(k): float(v) for k, v in option_raw.items()}

    option_delta_map = raw.get("option_delta_map", None)
    delta2option = build_delta2option(
        {"option_delta_map": option_delta_map} if option_delta_map is not None else {},
        n=int(N),
    )

    return {
        "policy_name": str(raw.get("policy_name", raw.get("route_name", run_policy.route_name))),
        "p_intra": p_intra,
        "default_p_inter": default_p_inter,
        "option_p_inter": option_p_inter,
        "delta2option": delta2option,
        "unknown_option_action": str(raw.get("unknown_option_action", "use_default")),
    }

def ensure_hop_columns(df: pd.DataFrame, *, N: int, path_col: str = "path") -> pd.DataFrame:
    out = df.copy()
    has_hops = {"intra_hops", "inter_hops"}.issubset(out.columns)
    if has_hops:
        out["intra_hops"] = pd.to_numeric(out["intra_hops"], errors="coerce")
        out["inter_hops"] = pd.to_numeric(out["inter_hops"], errors="coerce")
        if "total_hops" not in out.columns:
            out["total_hops"] = out["intra_hops"] + out["inter_hops"]
        else:
            out["total_hops"] = pd.to_numeric(out["total_hops"], errors="coerce")
        return out

    if path_col not in out.columns:
        out[path_col] = ""

    unique_paths = out[path_col].fillna("").astype(str).unique().tolist()
    hop_map = {p: count_hops_from_path_str(p, N=N) for p in unique_paths}
    hop_df = out[path_col].fillna("").astype(str).map(hop_map)
    out["intra_hops"] = [x[0] for x in hop_df]
    out["inter_hops"] = [x[1] for x in hop_df]
    out["total_hops"] = [x[2] for x in hop_df]
    return out


# def add_reliability_column(
#     df: pd.DataFrame,
#     *,
#     p_intra: float,
#     p_inter: float,
#     rel_col: str,
# ) -> pd.DataFrame:
#     out = df.copy()
#     intra = pd.to_numeric(out["intra_hops"], errors="coerce")
#     inter = pd.to_numeric(out["inter_hops"], errors="coerce")
#     rel = (float(p_intra) ** intra.astype(float)) * (float(p_inter) ** inter.astype(float))
#
#     if "path" in out.columns:
#         empty = ~out["path"].fillna("").astype(str).str.strip().astype(bool)
#         rel.loc[empty] = np.nan
#     if "min_shortest_path" in out.columns:
#         missing = pd.to_numeric(out["min_shortest_path"], errors="coerce").isna()
#         rel.loc[missing] = np.nan
#
#     out[rel_col] = rel
#     return out
def add_reliability_column(
    df: pd.DataFrame,
    *,
    N: int,
    p_intra: float,
    default_p_inter: float,
    option_p_inter: dict[int, float],
    delta2option: dict[tuple[int, int], int],
    rel_col: str,
    unknown_option_action: str = "use_default",
) -> pd.DataFrame:
    out = df.copy()
    if "path" not in out.columns:
        out["path"] = ""

    path_s = out["path"].fillna("").astype(str)
    uniq_paths = path_s.unique().tolist()

    cache: dict[str, dict[str, Any]] = {}
    option_cols: set[str] = set()

    for p in uniq_paths:
        if not p.strip():
            cache[p] = {
                "rel": math.nan,
                "intra_hops": math.nan,
                "inter_hops": math.nan,
                "total_hops": math.nan,
                "unknown_option_edges": 0,
            }
            continue

        rel, intra, inter, unknown_cnt, option_counter = compute_path_reliability_and_hops(
            p,
            n=int(N),
            p_intra=float(p_intra),
            default_p_inter=float(default_p_inter),
            option_p_inter=option_p_inter,
            delta2option=delta2option,
            unknown_option_action=unknown_option_action,
        )

        row = {
            "rel": float(rel),
            "intra_hops": float(intra),
            "inter_hops": float(inter),
            "total_hops": float(intra + inter),
            "unknown_option_edges": int(unknown_cnt),
        }
        for op, cnt in option_counter.items():
            c = f"option{int(op)}_hops"
            row[c] = int(cnt)
            option_cols.add(c)

        cache[p] = row

    out[rel_col] = path_s.map(lambda x: cache[x]["rel"])
    out["intra_hops"] = path_s.map(lambda x: cache[x]["intra_hops"])
    out["inter_hops"] = path_s.map(lambda x: cache[x]["inter_hops"])
    out["total_hops"] = path_s.map(lambda x: cache[x]["total_hops"])
    out["unknown_option_edges"] = path_s.map(lambda x: cache[x]["unknown_option_edges"])

    for c in sorted(option_cols):
        out[c] = path_s.map(lambda x: cache[x].get(c, 0))

    if "min_shortest_path" in out.columns:
        missing = pd.to_numeric(out["min_shortest_path"], errors="coerce").isna()
        out.loc[missing, [rel_col, "intra_hops", "inter_hops", "total_hops"]] = np.nan

    return out


def reliability_stats_from_series(x: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "p10": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max": math.nan,
            "time_ratio_rel_lt_0.95": math.nan,
            "time_ratio_rel_lt_0.98": math.nan,
            "time_ratio_rel_ge_0.99": math.nan,
        }
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "std": float(v.std(ddof=1)),
        "min": float(v.min()),
        "p05": float(v.quantile(0.05)),
        "p10": float(v.quantile(0.10)),
        "p90": float(v.quantile(0.90)),
        "p95": float(v.quantile(0.95)),
        "max": float(v.max()),
        "time_ratio_rel_lt_0.95": float((v < 0.95).mean()),
        "time_ratio_rel_lt_0.98": float((v < 0.98).mean()),
        "time_ratio_rel_ge_0.99": float((v >= 0.99).mean()),
    }


def enrich_pair_csv_with_reliability(
    csv_path: str | Path,
    *,
    N: int,
    p_intra: float,
    default_p_inter: float,
    option_p_inter: dict[int, float],
    delta2option: dict[tuple[int, int], int],
    unknown_option_action: str,
    rel_col: str,
) -> pd.DataFrame:
    df = read_route_pair_csv_minimal(csv_path)
    df = add_reliability_column(
        df,
        N=N,
        p_intra=p_intra,
        default_p_inter=default_p_inter,
        option_p_inter=option_p_inter,
        delta2option=delta2option,
        rel_col=rel_col,
        unknown_option_action=unknown_option_action,
    )
    return df



def process_one_pair_csv(
    csv_path: str | Path,
    *,
    out_pair_dir: str | Path,
    N: int,
    p_intra: float,
    default_p_inter: float,
    option_p_inter: dict[int, float],
    delta2option: dict[tuple[int, int], int],
    unknown_option_action: str,
    policy_name: str,
    rel_col: str,
    save_timeseries: bool,
    encoding: str,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    df = enrich_pair_csv_with_reliability(
        csv_path,
        N=N,
        p_intra=p_intra,
        default_p_inter=default_p_inter,
        option_p_inter=option_p_inter,
        delta2option=delta2option,
        unknown_option_action=unknown_option_action,
        rel_col=rel_col,
    )

    if save_timeseries:
        out_pair_dir = Path(out_pair_dir)
        out_pair_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_pair_dir / csv_path.name, index=False, encoding=encoding)

    rec: dict[str, Any] = {
        "pair_csv_name": csv_path.name,
        "station_a": int(pd.to_numeric(df["station_a"], errors="coerce").dropna().iloc[0]),
        "station_b": int(pd.to_numeric(df["station_b"], errors="coerce").dropna().iloc[0]),
        "policy_name": str(policy_name),
        "p_intra_used": float(p_intra),
        "default_p_inter_used": float(default_p_inter),
    }

    if "region_a" in df.columns:
        ra = df["region_a"].dropna()
        rec["region_a"] = ra.iloc[0] if not ra.empty else ""
    if "region_b" in df.columns:
        rb = df["region_b"].dropna()
        rec["region_b"] = rb.iloc[0] if not rb.empty else ""

    rec.update(reliability_stats_from_series(df[rel_col]))

    if "unknown_option_edges" in df.columns:
        rec["unknown_option_edge_total"] = int(
            pd.to_numeric(df["unknown_option_edges"], errors="coerce").fillna(0).sum()
        )

    for c in df.columns:
        if re.match(r"^option\d+_hops$", c):
            rec[f"{c}_total"] = int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())

    return rec


def export_global_probability_stats_for_motif(
    motif_name: str,
    *,
    data_root: str | Path,
    route_policy_path: str | Path,
    save_timeseries: bool = True,
    pair_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    policy = load_run_policy(route_policy_path)

    _, N, _ = load_g60_constants()

    in_dir = path_output_dir(data_root, policy, motif_name)
    if not in_dir.exists():
        raise FileNotFoundError(f"Route path directory not found: {in_dir}")
    pair_csvs = sorted(in_dir.glob("*.csv"))
    if not pair_csvs:
        raise FileNotFoundError(f"No pair CSV found under: {in_dir}")

    out_pair_dir = probability_pair_dir(data_root, policy, motif_name)
    out_stat_dir = global_stat_dir(data_root, policy, motif_name)
    out_stat_dir.mkdir(parents=True, exist_ok=True)
    rel_col = policy.rel_col
    opt = _resolve_option_policy(policy, N=N)

    workers = max(1, int(pair_workers or policy.pair_workers))
    common = dict(
        out_pair_dir=out_pair_dir,
        N=N,
        p_intra=opt["p_intra"],
        default_p_inter=opt["default_p_inter"],
        option_p_inter=opt["option_p_inter"],
        delta2option=opt["delta2option"],
        unknown_option_action=opt["unknown_option_action"],
        policy_name=opt["policy_name"],
        rel_col=rel_col,
        save_timeseries=save_timeseries,
        encoding=policy.encoding,
    )

    rows: list[dict[str, Any]] = []
    if workers == 1:
        for p in pair_csvs:
            rows.append(process_one_pair_csv(p, **common))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one_pair_csv, p, **common): p for p in pair_csvs}
            for fut in as_completed(futures):
                rows.append(fut.result())

    stat_df = pd.DataFrame(rows)
    sort_cols = [c for c in ["region_a", "region_b", "station_a", "station_b"] if c in stat_df.columns]
    if sort_cols:
        stat_df = stat_df.sort_values(sort_cols).reset_index(drop=True)

    stat_path = out_stat_dir / "all_pair_global_stat.csv"
    stat_df.to_csv(stat_path, index=False, encoding=policy.encoding)

    return {
        "motif": motif_name,
        "route_name": policy.route_name,
        "input_dir": str(in_dir),
        "pair_csvs": len(pair_csvs),
        "saved_pair_timeseries": bool(save_timeseries),
        "pair_timeseries_dir": str(out_pair_dir) if save_timeseries else "",
        "global_stat_csv": str(stat_path),
        "elapsed_sec": round(time.time() - t0, 3),
    }
