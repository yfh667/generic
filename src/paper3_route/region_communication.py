from pathlib import Path
from typing import Any
import time
import re

import numpy as np
import pandas as pd

from .route_policy import (
    load_route_policy,
    path_output_dir,
    probability_pair_dir,
    region_communication_dir,
)
from .probability import (
    enrich_pair_csv_with_reliability,
    reliability_stats_from_series,
)
from src.model.route_policy_probability import build_delta2option
from .static_paths import load_g60_constants, load_station_groups_from_g60

def station_to_region_map() -> dict[int, Any]:
    groups = load_station_groups_from_g60()
    out: dict[int, Any] = {}
    for region, stations in groups.items():
        for sid in stations:
            out[int(sid)] = region
    return out


def _read_probability_csv(csv_path: Path, *, rel_col: str) -> pd.DataFrame:
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    desired = [
        "time", "region_a", "region_b", "station_a", "station_b",
        rel_col, "path", "intra_hops", "inter_hops", "total_hops", "min_shortest_path",
    ]
    usecols = [c for c in desired if c in cols]
    return pd.read_csv(csv_path, usecols=usecols)


def _ensure_region_columns(df: pd.DataFrame, station_region: dict[int, Any]) -> pd.DataFrame:
    out = df.copy()
    if "region_a" not in out.columns:
        out["region_a"] = pd.to_numeric(out["station_a"], errors="coerce").map(
            lambda x: station_region.get(int(x)) if pd.notna(x) else None
        )
    if "region_b" not in out.columns:
        out["region_b"] = pd.to_numeric(out["station_b"], errors="coerce").map(
            lambda x: station_region.get(int(x)) if pd.notna(x) else None
        )
    return out


def _resolve_option_policy_from_route(policy, *, N: int) -> dict[str, Any]:
    raw = dict(getattr(policy, "raw", {}) or {})
    lp = raw.get("link_probability", {}) if isinstance(raw.get("link_probability", {}), dict) else {}

    p_intra = float(lp.get("p_intra", raw.get("p_intra", policy.p_intra)))
    default_p_inter = float(
        lp.get(
            "default_p_inter",
            lp.get("p_inter", raw.get("default_p_inter", raw.get("p_inter", policy.p_inter))),
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
        "p_intra": p_intra,
        "default_p_inter": default_p_inter,
        "option_p_inter": option_p_inter,
        "delta2option": delta2option,
        "unknown_option_action": str(raw.get("unknown_option_action", "use_default")),
    }


def _normalize_region_label(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "region_unknown"
    if isinstance(v, (int, np.integer)):
        return f"region{int(v) + 1}"
    s = str(v).strip()
    m = re.fullmatch(r"region(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"region{int(m.group(1))}"
    if re.fullmatch(r"\d+", s):
        return f"region{int(s) + 1}"
    return s


def _hops_from_path(path_str: Any, *, N: int) -> tuple[float, float, float]:
    if not isinstance(path_str, str) or not path_str.strip():
        return np.nan, np.nan, np.nan
    try:
        nodes = [int(x) for x in path_str.split("->") if str(x).strip()]
    except Exception:
        return np.nan, np.nan, np.nan
    intra = 0
    inter = 0
    for u, v in zip(nodes, nodes[1:]):
        if int(u) // int(N) == int(v) // int(N):
            intra += 1
        else:
            inter += 1
    return float(intra), float(inter), float(intra + inter)


def _ensure_hop_columns(df: pd.DataFrame, *, N: int) -> pd.DataFrame:
    out = df.copy()
    has = {"intra_hops", "inter_hops", "total_hops"}.issubset(out.columns)
    if has:
        out["intra_hops"] = pd.to_numeric(out["intra_hops"], errors="coerce")
        out["inter_hops"] = pd.to_numeric(out["inter_hops"], errors="coerce")
        out["total_hops"] = pd.to_numeric(out["total_hops"], errors="coerce")
        return out

    if "path" not in out.columns:
        out["intra_hops"] = np.nan
        out["inter_hops"] = np.nan
        out["total_hops"] = np.nan
        return out

    path_s = out["path"].fillna("").astype(str)
    uniq = path_s.unique().tolist()
    hop_map = {p: _hops_from_path(p, N=N) for p in uniq}
    hops = path_s.map(hop_map)
    out["intra_hops"] = [x[0] for x in hops]
    out["inter_hops"] = [x[1] for x in hops]
    out["total_hops"] = [x[2] for x in hops]
    return out






def load_pair_reliability_timeseries(
    csv_path: Path,
    *,
    source_has_reliability: bool,
    rel_col: str,
    N: int,
    p_intra: float,
    default_p_inter: float,
    option_p_inter: dict[int, float],
    delta2option: dict[tuple[int, int], int],
    unknown_option_action: str,
    station_region: dict[int, Any],
) -> pd.DataFrame:
    if source_has_reliability:
        df = _read_probability_csv(csv_path, rel_col=rel_col)
    else:
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

    df = _ensure_hop_columns(df, N=N)
    df = _ensure_region_columns(df, station_region)
    df[rel_col] = pd.to_numeric(df[rel_col], errors="coerce")

    keep = ["time", "region_a", "region_b", "station_a", "station_b", rel_col, "intra_hops", "inter_hops", "total_hops"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

def build_region_pair_probability_timeseries(all_df: pd.DataFrame, *, rel_col: str) -> pd.DataFrame:
    d = all_df.copy()
    d["time"] = pd.to_numeric(d["time"], errors="coerce")
    d = d.dropna(subset=["time"])
    d["time"] = d["time"].astype(int)
    d["_rel"] = pd.to_numeric(d[rel_col], errors="coerce")

    g = d.groupby(["region_a", "region_b", "time"], dropna=False, sort=True)["_rel"]
    out = g.agg(
        pair_count="size",
        reachable_pair_count=lambda s: int(s.notna().sum()),
        mean_reliability="mean",
        p05_reliability=lambda s: s.quantile(0.05),
        p95_reliability=lambda s: s.quantile(0.95),
        min_reliability="min",
        max_reliability="max",
    ).reset_index()
    return out


def build_region_pair_hops_timeseries(all_df: pd.DataFrame, *, rel_col: str) -> pd.DataFrame:
    d = all_df.copy()
    d["time"] = pd.to_numeric(d["time"], errors="coerce")
    d = d.dropna(subset=["time"])
    d["time"] = d["time"].astype(int)
    d["_rel"] = pd.to_numeric(d[rel_col], errors="coerce")

    for c in ["intra_hops", "inter_hops", "total_hops"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    key = ["region_a", "region_b", "time"]

    base = d.groupby(key, dropna=False, sort=True).size().rename("pair_count").reset_index()

    r = d[d["_rel"].notna()].copy()
    hop = r.groupby(key, dropna=False, sort=True).agg(
        reachable_pair_count=("station_a", "size"),
        mean_total_hops_on_reachable=("total_hops", "mean"),
        mean_inter_hops_on_reachable=("inter_hops", "mean"),
        mean_intra_hops_on_reachable=("intra_hops", "mean"),
        p05_total_hops_on_reachable=("total_hops", lambda s: s.quantile(0.05)),
        p95_total_hops_on_reachable=("total_hops", lambda s: s.quantile(0.95)),
    ).reset_index()

    out = base.merge(hop, on=key, how="left")
    out["reachable_pair_count"] = out["reachable_pair_count"].fillna(0).astype(int)
    return out


def _write_per_region_pair_files(df: pd.DataFrame, *, out_dir: Path, encoding: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for (ra, rb), sub in df.groupby(["region_a", "region_b"], dropna=False, sort=True):
        fn = f"{ra}_to_{rb}_timeseries.csv".replace("/", "_")
        sub.sort_values("time").to_csv(out_dir / fn, index=False, encoding=encoding)
        n += 1
    return n

def build_region_global_stats(all_df: pd.DataFrame, *, rel_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ra, rb), g in all_df.groupby(["region_a", "region_b"], dropna=False, sort=True):
        rec: dict[str, Any] = {
            "region_a": ra,
            "region_b": rb,
            "station_pair_count": int(g[["station_a", "station_b"]].drop_duplicates().shape[0]),
        }
        rec.update(reliability_stats_from_series(g[rel_col]))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["region_a", "region_b"]).reset_index(drop=True)


def build_region_time_stats(all_df: pd.DataFrame, *, rel_col: str) -> pd.DataFrame:
    d = all_df.copy()
    d["time"] = pd.to_numeric(d["time"], errors="coerce")
    d = d.dropna(subset=["time"])
    d["time"] = d["time"].astype(int)

    grouped = d.groupby(["region_a", "region_b", "time"], dropna=False, sort=True)[rel_col]
    out = grouped.agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
    ).reset_index()
    q = grouped.quantile([0.05, 0.95]).unstack(level=-1).reset_index()
    q = q.rename(columns={0.05: "p05", 0.95: "p95"})
    out = out.merge(q, on=["region_a", "region_b", "time"], how="left")
    cols = ["region_a", "region_b", "time", "count", "mean", "median", "p05", "p95", "min", "max"]
    return out.loc[:, cols]
_SUM_COLS = [
    "pair_count",
    "reachable_pair_count",
    "rel_sum",
    "total_hops_sum",
    "total_hops_count",
    "inter_hops_sum",
    "inter_hops_count",
    "intra_hops_sum",
    "intra_hops_count",
]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(num, errors="coerce") / den


def _merge_region_pair_part(
    acc: dict[tuple[str, str], pd.DataFrame],
    key: tuple[str, str],
    part: pd.DataFrame,
) -> None:
    """
    把一个 station-pair CSV 汇总后的 time-index DataFrame 合并进全局累加器。
    acc 只保存 region-pair × time 的聚合量，不保存 pair-level 明细。
    """
    if part.empty:
        return

    part = part.sort_index()

    if key not in acc:
        acc[key] = part.copy()
        return

    cur = acc[key]
    idx = cur.index.union(part.index)

    cur = cur.reindex(idx)
    part = part.reindex(idx)

    for c in _SUM_COLS:
        cur[c] = cur[c].fillna(0.0) + part[c].fillna(0.0)

    cur["rel_min"] = pd.concat(
        [cur["rel_min"], part["rel_min"]],
        axis=1,
    ).min(axis=1, skipna=True)

    cur["rel_max"] = pd.concat(
        [cur["rel_max"], part["rel_max"]],
        axis=1,
    ).max(axis=1, skipna=True)

    acc[key] = cur


def _build_one_region_pair_part(
    df: pd.DataFrame,
    *,
    rel_col: str,
) -> pd.DataFrame:
    """
    把一个 region-pair 子表按 time 压缩成可累加的中间量。
    注意：这里只做 sum/count/min/max，避免保存全量明细。
    """
    d = df.copy()

    if "time" not in d.columns:
        raise ValueError("缺少列: time")
    if rel_col not in d.columns:
        raise ValueError(f"缺少列: {rel_col}")

    d["time"] = pd.to_numeric(d["time"], errors="coerce")
    d = d.dropna(subset=["time"])
    if d.empty:
        return pd.DataFrame()

    d["time"] = d["time"].astype(int)

    rel = pd.to_numeric(d[rel_col], errors="coerce")
    reachable = rel.notna()

    d["_pair_count"] = 1
    d["_reachable_pair_count"] = reachable.astype(int)
    d["_rel_sum"] = rel.fillna(0.0)
    d["_rel_min"] = rel
    d["_rel_max"] = rel

    for hop_col in ["total_hops", "inter_hops", "intra_hops"]:
        if hop_col in d.columns:
            hop = pd.to_numeric(d[hop_col], errors="coerce")
        else:
            hop = pd.Series(np.nan, index=d.index)

        valid_hop = reachable & hop.notna()
        d[f"_{hop_col}_sum"] = hop.where(valid_hop, 0.0)
        d[f"_{hop_col}_count"] = valid_hop.astype(int)

    part = d.groupby("time", sort=True).agg(
        pair_count=("_pair_count", "sum"),
        reachable_pair_count=("_reachable_pair_count", "sum"),
        rel_sum=("_rel_sum", "sum"),
        rel_min=("_rel_min", "min"),
        rel_max=("_rel_max", "max"),
        total_hops_sum=("_total_hops_sum", "sum"),
        total_hops_count=("_total_hops_count", "sum"),
        inter_hops_sum=("_inter_hops_sum", "sum"),
        inter_hops_count=("_inter_hops_count", "sum"),
        intra_hops_sum=("_intra_hops_sum", "sum"),
        intra_hops_count=("_intra_hops_count", "sum"),
    )

    return part


def _accumulate_region_pair_df(
    acc: dict[tuple[str, str], pd.DataFrame],
    df: pd.DataFrame,
    *,
    rel_col: str,
) -> int:
    """
    把一个 station-pair 文件读出的 df 流式累加进 acc。
    返回该 df 的行数，用于日志统计。
    """
    if df.empty:
        return 0

    d = df.copy()
    d["region_a"] = d["region_a"].map(_normalize_region_label)
    d["region_b"] = d["region_b"].map(_normalize_region_label)

    for (ra, rb), sub in d.groupby(["region_a", "region_b"], dropna=False, sort=False):
        key = (str(ra), str(rb))
        part = _build_one_region_pair_part(sub, rel_col=rel_col)
        _merge_region_pair_part(acc, key, part)

    return int(len(d))


def _finalize_probability_timeseries(
    key: tuple[str, str],
    acc_df: pd.DataFrame,
) -> pd.DataFrame:
    ra, rb = key
    out = pd.DataFrame(index=acc_df.index).reset_index(names="time")

    out["region_a"] = ra
    out["region_b"] = rb
    out["pair_count"] = acc_df["pair_count"].fillna(0).astype(int).to_numpy()
    out["reachable_pair_count"] = acc_df["reachable_pair_count"].fillna(0).astype(int).to_numpy()

    out["mean_reliability"] = _safe_divide(
        acc_df["rel_sum"],
        acc_df["reachable_pair_count"],
    ).to_numpy()

    out["min_reliability"] = acc_df["rel_min"].to_numpy()
    out["max_reliability"] = acc_df["rel_max"].to_numpy()

    cols = [
        "time",
        "region_a",
        "region_b",
        "pair_count",
        "reachable_pair_count",
        "mean_reliability",
        "min_reliability",
        "max_reliability",
    ]
    return out.loc[:, cols].sort_values("time").reset_index(drop=True)


def _finalize_hops_timeseries(
    key: tuple[str, str],
    acc_df: pd.DataFrame,
) -> pd.DataFrame:
    ra, rb = key
    out = pd.DataFrame(index=acc_df.index).reset_index(names="time")

    out["region_a"] = ra
    out["region_b"] = rb
    out["pair_count"] = acc_df["pair_count"].fillna(0).astype(int).to_numpy()
    out["reachable_pair_count"] = acc_df["reachable_pair_count"].fillna(0).astype(int).to_numpy()

    out["mean_total_hops_on_reachable"] = _safe_divide(
        acc_df["total_hops_sum"],
        acc_df["total_hops_count"],
    ).to_numpy()

    out["mean_inter_hops_on_reachable"] = _safe_divide(
        acc_df["inter_hops_sum"],
        acc_df["inter_hops_count"],
    ).to_numpy()

    out["mean_intra_hops_on_reachable"] = _safe_divide(
        acc_df["intra_hops_sum"],
        acc_df["intra_hops_count"],
    ).to_numpy()

    out["total_hops_observation_count"] = acc_df["total_hops_count"].fillna(0).astype(int).to_numpy()
    out["inter_hops_observation_count"] = acc_df["inter_hops_count"].fillna(0).astype(int).to_numpy()
    out["intra_hops_observation_count"] = acc_df["intra_hops_count"].fillna(0).astype(int).to_numpy()

    cols = [
        "time",
        "region_a",
        "region_b",
        "pair_count",
        "reachable_pair_count",
        "mean_total_hops_on_reachable",
        "mean_inter_hops_on_reachable",
        "mean_intra_hops_on_reachable",
        "total_hops_observation_count",
        "inter_hops_observation_count",
        "intra_hops_observation_count",
    ]
    return out.loc[:, cols].sort_values("time").reset_index(drop=True)


def _safe_region_pair_filename(ra: str, rb: str) -> str:
    return f"{ra}_to_{rb}_timeseries.csv".replace("/", "_").replace("\\", "_")


def _write_streaming_region_outputs(
    acc: dict[tuple[str, str], pd.DataFrame],
    *,
    out_dir: Path,
    encoding: str,
) -> dict[str, Any]:
    prob_dir = out_dir / "region_pair_probability_timeseries"
    hop_dir = out_dir / "region_pair_hops_timeseries"

    prob_dir.mkdir(parents=True, exist_ok=True)
    hop_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for key in sorted(acc.keys()):
        ra, rb = key
        acc_df = acc[key].sort_index()

        prob_df = _finalize_probability_timeseries(key, acc_df)
        hop_df = _finalize_hops_timeseries(key, acc_df)

        fn = _safe_region_pair_filename(ra, rb)

        prob_csv = prob_dir / fn
        hop_csv = hop_dir / fn

        prob_df.to_csv(prob_csv, index=False, encoding=encoding)
        hop_df.to_csv(hop_csv, index=False, encoding=encoding)

        summary_rows.append(
            {
                "region_a": ra,
                "region_b": rb,
                "time_count": int(prob_df["time"].nunique()),
                "mean_reliability_over_time": float(
                    pd.to_numeric(prob_df["mean_reliability"], errors="coerce").mean()
                ),
                "mean_total_hops_over_time": float(
                    pd.to_numeric(hop_df["mean_total_hops_on_reachable"], errors="coerce").mean()
                ),
                "mean_inter_hops_over_time": float(
                    pd.to_numeric(hop_df["mean_inter_hops_on_reachable"], errors="coerce").mean()
                ),
                "mean_intra_hops_over_time": float(
                    pd.to_numeric(hop_df["mean_intra_hops_on_reachable"], errors="coerce").mean()
                ),
                "probability_csv": str(prob_csv),
                "hops_csv": str(hop_csv),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "region_pair_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding=encoding)

    return {
        "probability_timeseries_dir": str(prob_dir),
        "hop_timeseries_dir": str(hop_dir),
        "summary_csv": str(summary_csv),
        "region_pair_count": int(len(acc)),
    }


def export_region_communication_for_motif(
    motif_name: str,
    *,
    data_root: str | Path,
    route_policy_path: str | Path,
    prefer_probability_timeseries: bool = True,
) -> dict[str, Any]:
    """
    流式导出区域对通信结果。

    输出：
    1. region_pair_probability_timeseries/<region_a>_to_<region_b>_timeseries.csv
    2. region_pair_hops_timeseries/<region_a>_to_<region_b>_timeseries.csv
    3. region_pair_summary.csv

    这个版本不再把所有 station-pair 明细 concat 成一个巨大 all_df。
    """
    t0 = time.time()

    policy = load_route_policy(route_policy_path)
    _, N, _ = load_g60_constants()

    rel_col = policy.rel_col
    opt_cfg = _resolve_option_policy_from_route(policy, N=N)

    prob_dir = probability_pair_dir(data_root, policy, motif_name)
    path_dir = path_output_dir(data_root, policy, motif_name)

    source_has_reliability = False
    source_dir = path_dir

    if prefer_probability_timeseries and prob_dir.exists() and list(prob_dir.glob("*.csv")):
        source_dir = prob_dir
        source_has_reliability = True
    elif not path_dir.exists():
        raise FileNotFoundError(
            f"Neither probability timeseries nor route path directory exists: {prob_dir} / {path_dir}"
        )

    pair_csvs = sorted(source_dir.glob("*.csv"))
    if not pair_csvs:
        raise FileNotFoundError(f"No pair CSV found under: {source_dir}")

    station_region = station_to_region_map()

    out_dir = region_communication_dir(data_root, policy, motif_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc: dict[tuple[str, str], pd.DataFrame] = {}

    total_rows = 0
    for i, csv_path in enumerate(pair_csvs, start=1):
        df = load_pair_reliability_timeseries(
            csv_path,
            source_has_reliability=source_has_reliability,
            rel_col=rel_col,
            N=N,
            p_intra=opt_cfg["p_intra"],
            default_p_inter=opt_cfg["default_p_inter"],
            option_p_inter=opt_cfg["option_p_inter"],
            delta2option=opt_cfg["delta2option"],
            unknown_option_action=opt_cfg["unknown_option_action"],
            station_region=station_region,
        )

        total_rows += _accumulate_region_pair_df(
            acc,
            df,
            rel_col=rel_col,
        )

        if i == 1 or i % 100 == 0 or i == len(pair_csvs):
            elapsed = time.time() - t0
            print(
                f"[{motif_name}] region communication: "
                f"{i}/{len(pair_csvs)} pair csvs, "
                f"rows={total_rows}, "
                f"elapsed={elapsed:.1f}s"
            )

    output_info = _write_streaming_region_outputs(
        acc,
        out_dir=out_dir,
        encoding=policy.encoding,
    )

    return {
        "motif": motif_name,
        "route_name": policy.route_name,
        "source_dir": str(source_dir),
        "source_has_reliability": bool(source_has_reliability),
        "pair_csvs": int(len(pair_csvs)),
        "rows_read": int(total_rows),
        "region_pair_count": int(output_info["region_pair_count"]),
        "probability_timeseries_dir": output_info["probability_timeseries_dir"],
        "hop_timeseries_dir": output_info["hop_timeseries_dir"],
        "summary_csv": output_info["summary_csv"],
        "elapsed_sec": round(time.time() - t0, 3),
    }


