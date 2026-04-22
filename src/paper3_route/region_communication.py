from __future__ import annotations

from pathlib import Path
from typing import Any
import time

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
    desired = ["time", "region_a", "region_b", "station_a", "station_b", rel_col]
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


def load_pair_reliability_timeseries(
    csv_path: Path,
    *,
    source_has_reliability: bool,
    rel_col: str,
    N: int,
    p_intra: float,
    p_inter: float,
    station_region: dict[int, Any],
) -> pd.DataFrame:
    if source_has_reliability:
        df = _read_probability_csv(csv_path, rel_col=rel_col)
    else:
        df = enrich_pair_csv_with_reliability(
            csv_path,
            N=N,
            p_intra=p_intra,
            p_inter=p_inter,
            rel_col=rel_col,
        )
        keep = [c for c in ["time", "region_a", "region_b", "station_a", "station_b", rel_col] if c in df.columns]
        df = df.loc[:, keep].copy()

    df = _ensure_region_columns(df, station_region)
    df[rel_col] = pd.to_numeric(df[rel_col], errors="coerce")
    return df[["time", "region_a", "region_b", "station_a", "station_b", rel_col]].copy()


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


def export_region_communication_for_motif(
    motif_name: str,
    *,
    data_root: str | Path,
    route_policy_path: str | Path,
    prefer_probability_timeseries: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    policy = load_route_policy(route_policy_path)
    _, N, _ = load_g60_constants()
    rel_col = policy.rel_col

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
    frames: list[pd.DataFrame] = []
    for p in pair_csvs:
        frames.append(
            load_pair_reliability_timeseries(
                p,
                source_has_reliability=source_has_reliability,
                rel_col=rel_col,
                N=N,
                p_intra=policy.p_intra,
                p_inter=policy.p_inter,
                station_region=station_region,
            )
        )

    all_df = pd.concat(frames, ignore_index=True)
    out_dir = region_communication_dir(data_root, policy, motif_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_region_global_stats(all_df, rel_col=rel_col)
    time_df = build_region_time_stats(all_df, rel_col=rel_col)

    global_path = out_dir / "region_pair_global_stat.csv"
    time_path = out_dir / "region_pair_timeseries_stat.csv"
    raw_path = out_dir / "region_pair_probability_raw.csv"

    global_df.to_csv(global_path, index=False, encoding=policy.encoding)
    time_df.to_csv(time_path, index=False, encoding=policy.encoding)

    # This raw file keeps the minimum intermediate data needed to reproduce
    # region-level probability distributions without re-parsing paths.
    all_df.to_csv(raw_path, index=False, encoding=policy.encoding)

    return {
        "motif": motif_name,
        "route_name": policy.route_name,
        "source_dir": str(source_dir),
        "source_has_reliability": bool(source_has_reliability),
        "pair_csvs": len(pair_csvs),
        "rows": int(len(all_df)),
        "region_global_csv": str(global_path),
        "region_timeseries_csv": str(time_path),
        "region_raw_csv": str(raw_path),
        "elapsed_sec": round(time.time() - t0, 3),
    }
