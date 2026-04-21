from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

import src.model.get_intra_inter_link as get_intra_inter_link
from src.config.viewer_config import G60_CONFIG


PAIR_RE = re.compile(
    r"^region\d+--station(?P<sa>\d+)-region\d+--station(?P<sb>\d+)\.csv$"
)
REGION_RE = re.compile(r"region(\d+)$")


def _region_sort_key(r: str):
    m = REGION_RE.match(str(r))
    return (int(m.group(1)) if m else 9999, str(r))


def _build_station_to_region(cfg) -> dict[int, str]:
    out = {}
    for gid, info in cfg.station_groups.items():
        rn = f"region{int(gid) + 1}"
        for s in info["stations"]:
            out[int(s)] = rn
    return out


def _load_n(data_root: Path, topology_version: str) -> int:
    p = data_root / "topology_design" / topology_version / "config" / "motif.json"
    return int(json.loads(p.read_text(encoding="utf-8"))["N"])


def _compute_hops_from_paths(path_series: pd.Series, n: int, cache: dict):
    arr = path_series.fillna("").astype(str).to_numpy()
    intra_h = np.full(len(arr), np.nan, dtype=np.float64)
    inter_h = np.full(len(arr), np.nan, dtype=np.float64)
    total_h = np.full(len(arr), np.nan, dtype=np.float64)
    reachable = np.zeros(len(arr), dtype=np.float64)

    for i, path_str in enumerate(arr):
        path_str = path_str.strip()
        if not path_str:
            continue

        v = cache.get(path_str)
        if v is None:
            try:
                intra_links, inter_links = get_intra_inter_link.parse_path_links(path_str, N=n)
                ih = float(len(intra_links))
                eh = float(len(inter_links))
                th = ih + eh
                v = (ih, eh, th, 1.0)
            except Exception:
                v = (np.nan, np.nan, np.nan, 0.0)
            cache[path_str] = v

        ih, eh, th, ok = v
        intra_h[i] = ih
        inter_h[i] = eh
        total_h[i] = th
        reachable[i] = ok

    return intra_h, inter_h, total_h, reachable


def export_region_pair_hop_timeseries_grid_four(
    *,
    data_root=r"D:\paper3\data",
    topology_version="grid_four",
    csv_dir_name="region_pairs_0_86164",
):
    data_root = Path(data_root)
    pair_dir = data_root / "topology_design" / topology_version / "path" / csv_dir_name
    out_root = data_root / "topology_design" / topology_version / "analysis_link" / "region_pair_hop_timeseries_simple"
    out_dir = out_root / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    station2region = _build_station_to_region(G60_CONFIG)
    n = _load_n(data_root, topology_version)

    files = sorted(pair_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到输入文件: {pair_dir}")

    # agg[(region_a, region_b)] = time 对齐后的累计量
    agg = {}
    hop_cache = {}

    for idx, f in enumerate(files, start=1):
        m = PAIR_RE.match(f.name)
        if not m:
            continue

        sa = int(m.group("sa"))
        sb = int(m.group("sb"))

        ra = station2region.get(sa)
        rb = station2region.get(sb)
        if ra is None or rb is None or ra == rb:
            continue

        if _region_sort_key(ra) > _region_sort_key(rb):
            ra, rb = rb, ra

        d = pd.read_csv(f, usecols=["time", "path"])
        t = pd.to_numeric(d["time"], errors="coerce")
        keep = t.notna()
        if not keep.any():
            continue

        t = t[keep].astype(np.int64).to_numpy()
        intra_h, inter_h, total_h, reachable = _compute_hops_from_paths(d.loc[keep, "path"], n=n, cache=hop_cache)

        tmp = pd.DataFrame(
            {
                "time": t,
                "intra_hop": intra_h,
                "inter_hop": inter_h,
                "total_hop": total_h,
                "reachable": reachable,
            }
        )
        tmp["reachable_flag"] = tmp["reachable"] > 0.5

        grp = tmp.groupby("time", as_index=False).agg(
            intra_hop_sum=("intra_hop", "sum"),
            inter_hop_sum=("inter_hop", "sum"),
            total_hop_sum=("total_hop", "sum"),
            reachable_pair_count=("reachable_flag", "sum"),
            pair_count=("time", "size"),
        )

        cur = {
            "time": grp["time"].to_numpy(dtype=np.int64),
            "intra_hop_sum": grp["intra_hop_sum"].to_numpy(dtype=np.float64),
            "inter_hop_sum": grp["inter_hop_sum"].to_numpy(dtype=np.float64),
            "total_hop_sum": grp["total_hop_sum"].to_numpy(dtype=np.float64),
            "reachable_pair_count": grp["reachable_pair_count"].to_numpy(dtype=np.float64),
            "pair_count": grp["pair_count"].to_numpy(dtype=np.float64),
        }

        key = (ra, rb)
        if key not in agg:
            agg[key] = cur
        else:
            a = agg[key]
            union_t = np.union1d(a["time"], cur["time"])

            for k in ["intra_hop_sum", "inter_hop_sum", "total_hop_sum", "reachable_pair_count", "pair_count"]:
                a[k] = pd.Series(a[k], index=a["time"]).reindex(union_t, fill_value=0.0).to_numpy(dtype=np.float64)
                cur[k] = pd.Series(cur[k], index=cur["time"]).reindex(union_t, fill_value=0.0).to_numpy(dtype=np.float64)

            a["time"] = union_t
            for k in ["intra_hop_sum", "inter_hop_sum", "total_hop_sum", "reachable_pair_count", "pair_count"]:
                a[k] += cur[k]

        if idx % 50 == 0 or idx == len(files):
            print(f"[{topology_version}] {idx}/{len(files)}")

    summary_rows = []
    for (ra, rb), a in sorted(agg.items(), key=lambda x: (_region_sort_key(x[0][0]), _region_sort_key(x[0][1]))):
        reachable_den = a["reachable_pair_count"]
        pair_den = a["pair_count"]

        mean_total = np.divide(
            a["total_hop_sum"], reachable_den,
            out=np.full_like(reachable_den, np.nan, dtype=np.float64),
            where=reachable_den > 0,
        )
        mean_intra = np.divide(
            a["intra_hop_sum"], reachable_den,
            out=np.full_like(reachable_den, np.nan, dtype=np.float64),
            where=reachable_den > 0,
        )
        mean_inter = np.divide(
            a["inter_hop_sum"], reachable_den,
            out=np.full_like(reachable_den, np.nan, dtype=np.float64),
            where=reachable_den > 0,
        )
        reachable_ratio = np.divide(
            reachable_den, pair_den,
            out=np.zeros_like(pair_den, dtype=np.float64),
            where=pair_den > 0,
        )

        out_df = pd.DataFrame(
            {
                "time": a["time"].astype(np.int64),
                "region_a": ra,
                "region_b": rb,
                "pair_count": pair_den.astype(np.int64),
                "reachable_pair_count": reachable_den.astype(np.int64),
                "reachable_ratio": reachable_ratio,
                "mean_total_hops_on_reachable": mean_total,
                "mean_intra_hops_on_reachable": mean_intra,
                "mean_inter_hops_on_reachable": mean_inter,
            }
        )
        out_df.to_csv(out_dir / f"{ra}_to_{rb}_timeseries.csv", index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "region_a": ra,
                "region_b": rb,
                "time_points": int(len(out_df)),
                "mean_of_mean_total_hops_on_reachable": float(np.nanmean(out_df["mean_total_hops_on_reachable"])),
                "mean_reachable_ratio": float(np.nanmean(out_df["reachable_ratio"])),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] output -> {out_root}")


# ===== 批量 motif（和你原脚本风格一致）=====
# for tv in ["gridx", "grid_plane_alternating", "grid_x_sparse", "grid+",]:
#     export_region_pair_hop_timeseries_grid_four(
#         data_root=r"D:\paper3\data",
#         topology_version=tv,
#         csv_dir_name="region_pairs_0_86164",
#     )
for tv in ["grid_four"]:
    export_region_pair_hop_timeseries_grid_four(
        data_root=r"D:\paper3\data",
        topology_version=tv,
        csv_dir_name="region_pairs_0_86164",
    )
