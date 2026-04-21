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


def _compute_rel_from_paths(path_series: pd.Series, n: int, p_intra: float, p_inter: float, cache: dict) -> np.ndarray:
    arr = path_series.fillna("").astype(str).to_numpy()
    out = np.zeros(len(arr), dtype=np.float64)

    for i, path_str in enumerate(arr):
        path_str = path_str.strip()
        if not path_str:
            out[i] = 0.0
            continue

        v = cache.get(path_str)
        if v is None:
            try:
                intra_links, inter_links = get_intra_inter_link.parse_path_links(path_str, N=n)
                v = float((p_intra ** len(intra_links)) * (p_inter ** len(inter_links)))
            except Exception:
                v = 0.0
            cache[path_str] = v

        out[i] = v

    return out


def export_region_pair_prob_timeseries_grid_four(
    *,
    data_root=r"D:\paper3\data",
    topology_version="grid_four",
    csv_dir_name="region_pairs_0_86164",
    p_intra=0.999,
    p_inter=0.99,
):
    data_root = Path(data_root)
    pair_dir = data_root / "topology_design" / topology_version / "path" / csv_dir_name
    out_root = data_root / "topology_design" / topology_version / "analysis_link" / f"region_pair_prob_timeseries_simple_pi{p_intra}_pe{p_inter}"
    out_dir = out_root / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    station2region = _build_station_to_region(G60_CONFIG)
    n = _load_n(data_root, topology_version)

    files = sorted(pair_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到输入文件: {pair_dir}")

    # agg[(region_a, region_b)] = {"time":..., "rel_sum":..., "pair_cnt":...}
    agg = {}
    rel_cache = {}

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
        rel = _compute_rel_from_paths(d.loc[keep, "path"], n=n, p_intra=p_intra, p_inter=p_inter, cache=rel_cache)

        tmp = pd.DataFrame({"time": t, "rel": rel}).groupby("time", as_index=False)["rel"].mean()
        t = tmp["time"].to_numpy(dtype=np.int64)
        rel = tmp["rel"].to_numpy(dtype=np.float64)

        key = (ra, rb)
        if key not in agg:
            agg[key] = {
                "time": t,
                "rel_sum": rel.copy(),
                "pair_cnt": np.ones(len(t), dtype=np.float64),
            }
        else:
            a = agg[key]
            if not np.array_equal(a["time"], t):
                union_t = np.union1d(a["time"], t)

                a["rel_sum"] = pd.Series(a["rel_sum"], index=a["time"]).reindex(union_t, fill_value=0.0).to_numpy(dtype=np.float64)
                a["pair_cnt"] = pd.Series(a["pair_cnt"], index=a["time"]).reindex(union_t, fill_value=0.0).to_numpy(dtype=np.float64)

                rel = pd.Series(rel, index=t).reindex(union_t, fill_value=0.0).to_numpy(dtype=np.float64)
                t = union_t
                a["time"] = union_t

            a["rel_sum"] += rel
            a["pair_cnt"] += 1.0

        if idx % 50 == 0 or idx == len(files):
            print(f"[{topology_version}] {idx}/{len(files)}")

    summary_rows = []
    for (ra, rb), a in sorted(agg.items(), key=lambda x: (_region_sort_key(x[0][0]), _region_sort_key(x[0][1]))):
        den = np.maximum(a["pair_cnt"], 1.0)
        mean_rel = a["rel_sum"] / den

        out_df = pd.DataFrame(
            {
                "time": a["time"].astype(np.int64),
                "region_a": ra,
                "region_b": rb,
                "pair_count": a["pair_cnt"].astype(np.int64),
                "mean_reliability": mean_rel,
            }
        )
        out_df.to_csv(out_dir / f"{ra}_to_{rb}_timeseries.csv", index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "region_a": ra,
                "region_b": rb,
                "time_points": int(len(out_df)),
                "mean_of_mean_reliability": float(out_df["mean_reliability"].mean()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] output -> {out_root}")


# export_region_pair_prob_timeseries_grid_four(
#     data_root=r"D:\paper3\data",
#     topology_version="grid_four",
#     csv_dir_name="region_pairs_0_86164",
#     p_intra=0.999,
#     p_inter=0.99,
# )
# export_region_pair_prob_timeseries_grid_four(
#     data_root=r"D:\paper3\data",
#     topology_version="gridx",
#     csv_dir_name="region_pairs_0_86164",
#     p_intra=0.999,
#     p_inter=0.99,
# )
# export_region_pair_prob_timeseries_grid_four(
#     data_root=r"D:\paper3\data",
#     topology_version="grid_plane_alternating",
#     csv_dir_name="region_pairs_0_86164",
#     p_intra=0.999,
#     p_inter=0.99,
# )
# export_region_pair_prob_timeseries_grid_four(
#     data_root=r"D:\paper3\data",
#     topology_version="grid_x_sparse",
#     csv_dir_name="region_pairs_0_86164",
#     p_intra=0.999,
#     p_inter=0.99,
# )
# export_region_pair_prob_timeseries_grid_four(
#     data_root=r"D:\paper3\data",
#     topology_version="grid+",
#     csv_dir_name="region_pairs_0_86164",
#     p_intra=0.999,
#     p_inter=0.99,
# )