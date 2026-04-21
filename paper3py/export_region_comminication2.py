from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import src.model.route_policy_probability as route_policy_probability

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

def _compute_rel_from_paths(
    path_series: pd.Series,
    *,
    n: int,
    p_intra: float,
    default_p_inter: float,
    option_p_inter: dict,
    delta2option: dict,
    cache: dict,
) -> np.ndarray:
    arr = path_series.fillna("").astype(str).to_numpy()
    out = np.zeros(len(arr), dtype=np.float64)

    for i, path_str in enumerate(arr):
        path_str = path_str.strip()
        if not path_str:
            out[i] = 0.0
            continue

        v = cache.get(path_str)
        if v is None:
            rel, _, _, _, _ = route_policy_probability.compute_path_reliability_and_hops(
                path_str,
                n=n,
                p_intra=p_intra,
                default_p_inter=default_p_inter,
                option_p_inter=option_p_inter,
                delta2option=delta2option,
                unknown_option_action="use_default",
            )
            v = float(rel)
            cache[path_str] = v

        out[i] = v

    return out



def export_region_pair_prob_timeseries_grid_four(
    *,
    data_root=r"D:\paper3\data",
    topology_version="grid_four",
    csv_dir_name="region_pairs_0_86164",
    # p_intra=0.999,
    # p_inter=0.99,
):
    data_root = Path(data_root)
    pair_dir = data_root / "topology_design" / topology_version / "path" / csv_dir_name
    station2region = _build_station_to_region(G60_CONFIG)
    n = _load_n(data_root, topology_version)

    policy_path = data_root / "topology_design" / topology_version / "config" / "route_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"缺少策略文件: {policy_path}")

    policy = route_policy_probability.load_route_policy(
        policy_path,
        n=n,
        fallback_p_intra=1.0,
        fallback_p_inter=1.0,
    )

    p_intra_used = float(policy["p_intra"])
    default_p_inter_used = policy["default_p_inter"]
    option_p_inter = dict(policy["option_p_inter"])
    delta2option = dict(policy["delta2option"])
    policy_name = str(policy["policy_name"]).replace(" ", "_")


   # out_root = data_root / "topology_design" / topology_version / "analysis_link" / f"region_pair_prob_timeseries_simple_pi{p_intra}_pe{p_inter}"
    out_root = data_root / "topology_design" / topology_version / "analysis_link" / f"region_pair_prob_timeseries_policy_{policy_name}"

    out_dir = out_root / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)




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
      #  rel = _compute_rel_from_paths(d.loc[keep, "path"], n=n, p_intra=p_intra, p_inter=p_inter, cache=rel_cache)
        rel = _compute_rel_from_paths(
            d.loc[keep, "path"],
            n=n,
            p_intra=p_intra_used,
            default_p_inter=default_p_inter_used,
            option_p_inter=option_p_inter,
            delta2option=delta2option,
            cache=rel_cache,
        )

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

# 文件末尾新增
if __name__ == "__main__":
    import argparse
    import sys

    # 默认配置：直接点运行就用这组
    DEFAULT_DATA_ROOT = r"D:\paper3\data"
    DEFAULT_TOPOLOGY_VERSION = "grid_four"
    DEFAULT_CSV_DIR_NAME = "region_pairs_0_86164_minhop_v1"

    # 无参数：直接跑默认配置
    if len(sys.argv) == 1:
        export_region_pair_prob_timeseries_grid_four(
            data_root=DEFAULT_DATA_ROOT,
            topology_version=DEFAULT_TOPOLOGY_VERSION,
            csv_dir_name=DEFAULT_CSV_DIR_NAME,
        )
    else:
        # 有参数：可覆盖默认配置
        ap = argparse.ArgumentParser()
        ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
        ap.add_argument("--topology-version", default=DEFAULT_TOPOLOGY_VERSION)
        ap.add_argument("--csv-dir-name", default=DEFAULT_CSV_DIR_NAME)
        args = ap.parse_args()

        export_region_pair_prob_timeseries_grid_four(
            data_root=args.data_root,
            topology_version=args.topology_version,
            csv_dir_name=args.csv_dir_name,
        )







# python C:\user\generic\paper3py\export_region_comminication2.py `
#   --data-root D:\paper3\data `
#   --topology-version grid_four `
#   --csv-dir-name region_pairs_0_86164_maxrel_option_weighted_v1
