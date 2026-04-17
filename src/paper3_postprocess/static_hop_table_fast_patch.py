from pathlib import Path
import csv
import numpy as np
import pandas as pd


def _to_i(x):
    try:
        return int(x)
    except Exception:
        return None


def reconstruct_path(next_hop, s, d):
    n = next_hop.shape[0]
    s = int(s)
    d = int(d)
    if s < 0 or d < 0 or s >= n or d >= n:
        return []

    nh = int(next_hop[s, d])
    if nh < 0:
        return []

    path = [s]
    cur = s
    guard = 0
    while cur != d:
        cur = int(next_hop[cur, d])
        if cur < 0:
            return []
        path.append(cur)
        guard += 1
        if guard > n:
            return []
    return path


def prepare_series_index(series_by_station, n):
    """把 station/time 的可见卫星集合一次性转成 numpy 数组，避免内层循环重复转换。"""
    empty = np.empty(0, dtype=np.int16)
    out = {}

    for sid, ts in series_by_station.items():
        cur = {}
        for t, sats in ts.items():
            arr = np.fromiter(
                (x for x in (_to_i(v) for v in sats) if x is not None and 0 <= x < n),
                dtype=np.int16,
            )
            cur[int(t)] = arr if arr.size else empty
        out[int(sid)] = cur

    return out, empty


def compute_region_pair_timeseries_fast(
    dist,
    next_hop,
    series_by_station,
    station_pairs,
    steps,
    include_path=True,
    include_path_indexed=False,
):
    n = dist.shape[0]
    series_idx, empty = prepare_series_index(series_by_station, n)
    path_cache = {}
    rows = []

    for t in steps:
        t = int(t)
        for sa, sb in station_pairs:
            sa = int(sa)
            sb = int(sb)
            ai = series_idx.get(sa, {}).get(t, empty)
            bi = series_idx.get(sb, {}).get(t, empty)

            best_s = ""
            best_d = ""
            path = ""
            path_indexed = ""

            if ai.size and bi.size:
                sub = dist[np.ix_(ai, bi)]
                k = int(sub.argmin())
                m = float(sub.flat[k])

                if not np.isinf(m):
                    i, j = np.unravel_index(k, sub.shape)
                    bs = int(ai[i])
                    bd = int(bi[j])
                    best_s = str(bs)
                    best_d = str(bd)

                    if include_path:
                        key = (bs, bd, include_path_indexed)
                        cached = path_cache.get(key)
                        if cached is None:
                            nodes = reconstruct_path(next_hop, bs, bd)
                            path = "->".join(map(str, nodes)) if nodes else ""
                            path_indexed = (
                                ",".join(f"{idx + 1}:{node}" for idx, node in enumerate(nodes))
                                if include_path_indexed and nodes
                                else ""
                            )
                            cached = (path, path_indexed)
                            path_cache[key] = cached
                        else:
                            path, path_indexed = cached

            rows.append((t, sa, sb, best_s, best_d, path, path_indexed))

    return pd.DataFrame.from_records(
        rows,
        columns=["time", "station_a", "station_b", "best_s", "best_d", "path", "path_indexed"],
    )


def export_pair_csvs_streaming(
    dist,
    next_hop,
    series_by_station,
    station_pairs,
    steps,
    out_dir,
    left="region1",
    right="region2",
    include_path=True,
    include_path_indexed=False,
):
    """逐 pair 直接写 CSV，避免先堆一个巨型 DataFrame 再 groupby 导出。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = dist.shape[0]
    series_idx, empty = prepare_series_index(series_by_station, n)
    path_cache = {}
    paths = []

    for sa, sb in station_pairs:
        sa = int(sa)
        sb = int(sb)
        A = series_idx.get(sa, {})
        B = series_idx.get(sb, {})

        name = f"{left}--station{sa}-{right}--station{sb}".replace("/", "_")
        p = out / f"{name}.csv"

        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["time", "station_a", "station_b", "best_s", "best_d", "path"]
            if include_path_indexed:
                header.append("path_indexed")
            w.writerow(header)

            for t in steps:
                t = int(t)
                ai = A.get(t, empty)
                bi = B.get(t, empty)

                if ai.size == 0 or bi.size == 0:
                    row = [t, sa, sb, "", "", ""]
                    if include_path_indexed:
                        row.append("")
                    w.writerow(row)
                    continue

                sub = dist[np.ix_(ai, bi)]
                k = int(sub.argmin())
                m = float(sub.flat[k])
                if np.isinf(m):
                    row = [t, sa, sb, "", "", ""]
                    if include_path_indexed:
                        row.append("")
                    w.writerow(row)
                    continue

                i, j = np.unravel_index(k, sub.shape)
                bs = int(ai[i])
                bd = int(bi[j])

                path = ""
                path_indexed = ""
                if include_path:
                    key = (bs, bd, include_path_indexed)
                    cached = path_cache.get(key)
                    if cached is None:
                        nodes = reconstruct_path(next_hop, bs, bd)
                        path = "->".join(map(str, nodes)) if nodes else ""
                        path_indexed = (
                            ",".join(f"{idx + 1}:{node}" for idx, node in enumerate(nodes))
                            if include_path_indexed and nodes
                            else ""
                        )
                        cached = (path, path_indexed)
                        path_cache[key] = cached
                    else:
                        path, path_indexed = cached

                row = [t, sa, sb, str(bs), str(bd), path]
                if include_path_indexed:
                    row.append(path_indexed)
                w.writerow(row)

        paths.append(p)

    return paths
