# src/model/static_hop_table.py
import numpy as np, pandas as pd, networkx as nx
from pathlib import Path

def _to_i(x):
    try: return int(x)
    except: return None

def build_static_graph(all_edges, total_sats, ref_step=None, undirected=True):
    steps = sorted(s for s in (_to_i(k) for k in all_edges.keys()) if s is not None)
    if not steps: raise ValueError("all_edges is empty")
    ref_step = steps[0] if ref_step is None else int(ref_step)
    adj = all_edges.get(ref_step, {}) or {}
    G = nx.Graph() if undirected else nx.DiGraph()
    for u, vs in adj.items():
        uu = _to_i(u)
        if uu is None: continue
        if not vs: G.add_node(uu); continue
        for v in vs:
            vv = _to_i(v)
            if vv is None or vv == uu: continue
            G.add_edge(uu, vv)
    G.add_nodes_from(range(int(total_sats)))
    return G

def precompute_hop_matrix(G, total_sats, method="all_pairs_bfs"):
    n = int(total_sats)
    if method == "floyd":
        return np.asarray(nx.floyd_warshall_numpy(G, nodelist=list(range(n))), dtype=np.float32)
    dist = np.full((n, n), np.inf, dtype=np.float32); np.fill_diagonal(dist, 0.0)
    for s, lens in nx.all_pairs_shortest_path_length(G):
        if not (0 <= s < n): continue
        for d, h in lens.items():
            if 0 <= d < n: dist[s, d] = float(h)
    return dist

# def compute_region_pair_timeseries(dist, series_by_station, station_pairs, steps, left="region1", right="region2"):
#     n = dist.shape[0]; rows = []
#     for t in steps:
#         for sa, sb in station_pairs:
#             rec = {"time": int(t), "station_a": int(sa), "station_b": int(sb),
#                    "pair_name": f"{left}--station{sa}-{right}--station{sb}",
#                    "min_shortest_path": np.nan, "best_s": "", "best_d": ""}
#             A = series_by_station.get(sa, {}).get(t, set()) or set()
#             B = series_by_station.get(sb, {}).get(t, set()) or set()
#             ai = np.fromiter((x for x in (_to_i(v) for v in A) if x is not None and 0 <= x < n), dtype=np.int32)
#             bi = np.fromiter((x for x in (_to_i(v) for v in B) if x is not None and 0 <= x < n), dtype=np.int32)
#             if ai.size == 0 or bi.size == 0: rows.append(rec); continue
#             sub = dist[np.ix_(ai, bi)]
#             k = int(np.argmin(sub)); m = float(sub.flat[k])
#             if np.isinf(m): rows.append(rec); continue
#             i, j = np.unravel_index(k, sub.shape)
#             rec["min_shortest_path"], rec["best_s"], rec["best_d"] = m, str(int(ai[i])), str(int(bi[j]))
#             rows.append(rec)
#     return pd.DataFrame(rows, columns=["time","station_a","station_b","pair_name","min_shortest_path","best_s","best_d"])

# def compute_region_pair_timeseries(dist, series_by_station, station_pairs, steps, left="region1", right="region2"):
#     n = dist.shape[0]
#     rows = []
#     for t in steps:
#         for sa, sb in station_pairs:
#             rec = {
#                 "time": int(t),
#                 "station_a": int(sa),
#                 "station_b": int(sb),
#                 "min_shortest_path": np.nan,
#                 "best_s": "",
#                 "best_d": "",
#             }
#
#             A = series_by_station.get(sa, {}).get(t, set()) or set()
#             B = series_by_station.get(sb, {}).get(t, set()) or set()
#
#             ai = np.fromiter((x for x in (_to_i(v) for v in A) if x is not None and 0 <= x < n), dtype=np.int32)
#             bi = np.fromiter((x for x in (_to_i(v) for v in B) if x is not None and 0 <= x < n), dtype=np.int32)
#
#             if ai.size == 0 or bi.size == 0:
#                 rows.append(rec)
#                 continue
#
#             sub = dist[np.ix_(ai, bi)]
#             k = int(np.argmin(sub))
#             m = float(sub.flat[k])
#
#             if np.isinf(m):
#                 rows.append(rec)
#                 continue
#
#             i, j = np.unravel_index(k, sub.shape)
#             rec["min_shortest_path"] = m
#             rec["best_s"] = str(int(ai[i]))
#             rec["best_d"] = str(int(bi[j]))
#             rows.append(rec)
#
#     return pd.DataFrame(
#         rows,
#         columns=["time", "station_a", "station_b", "min_shortest_path", "best_s", "best_d"]
#     )


def compute_region_pair_timeseries(dist, next_hop, series_by_station, station_pairs, steps, left="region1", right="region2"):
    n = dist.shape[0]
    rows = []

    for t in steps:
        for sa, sb in station_pairs:
            rec = {
                "time": int(t),
                "station_a": int(sa),
                "station_b": int(sb),
                "best_s": "",
                "best_d": "",
                "path": "",            # 例: 12->45->89
                "path_indexed": "",    # 例: 1:12,2:45,3:89
            }

            A = series_by_station.get(sa, {}).get(t, set()) or set()
            B = series_by_station.get(sb, {}).get(t, set()) or set()

            ai = np.fromiter((x for x in (_to_i(v) for v in A) if x is not None and 0 <= x < n), dtype=np.int32)
            bi = np.fromiter((x for x in (_to_i(v) for v in B) if x is not None and 0 <= x < n), dtype=np.int32)

            if ai.size == 0 or bi.size == 0:
                rows.append(rec)
                continue

            sub = dist[np.ix_(ai, bi)]
            k = int(np.argmin(sub))
            m = float(sub.flat[k])
            if np.isinf(m):
                rows.append(rec)
                continue

            i, j = np.unravel_index(k, sub.shape)
            best_s = int(ai[i]); best_d = int(bi[j])

            path_nodes = reconstruct_path(next_hop, best_s, best_d)

            rec["best_s"] = str(best_s)
            rec["best_d"] = str(best_d)
            rec["path"] = "->".join(map(str, path_nodes)) if path_nodes else ""
            rec["path_indexed"] = ",".join(f"{idx+1}:{node}" for idx, node in enumerate(path_nodes)) if path_nodes else ""
            rows.append(rec)

    return pd.DataFrame(rows, columns=[
        "time", "station_a", "station_b", "best_s", "best_d", "path", "path_indexed"
    ])

def precompute_hop_and_next_hop(G, total_sats):
    n = int(total_sats)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    next_hop = np.full((n, n), -1, dtype=np.int32)

    for i in range(n):
        dist[i, i] = 0.0
        next_hop[i, i] = i

    # 一次性全图最短路径（无权）
    for s, paths in nx.all_pairs_shortest_path(G):
        if not (0 <= s < n):
            continue
        for d, p in paths.items():
            if not (0 <= d < n):
                continue
            h = len(p) - 1
            dist[s, d] = float(h)
            next_hop[s, d] = s if h == 0 else int(p[1])

    return dist, next_hop
def reconstruct_path(next_hop, s, d):
    n = next_hop.shape[0]
    s = int(s); d = int(d)
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
        if guard > n:   # 防御性保护
            return []
    return path

def export_pair_csvs(df, out_dir, left="region1", right="region2"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for (sa, sb), g in df.groupby(["station_a", "station_b"], sort=True):
        name = f"{left}--station{int(sa)}-{right}--station{int(sb)}".replace("/", "_")
        p = out / f"{name}.csv"
        g.to_csv(p, index=False)
        paths.append(p)

    return paths
