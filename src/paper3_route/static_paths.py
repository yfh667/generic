from __future__ import annotations
from datetime import datetime

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
import math
import time
import csv

import numpy as np
import pandas as pd

from .route_policy import (
    RoutePolicy,
    load_route_policy,
    motif_config_path,
    path_output_dir,
    resolve_visibility_xml,
)


EMPTY_I32 = np.empty(0, dtype=np.int32)


@dataclass(frozen=True)
class StationPair:
    region_a: Any
    region_b: Any
    station_a: int
    station_b: int


def load_g60_constants() -> tuple[int, int, int]:
    from src.config.viewer_config import G60_CONFIG

    return int(G60_CONFIG.P), int(G60_CONFIG.N), int(G60_CONFIG.total_sats)


def load_station_groups_from_g60() -> dict[Any, list[int]]:
    from src.config.viewer_config import G60_CONFIG

    groups: dict[Any, list[int]] = {}
    for gid, info in G60_CONFIG.station_groups.items():
        groups[gid] = [int(sid) for sid in info["stations"]]
    return groups


def make_station_pairs(station_groups: dict[Any, list[int]]) -> list[StationPair]:
    region_ids = sorted(station_groups.keys())
    pairs: list[StationPair] = []
    for ra, rb in combinations(region_ids, 2):
        for sa in station_groups[ra]:
            for sb in station_groups[rb]:
                pairs.append(StationPair(ra, rb, int(sa), int(sb)))
    return pairs


def all_station_ids(station_groups: dict[Any, list[int]]) -> list[int]:
    return sorted({int(sid) for stations in station_groups.values() for sid in stations})


def _safe_name(x: Any) -> str:
    s = str(x).replace("/", "_").replace("\\", "_").replace(":", "_")
    return s.replace(" ", "_")


def pair_csv_name(pair: StationPair) -> str:
    return (
        f"region{_safe_name(pair.region_a)}--station{pair.station_a}-"
        f"region{_safe_name(pair.region_b)}--station{pair.station_b}.csv"
    )


def build_static_edges_from_motif(
    motif_json: str | Path,
    *,
    t: int,
    start_ts: int,
    end_ts: int,
) -> tuple[dict[int, set[int]], int, int, int]:
    """
    Load data/topology_design/<motif>/config/motif.json and build one static graph.
    """
    import src.model.basiclink as basiclink
    from draw.basic_functio.topology_config import TopologyRecorder, load_config

    cfg = load_config(Path(motif_json))
    P, N = int(cfg.P), int(cfg.N)
    total_sats = P * N

    rec = TopologyRecorder(P, N)
    rec._motifs = cfg.motifs

    inter_once = rec.render_adj_at(
        t=int(t),
        eval_env={"start_ts": int(start_ts), "end_ts": int(end_ts)},
    )
    raw_inter_once = basiclink.make_edges_bidirectional(inter_once)

    static_edges: dict[int, set[int]] = {
        i * N + j: {
            i * N + ((j + 1) % N),
            i * N + ((j - 1) % N),
        }
        for i in range(P)
        for j in range(N)
    }
    for src, dsts in raw_inter_once.items():
        src_i = int(src)
        static_edges.setdefault(src_i, set()).update(int(dst) for dst in dsts)

    # Stabilize edge insertion order downstream by normalizing all nodes.
    for node in range(total_sats):
        static_edges.setdefault(node, set())
    return static_edges, P, N, total_sats


def precompute_shortest_tables(
    static_edges: dict[int, set[int]],
    *,
    total_sats: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    import src.model.static_hop_table as static_hop_table

    G = static_hop_table.build_static_graph({0: static_edges}, total_sats)
    dist, next_hop = static_hop_table.precompute_hop_and_next_hop(G, total_sats)
    return dist, next_hop, int(G.number_of_edges())


def load_station_visibility(
    xml_file: str | Path,
    station_ids: list[int],
    *,
    start: int,
    end: int,
) -> dict[int, dict[int, Any]]:
    from src.io import read_snap_xml

    series_list = read_snap_xml.parse_station_timeseries(
        Path(xml_file),
        station_ids,
        int(start),
        int(end),
    )
    return {int(sid): ts for sid, ts in zip(station_ids, series_list)}


def collect_steps(series_by_station: dict[int, dict[int, Any]], policy: RoutePolicy) -> list[int]:
    if policy.step_mode in {"range", "full", "dense"}:
        return list(range(policy.start, policy.end + 1, policy.stride))

    if policy.step_mode not in {"actual", "visible", "sparse"}:
        raise ValueError(
            f"Unsupported time_window.step_mode={policy.step_mode!r}; "
            "use 'actual' or 'range'."
        )

    steps: set[int] = set()
    for ts in series_by_station.values():
        for t in ts.keys():
            ti = int(t)
            if policy.start <= ti <= policy.end and ((ti - policy.start) % policy.stride == 0):
                steps.add(ti)
    return sorted(steps)


def _visible_to_array(values: Any, *, total_sats: int) -> np.ndarray:
    if not values:
        return EMPTY_I32
    out: list[int] = []
    for v in values:
        try:
            x = int(v)
        except Exception:
            continue
        if 0 <= x < total_sats:
            out.append(x)
    if not out:
        return EMPTY_I32
    return np.asarray(sorted(set(out)), dtype=np.int32)


def build_visibility_cache(
    series_by_station: dict[int, dict[int, Any]],
    *,
    steps: list[int],
    total_sats: int,
) -> dict[int, dict[int, np.ndarray]]:
    step_set = set(int(t) for t in steps)
    cache: dict[int, dict[int, np.ndarray]] = {}
    for sid, ts in series_by_station.items():
        station_cache: dict[int, np.ndarray] = {}
        for t, values in ts.items():
            ti = int(t)
            if ti not in step_set:
                continue
            arr = _visible_to_array(values, total_sats=total_sats)
            if arr.size:
                station_cache[ti] = arr
        cache[int(sid)] = station_cache
    return cache


def reconstruct_path_nodes(next_hop: np.ndarray, s: int, d: int) -> list[int]:
    n = int(next_hop.shape[0])
    s = int(s)
    d = int(d)
    if s < 0 or d < 0 or s >= n or d >= n:
        return []
    if int(next_hop[s, d]) < 0:
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


def path_hop_counts(path_nodes: list[int], *, N: int) -> tuple[int, int, int]:
    intra = 0
    inter = 0
    for u, v in zip(path_nodes, path_nodes[1:]):
        if int(u) // int(N) == int(v) // int(N):
            intra += 1
        else:
            inter += 1
    return intra, inter, intra + inter


def path_info_cached(
    next_hop: np.ndarray,
    s: int,
    d: int,
    *,
    N: int,
    cache: dict[tuple[int, int], dict[str, Any]],
    include_path_indexed: bool,
) -> dict[str, Any]:
    key = (int(s), int(d))
    hit = cache.get(key)
    if hit is not None:
        return hit

    nodes = reconstruct_path_nodes(next_hop, int(s), int(d))
    if not nodes:
        info = {
            "path": "",
            "path_indexed": "",
            "intra_hops": math.nan,
            "inter_hops": math.nan,
            "total_hops": math.nan,
        }
    else:
        intra, inter, total = path_hop_counts(nodes, N=N)
        info = {
            "path": "->".join(map(str, nodes)),
            "path_indexed": ",".join(f"{i + 1}:{node}" for i, node in enumerate(nodes))
            if include_path_indexed
            else "",
            "intra_hops": int(intra),
            "inter_hops": int(inter),
            "total_hops": int(total),
        }
    cache[key] = info
    return info


def select_shortest_access_pair(
    dist: np.ndarray,
    ai: np.ndarray,
    bi: np.ndarray,
) -> tuple[int | None, int | None, float]:
    if ai.size == 0 or bi.size == 0:
        return None, None, math.nan
    sub = dist[np.ix_(ai, bi)]
    k = int(np.argmin(sub))
    hop = float(sub.flat[k])
    if np.isinf(hop):
        return None, None, math.nan
    i, j = np.unravel_index(k, sub.shape)
    return int(ai[i]), int(bi[j]), hop


def _empty_record(pair: StationPair, t: int) -> dict[str, Any]:
    return {
        "time": int(t),
        "region_a": pair.region_a,
        "region_b": pair.region_b,
        "station_a": int(pair.station_a),
        "station_b": int(pair.station_b),
        "best_s": "",
        "best_d": "",
        "min_shortest_path": math.nan,
        "path": "",
        "path_indexed": "",
        "intra_hops": math.nan,
        "inter_hops": math.nan,
        "total_hops": math.nan,
    }


def compute_pair_rows(
    pair: StationPair,
    *,
    steps: list[int],
    dist: np.ndarray,
    next_hop: np.ndarray,
    visibility_cache: dict[int, dict[int, np.ndarray]],
    N: int,
    policy: RoutePolicy,
    path_cache: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    if policy.objective != "shortest_hop":
        raise NotImplementedError(f"objective={policy.objective!r} is not implemented here.")

    rows: list[dict[str, Any]] = []
    cache_a = visibility_cache.get(int(pair.station_a), {})
    cache_b = visibility_cache.get(int(pair.station_b), {})

    for t in steps:
        ai = cache_a.get(int(t), EMPTY_I32)
        bi = cache_b.get(int(t), EMPTY_I32)
        best_s, best_d, min_hop = select_shortest_access_pair(dist, ai, bi)
        if best_s is None or best_d is None:
            rows.append(_empty_record(pair, int(t)))
            continue

        info = path_info_cached(
            next_hop,
            best_s,
            best_d,
            N=N,
            cache=path_cache,
            include_path_indexed=policy.include_path_indexed,
        )
        rec = _empty_record(pair, int(t))
        rec.update(
            {
                "best_s": int(best_s),
                "best_d": int(best_d),
                "min_shortest_path": int(min_hop) if float(min_hop).is_integer() else float(min_hop),
                "path": info["path"] if policy.include_path else "",
                "path_indexed": info["path_indexed"] if policy.include_path_indexed else "",
                "intra_hops": info["intra_hops"] if policy.include_hops else math.nan,
                "inter_hops": info["inter_hops"] if policy.include_hops else math.nan,
                "total_hops": info["total_hops"] if policy.include_hops else math.nan,
            }
        )
        rows.append(rec)
    return rows


def output_columns(policy: RoutePolicy) -> list[str]:
    cols = [
        "time",
        "region_a",
        "region_b",
        "station_a",
        "station_b",
        "best_s",
        "best_d",
        "min_shortest_path",
    ]
    if policy.include_path:
        cols.append("path")
    if policy.include_path_indexed:
        cols.append("path_indexed")
    if policy.include_hops:
        cols.extend(["intra_hops", "inter_hops", "total_hops"])
    return cols

def write_pair_csv_streaming(
    pair: StationPair,
    *,
    steps: list[int],
    dist: np.ndarray,
    next_hop: np.ndarray,
    visibility_cache: dict[int, dict[int, np.ndarray]],
    N: int,
    policy: RoutePolicy,
    out_dir: str | Path,
    path_cache: dict[tuple[int, int], dict[str, Any]],
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / pair_csv_name(pair)
    cols = output_columns(policy)

    cache_a = visibility_cache.get(int(pair.station_a), {})
    cache_b = visibility_cache.get(int(pair.station_b), {})

    with csv_path.open("w", newline="", encoding=policy.encoding) as f:
        w = csv.writer(f)
        w.writerow(cols)

        for t in steps:
            ai = cache_a.get(int(t), EMPTY_I32)
            bi = cache_b.get(int(t), EMPTY_I32)
            best_s, best_d, min_hop = select_shortest_access_pair(dist, ai, bi)

            if best_s is None or best_d is None:
                row = [int(t), pair.region_a, pair.region_b, int(pair.station_a), int(pair.station_b), "", "", math.nan]
                if policy.include_path:
                    row.append("")
                if policy.include_path_indexed:
                    row.append("")
                if policy.include_hops:
                    row.extend([math.nan, math.nan, math.nan])
                w.writerow(row)
                continue

            info = path_info_cached(
                next_hop,
                best_s,
                best_d,
                N=N,
                cache=path_cache,
                include_path_indexed=policy.include_path_indexed,
            )

            min_val = int(min_hop) if float(min_hop).is_integer() else float(min_hop)
            row = [int(t), pair.region_a, pair.region_b, int(pair.station_a), int(pair.station_b), int(best_s), int(best_d), min_val]
            if policy.include_path:
                row.append(info["path"])
            if policy.include_path_indexed:
                row.append(info["path_indexed"])
            if policy.include_hops:
                row.extend([info["intra_hops"], info["inter_hops"], info["total_hops"]])

            w.writerow(row)

    return csv_path

# def write_pair_csv_streaming(
#     pair: StationPair,
#     *,
#     steps: list[int],
#     dist: np.ndarray,
#     next_hop: np.ndarray,
#     visibility_cache: dict[int, dict[int, np.ndarray]],
#     N: int,
#     policy: RoutePolicy,
#     out_dir: str | Path,
#     path_cache: dict[tuple[int, int], dict[str, Any]],
# ) -> Path:
#     out = Path(out_dir)
#     out.mkdir(parents=True, exist_ok=True)
#     csv_path = out / pair_csv_name(pair)
#     cols = output_columns(policy)
#
#     first = True
#     for start in range(0, len(steps), policy.chunk_rows):
#         chunk_steps = steps[start : start + policy.chunk_rows]
#         rows = compute_pair_rows(
#             pair,
#             steps=chunk_steps,
#             dist=dist,
#             next_hop=next_hop,
#             visibility_cache=visibility_cache,
#             N=N,
#             policy=policy,
#             path_cache=path_cache,
#         )
#         pd.DataFrame(rows, columns=cols).to_csv(
#             csv_path,
#             index=False,
#             mode="w" if first else "a",
#             header=first,
#             encoding=policy.encoding,
#         )
#         first = False
#     return csv_path


def export_shortest_paths_for_motif(
    motif_name: str,
    *,
    data_root: str | Path,
    route_policy_path: str | Path,
    force: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    step_seconds: dict[str, float] = {}

    def _log(msg: str) -> None:
        if verbose:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')} | {motif_name} | +{time.time() - t0:8.2f}s] {msg}",
                flush=True,
            )

    def _step_done(name: str, ts: float) -> None:
        dt = time.time() - ts
        step_seconds[name] = round(dt, 3)
        _log(f"{name} done in {dt:.3f}s")

    ts = time.time()
    policy = load_route_policy(route_policy_path)

    motif_json = motif_config_path(data_root, policy, motif_name)
    if not motif_json.exists():
        raise FileNotFoundError(f"motif.json not found: {motif_json}")

    out_dir = path_output_dir(data_root, policy, motif_name)
    if out_dir.exists() and force:
        pass
    out_dir.mkdir(parents=True, exist_ok=True)
    _step_done("load_policy_and_paths", ts)

    ts = time.time()
    static_edges, P, N, total_sats = build_static_edges_from_motif(
        motif_json,
        t=policy.start,
        start_ts=policy.start,
        end_ts=policy.end,
    )
    dist, next_hop, edge_count = precompute_shortest_tables(static_edges, total_sats=total_sats)
    _step_done("build_topology_and_precompute", ts)

    ts = time.time()
    station_groups = load_station_groups_from_g60()
    pairs = make_station_pairs(station_groups)
    stations = all_station_ids(station_groups)

    xml_file = resolve_visibility_xml(data_root, policy)
    series_by_station = load_station_visibility(
        xml_file,
        stations,
        start=policy.start,
        end=policy.end,
    )
    steps = collect_steps(series_by_station, policy)
    if not steps:
        raise RuntimeError(
            f"No time steps found for motif={motif_name}, window=[{policy.start}, {policy.end}], "
            f"xml={xml_file}"
        )
    visibility_cache = build_visibility_cache(
        series_by_station,
        steps=steps,
        total_sats=total_sats,
    )
    _step_done("load_visibility_and_steps", ts)

    ts = time.time()
    path_cache: dict[tuple[int, int], dict[str, Any]] = {}
    csv_paths: list[Path] = []
    n_pairs = len(pairs)
    for i, pair in enumerate(pairs, 1):
        if i == 1 or i % 20 == 0 or i == n_pairs:
            print(f"[{motif_name}] exporting pair {i}/{n_pairs}", flush=True)


        csv_paths.append(
            write_pair_csv_streaming(
                pair,
                steps=steps,
                dist=dist,
                next_hop=next_hop,
                visibility_cache=visibility_cache,
                N=N,
                policy=policy,
                out_dir=out_dir,
                path_cache=path_cache,
            )
        )
    _step_done("export_pair_csvs", ts)

    elapsed = round(time.time() - t0, 3)
    _log(f"finished: csv_files={len(csv_paths)}, elapsed={elapsed:.3f}s")

    return {
        "motif": motif_name,
        "route_name": policy.route_name,
        "objective": policy.objective,
        "P": P,
        "N": N,
        "total_sats": total_sats,
        "static_edges": edge_count,
        "time_steps": len(steps),
        "station_pairs": len(pairs),
        "path_cache_size": len(path_cache),
        "out_dir": str(out_dir),
        "csv_files": len(csv_paths),
        "elapsed_sec": elapsed,
        "step_seconds": step_seconds,
    }


# def export_shortest_paths_for_motif(
#     motif_name: str,
#     *,
#     data_root: str | Path,
#     route_policy_path: str | Path,
#     force: bool = True,
#     verbose: bool = False,
# ) -> dict[str, Any]:
#     t0 = time.time()
#     step_sec: dict[str, float] = {}
#
#     def _log(msg: str) -> None:
#         if verbose:
#             print(f"[{datetime.now().strftime('%H:%M:%S')} | {motif_name} | +{time.time() - t0:8.2f}s] {msg}",
#                   flush=True)
#
#     def _step_done(name: str, ts: float) -> None:
#         dt = time.time() - ts
#         step_sec[name] = round(dt, 3)
#         _log(f"{name} done in {dt:.3f}s")
#
#     t0 = time.time()
#     policy = load_route_policy(route_policy_path)
#
#     motif_json = motif_config_path(data_root, policy, motif_name)
#     if not motif_json.exists():
#         raise FileNotFoundError(f"motif.json not found: {motif_json}")
#
#     out_dir = path_output_dir(data_root, policy, motif_name)
#     if out_dir.exists() and force:
#         # Keep directory; overwrite each pair file as it is generated.
#         pass
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     static_edges, P, N, total_sats = build_static_edges_from_motif(
#         motif_json,
#         t=policy.start,
#         start_ts=policy.start,
#         end_ts=policy.end,
#     )
#     dist, next_hop, edge_count = precompute_shortest_tables(static_edges, total_sats=total_sats)
#
#     station_groups = load_station_groups_from_g60()
#     pairs = make_station_pairs(station_groups)
#     stations = all_station_ids(station_groups)
#
#     xml_file = resolve_visibility_xml(data_root, policy)
#     series_by_station = load_station_visibility(
#         xml_file,
#         stations,
#         start=policy.start,
#         end=policy.end,
#     )
#     steps = collect_steps(series_by_station, policy)
#     if not steps:
#         raise RuntimeError(
#             f"No time steps found for motif={motif_name}, window=[{policy.start}, {policy.end}], "
#             f"xml={xml_file}"
#         )
#     visibility_cache = build_visibility_cache(
#         series_by_station,
#         steps=steps,
#         total_sats=total_sats,
#     )
#
#     path_cache: dict[tuple[int, int], dict[str, Any]] = {}
#     csv_paths: list[Path] = []
#     for pair in pairs:
#         csv_paths.append(
#             write_pair_csv_streaming(
#                 pair,
#                 steps=steps,
#                 dist=dist,
#                 next_hop=next_hop,
#                 visibility_cache=visibility_cache,
#                 N=N,
#                 policy=policy,
#                 out_dir=out_dir,
#                 path_cache=path_cache,
#             )
#         )
#
#     return {
#         "motif": motif_name,
#         "route_name": policy.route_name,
#         "objective": policy.objective,
#         "P": P,
#         "N": N,
#         "total_sats": total_sats,
#         "static_edges": edge_count,
#         "time_steps": len(steps),
#         "station_pairs": len(pairs),
#         "path_cache_size": len(path_cache),
#         "out_dir": str(out_dir),
#         "csv_files": len(csv_paths),
#         "elapsed_sec": round(time.time() - t0, 3),
#     }
