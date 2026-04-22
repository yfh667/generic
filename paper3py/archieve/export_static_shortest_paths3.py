from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import pandas as pd

import src.model.basiclink as basiclink
import src.model.static_hop_table as static_hop_table
from draw.basic_functio import motif as motif_mod
from draw.basic_functio.topology_config import TopologyRecorder, load_config
from src.config.viewer_config import G60_CONFIG
from src.io import read_snap_xml
from src.paper3_postprocess.static_hop_table_fast_patch import export_pair_csvs_streaming


# =========================
# Defaults
# =========================
DEFAULT_DATA_ROOT = r"D:\paper3\data"
DEFAULT_TOPOLOGY_VERSION = "grid_x_sparse"
DEFAULT_ROUTE_POLICY = "route_policy.json"
DEFAULT_ROUTE_MODE = "max_reliability"  # min_hop | max_reliability
DEFAULT_ROUTE_P_INTRA = 0.995
DEFAULT_ROUTE_P_INTER = 0.99
DEFAULT_WIN_START = 0
DEFAULT_WIN_END = 86164
DEFAULT_XML_REL = Path("satellitesposition") / "station_visible_satellites_20250106.xml"


# =========================
# Generic parallel runner
# =========================
class ParallelTaskRunner:
    """
    通用并行执行器（可复用于其他脚本）
    """
    @staticmethod
    def run(
        tasks: list[dict],
        worker_fn: Callable[[dict], dict],
        *,
        max_workers: int = 0,
        fail_fast: bool = False,
        progress_prefix: str = "[batch]",
    ) -> list[dict]:
        if not tasks:
            return []

        if max_workers <= 0:
            max_workers = min(len(tasks), max(1, os.cpu_count() or 1))

        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(worker_fn, t): t for t in tasks}
            total = len(fut_map)
            done = 0

            for fut in as_completed(fut_map):
                done += 1
                res = fut.result()
                results.append(res)

                status = res.get("status", "unknown")
                task_id = res.get("task_id", "unknown")
                print(f"{progress_prefix} [{done}/{total}] {status}: {task_id}")

                if fail_fast and status == "failed":
                    for f in fut_map:
                        f.cancel()
                    break

        return results


# =========================
# Utility
# =========================
def _log(msg: str, t0: float | None = None):
    if t0 is None:
        print(msg)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')} | +{time.time() - t0:8.2f}s] {msg}")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def _split_csv_arg(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _validate_prob(name: str, v, allow_none: bool = False):
    if v is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be None")
    x = float(v)
    if not (0 < x <= 1):
        raise ValueError(f"{name} must be in (0,1], got {v}")
    return x


def _dedup_paths(paths: list[Path]) -> list[Path]:
    out = []
    seen = set()
    for p in paths:
        k = str(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


# =========================
# Policy handling
# =========================
def resolve_route_policy_path(data_root: Path, topology_version: str, policy_spec: str) -> Path:
    cfg_dir = data_root / "topology_design" / topology_version / "config"
    spec = (policy_spec or "").strip()

    candidates: list[Path] = []

    if spec:
        p = Path(spec)

        if p.is_absolute():
            candidates.append(p)
        else:
            # cwd 相对
            candidates.append(Path.cwd() / p)
            # motif config 相对
            candidates.append(cfg_dir / p)
            candidates.append(cfg_dir / "route_policies" / p)

            if p.suffix.lower() != ".json":
                candidates.append(cfg_dir / f"{spec}.json")
                candidates.append(cfg_dir / "route_policies" / f"{spec}.json")

    # 默认回退
    candidates.append(cfg_dir / "route_policy.json")
    candidates.append(cfg_dir / "route_policies" / "route_policy.json")

    for c in _dedup_paths(candidates):
        if c.exists():
            return c

    tried = "\n".join(str(x) for x in _dedup_paths(candidates))
    raise FileNotFoundError(f"route policy not found, tried:\n{tried}")


def load_route_policy_json(
    policy_path: Path,
    *,
    fallback_mode: str,
    fallback_p_intra: float,
    fallback_p_inter: float,
) -> dict:
    if not policy_path.exists():
        raise FileNotFoundError(f"route policy file not found: {policy_path}")

    with policy_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    route_mode = str(raw.get("route_mode", fallback_mode))
    if route_mode not in {"min_hop", "max_reliability"}:
        raise ValueError(f"invalid route_mode={route_mode!r}")

    p_intra = _validate_prob("p_intra", raw.get("p_intra", fallback_p_intra))
    default_p_inter = _validate_prob("default_p_inter", raw.get("default_p_inter", fallback_p_inter), allow_none=True)

    option_p_inter = {}
    for k, v in (raw.get("option_p_inter", {}) or {}).items():
        op = int(k)
        option_p_inter[op] = _validate_prob(f"option_p_inter[{op}]", v)

    return {
        "policy_name": str(raw.get("policy_name", "route_policy")),
        "route_mode": route_mode,
        "p_intra": p_intra,
        "default_p_inter": default_p_inter,
        "option_p_inter": option_p_inter,
        "conflict_policy": str(raw.get("conflict_policy", "max_probability")),
    }


# =========================
# Core task functions
# =========================
def build_option_edge_keys_map(cfg, *, t: int, eval_env: dict) -> dict[int, set[tuple[int, int]]]:
    rec_tmp = TopologyRecorder(cfg.P, cfg.N)
    rec_tmp._motifs = cfg.motifs

    out: dict[int, set[tuple[int, int]]] = {}
    for m in rec_tmp._motifs_active_at(t, eval_env):
        nodes = {}
        motif_mod.write_distinct_motif(
            m.p_start, m.p_end, m.y_start, m.y_end,
            cfg.P, cfg.N, nodes, option=m.option
        )
        adj = motif_mod.transform_nodes_2_adjacent(nodes, cfg.P, cfg.N)
        keys = static_hop_table.build_undirected_edge_keyset(adj)
        out.setdefault(int(m.option), set()).update(keys)
    return out


def build_static_graph(cfg, *, win_start: int, win_end: int):
    rec = TopologyRecorder(cfg.P, cfg.N)
    rec._motifs = cfg.motifs

    inter_once = rec.render_adj_at(
        t=win_start,
        eval_env={"start_ts": win_start, "end_ts": win_end},
    )
    raw_inter_once = basiclink.make_edges_bidirectional(inter_once)

    base_neighbors = {
        i * cfg.N + j: ((i * cfg.N + (j + 1) % cfg.N), (i * cfg.N + (j - 1) % cfg.N))
        for i in range(cfg.P) for j in range(cfg.N)
    }

    static_edges = {node: {r, l} for node, (r, l) in base_neighbors.items()}
    for src, dsts in raw_inter_once.items():
        static_edges.setdefault(src, set()).update(dsts)

    total_sats = int(cfg.P) * int(cfg.N)
    G = static_hop_table.build_static_graph({0: static_edges}, total_sats)

    return G, raw_inter_once, total_sats


def compute_route_tables(
    G,
    cfg,
    raw_inter_once,
    route_policy: dict,
    *,
    route_mode: str,
    win_start: int,
    win_end: int,
):
    if route_mode == "max_reliability":
        option_edge_keys_map = build_option_edge_keys_map(
            cfg,
            t=win_start,
            eval_env={"start_ts": win_start, "end_ts": win_end},
        )

        summary = static_hop_table.assign_option_probability_cost_to_graph_edges(
            G,
            option_edge_keys_map,
            p_intra=route_policy["p_intra"],
            option_p_inter=route_policy.get("option_p_inter", {}),
            default_p_inter=route_policy.get("default_p_inter", None),
            conflict_policy=route_policy.get("conflict_policy", "max_probability"),
            cost_attr="cost",
        )

        total_sats = int(cfg.P) * int(cfg.N)
        dist, next_hop = static_hop_table.precompute_weighted_cost_and_next_hop(
            G, total_sats, weight="cost"
        )
        return dist, next_hop, summary

    total_sats = int(cfg.P) * int(cfg.N)
    dist, next_hop = static_hop_table.precompute_hop_and_next_hop(G, total_sats)
    return dist, next_hop, {}


def load_station_timeseries(xml_file: Path, *, win_start: int, win_end: int):
    all_regions = {gid: info["stations"] for gid, info in G60_CONFIG.station_groups.items()}
    all_station_ids = sorted(set(sid for stations in all_regions.values() for sid in stations))

    series_list = read_snap_xml.parse_station_timeseries(
        xml_file, all_station_ids, win_start, win_end
    )
    series_by_station = {sid: ts for sid, ts in zip(all_station_ids, series_list)}

    region_ids = sorted(all_regions.keys())
    station_pairs = []
    for ra, rb in combinations(region_ids, 2):
        for sa in all_regions[ra]:
            for sb in all_regions[rb]:
                station_pairs.append((sa, sb))

    steps = sorted({int(t) for ts in series_by_station.values() for t in ts.keys()})
    if not steps:
        raise ValueError("no time steps found in station timeseries")

    return all_regions, all_station_ids, series_by_station, station_pairs, steps


def write_run_meta(output_dir: Path, meta: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_export_task(task: dict) -> dict:
    t0 = time.time()

    data_root = Path(task["data_root"])
    topology_version = str(task["topology_version"])
    route_policy_spec = str(task.get("route_policy", DEFAULT_ROUTE_POLICY))
    fallback_mode = str(task.get("route_mode", DEFAULT_ROUTE_MODE))
    fallback_p_intra = float(task.get("route_p_intra", DEFAULT_ROUTE_P_INTRA))
    fallback_p_inter = float(task.get("route_p_inter", DEFAULT_ROUTE_P_INTER))
    win_start = int(task.get("win_start", DEFAULT_WIN_START))
    win_end = int(task.get("win_end", DEFAULT_WIN_END))
    skip_if_exists = bool(task.get("skip_if_exists", False))
    task_id = str(task.get("task_id", f"{_slug(topology_version)}__{_slug(route_policy_spec)}"))

    topology_root = data_root / "topology_design" / topology_version
    motif_cfg_path = topology_root / "config" / "motif.json"
    if not motif_cfg_path.exists():
        raise FileNotFoundError(f"motif config missing: {motif_cfg_path}")

    xml_file = Path(task["xml_file"]) if str(task.get("xml_file", "")).strip() else (data_root / DEFAULT_XML_REL)
    if not xml_file.exists():
        raise FileNotFoundError(f"xml file missing: {xml_file}")

    policy_path = resolve_route_policy_path(data_root, topology_version, route_policy_spec)
    route_policy = load_route_policy_json(
        policy_path,
        fallback_mode=fallback_mode,
        fallback_p_intra=fallback_p_intra,
        fallback_p_inter=fallback_p_inter,
    )
    route_mode = str(route_policy.get("route_mode", fallback_mode))
    policy_name = str(route_policy.get("policy_name", "route_policy"))
    route_tag = _slug(policy_name)

    output_dir = topology_root / "path" / f"region_pairs_{win_start}_{win_end}_{route_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and any(output_dir.glob("*.csv")):
        meta = {
            "task_id": task_id,
            "status": "skipped",
            "reason": "output exists and skip_if_exists=true",
            "topology_version": topology_version,
            "policy_path": str(policy_path),
            "policy_name": policy_name,
            "route_mode": route_mode,
            "output_dir": str(output_dir),
            "win_start": win_start,
            "win_end": win_end,
            "duration_sec": time.time() - t0,
        }
        write_run_meta(output_dir, meta)
        return meta

    _log(f"[{task_id}] Step 1: load motif + build static graph", t0)
    cfg = load_config(motif_cfg_path)
    G, raw_inter_once, total_sats = build_static_graph(cfg, win_start=win_start, win_end=win_end)
    _log(f"[{task_id}]   graph nodes={G.number_of_nodes()} edges={G.number_of_edges()} total_sats={total_sats}", t0)

    _log(f"[{task_id}] Step 2: compute route tables ({route_mode})", t0)
    dist, next_hop, summary = compute_route_tables(
        G,
        cfg,
        raw_inter_once,
        route_policy,
        route_mode=route_mode,
        win_start=win_start,
        win_end=win_end,
    )

    _log(f"[{task_id}] Step 3: load station timeseries", t0)
    all_regions, all_station_ids, series_by_station, station_pairs, steps = load_station_timeseries(
        xml_file,
        win_start=win_start,
        win_end=win_end,
    )

    _log(f"[{task_id}]   regions={len(all_regions)} stations={len(all_station_ids)}", t0)
    _log(f"[{task_id}]   station_pairs={len(station_pairs)} steps={len(steps)}", t0)

    _log(f"[{task_id}] Step 4: export pair csvs", t0)
    csv_paths = export_pair_csvs_streaming(
        dist=dist,
        next_hop=next_hop,
        series_by_station=series_by_station,
        station_pairs=station_pairs,
        steps=steps,
        out_dir=output_dir,
        left="region1",
        right="region2",
        include_path=True,
        include_path_indexed=False,
    )

    result = {
        "task_id": task_id,
        "status": "ok",
        "topology_version": topology_version,
        "policy_path": str(policy_path),
        "policy_name": policy_name,
        "route_mode": route_mode,
        "output_dir": str(output_dir),
        "csv_count": int(len(csv_paths)),
        "num_regions": int(len(all_regions)),
        "num_stations": int(len(all_station_ids)),
        "num_station_pairs": int(len(station_pairs)),
        "num_steps": int(len(steps)),
        "win_start": win_start,
        "win_end": win_end,
        "duration_sec": float(time.time() - t0),
    }
    if summary:
        result["route_summary"] = summary

    write_run_meta(output_dir, result)
    _log(f"[{task_id}] done -> {output_dir}", t0)
    return result


def _worker_run_export_task(task: dict) -> dict:
    try:
        return run_export_task(task)
    except Exception as exc:
        return {
            "task_id": task.get("task_id", "unknown"),
            "status": "failed",
            "topology_version": task.get("topology_version", ""),
            "route_policy": task.get("route_policy", ""),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "duration_sec": 0.0,
        }


# =========================
# Task builders
# =========================
def build_tasks_from_cli(args) -> list[dict]:
    motifs = _split_csv_arg(args.topology_list) or [args.topology_version]
    policies = _split_csv_arg(args.route_policy_list) or [args.route_policy]

    if args.batch_mode == "zip":
        if len(motifs) != len(policies):
            raise ValueError("batch-mode=zip requires topology-list and route-policy-list with same length")
        pairs = list(zip(motifs, policies))
    else:
        pairs = [(m, p) for m in motifs for p in policies]

    tasks = []
    for idx, (motif, policy) in enumerate(pairs, start=1):
        policy_tag = Path(policy).stem if Path(policy).suffix else str(policy)
        task_id = f"{_slug(motif)}__{_slug(policy_tag)}__{idx:03d}"
        tasks.append(
            {
                "task_id": task_id,
                "data_root": args.data_root,
                "topology_version": motif,
                "route_policy": policy,
                "route_mode": args.route_mode,
                "route_p_intra": args.route_p_intra,
                "route_p_inter": args.route_p_inter,
                "xml_file": args.xml_file,
                "win_start": args.win_start,
                "win_end": args.win_end,
                "skip_if_exists": bool(args.skip_if_exists),
            }
        )
    return tasks


def build_tasks_from_run_config(config_path: Path, args_fallback):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    defaults = {
        "data_root": args_fallback.data_root,
        "route_mode": args_fallback.route_mode,
        "route_p_intra": args_fallback.route_p_intra,
        "route_p_inter": args_fallback.route_p_inter,
        "xml_file": args_fallback.xml_file,
        "win_start": args_fallback.win_start,
        "win_end": args_fallback.win_end,
        "skip_if_exists": bool(args_fallback.skip_if_exists),
    }
    defaults.update(cfg.get("defaults", {}))

    if "tasks" in cfg:
        raw_tasks = list(cfg["tasks"])
    elif "task" in cfg:
        raw_tasks = [cfg["task"]]
    else:
        raise ValueError("run-config must contain 'task' or 'tasks'")

    tasks = []
    for idx, t in enumerate(raw_tasks, start=1):
        merged = dict(defaults)
        merged.update(t)

        if "topology_version" not in merged:
            raise ValueError(f"task #{idx} missing topology_version")

        if "route_policy" not in merged:
            merged["route_policy"] = DEFAULT_ROUTE_POLICY

        if "task_id" not in merged:
            policy_tag = Path(str(merged["route_policy"])).stem if Path(str(merged["route_policy"])).suffix else str(merged["route_policy"])
            merged["task_id"] = f"{_slug(merged['topology_version'])}__{_slug(policy_tag)}__{idx:03d}"

        tasks.append(merged)

    run_opts = {
        "max_workers": int(cfg.get("max_workers", args_fallback.max_workers)),
        "fail_fast": bool(cfg.get("fail_fast", args_fallback.fail_fast)),
        "dry_run": bool(cfg.get("dry_run", args_fallback.dry_run)),
        "batch_name": str(cfg.get("batch_name", args_fallback.batch_name)),
    }
    return tasks, run_opts


# =========================
# Batch orchestration
# =========================
def run_batch_tasks(
    tasks: list[dict],
    *,
    max_workers: int,
    fail_fast: bool,
    dry_run: bool,
    batch_name: str,
):
    if not tasks:
        raise ValueError("no tasks to run")

    batch_root_data = Path(tasks[0]["data_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = _slug(batch_name) if batch_name else f"static_paths_{stamp}"

    batch_dir = batch_root_data / "topology_design" / "_batch_runs" / name
    batch_dir.mkdir(parents=True, exist_ok=True)

    df_tasks = pd.DataFrame(tasks)
    df_tasks.to_csv(batch_dir / "00_tasks.csv", index=False, encoding="utf-8-sig")

    print(f"[batch] tasks={len(tasks)}")
    print(f"[batch] out={batch_dir}")

    if dry_run:
        print("[batch] dry-run only")
        return batch_dir, pd.DataFrame()

    results = ParallelTaskRunner.run(
        tasks,
        _worker_run_export_task,
        max_workers=max_workers,
        fail_fast=fail_fast,
        progress_prefix="[batch]",
    )

    df_res = pd.DataFrame(results)
    df_res.to_csv(batch_dir / "01_results.csv", index=False, encoding="utf-8-sig")

    if not df_res.empty:
        df_fail = df_res[df_res["status"] != "ok"]
        if not df_fail.empty:
            df_fail.to_csv(batch_dir / "02_failed.csv", index=False, encoding="utf-8-sig")

    return batch_dir, df_res


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    # single-task params
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--topology-version", default=DEFAULT_TOPOLOGY_VERSION)
    p.add_argument("--route-policy", default=DEFAULT_ROUTE_POLICY)
    p.add_argument("--route-mode", choices=["min_hop", "max_reliability"], default=DEFAULT_ROUTE_MODE)
    p.add_argument("--route-p-intra", type=float, default=DEFAULT_ROUTE_P_INTRA)
    p.add_argument("--route-p-inter", type=float, default=DEFAULT_ROUTE_P_INTER)
    p.add_argument("--xml-file", default="")
    p.add_argument("--win-start", type=int, default=DEFAULT_WIN_START)
    p.add_argument("--win-end", type=int, default=DEFAULT_WIN_END)
    p.add_argument("--skip-if-exists", action="store_true")

    # batch-from-cli params
    p.add_argument("--batch", action="store_true")
    p.add_argument("--topology-list", default="")
    p.add_argument("--route-policy-list", default="")
    p.add_argument("--batch-mode", choices=["product", "zip"], default="product")
    p.add_argument("--max-workers", type=int, default=0)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch-name", default="")

    # external config
    p.add_argument("--run-config", default="", help="json config path")

    return p.parse_args()


def print_single_result(res: dict):
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    for k in [
        "status",
        "task_id",
        "topology_version",
        "policy_name",
        "route_mode",
        "output_dir",
        "csv_count",
        "num_steps",
        "num_station_pairs",
        "duration_sec",
    ]:
        if k in res:
            print(f"{k:>20}: {res[k]}")
    if res.get("status") != "ok":
        print(f"{'error':>20}: {res.get('error', '')}")
    print("=" * 60)


def main():
    args = parse_args()

    # 1) external json config mode
    if args.run_config:
        cfg_path = Path(args.run_config)
        tasks, run_opts = build_tasks_from_run_config(cfg_path, args)
        batch_dir, df_res = run_batch_tasks(
            tasks,
            max_workers=run_opts["max_workers"],
            fail_fast=run_opts["fail_fast"],
            dry_run=run_opts["dry_run"],
            batch_name=run_opts["batch_name"],
        )
        print(f"[batch] summary={batch_dir / '01_results.csv'}")
        if not df_res.empty and (df_res["status"] != "ok").any():
            sys.exit(1)
        return

    # 2) cli batch mode
    use_batch = bool(args.batch or args.topology_list or args.route_policy_list)
    if use_batch:
        tasks = build_tasks_from_cli(args)
        batch_dir, df_res = run_batch_tasks(
            tasks,
            max_workers=args.max_workers,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
            batch_name=args.batch_name,
        )
        print(f"[batch] summary={batch_dir / '01_results.csv'}")
        if not df_res.empty and (df_res["status"] != "ok").any():
            sys.exit(1)
        return

    # 3) single task mode
    task = {
        "task_id": f"{_slug(args.topology_version)}__{_slug(args.route_policy)}",
        "data_root": args.data_root,
        "topology_version": args.topology_version,
        "route_policy": args.route_policy,
        "route_mode": args.route_mode,
        "route_p_intra": args.route_p_intra,
        "route_p_inter": args.route_p_inter,
        "xml_file": args.xml_file,
        "win_start": args.win_start,
        "win_end": args.win_end,
        "skip_if_exists": bool(args.skip_if_exists),
    }

    res = _worker_run_export_task(task)
    print_single_result(res)
    if res.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
