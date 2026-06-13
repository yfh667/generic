from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.link_delay.module.position_cache import PositionCacheStore, open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data

from .config import (
    load_workflow_yaml,
    optional_path,
    time_axis_from_config,
    viewer_config_from_workflow,
)
from .edge_tables import build_edge_table_from_topology_config
from .shortest_delay import compute_shortest_delay_timeseries
from .shortest_hops import compute_shortest_hops_timeseries


def _with_override(value, override):
    return value if override is None else override


def _metric_out_dir(base_out_dir: Path, topology_name: str, metric_type: str) -> Path:
    return base_out_dir / str(topology_name) / str(metric_type)


def run_workflow_from_yaml(
    config_path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
    stride: int | None = None,
    out_dir: str | Path | None = None,
    force_group_cache: bool = False,
) -> Path:
    config_path = Path(config_path)
    raw = load_workflow_yaml(config_path)
    config = viewer_config_from_workflow(raw)

    cfg_start, cfg_end, cfg_stride = time_axis_from_config(raw)
    start = int(_with_override(cfg_start, start))
    end = int(_with_override(cfg_end, end))
    stride = int(_with_override(cfg_stride, stride))
    if end < start:
        raise ValueError("end must be >= start")
    if stride <= 0:
        raise ValueError("stride must be positive")

    outputs_raw = raw.get("outputs", {})
    if not isinstance(outputs_raw, dict):
        raise ValueError("workflow.outputs must be a mapping when present")
    base_out_dir = Path(out_dir if out_dir is not None else outputs_raw["out_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, base_out_dir / "workflow_config.yaml")

    data_raw = raw.get("data", {})
    if not isinstance(data_raw, dict):
        raise ValueError("workflow.data must be a mapping")
    delay_raw = data_raw.get("delay_store", {})
    if not isinstance(delay_raw, dict):
        raise ValueError("workflow.data.delay_store must be a mapping")
    delay_store = open_delay_store_for_interval(
        start,
        end,
        stride=stride,
        store_dir=optional_path(delay_raw.get("store_dir")),
        output_base=optional_path(delay_raw.get("output_base")),
        constellation_name=config.name,
    )
    delay_rows = delay_store.rows_for_interval(start, end, stride)
    steps = [int(x) for x in np.asarray(delay_store.time_indices[delay_rows], dtype=np.int64)]

    position_store: PositionCacheStore | None = None
    position_rows = None
    position_raw = data_raw.get("position", {})
    if isinstance(position_raw, dict) and bool(position_raw.get("enable_fallback", False)):
        position_store = open_position_cache_for_interval(
            start,
            end,
            stride=stride,
            cache_dir=optional_path(position_raw.get("cache_dir")),
            cache_root=optional_path(position_raw.get("cache_root")),
            full_cache_dir=optional_path(position_raw.get("full_cache_dir")),
        )
        position_rows = position_store.rows_for_interval(start, end, stride)
        position_steps = [int(x) for x in np.asarray(position_store.times_s[position_rows], dtype=np.int64)]
        if steps != position_steps:
            raise ValueError("delay store time steps and position cache time steps are not aligned")

    group_raw = data_raw.get("group_data", {})
    if not isinstance(group_raw, dict):
        raise ValueError("workflow.data.group_data must be a mapping")
    group_data = load_or_build_group_data(
        xml_file=optional_path(group_raw.get("xml_file")),
        group_cache_dir=Path(group_raw["cache_dir"]),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=bool(group_raw.get("enabled", True)),
        force=bool(force_group_cache or group_raw.get("force", False)),
    )

    topology_raw = raw.get("topology", {})
    if not isinstance(topology_raw, dict):
        raise ValueError("workflow.topology must be a mapping")
    topology_name = str(topology_raw.get("name", topology_raw.get("kind", "topology")))
    edge_table = build_edge_table_from_topology_config(topology_raw=topology_raw, config=config)

    metric_raw = raw.get("metric", {})
    if not isinstance(metric_raw, dict):
        raise ValueError("workflow.metric must be a mapping")
    metric_type = str(metric_raw.get("type", "shortest_delay"))

    if metric_type == "shortest_delay":
        metric_dir = _metric_out_dir(base_out_dir, topology_name, metric_type)
        compute_shortest_delay_timeseries(
            topology_name=topology_name,
            edge_table=edge_table,
            config=config,
            group_data=group_data,
            steps=steps,
            delay_rows=delay_rows,
            position_rows=position_rows,
            delay_store=delay_store,
            position_store=position_store,
            source_group_id=int(metric_raw["source_group_id"]),
            target_group_id=int(metric_raw["target_group_id"]),
            out_dir=metric_dir,
            engine=str(metric_raw.get("engine", "auto")),
            sample_steps=int(metric_raw.get("sample_steps", 3)),
            sample_pairs_per_step=int(metric_raw.get("sample_pairs_per_step", 60)),
            progress_every=int(metric_raw.get("progress_every", 200)),
        )
    elif metric_type == "shortest_hops":
        metric_dir = _metric_out_dir(base_out_dir, topology_name, metric_type)
        compute_shortest_hops_timeseries(
            topology_name=topology_name,
            edge_table=edge_table,
            config=config,
            group_data=group_data,
            steps=steps,
            source_group_id=int(metric_raw["source_group_id"]),
            target_group_id=int(metric_raw["target_group_id"]),
            out_dir=metric_dir,
            sample_steps=int(metric_raw.get("sample_steps", 3)),
            sample_pairs_per_step=int(metric_raw.get("sample_pairs_per_step", 60)),
        )
    else:
        raise NotImplementedError(
            f"Unsupported metric.type={metric_type!r}. "
            "Supported metrics: shortest_delay, shortest_hops."
        )

    meta: dict[str, Any] = {
        "workflow_config": str(config_path),
        "constellation": config.name,
        "P": int(config.P),
        "N": int(config.N),
        "total_sats": int(config.total_sats),
        "start": int(start),
        "end": int(end),
        "stride": int(stride),
        "steps": int(len(steps)),
        "topology": topology_name,
        "topology_kind": str(topology_raw.get("kind", "single_motif")),
        "metric": metric_type,
        "delay_store_dir": str(delay_store.store_dir),
        "position_cache_dir": str(position_store.cache_dir) if position_store is not None else None,
        "group_xml": str(optional_path(group_raw.get("xml_file"))),
        "out_dir": str(base_out_dir),
    }
    (base_out_dir / "workflow_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[topology-workflow] done out_dir={base_out_dir}", flush=True)
    return base_out_dir
