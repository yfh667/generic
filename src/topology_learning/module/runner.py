from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_io import load_yaml_dict, optional_path
from .datasets import MetricSpec, build_topology_learning_dataset, write_topology_learning_dataset


def _metric_specs_from_raw(raw_metrics: Any) -> list[MetricSpec]:
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError("metrics must be a non-empty list")
    specs: list[MetricSpec] = []
    for item in raw_metrics:
        if not isinstance(item, dict):
            raise ValueError("each metric item must be a mapping")
        specs.append(
            MetricSpec(
                name=str(item["name"]),
                path=Path(str(item["path"])),
                weight=float(item.get("weight", 1.0)),
                normalize=str(item.get("normalize", "none")),
            )
        )
    return specs


def run_topology_learning_dataset_from_yaml(
    config_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    max_rows: int | None = None,
) -> Path:
    config_path = Path(config_path)
    raw = load_yaml_dict(config_path)

    metrics = _metric_specs_from_raw(raw.get("metrics"))
    teacher_raw = raw.get("teacher", {})
    if not isinstance(teacher_raw, dict):
        raise ValueError("teacher must be a mapping when present")
    features_raw = raw.get("features", {})
    if not isinstance(features_raw, dict):
        raise ValueError("features must be a mapping when present")
    outputs_raw = raw.get("outputs", {})
    if not isinstance(outputs_raw, dict):
        raise ValueError("outputs must be a mapping when present")
    inputs_raw = raw.get("inputs", {})
    if not isinstance(inputs_raw, dict):
        raise ValueError("inputs must be a mapping when present")

    run_name = str(raw.get("name", config_path.stem))
    output_dir = Path(out_dir if out_dir is not None else outputs_raw["out_dir"])
    topology_library_csv = optional_path(inputs_raw.get("topology_library_csv"))

    dataset = build_topology_learning_dataset(
        metric_specs=metrics,
        topology_prefix=teacher_raw.get("topology_prefix"),
        step_column=str(teacher_raw.get("step_column", "step")),
        max_rows=max_rows,
        time_period_seconds=float(features_raw.get("time_period_seconds", 86400.0)),
        meta={
            "name": run_name,
            "config_path": str(config_path),
            "teacher": dict(teacher_raw),
            "features": dict(features_raw),
        },
    )
    write_topology_learning_dataset(
        dataset,
        output_dir,
        metric_specs=metrics,
        topology_library_csv=topology_library_csv,
        source_config=config_path,
    )

    summary = {
        "out_dir": str(output_dir),
        "dataset": str(output_dir / "dataset.npz"),
        "num_steps": int(dataset.steps.size),
        "num_actions": int(len(dataset.topology_names)),
        "num_teacher_classes": int(len(set(int(x) for x in dataset.labels.tolist()))),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return output_dir

