from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WideMetricTable:
    """One wide time-series table: rows are time steps, columns are topologies."""

    name: str
    path: Path
    steps: np.ndarray
    topology_names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class MetricSpec:
    name: str
    path: Path
    weight: float = 1.0
    normalize: str = "none"


@dataclass(frozen=True)
class TopologyLearningDataset:
    """Supervised topology-selection data generated from metric tables."""

    steps: np.ndarray
    topology_names: tuple[str, ...]
    action_values: np.ndarray
    action_valid_mask: np.ndarray
    labels: np.ndarray
    label_values: np.ndarray
    sample_features: np.ndarray
    feature_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    metric_values: dict[str, np.ndarray]
    meta: dict[str, Any]


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        return list(reader.fieldnames), list(reader)


def read_wide_metric_csv(
    path: str | Path,
    *,
    name: str | None = None,
    step_column: str = "step",
    topology_prefix: str | None = None,
    topology_columns: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> WideMetricTable:
    """Read a compare_*.csv metric table.

    The file is expected to contain one time column and many topology columns.
    Lower metric values are assumed to be better by downstream teacher builders.
    """

    path = Path(path)
    fieldnames, rows = _read_csv_rows(path)
    if step_column not in fieldnames:
        raise ValueError(f"step column {step_column!r} not found in {path}")

    if topology_columns is None:
        candidates = [col for col in fieldnames if col != step_column]
        if topology_prefix not in (None, ""):
            candidates = [col for col in candidates if str(col).startswith(str(topology_prefix))]
    else:
        candidates = [str(col) for col in topology_columns]
        missing = [col for col in candidates if col not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing topology columns: {missing[:5]}")
    if not candidates:
        raise ValueError(f"No topology columns selected from {path}")

    if max_rows is not None:
        rows = rows[: int(max_rows)]
    steps = np.asarray([int(float(row[step_column])) for row in rows], dtype=np.int64)
    values = np.empty((len(rows), len(candidates)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(candidates):
            raw = row.get(col, "")
            values[row_idx, col_idx] = math.inf if raw in ("", None) else float(raw)

    return WideMetricTable(
        name=str(name or path.stem),
        path=path,
        steps=steps,
        topology_names=tuple(candidates),
        values=values,
    )


def common_topology_names(tables: Sequence[WideMetricTable]) -> tuple[str, ...]:
    if not tables:
        raise ValueError("tables must be non-empty")
    common = set(tables[0].topology_names)
    for table in tables[1:]:
        common &= set(table.topology_names)
    return tuple(name for name in tables[0].topology_names if name in common)


def align_metric_tables(
    tables: Sequence[WideMetricTable],
    *,
    topology_names: Sequence[str] | None = None,
) -> list[WideMetricTable]:
    """Align metric tables to identical steps and topology-column order."""

    if not tables:
        raise ValueError("tables must be non-empty")
    base_steps = np.asarray(tables[0].steps, dtype=np.int64)
    names = tuple(str(x) for x in (topology_names or common_topology_names(tables)))
    aligned: list[WideMetricTable] = []
    for table in tables:
        if not np.array_equal(base_steps, np.asarray(table.steps, dtype=np.int64)):
            raise ValueError(f"metric steps are not aligned: {table.path}")
        col_lookup = {name: idx for idx, name in enumerate(table.topology_names)}
        missing = [name for name in names if name not in col_lookup]
        if missing:
            raise ValueError(f"missing topology columns in {table.path}: {missing[:5]}")
        order = np.asarray([col_lookup[name] for name in names], dtype=np.int64)
        aligned.append(
            WideMetricTable(
                name=table.name,
                path=table.path,
                steps=np.asarray(table.steps, dtype=np.int64),
                topology_names=names,
                values=np.asarray(table.values[:, order], dtype=np.float64),
            )
        )
    return aligned


def normalize_metric_values(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mode = str(mode or "none").lower()
    if mode == "none":
        return values.copy()
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)

    if mode == "minmax_per_step":
        for row in range(values.shape[0]):
            mask = finite[row]
            if not np.any(mask):
                out[row, :] = math.inf
                continue
            lo = float(np.min(values[row, mask]))
            hi = float(np.max(values[row, mask]))
            denom = hi - lo
            out[row, mask] = 0.0 if denom <= 0 else (values[row, mask] - lo) / denom
            out[row, ~mask] = math.inf
        return out

    if mode == "zscore_global":
        if not np.any(finite):
            return np.full_like(values, math.inf, dtype=np.float64)
        mu = float(np.mean(values[finite]))
        sigma = float(np.std(values[finite]))
        if sigma <= 0:
            out[finite] = 0.0
        else:
            out[finite] = (values[finite] - mu) / sigma
        out[~finite] = math.inf
        return out

    if mode == "zscore_per_step":
        for row in range(values.shape[0]):
            mask = finite[row]
            if not np.any(mask):
                out[row, :] = math.inf
                continue
            mu = float(np.mean(values[row, mask]))
            sigma = float(np.std(values[row, mask]))
            out[row, mask] = 0.0 if sigma <= 0 else (values[row, mask] - mu) / sigma
            out[row, ~mask] = math.inf
        return out

    raise ValueError(f"unsupported normalization mode: {mode!r}")


def make_time_features(steps: Sequence[int], *, period_seconds: float = 86400.0) -> tuple[np.ndarray, tuple[str, ...]]:
    steps_arr = np.asarray(steps, dtype=np.float64)
    period = float(period_seconds)
    if period <= 0:
        raise ValueError("period_seconds must be positive")
    phase = (steps_arr % period) / period
    features = np.column_stack(
        [
            phase,
            np.sin(2.0 * np.pi * phase),
            np.cos(2.0 * np.pi * phase),
            np.sin(4.0 * np.pi * phase),
            np.cos(4.0 * np.pi * phase),
        ]
    ).astype(np.float32)
    return features, ("time_phase", "time_sin1", "time_cos1", "time_sin2", "time_cos2")


def build_weighted_action_values(
    tables: Sequence[WideMetricTable],
    *,
    weights: Sequence[float],
    normalizations: Sequence[str],
) -> np.ndarray:
    if len(tables) != len(weights) or len(tables) != len(normalizations):
        raise ValueError("tables, weights, and normalizations must have the same length")
    out = np.zeros_like(tables[0].values, dtype=np.float64)
    for table, weight, normalize in zip(tables, weights, normalizations):
        weight = float(weight)
        if weight == 0.0:
            continue
        out += weight * normalize_metric_values(table.values, str(normalize))
    return out


def _load_action_meta(topology_library_csv: str | Path | None) -> dict[str, dict[str, Any]]:
    if topology_library_csv in (None, "", False):
        return {}
    path = Path(topology_library_csv)
    if not path.exists():
        raise FileNotFoundError(path)
    fieldnames, rows = _read_csv_rows(path)
    name_col = "name" if "name" in fieldnames else fieldnames[0]
    return {str(row[name_col]): dict(row) for row in rows if row.get(name_col)}


def build_action_summary_rows(
    *,
    topology_names: Sequence[str],
    action_values: np.ndarray,
    labels: np.ndarray,
    topology_library_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    meta = _load_action_meta(topology_library_csv)
    values = np.asarray(action_values, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(topology_names):
        col = values[:, int(idx)]
        finite = col[np.isfinite(col)]
        row = {
            "action_idx": int(idx),
            "topology": str(name),
            "teacher_selected_steps": int(np.count_nonzero(labels_arr == int(idx))),
            "mean_objective": float(np.mean(finite)) if finite.size else math.inf,
            "min_objective": float(np.min(finite)) if finite.size else math.inf,
            "max_objective": float(np.max(finite)) if finite.size else math.inf,
        }
        for key, value in meta.get(str(name), {}).items():
            if key not in row:
                row[f"meta_{key}"] = value
        rows.append(row)
    return rows


def build_topology_learning_dataset(
    *,
    metric_specs: Sequence[MetricSpec],
    topology_names: Sequence[str] | None = None,
    topology_prefix: str | None = None,
    step_column: str = "step",
    max_rows: int | None = None,
    time_period_seconds: float = 86400.0,
    meta: Mapping[str, Any] | None = None,
) -> TopologyLearningDataset:
    """Build a supervised teacher dataset from one or more metric tables."""

    if not metric_specs:
        raise ValueError("metric_specs must be non-empty")
    raw_tables = [
        read_wide_metric_csv(
            spec.path,
            name=spec.name,
            step_column=step_column,
            topology_prefix=topology_prefix,
            topology_columns=topology_names,
            max_rows=max_rows,
        )
        for spec in metric_specs
    ]
    names = tuple(str(x) for x in (topology_names or common_topology_names(raw_tables)))
    tables = align_metric_tables(raw_tables, topology_names=names)
    action_values = build_weighted_action_values(
        tables,
        weights=[spec.weight for spec in metric_specs],
        normalizations=[spec.normalize for spec in metric_specs],
    )
    action_valid_mask = np.isfinite(action_values)
    if np.any(~np.any(action_valid_mask, axis=1)):
        first_bad = int(np.flatnonzero(~np.any(action_valid_mask, axis=1))[0])
        raise ValueError(f"no valid topology action at row {first_bad}, step={int(tables[0].steps[first_bad])}")
    labels = np.argmin(action_values, axis=1).astype(np.int32)
    label_values = action_values[np.arange(action_values.shape[0]), labels].astype(np.float64)
    features, feature_names = make_time_features(tables[0].steps, period_seconds=float(time_period_seconds))
    metric_values = {table.name: np.asarray(table.values, dtype=np.float64) for table in tables}
    return TopologyLearningDataset(
        steps=np.asarray(tables[0].steps, dtype=np.int64),
        topology_names=names,
        action_values=np.asarray(action_values, dtype=np.float64),
        action_valid_mask=np.asarray(action_valid_mask, dtype=bool),
        labels=labels,
        label_values=label_values,
        sample_features=features,
        feature_names=feature_names,
        metric_names=tuple(table.name for table in tables),
        metric_values=metric_values,
        meta=dict(meta or {}),
    )


def _write_dict_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_topology_learning_dataset(
    dataset: TopologyLearningDataset,
    out_dir: str | Path,
    *,
    metric_specs: Sequence[MetricSpec] = (),
    topology_library_csv: str | Path | None = None,
    source_config: str | Path | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_arrays = {f"metric_{name}": values.astype(np.float32) for name, values in dataset.metric_values.items()}
    np.savez_compressed(
        out_dir / "dataset.npz",
        steps=dataset.steps.astype(np.int64),
        topology_names=np.asarray(dataset.topology_names, dtype="U128"),
        action_values=dataset.action_values.astype(np.float32),
        action_valid_mask=dataset.action_valid_mask.astype(np.bool_),
        labels=dataset.labels.astype(np.int32),
        label_values=dataset.label_values.astype(np.float32),
        sample_features=dataset.sample_features.astype(np.float32),
        feature_names=np.asarray(dataset.feature_names, dtype="U64"),
        metric_names=np.asarray(dataset.metric_names, dtype="U64"),
        **metric_arrays,
    )

    teacher_rows = []
    for row_idx, step in enumerate(dataset.steps):
        label = int(dataset.labels[row_idx])
        item: dict[str, Any] = {
            "step": int(step),
            "label_idx": label,
            "label_topology": str(dataset.topology_names[label]),
            "objective_value": float(dataset.label_values[row_idx]),
        }
        for metric_name, values in dataset.metric_values.items():
            item[f"{metric_name}_value"] = float(values[row_idx, label])
        teacher_rows.append(item)
    _write_dict_rows(out_dir / "teacher_by_step.csv", teacher_rows)

    action_rows = build_action_summary_rows(
        topology_names=dataset.topology_names,
        action_values=dataset.action_values,
        labels=dataset.labels,
        topology_library_csv=topology_library_csv,
    )
    _write_dict_rows(out_dir / "action_summary.csv", action_rows)

    meta = {
        **dataset.meta,
        "num_steps": int(dataset.steps.size),
        "num_actions": int(len(dataset.topology_names)),
        "metrics": [asdict(spec) | {"path": str(spec.path)} for spec in metric_specs],
        "topology_library_csv": str(topology_library_csv) if topology_library_csv not in (None, "", False) else None,
        "outputs": {
            "dataset_npz": str(out_dir / "dataset.npz"),
            "teacher_by_step_csv": str(out_dir / "teacher_by_step.csv"),
            "action_summary_csv": str(out_dir / "action_summary.csv"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if source_config not in (None, "", False):
        shutil.copy2(Path(source_config), out_dir / "learning_config.yaml")
    return out_dir
