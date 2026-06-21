from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def max_normalize_usage(values: np.ndarray) -> np.ndarray:
    """Normalize edge-usage values by their finite maximum.

    The input is usually an edge usage-share vector such as
    ``combined_usage_share_max_over_time.npy``. If all values are zero, the
    result stays zero instead of producing NaNs.
    """

    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return out
    max_value = float(np.max(arr[finite]))
    if max_value <= 0.0:
        return out
    out[finite] = arr[finite] / max_value
    return out


def combine_delay_hop_usage_share(
    *,
    delay_usage_share: np.ndarray,
    hop_usage_share: np.ndarray,
    delay_weight: float = 0.5,
    hop_weight: float = 0.5,
    normalize: str = "max",
) -> np.ndarray:
    """Combine weighted-delay and shortest-hop edge usage shares.

    Current paper1 convention: edge criticality/betweenness for topology
    selection is a weighted sum of shortest-delay usage share and shortest-hop
    usage share. With ``normalize="max"``, each input is first divided by its
    own maximum so both terms live on comparable [0, 1] scales.
    """

    delay = np.asarray(delay_usage_share, dtype=np.float32)
    hop = np.asarray(hop_usage_share, dtype=np.float32)
    if delay.shape != hop.shape:
        raise ValueError(f"delay_usage_share shape {delay.shape} != hop_usage_share shape {hop.shape}")

    mode = str(normalize or "max").lower()
    if mode == "max":
        delay_term = max_normalize_usage(delay)
        hop_term = max_normalize_usage(hop)
    elif mode == "none":
        delay_term = delay
        hop_term = hop
    else:
        raise ValueError("normalize must be 'max' or 'none'")

    return (
        float(delay_weight) * np.asarray(delay_term, dtype=np.float32)
        + float(hop_weight) * np.asarray(hop_term, dtype=np.float32)
    ).astype(np.float32, copy=False)


def write_delay_hop_combined_usage_store(
    *,
    out_dir: str | Path,
    edges_csv: str | Path,
    delay_usage_share: np.ndarray,
    hop_usage_share: np.ndarray,
    delay_weight: float = 0.5,
    hop_weight: float = 0.5,
    normalize: str = "max",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a small store for combined delay/hop edge usage share."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(edges_csv), out / "edges.csv")

    delay = np.asarray(delay_usage_share, dtype=np.float32)
    hop = np.asarray(hop_usage_share, dtype=np.float32)
    combined = combine_delay_hop_usage_share(
        delay_usage_share=delay,
        hop_usage_share=hop,
        delay_weight=float(delay_weight),
        hop_weight=float(hop_weight),
        normalize=str(normalize),
    )

    np.save(out / "delay_usage_share.npy", delay)
    np.save(out / "hop_usage_share.npy", hop)
    np.save(out / "combined_usage_share_max_over_time.npy", combined)
    meta: dict[str, Any] = {
        "metric": "combined_shortest_delay_and_hop_edge_usage_share",
        "counting_rule": (
            "delay_usage_share and hop_usage_share are shortest-path edge usage counts "
            "divided by reachable source-target pairs; combined = "
            "delay_weight * norm(delay_usage_share) + hop_weight * norm(hop_usage_share)."
        ),
        "normalize": str(normalize),
        "delay_weight": float(delay_weight),
        "hop_weight": float(hop_weight),
        "shape": [int(x) for x in combined.shape],
        "value_min": float(np.nanmin(combined)) if combined.size else 0.0,
        "value_max": float(np.nanmax(combined)) if combined.size else 0.0,
        "storage": {
            "edges_csv": "edges.csv",
            "delay_usage_share": "delay_usage_share.npy",
            "hop_usage_share": "hop_usage_share.npy",
            "combined_usage_share": "combined_usage_share_max_over_time.npy",
        },
    }
    if extra_meta:
        meta["extra"] = dict(extra_meta)
    (out / "combined_usage_share_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
