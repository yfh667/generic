# -*- coding: utf-8 -*-
"""Merge row-mask oracle discoveries into an existing action library."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


G60_RUN_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
DEFAULT_BASE_LIBRARY = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_row_mask_action_scorer_hard96"
    / "row_mask_action_library.npy"
)
DEFAULT_ORACLE_CSV = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_row_mask_oracle_hard8_beam16"
    / "row_mask_oracle_hard_steps.csv"
)
DEFAULT_OUT_DIR = (
    G60_RUN_ROOT
    / "motif0040_0056_region_internal_plus_grid_row_mask_action_library_extended_hard8_beam"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an extended row-mask action library from oracle CSV masks.")
    parser.add_argument("--base-library", type=Path, default=DEFAULT_BASE_LIBRARY)
    parser.add_argument("--oracle-csv", type=Path, default=DEFAULT_ORACLE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--columns", nargs="+", default=("row_oracle_rows", "beam_rows"))
    return parser.parse_args()


def token_to_mask(text: str, n: int) -> np.ndarray:
    mask = np.zeros(int(n), dtype=np.int8)
    for value in re.findall(r"\d+", str(text)):
        mask[int(value) % int(n)] = 1
    return mask


def mask_key(mask: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(mask, dtype=np.int8).tolist())


def mask_token(mask: Sequence[int]) -> str:
    rows = [idx for idx, value in enumerate(np.asarray(mask, dtype=np.int8).tolist()) if int(value)]
    return "{" + ",".join(f"{idx:02d}" for idx in rows) + "}"


def write_actions_csv(path: Path, actions: np.ndarray, sources: Sequence[str]) -> None:
    fields = ["action", "action_index", "mask", "row_count", "source"] + [f"y{i:02d}" for i in range(actions.shape[1])]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(actions.astype(np.int8)):
            item: dict[str, Any] = {
                "action": f"action_{idx:03d}",
                "action_index": int(idx),
                "mask": mask_token(row),
                "row_count": int(np.sum(row)),
                "source": str(sources[idx]),
            }
            for y, value in enumerate(row.tolist()):
                item[f"y{y:02d}"] = int(value)
            writer.writerow(item)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = np.load(Path(args.base_library))
    if base.ndim != 2 or base.shape[1] != int(args.n):
        raise ValueError(f"base library has bad shape {base.shape}; expected (*,{int(args.n)})")
    base = (base != 0).astype(np.int8)
    actions = [row.copy() for row in base]
    sources = ["base"] * len(actions)
    seen = {mask_key(row): idx for idx, row in enumerate(actions)}
    additions: list[dict[str, Any]] = []

    with Path(args.oracle_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            step = int(float(row["step"]))
            for column in args.columns:
                mask = token_to_mask(row.get(column, ""), int(args.n))
                key = mask_key(mask)
                if key in seen:
                    continue
                seen[key] = len(actions)
                actions.append(mask)
                source = f"oracle_step{step}_{column}"
                sources.append(source)
                additions.append(
                    {
                        "action_index": int(seen[key]),
                        "source": source,
                        "step": step,
                        "column": str(column),
                        "mask": mask_token(mask),
                        "row_count": int(np.sum(mask)),
                    }
                )

    action_arr = np.asarray(actions, dtype=np.int8)
    npy_path = out_dir / "row_mask_action_library.npy"
    csv_path = out_dir / "row_mask_action_library_actions.csv"
    additions_path = out_dir / "added_actions.csv"
    np.save(npy_path, action_arr)
    write_actions_csv(csv_path, action_arr, sources)
    with additions_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["action_index", "source", "step", "column", "mask", "row_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(additions)
    meta = {
        "base_library": str(Path(args.base_library)),
        "oracle_csv": str(Path(args.oracle_csv)),
        "out_dir": str(out_dir),
        "npy": str(npy_path),
        "actions_csv": str(csv_path),
        "added_actions_csv": str(additions_path),
        "base_actions": int(base.shape[0]),
        "added_actions": int(len(additions)),
        "total_actions": int(action_arr.shape[0]),
        "columns": [str(x) for x in args.columns],
    }
    meta_path = out_dir / "row_mask_action_library_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
