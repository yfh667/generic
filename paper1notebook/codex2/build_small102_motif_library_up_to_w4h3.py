from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from box_motif_enumerator_v2 import canon_graph
from src.motif_generator.module.exact_box import (
    enumerate_primitive_exact_box,
    motif_edges_as_user_ids,
    pretty_motif,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "motif_up_to_w4h3_small102_translation_dedup"
ROW_PITCH = 36


def motif_to_assign(motif):
    return {
        (int(c), int(r)): motif[c][r]
        for c in range(len(motif))
        for r in range(len(motif[0]))
    }


def canonical_primitive_representatives(w: int, h: int):
    primitive = enumerate_primitive_exact_box(w, h)
    by_key = {}
    for motif in primitive:
        key = canon_graph(motif_to_assign(motif), w, h)
        current = by_key.get(key)
        if current is None or pretty_motif(motif) < pretty_motif(current):
            by_key[key] = motif
    return sorted(by_key.values(), key=pretty_motif), len(primitive)


def motif_support_label(motif) -> str:
    entries = []
    for c, column in enumerate(motif):
        for r, symbol in enumerate(column):
            if symbol is not None:
                entries.append(f"({c},{r},{symbol})")
    return "[" + ", ".join(entries) + "]"


def main() -> int:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    counts: list[dict[str, int]] = []
    motif_id = 1
    for w in range(2, 5):
        for h in range(1, 4):
            if w == 4 and h == 3:
                continue
            motifs, primitive_count = canonical_primitive_representatives(w, h)
            counts.append(
                {
                    "w": int(w),
                    "h": int(h),
                    "function_a_count": int(len(motifs)),
                    "primitive_matrix_count": int(primitive_count),
                }
            )
            for local_id, motif in enumerate(motifs, start=1):
                rows.append(
                    {
                        "motif_id": int(motif_id),
                        "source_w": int(w),
                        "source_h": int(h),
                        "local_motif_id": int(local_id),
                        "motif": pretty_motif(motif),
                        "edge_count": int(sum(1 for col in motif for symbol in col if symbol is not None)),
                        "support": motif_support_label(motif),
                        "edges": ", ".join(motif_edges_as_user_ids(motif, row_pitch=ROW_PITCH)),
                    }
                )
                motif_id += 1

    csv_path = out_dir / "small102_up_to_w4h3_excluding_w4h3.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "motif_id",
            "source_w",
            "source_h",
            "local_motif_id",
            "motif",
            "edge_count",
            "support",
            "edges",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "definition": (
            "For every exact-box size w=2..4, h=1..3 except w=4,h=3, "
            "run function_a = primitive_exact_box followed by canon_graph "
            "translation/global-graph deduplication, then concatenate the rows. "
            "No cross-size final topology deduplication is applied."
        ),
        "row_pitch": ROW_PITCH,
        "total_motifs": int(len(rows)),
        "counts_by_size": counts,
        "csv": str(csv_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
