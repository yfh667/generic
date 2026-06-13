from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .exact_box import (
    OFFSETS,
    Motif,
    enumerate_primitive_exact_box,
    motif_edges_as_user_ids,
    pretty_motif,
)


@dataclass(frozen=True)
class CanonicalMotifRow:
    motif_id: int
    w: int
    h: int
    primitive_matrix_count: int
    motif: str
    edge_count: int
    support: str
    edges: str


def motif_to_assignment(motif: Motif) -> dict[tuple[int, int], str | None]:
    return {
        (int(c), int(r)): motif[c][r]
        for c in range(len(motif))
        for r in range(len(motif[0]))
    }


def tiled_inter_edges_for_canonical_key(
    motif: Motif,
    *,
    p_count: int | None = None,
    y_count: int | None = None,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Tile a local motif on a small torus for translation canonicalization.

    The torus is only used to identify motifs that are equivalent under global
    x/y translation. It is not the physical constellation grid.
    """

    m = len(motif)
    h = len(motif[0]) if m else 0
    if m <= 0 or h <= 0:
        return tuple()

    p_count = int(p_count if p_count is not None else 4 * m)
    y_count = int(y_count if y_count is not None else 2 * h)
    if p_count <= 0 or y_count <= 0:
        raise ValueError("canonical torus dimensions must be positive")
    if y_count % h != 0:
        raise ValueError(f"canonical y_count={y_count} must be divisible by motif h={h}")

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for p in range(p_count):
        local_col = p % m
        for block in range(y_count // h):
            for row in range(h):
                symbol = motif[local_col][row]
                if symbol is None:
                    continue
                dp, dy = OFFSETS[str(symbol)]
                src = (p, block * h + row)
                dst = ((p + dp) % p_count, block * h + row + dy)
                edges.add(tuple(sorted((src, dst))))
    return tuple(sorted(edges))


def canonical_torus_key(motif: Motif) -> tuple:
    """Canonical edge-set key under global torus translation."""

    m = len(motif)
    h = len(motif[0]) if m else 0
    if m <= 0 or h <= 0:
        return tuple()

    p_count = 4 * m
    y_count = 2 * h
    edges = tiled_inter_edges_for_canonical_key(motif, p_count=p_count, y_count=y_count)
    best = None
    for dp in range(p_count):
        for dy in range(y_count):
            shifted = tuple(
                sorted(
                    tuple(
                        sorted(
                            (
                                ((u[0] + dp) % p_count, (u[1] + dy) % y_count),
                                ((v[0] + dp) % p_count, (v[1] + dy) % y_count),
                            )
                        )
                    )
                    for u, v in edges
                )
            )
            if best is None or shifted < best:
                best = shifted
    return best or tuple()


def canonical_primitive_representatives(
    w: int,
    h: int,
    *,
    phase_count: int | None = None,
) -> tuple[list[Motif], int]:
    """Enumerate primitive motifs and keep one representative per translation class.

    This is the reusable version of the experimental "function_a":

    ``enumerate_primitive_exact_box(w, h)`` followed by global translation
    canonicalization of the tiled inter-edge graph.
    """

    primitive = enumerate_primitive_exact_box(int(w), int(h), phase_count=phase_count)
    by_key: dict[tuple, Motif] = {}
    for motif in primitive:
        key = canonical_torus_key(motif)
        current = by_key.get(key)
        if current is None or pretty_motif(motif) < pretty_motif(current):
            by_key[key] = motif
    return sorted(by_key.values(), key=pretty_motif), len(primitive)


def motif_matrix_support_label(motif: Motif) -> str:
    entries: list[str] = []
    for c, column in enumerate(motif):
        for r, symbol in enumerate(column):
            if symbol is not None:
                entries.append(f"({c},{r},{symbol})")
    return "[" + ", ".join(entries) + "]"


def canonical_motif_rows(
    w: int,
    h: int,
    *,
    row_pitch: int = 36,
    phase_count: int | None = None,
    start_motif_id: int = 1,
) -> tuple[list[CanonicalMotifRow], int]:
    motifs, primitive_count = canonical_primitive_representatives(w, h, phase_count=phase_count)
    rows: list[CanonicalMotifRow] = []
    for offset, motif in enumerate(motifs):
        rows.append(
            CanonicalMotifRow(
                motif_id=int(start_motif_id) + int(offset),
                w=int(w),
                h=int(h),
                primitive_matrix_count=int(primitive_count),
                motif=pretty_motif(motif),
                edge_count=int(sum(1 for col in motif for symbol in col if symbol is not None)),
                support=motif_matrix_support_label(motif),
                edges=", ".join(motif_edges_as_user_ids(motif, row_pitch=int(row_pitch))),
            )
        )
    return rows, primitive_count


def write_canonical_motif_library_csv(
    path: str | Path,
    *,
    w: int,
    h: int,
    row_pitch: int = 36,
    phase_count: int | None = None,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, primitive_count = canonical_motif_rows(
        w=w,
        h=h,
        row_pitch=row_pitch,
        phase_count=phase_count,
    )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(CanonicalMotifRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    meta = {
        "w": int(w),
        "h": int(h),
        "row_pitch": int(row_pitch),
        "phase_count": None if phase_count is None else int(phase_count),
        "primitive_matrix_count": int(primitive_count),
        "canonical_count": int(len(rows)),
        "csv": str(path),
        "definition": "primitive_exact_box followed by canonical_torus_key translation deduplication",
    }
    (path.parent / f"canonical_w{int(w)}_h{int(h)}_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
