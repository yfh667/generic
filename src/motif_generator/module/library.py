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


@dataclass(frozen=True)
class CombinedCanonicalMotifRow:
    motif_id: int
    source_w: int
    source_h: int
    local_motif_id: int
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


def combined_canonical_motif_rows(
    *,
    max_w: int,
    max_h: int,
    min_w: int = 2,
    min_h: int = 1,
    include_max_size: bool = True,
    row_pitch: int = 36,
    phase_count: int | None = None,
    start_motif_id: int = 1,
) -> tuple[list[CombinedCanonicalMotifRow], list[dict]]:
    """Build one combined canonical motif library over a size rectangle.

    This is the reusable form of the paper experiments that concatenate
    all translation-deduplicated primitive motif libraries up to `max_w,max_h`.
    Set `include_max_size=False` to get the "small102" style library for
    `max_w=4,max_h=3`; set it to `True` to get small102 plus the 4x3 706
    library, i.e. 808 rows.
    """

    max_w = int(max_w)
    max_h = int(max_h)
    min_w = int(min_w)
    min_h = int(min_h)
    if min_w < 2:
        raise ValueError("min_w must be >= 2")
    if min_h < 1:
        raise ValueError("min_h must be >= 1")
    if max_w < min_w or max_h < min_h:
        raise ValueError("max_w/max_h must be >= min_w/min_h")

    rows: list[CombinedCanonicalMotifRow] = []
    counts: list[dict] = []
    motif_id = int(start_motif_id)
    for w in range(min_w, max_w + 1):
        for h in range(min_h, max_h + 1):
            if not include_max_size and w == max_w and h == max_h:
                continue
            motifs, primitive_count = canonical_primitive_representatives(w, h, phase_count=phase_count)
            counts.append(
                {
                    "w": int(w),
                    "h": int(h),
                    "canonical_count": int(len(motifs)),
                    "primitive_matrix_count": int(primitive_count),
                }
            )
            for local_motif_id, motif in enumerate(motifs, start=1):
                rows.append(
                    CombinedCanonicalMotifRow(
                        motif_id=int(motif_id),
                        source_w=int(w),
                        source_h=int(h),
                        local_motif_id=int(local_motif_id),
                        primitive_matrix_count=int(primitive_count),
                        motif=pretty_motif(motif),
                        edge_count=int(sum(1 for col in motif for symbol in col if symbol is not None)),
                        support=motif_matrix_support_label(motif),
                        edges=", ".join(motif_edges_as_user_ids(motif, row_pitch=int(row_pitch))),
                    )
                )
                motif_id += 1
    return rows, counts


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


def write_combined_canonical_motif_library_csv(
    path: str | Path,
    *,
    max_w: int,
    max_h: int,
    min_w: int = 2,
    min_h: int = 1,
    include_max_size: bool = True,
    row_pitch: int = 36,
    phase_count: int | None = None,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, counts = combined_canonical_motif_rows(
        max_w=max_w,
        max_h=max_h,
        min_w=min_w,
        min_h=min_h,
        include_max_size=include_max_size,
        row_pitch=row_pitch,
        phase_count=phase_count,
    )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(CombinedCanonicalMotifRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    meta = {
        "min_w": int(min_w),
        "min_h": int(min_h),
        "max_w": int(max_w),
        "max_h": int(max_h),
        "include_max_size": bool(include_max_size),
        "row_pitch": int(row_pitch),
        "phase_count": None if phase_count is None else int(phase_count),
        "total_motifs": int(len(rows)),
        "counts_by_size": counts,
        "csv": str(path),
        "definition": (
            "Concatenate canonical_primitive_representatives(w,h) for "
            "min_w<=w<=max_w and min_h<=h<=max_h."
        ),
    }
    (path.parent / f"combined_canonical_w{int(min_w)}_{int(max_w)}_h{int(min_h)}_{int(max_h)}_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
