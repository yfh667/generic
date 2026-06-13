# -*- coding: utf-8 -*-
"""
Self-contained box motif enumerator v2.

Constraint:
- each planned satellite has out-degree <= 1
- each satellite has in-degree <= 1
- unused terminals are allowed

Box:
- m: box width in the plane/x direction
- n: box height in the y phase direction
- planned columns are 0..m-2
- targets must stay inside the m x n box
- global tiling step is (m-1) in plane direction and n in phase direction

Maximality:
- a solution is maximal if no extra edge can be added without violating constraints
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Iterable


OFFS = {"A": (1, 0), "B": (1, -1), "C": (1, 1), "D": (2, 0)}


def validate_mn(m: int, n: int) -> tuple[int, int]:
    m = int(m)
    n = int(n)
    if m < 2:
        raise ValueError("m must be >= 2 because the planned width is m-1")
    if n < 1:
        raise ValueError("n must be >= 1")
    return m, n


def planned_cells(m: int, n: int) -> list[tuple[int, int]]:
    m, n = validate_mn(m, n)
    return [(c, r) for c in range(m - 1) for r in range(n)]


def cell_options(c: int, r: int, m: int, n: int) -> list[str | None]:
    """Available choices for one planned node.

    None means this node leaves its outgoing terminal unused.
    """

    m, n = validate_mn(m, n)
    opts: list[str | None] = [None]
    for o, (dp, dn) in OFFS.items():
        if c + dp <= m - 1 and 0 <= r + dn <= n - 1:
            opts.append(o)
    return opts


def option_counts(m: int, n: int) -> list[int]:
    return [len(cell_options(c, r, m, n)) for c, r in planned_cells(m, n)]


def search_space_size(m: int, n: int) -> int:
    total = 1
    for count in option_counts(m, n):
        total *= count
    return total


def indeg_map(assign: dict[tuple[int, int], str | None], m: int, n: int) -> dict[tuple[int, int], int]:
    """Incoming degree by tiled node class: (plane class mod m-1, row)."""

    m, n = validate_mn(m, n)
    incoming: dict[tuple[int, int], int] = {}
    for (c, r), o in assign.items():
        if o is None:
            continue
        dp, dn = OFFS[o]
        key = ((c + dp) % (m - 1), r + dn)
        incoming[key] = incoming.get(key, 0) + 1
    return incoming


def legal(assign: dict[tuple[int, int], str | None], m: int, n: int) -> bool:
    return all(v <= 1 for v in indeg_map(assign, m, n).values())


def is_maximal(assign: dict[tuple[int, int], str | None], m: int, n: int) -> bool:
    for (c, r), o in assign.items():
        if o is not None:
            continue
        for cand in cell_options(c, r, m, n):
            if cand is None:
                continue
            a2 = dict(assign)
            a2[(c, r)] = cand
            if legal(a2, m, n):
                return False
    return True


def tiled_edges(
    assign: dict[tuple[int, int], str | None],
    m: int,
    n: int,
    p_count: int,
    y_count: int,
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for p in range(p_count):
        c = p % (m - 1)
        for block in range(y_count // n):
            for r in range(n):
                o = assign[(c, r)]
                if o is None:
                    continue
                dp, dn = OFFS[o]
                u = (p, block * n + r)
                v = ((p + dp) % p_count, block * n + r + dn)
                edges.add(tuple(sorted((u, v))))
    return edges


def canon_graph(assign: dict[tuple[int, int], str | None], m: int, n: int) -> tuple:
    """Canonical graph under global torus translation."""

    m, n = validate_mn(m, n)
    p_count = 4 * (m - 1)
    y_count = 2 * n
    edges = tiled_edges(assign, m, n, p_count, y_count)
    best = None
    for dp in range(p_count):
        for dn in range(y_count):
            shifted = tuple(
                sorted(
                    tuple(
                        sorted(
                            (
                                ((u[0] + dp) % p_count, (u[1] + dn) % y_count),
                                ((v[0] + dp) % p_count, (v[1] + dn) % y_count),
                            )
                        )
                    )
                    for (u, v) in edges
                )
            )
            if best is None or shifted < best:
                best = shifted
    return best or tuple()


def enumerate_mn(m: int, n: int) -> tuple[int, int, list[dict[tuple[int, int], str | None]]]:
    """Enumerate legal, maximal, translation-deduplicated box motifs."""

    m, n = validate_mn(m, n)
    cells = planned_cells(m, n)
    opts = [cell_options(c, r, m, n) for c, r in cells]

    legal_n = 0
    maximal: list[dict[tuple[int, int], str | None]] = []
    for combo in product(*opts):
        assign = dict(zip(cells, combo))
        if not legal(assign, m, n):
            continue
        legal_n += 1
        if is_maximal(assign, m, n):
            maximal.append(assign)

    seen = set()
    classes: list[dict[tuple[int, int], str | None]] = []
    for assign in maximal:
        key = canon_graph(assign, m, n)
        if key not in seen:
            seen.add(key)
            classes.append(assign)
    return legal_n, len(maximal), classes


def enumerate_v2(w: int, kn: int) -> tuple[int, int, list[dict[tuple[int, int], str | None]]]:
    """Backward-compatible alias for the original variable names."""

    return enumerate_mn(w, kn)


def describe(assign: dict[tuple[int, int], str | None], m: int, n: int) -> tuple[str, str, str]:
    incoming = indeg_map(assign, m, n)
    edge_count = sum(1 for o in assign.values() if o is not None)
    slots = (m - 1) * n
    saturated = all(
        incoming.get((c, r), 0) + int(assign[(c, r)] is not None) == 2
        for c in range(m - 1)
        for r in range(n)
    )

    cols = []
    for c in range(m - 1):
        cols.append("".join((assign[(c, r)] or "-") for r in range(n)))
    return " | ".join(cols), f"{edge_count}/{slots}边", "饱和" if saturated else "含空闲端子"


def edge_labels(assign: dict[tuple[int, int], str | None], row_pitch: int = 36) -> list[str]:
    labels = []
    for (c, r), o in sorted(assign.items()):
        if o is None:
            continue
        dp, dn = OFFS[o]
        src = c * row_pitch + r + 1
        dst = (c + dp) * row_pitch + (r + dn) + 1
        labels.append(f"{src}-{dst}({o})")
    return labels


def rows_for_csv(
    classes: Iterable[dict[tuple[int, int], str | None]],
    m: int,
    n: int,
    row_pitch: int,
) -> list[dict[str, str | int]]:
    rows = []
    for idx, assign in enumerate(classes, start=1):
        cols, edge_count, saturation = describe(assign, m, n)
        rows.append(
            {
                "motif_id": idx,
                "m": m,
                "n": n,
                "columns": cols,
                "edge_count": edge_count,
                "saturation": saturation,
                "edges": ", ".join(edge_labels(assign, row_pitch=row_pitch)),
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    m: int,
    n: int,
    legal_count: int,
    maximal_count: int,
    classes: list[dict[tuple[int, int], str | None]],
    row_pitch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"box_motif_v2_m{m}_n{n}"
    rows = rows_for_csv(classes, m, n, row_pitch)

    with (output_dir / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["motif_id", "m", "n", "columns", "edge_count", "saturation", "edges"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "m": m,
        "n": n,
        "offsets": {key: {"dp": value[0], "dn": value[1]} for key, value in OFFS.items()},
        "constraint": "out_degree<=1 and in_degree<=1; unused terminals allowed",
        "search_space_size": search_space_size(m, n),
        "legal_count": legal_count,
        "maximal_count": maximal_count,
        "translation_deduped_maximal_count": len(classes),
        "motifs": rows,
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate v2 box motifs for arbitrary m and n.")
    parser.add_argument("pos_m", nargs="?", type=int, help="Box width in plane/x direction.")
    parser.add_argument("pos_n", nargs="?", type=int, help="Box height in y phase direction.")
    parser.add_argument("--m", dest="flag_m", type=int, help="Box width in plane/x direction.")
    parser.add_argument("--n", dest="flag_n", type=int, help="Box height in y phase direction.")
    parser.add_argument("--row-pitch", type=int, default=36, help="Node label pitch, default is G60 N=36.")
    parser.add_argument("--limit", type=int, default=None, help="Only print the first K classes.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory for CSV/JSON output.")
    parser.add_argument("--max-combos", type=int, default=5_000_000, help="Refuse larger searches unless --force.")
    parser.add_argument("--force", action="store_true", help="Run even if the search space is large.")
    args = parser.parse_args()

    args.m = args.flag_m if args.flag_m is not None else (args.pos_m if args.pos_m is not None else 3)
    args.n = args.flag_n if args.flag_n is not None else (args.pos_n if args.pos_n is not None else 2)
    return args


def main() -> int:
    args = parse_args()
    m, n = validate_mn(args.m, args.n)
    combo_count = search_space_size(m, n)
    if combo_count > args.max_combos and not args.force:
        print(
            f"[motif-v2] m={m}, n={n} search_space_size={combo_count:,}, "
            f"larger than --max-combos={args.max_combos:,}. Use --force to run."
        )
        return 2

    legal_count, maximal_count, classes = enumerate_mn(m, n)
    print(f"{m}x{n} (宽{m}面 x 高{n}相位), 约束: 出<=1 入<=1, 目标在框内")
    print(f"搜索空间: {combo_count:,}")
    print(f"合法方案: {legal_count}   极大方案: {maximal_count}   极大且平移不等价: {len(classes)}\n")

    show_classes = classes if args.limit is None else classes[: max(args.limit, 0)]
    for idx, assign in enumerate(show_classes, start=1):
        cols, edge_count, saturation = describe(assign, m, n)
        print(f"#{idx}  [{cols}]  {edge_count}  {saturation}   边: {', '.join(edge_labels(assign, args.row_pitch))}")

    if args.limit is not None and len(classes) > args.limit:
        print(f"\n[motif-v2] only printed {args.limit}/{len(classes)} classes")

    if args.out_dir is not None:
        write_outputs(Path(args.out_dir), m, n, legal_count, maximal_count, classes, args.row_pitch)
        print(f"\n[motif-v2] wrote CSV/JSON to {Path(args.out_dir)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
