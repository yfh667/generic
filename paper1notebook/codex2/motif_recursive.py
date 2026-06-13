# -*- coding: utf-8 -*-
"""
motif_recursive.py

Recursive-pruning enumerator for self-contained-box LISL motifs.

Model
-----
- Constellation template: G60-like Walker-Star, 18 planes x 36 phases.
- We only plan inter-plane edges; intra-plane ring edges are fixed and omitted here.
- Candidate inter offsets from a source (p, n):
    A = (1,  0)
    B = (1, -1)
    C = (1, +1)
    D = (2,  0)

Self-contained box semantics
----------------------------
A motif box has width w planes and height h phase rows, where h | 36.
Let m = w - 1. The first m columns are *planning columns*; the last column is
a pure target column. The motif tiles:
    - horizontally with step m  (= w - 1),
    - vertically   with step h.
Targets must stay inside the box:
    - B forbidden on the top row,
    - C forbidden on the bottom row,
    - D forbidden on the last planning column.

Constraints
-----------
- Each source has out-degree <= 1 in the inter graph.
- Each target has in-degree  <= 1 in the inter graph.
- Maximality: a blank source is allowed only when every legal target of that
  source is already occupied.

Key design choice
-----------------
We recurse only in one direction (column by column, left to right). Therefore
the same exact-box motif is generated exactly once; no same-size global dedup
is needed during generation.

Primitive (exact-box version)
-----------------------------
A generated motif is called primitive if its symbol matrix is not obtained by
repeating a smaller exact-box motif horizontally and/or vertically.
This is checked by direct period testing on the symbol matrix.

Note
----
This module intentionally does *not* quotient by translation-equivalence of the
tiled whole-network graph. If you later want topology classes up to translation,
add a separate canonicalization layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# Basic offsets and helpers
# ---------------------------------------------------------------------

Symbol = Optional[str]  # None, 'A', 'B', 'C', 'D'
Column = Tuple[Symbol, ...]
Motif = Tuple[Column, ...]

OFFS: Dict[str, Tuple[int, int]] = {
    'A': (1,  0),
    'B': (1, -1),
    'C': (1, +1),
    'D': (2,  0),
}

def bit(r: int) -> int:
    return 1 << r

def divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]

def pretty_motif(motif: Motif) -> str:
    """Human-readable symbol view, column by column."""
    return " | ".join(
        "".join('-' if s is None else s for s in col)
        for col in motif
    )

# ---------------------------------------------------------------------
# Column pattern summary
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ColPat:
    """
    Summary of one planning-column pattern.

    syms      : row symbols (None / A / B / C / D)
    abc_mask  : rows occupied in the *next* target class by A/B/C from this col
    d_mask    : rows occupied in the *next-next* target class by D from this col
    need_occ  : rows that must be occupied in the next target class so that all
                current blanks are maximal w.r.t. A/B/C
    need_d    : rows that must be occupied in the next-next target class so that
                all current blanks are maximal w.r.t. D
    """
    syms: Column
    abc_mask: int
    d_mask: int
    need_occ: int
    need_d: int

def allowed_symbols(c: int, r: int, m: int, h: int) -> List[Symbol]:
    """
    Allowed symbols at planning position (c, r), with m = w - 1 planning cols.
    """
    out: List[Symbol] = [None, 'A']
    if r > 0:
        out.append('B')
    if r + 1 < h:
        out.append('C')
    if c <= m - 2:  # last planning column forbids D
        out.append('D')
    return out

@lru_cache(maxsize=None)
def column_patterns(c: int, m: int, h: int) -> Tuple[ColPat, ...]:
    """
    Generate all *column-internal* consistent patterns for planning column c.

    Local pruning done here:
      1) box-boundary pruning (B/C/D availability),
      2) A/B/C from the same column may not collide on the next target class.

    Maximality is not fully decided here; instead we accumulate the two demand
    masks (need_occ, need_d) that the outer frontier recursion will verify.
    """
    out: List[ColPat] = []

    def rec(
        r: int,
        syms: List[Symbol],
        abc_mask: int,
        d_mask: int,
        need_occ: int,
        need_d: int,
    ) -> None:
        if r == h:
            out.append(
                ColPat(
                    syms=tuple(syms),
                    abc_mask=abc_mask,
                    d_mask=d_mask,
                    need_occ=need_occ,
                    need_d=need_d,
                )
            )
            return

        for s in allowed_symbols(c, r, m, h):
            if s is None:
                # If left blank, then A/B/C must all already be blocked
                req = bit(r)                  # A -> next class, same row
                if r > 0:
                    req |= bit(r - 1)         # B -> next class, row-1
                if r + 1 < h:
                    req |= bit(r + 1)         # C -> next class, row+1

                # If D is legal at this column, then D must also be blocked
                nd = need_d
                if c <= m - 2:
                    nd |= bit(r)

                rec(
                    r + 1,
                    syms + [None],
                    abc_mask,
                    d_mask,
                    need_occ | req,
                    nd,
                )

            elif s == 'A':
                t = r
                if abc_mask & bit(t):
                    continue
                rec(r + 1, syms + ['A'], abc_mask | bit(t), d_mask, need_occ, need_d)

            elif s == 'B':
                t = r - 1
                if abc_mask & bit(t):
                    continue
                rec(r + 1, syms + ['B'], abc_mask | bit(t), d_mask, need_occ, need_d)

            elif s == 'C':
                t = r + 1
                if abc_mask & bit(t):
                    continue
                rec(r + 1, syms + ['C'], abc_mask | bit(t), d_mask, need_occ, need_d)

            elif s == 'D':
                rec(r + 1, syms + ['D'], abc_mask, d_mask | bit(r), need_occ, need_d)

            else:
                raise ValueError(f"Unknown symbol: {s!r}")

    rec(0, [], 0, 0, 0, 0)
    return tuple(out)

# ---------------------------------------------------------------------
# Core recursive-pruning generator
# ---------------------------------------------------------------------

def _validate_size(w: int, h: int) -> None:
    if w < 2:
        raise ValueError("w must be >= 2.")
    if h < 1:
        raise ValueError("h must be >= 1.")
    if 36 % h != 0:
        raise ValueError("h must divide 36.")

def enumerate_maximal_exact_box(w: int, h: int) -> List[Motif]:
    """
    Enumerate all maximal motifs under self-contained-box semantics.

    Important:
    - No same-size dedup is performed.
    - This is exact-box enumeration, not translation-class enumeration.

    Returns
    -------
    List[Motif]
        Each motif is a tuple of m = w - 1 planning columns.
        Each planning column is a tuple of h symbols.
    """
    _validate_size(w, h)
    m = w - 1
    catalogs = [column_patterns(c, m, h) for c in range(m)]
    sols: List[Motif] = []

    def dfs(c: int, d_prev: int, need_prev: int, cols: List[Column]) -> None:
        """
        Before choosing column c:
          d_prev    = rows already occupied in the current target class by D
                      coming from column c-1
          need_prev = rows that must be occupied in the current target class,
                      inherited from blanks in column c-1
        """
        if c == m:
            # The last recursion step is valid only if nothing is still pending
            if d_prev == 0 and need_prev == 0:
                sols.append(tuple(cols))
            return

        for pat in catalogs[c]:
            # The current target class receives:
            #   A/B/C from this column  +  D from the previous column
            if pat.abc_mask & d_prev:
                continue  # in-degree > 1 somewhere

            occ = pat.abc_mask | d_prev

            # Two sets of demands must be satisfied:
            #   1) inherited D-blocking demands from the previous column
            #   2) current blanks' A/B/C-blocking demands
            if ((need_prev | pat.need_occ) & ~occ) != 0:
                continue

            # Advance to the next column
            dfs(c + 1, pat.d_mask, pat.need_d, cols + [pat.syms])

    dfs(0, 0, 0, [])
    return sols

@lru_cache(maxsize=None)
def count_maximal_exact_box(w: int, h: int) -> int:
    """
    Fast count-only DP for maximal exact-box motifs.
    Much faster than materializing every motif when only counts are needed.
    """
    _validate_size(w, h)
    m = w - 1
    catalogs = [column_patterns(c, m, h) for c in range(m)]

    @lru_cache(maxsize=None)
    def dp(c: int, d_prev: int, need_prev: int) -> int:
        if c == m:
            return 1 if (d_prev == 0 and need_prev == 0) else 0

        total = 0
        for pat in catalogs[c]:
            if pat.abc_mask & d_prev:
                continue
            occ = pat.abc_mask | d_prev
            if ((need_prev | pat.need_occ) & ~occ) != 0:
                continue
            total += dp(c + 1, pat.d_mask, pat.need_d)
        return total

    return dp(0, 0, 0)

# ---------------------------------------------------------------------
# Primitive test (exact-box version, no translation quotient)
# ---------------------------------------------------------------------

def smaller_repeat_factors(motif: Motif) -> List[Tuple[int, int]]:
    """
    Return all smaller exact-box sizes (u, v) such that this motif is an exact
    horizontal/vertical repetition of some smaller motif of size (u, v).

    Here:
      - motif has m = w - 1 planning columns and height h,
      - smaller motif has mp = u - 1 planning columns and height v,
      - we require mp | m and v | h,
      - repetition is checked directly on the symbol matrix.

    Returns
    -------
    List[(u, v)]
        All smaller sizes whose repeated symbol matrix equals this motif.
    """
    m = len(motif)
    h = len(motif[0]) if m else 0
    out: List[Tuple[int, int]] = []

    for mp in divisors(m):
        for v in divisors(h):
            if mp == m and v == h:
                continue

            ok = True
            for c in range(m):
                for r in range(h):
                    if motif[c][r] != motif[c % mp][r % v]:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                out.append((mp + 1, v))

    return out

def is_primitive_exact_box(motif: Motif) -> bool:
    """
    Primitive under exact-box semantics:
    the symbol matrix is not obtained by repeating any smaller exact-box motif.
    """
    return len(smaller_repeat_factors(motif)) == 0

def enumerate_primitive_exact_box(w: int, h: int) -> List[Motif]:
    """
    Enumerate maximal motifs of size (w, h) and keep only exact-box primitive ones.
    """
    return [motif for motif in enumerate_maximal_exact_box(w, h)
            if is_primitive_exact_box(motif)]

# ---------------------------------------------------------------------
# Helpers: representative-box edges and full-constellation tiling
# ---------------------------------------------------------------------

def representative_box_edges_user_ids(motif: Motif) -> List[str]:
    """
    Show the motif edges on the top-left representative box, using the user's
    numbering convention:
        id = plane * 36 + phase + 1

    This only prints the edges implied inside one box:
      planning cols 0..m-1, target cols up to m.
    """
    m = len(motif)
    h = len(motif[0]) if m else 0
    edges: List[str] = []

    for c in range(m):
        for r, s in enumerate(motif[c]):
            if s is None:
                continue
            dp, dn = OFFS[s]
            src = c * 36 + r + 1
            dst = (c + dp) * 36 + (r + dn) + 1
            edges.append(f"{src}-{dst}")

    return edges

def tile_constellation_inter_edges(
    motif: Motif,
    planes: int = 18,
    phases: int = 36,
    seam_open: bool = True,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Tile one exact-box motif onto the actual Walker-Star-like constellation.

    Tiling:
      - horizontal step = m = w - 1
      - vertical step   = h

    Seam handling:
      - if seam_open=True, no wraparound in the plane direction:
          A/B/C need p + 1 < planes
          D     need p + 2 < planes
      - phase direction here follows the self-contained-box tiling; because h|36
        and B/C are forbidden at row boundaries, inter edges do not cross block
        boundaries vertically.

    Returns
    -------
    List[((p1, n1), (p2, n2))]
        Undirected inter-edge list in (plane, phase) coordinates.
    """
    m = len(motif)
    h = len(motif[0]) if m else 0
    if phases % h != 0:
        raise ValueError("phases must be a multiple of the motif height h.")

    edges = set()

    for p in range(planes):
        c = p % m
        for blk in range(phases // h):
            base = blk * h
            for r, s in enumerate(motif[c]):
                if s is None:
                    continue
                dp, dn = OFFS[s]
                q = p + dp
                n2 = base + r + dn

                if seam_open:
                    if q >= planes:
                        continue
                else:
                    q %= planes

                # Under self-contained-box semantics, n2 already stays in range
                # because B/C are disallowed on the local top/bottom rows.
                if not (0 <= n2 < phases):
                    continue

                u = (p, base + r)
                v = (q, n2)
                edges.add(tuple(sorted((u, v))))

    return sorted(edges)

def constellation_inter_edges_user_ids(
    motif: Motif,
    planes: int = 18,
    phases: int = 36,
    seam_open: bool = True,
) -> List[str]:
    """
    Same as tile_constellation_inter_edges(), but returned as user IDs.
    """
    out = []
    for (p1, n1), (p2, n2) in tile_constellation_inter_edges(
        motif, planes=planes, phases=phases, seam_open=seam_open
    ):
        id1 = p1 * 36 + n1 + 1
        id2 = p2 * 36 + n2 + 1
        out.append(f"{id1}-{id2}")
    return out

# ---------------------------------------------------------------------
# Convenience reports
# ---------------------------------------------------------------------

def summary(w: int, h: int, show_examples: int = 5) -> str:
    """
    Small text report for quick inspection.
    """
    total = count_maximal_exact_box(w, h)
    motifs = enumerate_maximal_exact_box(w, h)
    prim = [m for m in motifs if is_primitive_exact_box(m)]

    lines = [
        f"(w, h) = ({w}, {h})",
        f"maximal exact-box motifs      : {total}",
        f"primitive exact-box motifs    : {len(prim)}",
    ]

    if show_examples > 0:
        lines.append("")
        lines.append("examples:")
        for i, m in enumerate(motifs[:show_examples], 1):
            lines.append(f"  #{i:02d}  {pretty_motif(m)}")
            lines.append(f"       box edges: {', '.join(representative_box_edges_user_ids(m))}")

    return "\n".join(lines)

# ---------------------------------------------------------------------
# Minimal self-checks
# ---------------------------------------------------------------------

def _self_check() -> None:
    """
    Regression checks that should remain stable under the current definitions.
    """
    # Known exact-box maximal counts under the recursive-pruning semantics
    assert count_maximal_exact_box(3, 2) == 13
    assert count_maximal_exact_box(3, 3) == 82
    assert count_maximal_exact_box(4, 2) == 69

    # User-discussed primitive example: width-2, height-3, pattern A C B
    acb: Motif = (('A', 'C', 'B'),)
    assert acb in enumerate_maximal_exact_box(2, 3)
    assert is_primitive_exact_box(acb)

if __name__ == "__main__":
    _self_check()

    # Demo
    for (w, h) in [(2, 3), (3, 2), (4, 2)]:
        print(summary(w, h, show_examples=3))
        print("-" * 72)
