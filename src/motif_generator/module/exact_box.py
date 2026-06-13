from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


Symbol = Optional[str]
Column = tuple[Symbol, ...]
Motif = tuple[Column, ...]


OFFSETS: dict[str, tuple[int, int]] = {
    "A": (1, 0),
    "B": (1, -1),
    "C": (1, 1),
    "D": (2, 0),
}


@dataclass(frozen=True)
class ColPattern:
    """Summary of one planning-column pattern."""

    syms: Column
    abc_mask: int
    d_mask: int
    need_occ: int
    need_d: int


@dataclass(frozen=True)
class EdgeRecord:
    """One edge induced by a motif in local box coordinates."""

    src_col: int
    src_row: int
    dst_col: int
    dst_row: int
    symbol: str

    def user_id_pair(self, row_pitch: int) -> tuple[int, int]:
        src = self.src_col * int(row_pitch) + self.src_row + 1
        dst = self.dst_col * int(row_pitch) + self.dst_row + 1
        return src, dst


def bit(row: int) -> int:
    return 1 << int(row)


def divisors(value: int) -> list[int]:
    value = int(value)
    return [d for d in range(1, value + 1) if value % d == 0]


def validate_size(w: int, h: int, *, phase_count: int | None = None) -> tuple[int, int]:
    w = int(w)
    h = int(h)
    if w < 2:
        raise ValueError("w must be >= 2; the last box column is the target boundary.")
    if h < 1:
        raise ValueError("h must be >= 1.")
    if phase_count is not None and int(phase_count) % h != 0:
        raise ValueError(f"h={h} must divide phase_count={int(phase_count)} for exact vertical tiling.")
    return w, h


def pretty_motif(motif: Motif) -> str:
    """Human-readable symbol view, column by column."""

    return " | ".join("".join("-" if s is None else s for s in col) for col in motif)


def allowed_symbols(c: int, r: int, m: int, h: int) -> list[Symbol]:
    """Allowed symbols at planning position ``(c, r)``.

    ``m`` is the number of planning columns, i.e. ``w - 1``.
    """

    c = int(c)
    r = int(r)
    m = int(m)
    h = int(h)
    out: list[Symbol] = [None, "A"]
    if r > 0:
        out.append("B")
    if r + 1 < h:
        out.append("C")
    if c <= m - 2:
        out.append("D")
    return out


@lru_cache(maxsize=None)
def column_patterns(c: int, m: int, h: int) -> tuple[ColPattern, ...]:
    """Generate locally consistent patterns for one planning column."""

    c = int(c)
    m = int(m)
    h = int(h)
    out: list[ColPattern] = []

    def rec(
        r: int,
        syms: list[Symbol],
        abc_mask: int,
        d_mask: int,
        need_occ: int,
        need_d: int,
    ) -> None:
        if r == h:
            out.append(
                ColPattern(
                    syms=tuple(syms),
                    abc_mask=int(abc_mask),
                    d_mask=int(d_mask),
                    need_occ=int(need_occ),
                    need_d=int(need_d),
                )
            )
            return

        for symbol in allowed_symbols(c, r, m, h):
            if symbol is None:
                req = bit(r)
                if r > 0:
                    req |= bit(r - 1)
                if r + 1 < h:
                    req |= bit(r + 1)

                next_need_d = need_d
                if c <= m - 2:
                    next_need_d |= bit(r)

                rec(r + 1, syms + [None], abc_mask, d_mask, need_occ | req, next_need_d)

            elif symbol == "A":
                target_row = r
                if abc_mask & bit(target_row):
                    continue
                rec(r + 1, syms + ["A"], abc_mask | bit(target_row), d_mask, need_occ, need_d)

            elif symbol == "B":
                target_row = r - 1
                if abc_mask & bit(target_row):
                    continue
                rec(r + 1, syms + ["B"], abc_mask | bit(target_row), d_mask, need_occ, need_d)

            elif symbol == "C":
                target_row = r + 1
                if abc_mask & bit(target_row):
                    continue
                rec(r + 1, syms + ["C"], abc_mask | bit(target_row), d_mask, need_occ, need_d)

            elif symbol == "D":
                rec(r + 1, syms + ["D"], abc_mask, d_mask | bit(r), need_occ, need_d)

            else:
                raise ValueError(f"unknown symbol: {symbol!r}")

    rec(0, [], 0, 0, 0, 0)
    return tuple(out)


def enumerate_maximal_exact_box(w: int, h: int, *, phase_count: int | None = None) -> list[Motif]:
    """Enumerate maximal exact-box motifs for a given ``w`` and ``h``.

    The motif box has width ``w`` and height ``h``. The first ``w - 1`` columns
    are planning columns; the last column is a target boundary inside the box.

    This function intentionally does not quotient by translation equivalence.
    """

    w, h = validate_size(w, h, phase_count=phase_count)
    m = w - 1
    catalogs = [column_patterns(c, m, h) for c in range(m)]
    motifs: list[Motif] = []

    def dfs(c: int, d_prev: int, need_prev: int, cols: list[Column]) -> None:
        if c == m:
            if d_prev == 0 and need_prev == 0:
                motifs.append(tuple(cols))
            return

        for pat in catalogs[c]:
            if pat.abc_mask & d_prev:
                continue
            occ = pat.abc_mask | d_prev
            if ((need_prev | pat.need_occ) & ~occ) != 0:
                continue
            dfs(c + 1, pat.d_mask, pat.need_d, cols + [pat.syms])

    dfs(0, 0, 0, [])
    return motifs


@lru_cache(maxsize=None)
def _count_maximal_exact_box_cached(w: int, h: int) -> int:
    w, h = validate_size(w, h)
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


def count_maximal_exact_box(w: int, h: int, *, phase_count: int | None = None) -> int:
    """Count maximal exact-box motifs without materializing them."""

    w, h = validate_size(w, h, phase_count=phase_count)
    return _count_maximal_exact_box_cached(w, h)


def smaller_repeat_factors(motif: Motif) -> list[tuple[int, int]]:
    """Return smaller exact-box sizes that tile this motif's symbol matrix."""

    m = len(motif)
    h = len(motif[0]) if m else 0
    out: list[tuple[int, int]] = []

    for small_m in divisors(m):
        for small_h in divisors(h):
            if small_m == m and small_h == h:
                continue
            ok = True
            for c in range(m):
                for r in range(h):
                    if motif[c][r] != motif[c % small_m][r % small_h]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((small_m + 1, small_h))

    return out


def is_primitive_exact_box(motif: Motif) -> bool:
    return len(smaller_repeat_factors(motif)) == 0


def enumerate_primitive_exact_box(w: int, h: int, *, phase_count: int | None = None) -> list[Motif]:
    """Enumerate maximal exact-box motifs and keep only primitive matrices."""

    return [
        motif
        for motif in enumerate_maximal_exact_box(w, h, phase_count=phase_count)
        if is_primitive_exact_box(motif)
    ]


def motif_edge_records(motif: Motif) -> list[EdgeRecord]:
    """Convert a motif symbol matrix to local edge records."""

    records: list[EdgeRecord] = []
    for col, column in enumerate(motif):
        for row, symbol in enumerate(column):
            if symbol is None:
                continue
            dp, dy = OFFSETS[str(symbol)]
            records.append(
                EdgeRecord(
                    src_col=int(col),
                    src_row=int(row),
                    dst_col=int(col + dp),
                    dst_row=int(row + dy),
                    symbol=str(symbol),
                )
            )
    return records


def motif_edges_as_user_ids(motif: Motif, *, row_pitch: int) -> list[str]:
    """Return local representative-box edges as ``src-dst`` strings."""

    out = []
    for edge in motif_edge_records(motif):
        src, dst = edge.user_id_pair(row_pitch)
        out.append(f"{src}-{dst}")
    return out


def motif_from_columns(columns: list[str] | tuple[str, ...]) -> Motif:
    """Build a motif from strings like ``["DAD", "C--"]``."""

    motif: list[Column] = []
    for col in columns:
        values: list[Symbol] = []
        for char in str(col).strip():
            if char == "-":
                values.append(None)
            elif char in OFFSETS:
                values.append(char)
            else:
                raise ValueError(f"unsupported motif symbol {char!r}; use A/B/C/D/-")
        motif.append(tuple(values))
    if not motif:
        raise ValueError("motif must contain at least one planning column")
    height = len(motif[0])
    if any(len(col) != height for col in motif):
        raise ValueError("all motif columns must have the same height")
    return tuple(motif)
