from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .exact_box import OFFSETS, EdgeRecord


# This default matches the existing row-index convention used by the current
# motif generator code. Keep offsets explicit in YAML when a paper section uses
# another coordinate convention.
DEFAULT_SUPPORT_OFFSETS: dict[str, tuple[int, int]] = dict(OFFSETS)


@dataclass(frozen=True)
class SupportEntry:
    x: int
    y: int
    symbol: str


@dataclass(frozen=True)
class MotifSupport:
    w: int
    h: int
    support: tuple[SupportEntry, ...]
    offsets: dict[str, tuple[int, int]]
    name: str = ""


def normalize_offsets(raw: dict | None = None) -> dict[str, tuple[int, int]]:
    if raw is None:
        return dict(DEFAULT_SUPPORT_OFFSETS)

    offsets: dict[str, tuple[int, int]] = {}
    for symbol, value in raw.items():
        if symbol not in DEFAULT_SUPPORT_OFFSETS:
            raise ValueError(f"unsupported symbol in offsets: {symbol!r}")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"offset for {symbol!r} must be a 2-item list/tuple")
        offsets[str(symbol)] = (int(value[0]), int(value[1]))

    missing = sorted(set(DEFAULT_SUPPORT_OFFSETS) - set(offsets))
    if missing:
        raise ValueError(f"missing offsets for symbols: {missing}")
    return offsets


def normalize_support_entries(raw_support: Iterable) -> tuple[SupportEntry, ...]:
    entries: list[SupportEntry] = []
    seen: set[tuple[int, int]] = set()
    for item in raw_support:
        if isinstance(item, dict):
            x = int(item["x"])
            y = int(item["y"])
            symbol = str(item["symbol"])
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            x = int(item[0])
            y = int(item[1])
            symbol = str(item[2])
        else:
            raise ValueError(f"support entry must be [x, y, symbol] or a dict, got: {item!r}")

        if (x, y) in seen:
            raise ValueError(f"duplicated support source coordinate: {(x, y)}")
        seen.add((x, y))
        entries.append(SupportEntry(x=x, y=y, symbol=symbol))
    return tuple(entries)


def motif_support_from_dict(raw: dict) -> MotifSupport:
    motif_raw = raw.get("motif", raw)
    w = int(motif_raw["w"])
    h = int(motif_raw["h"])
    if w < 2 or h < 1:
        raise ValueError("motif w must be >= 2 and h must be >= 1")
    support = normalize_support_entries(motif_raw.get("support", []))
    offsets = normalize_offsets(motif_raw.get("offsets"))
    name = str(motif_raw.get("name", raw.get("name", "")))

    for entry in support:
        if entry.symbol not in offsets:
            raise ValueError(f"support symbol {entry.symbol!r} has no offset")
        if not (0 <= entry.x < w - 1 and 0 <= entry.y < h):
            raise ValueError(
                f"support source {(entry.x, entry.y)} outside planning area: "
                f"x must be 0..{w - 2}, y must be 0..{h - 1}"
            )
        dx, dy = offsets[entry.symbol]
        tx = entry.x + dx
        ty = entry.y + dy
        if not (0 <= tx < w and 0 <= ty < h):
            raise ValueError(
                f"support edge {(entry.x, entry.y, entry.symbol)} targets {(tx, ty)}, "
                f"outside motif box w={w}, h={h}"
            )

    return MotifSupport(w=w, h=h, support=support, offsets=offsets, name=name)


def motif_support_to_edge_records(motif: MotifSupport) -> list[EdgeRecord]:
    records: list[EdgeRecord] = []
    for entry in motif.support:
        dx, dy = motif.offsets[entry.symbol]
        records.append(
            EdgeRecord(
                src_col=int(entry.x),
                src_row=int(entry.y),
                dst_col=int(entry.x + dx),
                dst_row=int(entry.y + dy),
                symbol=str(entry.symbol),
            )
        )
    return records


def motif_support_label(motif: MotifSupport) -> str:
    return "{" + ", ".join(f"({e.x},{e.y},{e.symbol})" for e in motif.support) + "}"
