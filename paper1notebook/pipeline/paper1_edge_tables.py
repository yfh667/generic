from __future__ import annotations

import json
from pathlib import Path

from src.link_delay.module.edge_options import EdgeTable
from src.topology_workflow.module.edge_tables import add_intra_ring_records, make_edge_table_from_records


LEGACY_OPTION_DELTAS = {
    0: (1, 0),
    1: (1, -1),
    2: (2, 0),
    4: (1, 1),
    5: (2, -1),
    6: (1, 2),
}


def legacy_target(x: int, y: int, *, n: int, option: int) -> tuple[int, int]:
    if int(option) not in LEGACY_OPTION_DELTAS:
        raise ValueError(f"unsupported legacy option: {option}")
    dx, dy = LEGACY_OPTION_DELTAS[int(option)]
    return int(x) + int(dx), (int(y) + int(dy) + int(n)) % int(n)


def build_legacy_gridplus_edge_table(*, motif_json: str | Path, add_intra_ring: bool = True) -> EdgeTable:
    """Build the historical paper1 grid+ baseline from its motif.json.

    This remains in the paper1 pipeline layer because the file format and the
    overwrite semantics are a paper-specific legacy baseline, not a general
    topology module contract.
    """

    raw = json.loads(Path(motif_json).read_text(encoding="utf-8"))
    p = int(raw["P"])
    n = int(raw["N"])
    outgoing: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    incoming: dict[tuple[int, int], tuple[int, int]] = {}

    def set_edge(src: tuple[int, int], dst: tuple[int, int], option: int) -> None:
        old = outgoing.get(src)
        if old is not None:
            old_dst, _old_option = old
            if incoming.get(old_dst) == src:
                incoming.pop(old_dst, None)
        old_src = incoming.get(dst)
        if old_src is not None:
            old_out = outgoing.get(old_src)
            if old_out is not None and old_out[0] == dst:
                outgoing.pop(old_src, None)
        outgoing[src] = (dst, int(option))
        incoming[dst] = src

    for motif in raw.get("motifs", []):
        p_start = int(motif["p_start"])
        p_end = int(motif["p_end"])
        y_start = int(motif["y_start"])
        y_end = int(motif["y_end"])
        option = int(motif["option"])

        for x in range(p_start, p_end + 1):
            for y in range(y_start, y_end + 1):
                target_x, target_y = legacy_target(x, y, n=n, option=option)
                if target_x < p_start or target_x > p_end:
                    continue
                if target_y < y_start or target_y > y_end:
                    continue
                set_edge((x, y), (target_x, target_y), option)

    records: list[tuple[int, int, int, int, int]] = []
    for (x, y), ((target_x, target_y), option) in outgoing.items():
        if not (0 <= x < p - 1 and 0 <= y < n and 0 <= target_x < p and 0 <= target_y < n):
            continue
        records.append((x, y, target_x, target_y, option))

    if bool(add_intra_ring):
        add_intra_ring_records(records, p=p, n=n)
    return make_edge_table_from_records(p=p, n=n, records=records)
