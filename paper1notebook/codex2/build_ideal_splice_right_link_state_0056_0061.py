from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402


DEFAULT_DELAY_USAGE_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\delay_edge_usage_056_061_china_europe"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\ideal_splice_056_061_056_china_europe"
)
PAIR_KEY = "china_europe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-node right-link state for the ideal no-LST splice "
            "motif000056 -> motif000061 -> motif000056."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--middle-motif-id", type=int, default=61)
    parser.add_argument("--delay-usage-root", type=Path, default=DEFAULT_DELAY_USAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write-long-csv", action="store_true")
    parser.add_argument("--working-threshold", type=float, default=0.0)
    return parser.parse_args()


def build_topology(motif_id: int, *, start: int, end: int, stride: int) -> one_way.TopologyData:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    spec = one_way.build_topology_spec(int(motif_id), rows[int(motif_id)])
    return one_way.load_topology_data(spec, start=int(start), end=int(end), stride=int(stride))


def load_delay_usage(
    topo: one_way.TopologyData,
    *,
    delay_usage_root: Path,
    start: int,
    end: int,
    stride: int,
) -> np.ndarray:
    path = (
        Path(delay_usage_root)
        / f"t{int(start)}_{int(end)}_stride{int(stride)}"
        / topo.spec.name
        / PAIR_KEY
        / "edge_betweenness.npy"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing delay betweenness cache: {path}\n"
            "Run compute_delay_edge_usage_0056_0061_china_europe.py first."
        )
    arr = np.load(path, mmap_mode="r")
    if arr.shape != topo.values.shape:
        raise ValueError(f"delay usage shape mismatch for {topo.spec.name}: {arr.shape} != {topo.values.shape}")
    return np.asarray(arr, dtype=np.float32)


def topology_for_step(step: int, *, switch_step: int, return_switch_step: int, source_id: int, middle_id: int) -> int:
    if int(switch_step) <= int(step) < int(return_switch_step):
        return int(middle_id)
    return int(source_id)


def union_right_links(*topo_list: one_way.TopologyData) -> tuple[list[one_way.RightLink], dict[tuple[int, int], int]]:
    links: list[one_way.RightLink] = []
    key_to_idx: dict[tuple[int, int], int] = {}
    for topo in topo_list:
        for link in topo.right_by_owner.values():
            key = link.edge_key
            if key in key_to_idx:
                continue
            key_to_idx[key] = len(links)
            links.append(link)
    links.sort(key=lambda link: (link.owner, link.right, link.edge_key))
    key_to_idx = {link.edge_key: idx for idx, link in enumerate(links)}
    return links, key_to_idx


def write_right_edges(path: Path, links: list[one_way.RightLink]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "right_edge_idx",
                "owner",
                "owner_p",
                "owner_y",
                "right",
                "right_p",
                "right_y",
                "option",
                "symbol",
                "edge_key",
            ]
        )
        for idx, link in enumerate(links):
            writer.writerow(
                [
                    int(idx),
                    int(link.owner),
                    int(link.owner_p),
                    int(link.owner_y),
                    int(link.right),
                    int(link.right_p),
                    int(link.right_y),
                    int(link.option),
                    str(link.symbol),
                    str(link.edge_key_text),
                ]
            )


def write_node_static(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "p", "y"])
        for node in range(int(G60_CONFIG.total_sats)):
            writer.writerow([int(node), int(node // G60_CONFIG.N), int(node % G60_CONFIG.N)])


def write_long_csv(
    path: Path,
    *,
    steps: list[int],
    topology_ids: np.ndarray,
    right_neighbor: np.ndarray,
    right_option: np.ndarray,
    right_symbol_id: np.ndarray,
    right_edge_idx: np.ndarray,
    hop_betweenness: np.ndarray,
    delay_betweenness: np.ndarray,
    threshold: float,
) -> None:
    symbols = np.asarray(["", "A", "B", "C", "D"], dtype=object)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "hour",
                "topology_motif_id",
                "node",
                "node_p",
                "node_y",
                "right_node",
                "right_p",
                "right_y",
                "option",
                "symbol",
                "right_edge_idx",
                "hop_betweenness",
                "delay_betweenness",
                "working_hop",
                "working_delay",
                "working_any",
            ]
        )
        for row, step in enumerate(steps):
            for node in range(int(G60_CONFIG.total_sats)):
                right = int(right_neighbor[row, node])
                writer.writerow(
                    [
                        int(step),
                        float(step) / 3600.0,
                        int(topology_ids[row]),
                        int(node),
                        int(node // G60_CONFIG.N),
                        int(node % G60_CONFIG.N),
                        "" if right < 0 else right,
                        "" if right < 0 else int(right // G60_CONFIG.N),
                        "" if right < 0 else int(right % G60_CONFIG.N),
                        "" if right < 0 else int(right_option[row, node]),
                        "" if right < 0 else str(symbols[int(right_symbol_id[row, node])]),
                        "" if right < 0 else int(right_edge_idx[row, node]),
                        float(hop_betweenness[row, node]),
                        float(delay_betweenness[row, node]),
                        int(hop_betweenness[row, node] > float(threshold)),
                        int(delay_betweenness[row, node] > float(threshold)),
                        int(
                            (hop_betweenness[row, node] > float(threshold))
                            or (delay_betweenness[row, node] > float(threshold))
                        ),
                    ]
                )


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time axis")
    if int(args.switch_step) not in set(steps) or int(args.return_switch_step) not in set(steps):
        raise ValueError("switch steps must be on the requested time axis")

    source = build_topology(
        int(args.source_motif_id),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    middle = build_topology(
        int(args.middle_motif_id),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    delay_source = load_delay_usage(
        source,
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    delay_middle = load_delay_usage(
        middle,
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"
        / "right_link_state"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    right_links, right_key_to_idx = union_right_links(source, middle)
    num_steps = len(steps)
    num_nodes = int(G60_CONFIG.total_sats)
    num_edges = len(right_links)

    topology_ids = np.empty(num_steps, dtype=np.int16)
    active_right_edge_mask = np.zeros((num_steps, num_edges), dtype=bool)
    edge_hop_betweenness = np.zeros((num_steps, num_edges), dtype=np.float32)
    edge_delay_betweenness = np.zeros((num_steps, num_edges), dtype=np.float32)

    right_neighbor = np.full((num_steps, num_nodes), -1, dtype=np.int32)
    right_option = np.full((num_steps, num_nodes), -999, dtype=np.int16)
    right_symbol_id = np.zeros((num_steps, num_nodes), dtype=np.int8)
    right_edge_idx = np.full((num_steps, num_nodes), -1, dtype=np.int32)
    node_hop_betweenness = np.zeros((num_steps, num_nodes), dtype=np.float32)
    node_delay_betweenness = np.zeros((num_steps, num_nodes), dtype=np.float32)

    symbol_to_id = {"A": 1, "B": 2, "C": 3, "D": 4}
    topology_by_id = {int(args.source_motif_id): source, int(args.middle_motif_id): middle}
    delay_by_id = {int(args.source_motif_id): delay_source, int(args.middle_motif_id): delay_middle}

    for row, step in enumerate(steps):
        motif_id = topology_for_step(
            int(step),
            switch_step=int(args.switch_step),
            return_switch_step=int(args.return_switch_step),
            source_id=int(args.source_motif_id),
            middle_id=int(args.middle_motif_id),
        )
        topology_ids[row] = int(motif_id)
        topo = topology_by_id[int(motif_id)]
        delay_values = delay_by_id[int(motif_id)]
        for owner, link in topo.right_by_owner.items():
            union_idx = int(right_key_to_idx[link.edge_key])
            hop_value = float(topo.values[row, link.edge_idx])
            delay_value = float(delay_values[row, link.edge_idx])
            active_right_edge_mask[row, union_idx] = True
            edge_hop_betweenness[row, union_idx] = hop_value
            edge_delay_betweenness[row, union_idx] = delay_value

            owner = int(owner)
            right_neighbor[row, owner] = int(link.right)
            right_option[row, owner] = int(link.option)
            right_symbol_id[row, owner] = int(symbol_to_id.get(str(link.symbol), 0))
            right_edge_idx[row, owner] = union_idx
            node_hop_betweenness[row, owner] = hop_value
            node_delay_betweenness[row, owner] = delay_value

    threshold = float(args.working_threshold)
    np.save(out_dir / "steps.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "topology_motif_id.npy", topology_ids)
    np.save(out_dir / "active_right_edge_mask.npy", active_right_edge_mask)
    np.save(out_dir / "right_edge_hop_betweenness.npy", edge_hop_betweenness)
    np.save(out_dir / "right_edge_delay_betweenness.npy", edge_delay_betweenness)
    np.save(out_dir / "right_neighbor.npy", right_neighbor)
    np.save(out_dir / "right_option.npy", right_option)
    np.save(out_dir / "right_symbol_id.npy", right_symbol_id)
    np.save(out_dir / "right_edge_idx_by_node.npy", right_edge_idx)
    np.save(out_dir / "node_right_hop_betweenness.npy", node_hop_betweenness)
    np.save(out_dir / "node_right_delay_betweenness.npy", node_delay_betweenness)
    np.save(out_dir / "node_right_working_hop.npy", node_hop_betweenness > threshold)
    np.save(out_dir / "node_right_working_delay.npy", node_delay_betweenness > threshold)
    np.save(
        out_dir / "node_right_working_any.npy",
        (node_hop_betweenness > threshold) | (node_delay_betweenness > threshold),
    )
    write_right_edges(out_dir / "right_edges.csv", right_links)
    write_node_static(out_dir / "nodes.csv")

    if bool(args.write_long_csv):
        write_long_csv(
            out_dir / "node_right_link_state_long.csv.gz",
            steps=steps,
            topology_ids=topology_ids,
            right_neighbor=right_neighbor,
            right_option=right_option,
            right_symbol_id=right_symbol_id,
            right_edge_idx=right_edge_idx,
            hop_betweenness=node_hop_betweenness,
            delay_betweenness=node_delay_betweenness,
            threshold=threshold,
        )

    meta = {
        "description": "Ideal no-LST right-link state for motif000056 -> motif000061 -> motif000056.",
        "constellation": G60_CONFIG.name,
        "constellation_p": int(G60_CONFIG.P),
        "constellation_n": int(G60_CONFIG.N),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "source_motif_id": int(args.source_motif_id),
        "source_motif": source.spec.motif,
        "middle_motif_id": int(args.middle_motif_id),
        "middle_motif": middle.spec.motif,
        "pair_key": PAIR_KEY,
        "num_steps": int(num_steps),
        "num_nodes": int(num_nodes),
        "num_union_right_edges": int(num_edges),
        "working_threshold": threshold,
        "working_rule": "working_hop/delay/any is true when the corresponding right-link edge betweenness is > working_threshold.",
        "right_link_only": True,
        "outputs": {
            "right_neighbor": "right_neighbor.npy",
            "node_hop_betweenness": "node_right_hop_betweenness.npy",
            "node_delay_betweenness": "node_right_delay_betweenness.npy",
            "node_working_any": "node_right_working_any.npy",
            "right_edges": "right_edges.csv",
            "long_csv": "node_right_link_state_long.csv.gz" if bool(args.write_long_csv) else None,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"out_dir={out_dir}")
    print(f"steps={num_steps} nodes={num_nodes} union_right_edges={num_edges}")
    print(
        "working_any_entries="
        f"{int(np.count_nonzero((node_hop_betweenness > threshold) | (node_delay_betweenness > threshold)))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
