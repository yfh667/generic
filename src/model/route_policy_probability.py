from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
from functools import lru_cache


DEFAULT_OPTION_DELTA = {
    0: (1, 0),
    1: (1, -1),
    2: (2, 0),
    4: (1, 1),
    5: (2, -1),
    6: (1, 2),
}


@lru_cache(maxsize=128)
def _load_json_cached(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_prob(name: str, v):
    if v is None:
        return None
    x = float(v)
    if not (0 < x <= 1):
        raise ValueError(f"{name} must be in (0,1], got {v}")
    return x


def build_delta2option(policy: dict, n: int):
    raw = policy.get("option_delta_map", None)
    mapping = {}
    if raw:
        for k, vv in raw.items():
            op = int(k)
            dx = int(vv[0])
            dy = int(vv[1]) % int(n)
            mapping[(dx, dy)] = op
    else:
        for op, (dx, dy_raw) in DEFAULT_OPTION_DELTA.items():
            mapping[(int(dx), int(dy_raw) % int(n))] = int(op)
    return mapping


def load_route_policy(policy_path: Path, *, n: int, fallback_p_intra: float, fallback_p_inter: float):
    raw = _load_json_cached(str(policy_path))
    if raw is None:
        raw = {
            "policy_name": "fallback_constant_prob",
            "route_mode": "max_reliability",
            "p_intra": float(fallback_p_intra),
            "default_p_inter": float(fallback_p_inter),
            "option_p_inter": {},
            "conflict_policy": "max_probability",
        }

    p_intra = _validate_prob("p_intra", raw.get("p_intra", fallback_p_intra))
    default_p_inter = _validate_prob("default_p_inter", raw.get("default_p_inter", fallback_p_inter))

    option_p = {}
    for k, v in (raw.get("option_p_inter", {}) or {}).items():
        op = int(k)
        option_p[op] = _validate_prob(f"option_p_inter[{op}]", v)

    return {
        "policy_name": str(raw.get("policy_name", "route_policy")),
        "route_mode": str(raw.get("route_mode", "max_reliability")),
        "p_intra": p_intra,
        "default_p_inter": default_p_inter,
        "option_p_inter": option_p,
        "conflict_policy": str(raw.get("conflict_policy", "max_probability")),
        "delta2option": build_delta2option(raw, n=n),
    }


def edge_option_from_smaller_node(u: int, v: int, *, n: int, delta2option: dict):
    a = int(min(u, v))
    b = int(max(u, v))

    pa, ya = divmod(a, int(n))
    pb, yb = divmod(b, int(n))

    if pa == pb:
        return None

    dx = pb - pa
    dy = (yb - ya) % int(n)
    return delta2option.get((dx, dy), None)


def compute_path_reliability_and_hops(
    path_str: str,
    *,
    n: int,
    p_intra: float,
    default_p_inter: float,
    option_p_inter: dict,
    delta2option: dict,
    unknown_option_action: str = "use_default",  # use_default | raise
):
    if not isinstance(path_str, str) or (not path_str.strip()):
        return 0.0, 0, 0, 0, Counter()

    nodes = [int(x) for x in path_str.split("->")]
    rel = 1.0
    intra_hops = 0
    inter_hops = 0
    unknown_option_edges = 0
    option_counter = Counter()

    for u, v in zip(nodes[:-1], nodes[1:]):
        pu = int(u) // int(n)
        pv = int(v) // int(n)

        if pu == pv:
            intra_hops += 1
            rel *= float(p_intra)
            continue

        inter_hops += 1
        op = edge_option_from_smaller_node(u, v, n=n, delta2option=delta2option)

        if op is None:
            unknown_option_edges += 1
            if unknown_option_action == "raise":
                raise ValueError(f"cannot map inter edge to option: ({u}, {v})")
            p = default_p_inter
        else:
            option_counter[op] += 1
            p = option_p_inter.get(op, default_p_inter)

        if p is None:
            raise ValueError(f"missing probability for inter edge ({u},{v}), option={op}")

        rel *= float(p)

    return float(rel), int(intra_hops), int(inter_hops), int(unknown_option_edges), option_counter
