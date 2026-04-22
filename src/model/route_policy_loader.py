from __future__ import annotations

import json
import re
from pathlib import Path


def _safe_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "policy"


def _to_prob(name: str, v, allow_none: bool = False):
    if v is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be None")
    x = float(v)
    if not (0 < x <= 1):
        raise ValueError(f"{name} must be in (0,1], got {v}")
    return x


def resolve_route_policy_path(
    data_root: Path,
    policy_spec: str,
    *,
    motif: str | None = None,
) -> Path:
    """
    解析策略路径，优先级：
    1) policy_spec 绝对路径
    2) data_root/route_policy/<policy_spec>[.json]
    3) 兼容旧目录: data_root/topology_design/<motif>/config/...
    """
    data_root = Path(data_root)
    spec = (policy_spec or "").strip()
    cands: list[Path] = []

    if spec:
        p = Path(spec)
        if p.is_absolute():
            cands.append(p)
        else:
            cands.append(data_root / "route_policy" / spec)
            if not spec.lower().endswith(".json"):
                cands.append(data_root / "route_policy" / f"{spec}.json")

            if motif:
                cfg_dir = data_root / "topology_design" / motif / "config"
                cands.append(cfg_dir / spec)
                cands.append(cfg_dir / "route_policies" / spec)
                if not spec.lower().endswith(".json"):
                    cands.append(cfg_dir / f"{spec}.json")
                    cands.append(cfg_dir / "route_policies" / f"{spec}.json")

    if motif:
        cands.append(data_root / "topology_design" / motif / "config" / "route_policy.json")
        cands.append(data_root / "topology_design" / motif / "config" / "route_policies" / "route_policy.json")

    # 去重并查找
    seen = set()
    uniq = []
    for p in cands:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)

    for p in uniq:
        if p.exists():
            return p

    raise FileNotFoundError("route policy not found, tried:\n" + "\n".join(str(x) for x in uniq))


def load_route_policy(
    *,
    data_root: Path,
    policy_spec: str,
    motif: str | None = None,
    fallback_mode: str = "max_reliability",
    fallback_p_intra: float = 0.995,
    fallback_p_inter: float = 0.99,
) -> dict:
    path = resolve_route_policy_path(data_root, policy_spec, motif=motif)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    route_mode = str(raw.get("route_mode", fallback_mode))
    if route_mode not in {"min_hop", "max_reliability"}:
        raise ValueError(f"invalid route_mode={route_mode!r}")

    option_map = {}
    for k, v in (raw.get("option_p_inter", {}) or {}).items():
        option_map[int(k)] = _to_prob(f"option_p_inter[{k}]", v)

    policy_name = str(raw.get("policy_name", path.stem))
    out = {
        "policy_name": policy_name,
        "policy_tag": _safe_tag(policy_name),
        "route_mode": route_mode,
        "p_intra": _to_prob("p_intra", raw.get("p_intra", fallback_p_intra)),
        "default_p_inter": _to_prob(
            "default_p_inter",
            raw.get("default_p_inter", fallback_p_inter),
            allow_none=True,
        ),
        "option_p_inter": option_map,
        "conflict_policy": str(raw.get("conflict_policy", "max_probability")),
        "policy_path": str(path),
    }
    return out
