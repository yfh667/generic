from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


# _SUPPORTED_OBJECTIVES = {"shortest_hop"}

# 替换后
_SUPPORTED_OBJECTIVES = {"shortest_hop"}
_ROUTE_MODE_TO_OBJECTIVE = {
    "min_hop": "shortest_hop",
    "shortest_hop": "shortest_hop",
}

def _check_prob(name: str, v: Any) -> float:
    x = float(v)
    if not (0 < x <= 1):
        raise ValueError(f"{name} must be in (0, 1], got {v}")
    return x

def _normalize_objective(raw: dict[str, Any]) -> str:
    obj = raw.get("objective", None)
    if obj is not None and str(obj).strip():
        return str(obj).strip()
    mode = str(raw.get("route_mode", "")).strip().lower()
    return _ROUTE_MODE_TO_OBJECTIVE.get(mode, "shortest_hop")



def _deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()
def _safe_fs_tag(s: str) -> str:
    s = str(s).strip()
    if not s:
        return "route"
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in s)


class RoutePolicy:
    """
    Thin wrapper around route*.json.

    The only per-motif hard-coded file should remain:
        data/topology_design/<motif>/config/motif.json

    Everything else that describes route behavior, output naming, probability
    parameters, and time-step policy comes from this object.
    """

    def __init__(self, raw: dict[str, Any], *, path: str | Path | None = None):
        self.raw = raw
        self.path = Path(path).expanduser() if path else None
        self.route_name = str(raw.get("route_name") or raw.get("name") or "route_policy")
        # 替换后（__init__里）
        self.objective = _normalize_objective(raw)

      #  self.objective = str(raw.get("objective") or "shortest_hop")
        if self.objective not in _SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unsupported route objective {self.objective!r}. "
                f"Supported objectives: {sorted(_SUPPORTED_OBJECTIVES)}"
            )

    def get(self, dotted: str, default: Any = None) -> Any:
        return _deep_get(self.raw, dotted, default)

    @property
    def start(self) -> int:
        return int(self.get("time_window.start", 0))

    # 在 RoutePolicy 类里新增
    @property
    def output_layout(self) -> str:
        """
        by_route: <motif>/route_policy/<route_name>/...
        by_motif: <motif>/...
        """
        v = str(self.get("outputs.layout", "by_route")).strip().lower()
        if v not in {"by_route", "by_motif"}:
            raise ValueError(f"outputs.layout must be 'by_route' or 'by_motif', got {v!r}")
        return v

    @property
    def end(self) -> int:
        return int(self.get("time_window.end", self.start))

    @property
    def stride(self) -> int:
        return max(1, int(self.get("time_window.stride", 1)))

    @property
    def step_mode(self) -> str:
        return str(self.get("time_window.step_mode", "actual")).lower()

    # @property
    # def p_intra(self) -> float:
    #     return float(self.get("link_probability.p_intra", 0.999))
    #
    # @property
    # def p_inter(self) -> float:
    #     return float(self.get("link_probability.p_inter", 0.99))
    # 替换后


    @property
    def p_intra(self) -> float:
        return _check_prob("p_intra", self.get("link_probability.p_intra", 0.999))

    # @property
    # def default_p_inter(self) -> float:
    #     v = self.get("link_probability.default_p_inter", None)
    #     if v is None:
    #         v = self.get("link_probability.p_inter", 0.99)  # 兼容旧字段
    #     return _check_prob("default_p_inter", v)
    #
    # @property
    # def p_inter(self) -> float:
    #     # 兼容旧代码
    #     return self.default_p_inter

    @property
    def default_p_inter(self) -> float:
        v = self.get("link_probability.default_p_inter", None)
        if v is None:
            v = self.get("link_probability.p_inter", 0.99)
        return _check_prob("default_p_inter", v)

    @property
    def p_inter(self) -> float:
        return self.default_p_inter



    @property
    def option_p_inter(self) -> dict[int, float]:
        raw = self.get("link_probability.option_p_inter", {}) or {}
        out: dict[int, float] = {}
        for k, v in raw.items():
            out[int(k)] = _check_prob(f"option_p_inter[{k}]", v)
        return out

    @property
    def rel_col(self) -> str:
        # Stable column name derived from the policy instead of hard-coded p values.
        return str(self.get("link_probability.rel_col", f"rel_{self.route_name}"))

    @property
    def encoding(self) -> str:
        return str(self.get("path_export.encoding", "utf-8-sig"))

    @property
    def chunk_rows(self) -> int:
        return max(1, int(self.get("path_export.chunk_rows", 50000)))

    @property
    def include_path(self) -> bool:
        return bool(self.get("path_export.include_path", True))

    @property
    def include_path_indexed(self) -> bool:
        return bool(self.get("path_export.include_path_indexed", False))

    @property
    def include_hops(self) -> bool:
        return bool(self.get("path_export.include_hops", True))

    @property
    def motif_workers(self) -> int:
        return max(1, int(self.get("parallel.motif_workers", 1)))

    @property
    def pair_workers(self) -> int:
        return max(1, int(self.get("parallel.pair_workers", 1)))

    @property
    def topology_design_dirname(self) -> str:
        return str(self.get("data_layout.topology_design_dir", "topology_design"))

    @property
    def route_output_dirname(self) -> str:
        return str(self.get("data_layout.route_output_dirname", "route_policy"))

    @property
    def basic_file_dirname(self) -> str:
        return str(self.get("data_layout.basic_file_dir", "basic_file"))

    @property
    def visibility_xml_setting(self) -> str:
        return str(
            self.get(
                "data_layout.visibility_xml",
                "satellitesposition/station_visible_satellites_20250106.xml",
            )
        )

    def output_subdir(self, key: str, default: str) -> str:
        return str(self.get(f"outputs.{key}", default))


def load_route_policy(path: str | Path) -> RoutePolicy:
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return RoutePolicy(raw, path=p)


def list_motifs_from_topology_dir(data_root: str | Path, policy: RoutePolicy) -> list[str]:
    root = Path(data_root).expanduser() / policy.topology_design_dirname
    if not root.exists():
        return []
    motifs: list[str] = []
    for child in sorted(root.iterdir(), key=lambda x: x.name):
        if child.is_dir() and (child / "config" / "motif.json").exists():
            motifs.append(child.name)
    return motifs


def normalize_motif_names(
    motifs: Iterable[str] | None,
    *,
    data_root: str | Path,
    policy: RoutePolicy,
    fallback: Iterable[str] | None = None,
) -> list[str]:
    values = [str(x) for x in (motifs or []) if str(x).strip()]
    if not values:
        values = [str(x) for x in (fallback or []) if str(x).strip()]
    if len(values) == 1 and values[0].lower() == "all":
        values = list_motifs_from_topology_dir(data_root, policy)
    if not values:
        raise ValueError("No motif selected. Pass --motifs or set MOTIFS_DEFAULT in the script.")
    return values


def motif_config_path(data_root: str | Path, policy: RoutePolicy, motif_name: str) -> Path:
    return (
        Path(data_root).expanduser()
        / policy.topology_design_dirname
        / motif_name
        / "config"
        / "motif.json"
    )


# 替换 route_output_root(...)
def route_output_root(data_root: str | Path, policy: RoutePolicy, motif_name: str) -> Path:
    motif_root = (
        Path(data_root).expanduser()
        / policy.topology_design_dirname
        / motif_name
    )
    if policy.output_layout == "by_motif":
        return motif_root
    return motif_root / policy.route_output_dirname / policy.route_name



def path_output_dir(data_root: str | Path, policy: RoutePolicy, motif_name: str) -> Path:
    base = (
        route_output_root(data_root, policy, motif_name)
        / policy.output_subdir("path_subdir", "path")
    )

    prefix = f"region_pairs_{policy.start}_{policy.end}"
    append_tag = bool(policy.get("outputs.path_append_route_tag", True))
    if not append_tag:
        return base / prefix

    tag_source = str(policy.get("outputs.route_tag_source", "policy_file")).strip().lower()
    if tag_source == "policy_file" and policy.path is not None:
        raw_tag = policy.path.stem   # route1.json -> route1
    elif tag_source == "route_name":
        raw_tag = policy.route_name
    else:
        raw_tag = policy.route_name

    tag = _safe_fs_tag(raw_tag)
    return base / f"{prefix}_{tag}"



def probability_pair_dir(data_root: str | Path, policy: RoutePolicy, motif_name: str) -> Path:
    return (
        route_output_root(data_root, policy, motif_name)
        / policy.output_subdir("probability_subdir", "probability")
        / "pair_timeseries"
    )


def global_stat_dir(data_root: str | Path, policy: RoutePolicy, motif_name: str) -> Path:
    return route_output_root(data_root, policy, motif_name) / policy.output_subdir(
        "global_stat_subdir", "global_stat"
    )


def region_communication_dir(data_root: str | Path, policy: RoutePolicy, motif_name: str) -> Path:
    return route_output_root(data_root, policy, motif_name) / policy.output_subdir(
        "region_communication_subdir", "region_communication"
    )


def resolve_visibility_xml(data_root: str | Path, policy: RoutePolicy) -> Path:
    """
    Resolve XML path with backward-compatible fallbacks:
    1. absolute value in route JSON
    2. data_root / visibility_xml
    3. data_root / basic_file / visibility_xml
    """
    setting = Path(policy.visibility_xml_setting).expanduser()
    if setting.is_absolute():
        return setting

    root = Path(data_root).expanduser()
    candidates = [root / setting, root / policy.basic_file_dirname / setting]
    for p in candidates:
        if p.exists():
            return p
    # Return the primary candidate even when it does not exist, so the caller's
    # FileNotFoundError shows the exact path that was attempted.
    return candidates[0]
