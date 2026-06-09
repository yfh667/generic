from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG, ViewerConfig
from src.model.static_hop_table import precompute_hop_and_next_hop

try:
    from src.viz.pyqt_main2 import SatelliteViewer, getoption
    _VIEWER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _VIEWER_IMPORT_ERROR = exc

    class SatelliteViewer:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "PyQt5 is required to open EdgeBetweennessTimeViewer. "
                "Use build_betweenness_edges_by_step(...) in non-GUI environments."
            ) from _VIEWER_IMPORT_ERROR

    def getoption(x1, y1, x2, y2, n):
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        if dx == 1 and y2 == y1:
            return 0
        if dx == 1 and y2 == (y1 - 1 + n) % n:
            return 1
        if dx == 2 and y2 == y1:
            return 2
        if dx == 1 and y2 == (y1 + 1) % n:
            return 4
        return None

try:
    from paper1notebook.group_edge_betweenness import (
        build_full_option_graph,
        count_step_edge_betweenness,
    )
except ModuleNotFoundError:
    from group_edge_betweenness import (
        build_full_option_graph,
        count_step_edge_betweenness,
    )


def _counts_to_weighted_edges(
    counts: np.ndarray,
    idx_to_edge: list[tuple[int, int]],
    *,
    min_count: int,
    max_draw_edges: int,
) -> dict[int, dict[int, int]]:
    candidate_idx = np.flatnonzero(counts >= int(min_count))
    if int(max_draw_edges) > 0 and candidate_idx.size > int(max_draw_edges):
        order = np.argsort(counts[candidate_idx])[-int(max_draw_edges):]
        selected_idx = candidate_idx[order]
    else:
        selected_idx = candidate_idx

    selected_idx = sorted((int(i) for i in selected_idx), key=lambda i: int(counts[i]), reverse=True)
    weighted: dict[int, dict[int, int]] = {}
    for idx in selected_idx:
        u, v = idx_to_edge[idx]
        weighted.setdefault(int(u), {})[int(v)] = int(counts[idx])
    return weighted


def build_betweenness_edges_by_step(
    group_data,
    config: ViewerConfig = G60_CONFIG,
    *,
    group_a: int = 2,
    group_b: int = 3,
    steps=None,
    max_draw_edges: int = 500,
    min_count: int = 1,
    include_intra: bool = True,
    skip_first_last_source: bool = False,
):
    """
    Precompute weighted all_adj for the existing SatelliteViewer time axis.

    Returns:
      all_adj[step][src][dst] = count
      summaries[step] = per-step statistics
    """
    full_option_graph = build_full_option_graph(
        config,
        include_intra=include_intra,
        skip_first_last_source=skip_first_last_source,
    )
    _, next_hop = precompute_hop_and_next_hop(full_option_graph.graph, config.total_sats)

    if steps is None:
        steps = sorted(int(s) for s in group_data.keys())
    else:
        steps = [int(s) for s in steps if int(s) in group_data]

    all_adj: dict[int, dict[int, dict[int, int]]] = {}
    summaries: dict[int, dict] = {}

    for step in steps:
        counts, summary, _ = count_step_edge_betweenness(
            step=step,
            group_data=group_data,
            group_a=group_a,
            group_b=group_b,
            next_hop=next_hop,
            edge_to_idx=full_option_graph.edge_to_idx,
            total_sats=config.total_sats,
            inspect=False,
        )
        weighted = _counts_to_weighted_edges(
            counts,
            full_option_graph.idx_to_edge,
            min_count=min_count,
            max_draw_edges=max_draw_edges,
        )
        summary = dict(summary)
        summary["drawn_edge_count"] = sum(len(dsts) for dsts in weighted.values())
        summary["drawn_max_count"] = max(
            (count for dst_counts in weighted.values() for count in dst_counts.values()),
            default=0,
        )
        all_adj[int(step)] = weighted
        summaries[int(step)] = summary

    return all_adj, summaries


class EdgeBetweennessTimeViewer(SatelliteViewer):
    """
    Time-axis viewer for group-to-group edge path counts.

    This class does not modify src.viz.pyqt_main2.SatelliteViewer.  It reuses
    the existing time slider and satellite coloring, then overrides edge
    drawing so each visible step can use weighted edges_by_step:
        edges_by_step[step][src][dst] = path_count
    """

    def __init__(
        self,
        group_data,
        config: ViewerConfig = G60_CONFIG,
        *,
        group_a: int = 2,
        group_b: int = 3,
        max_draw_edges: int = 500,
        min_count: int = 1,
        width_min: float = 0.4,
        width_max: float = 6.0,
        color: str = "#d62728",
        include_intra: bool = True,
        skip_first_last_source: bool = False,
    ):
        self.group_a = int(group_a)
        self.group_b = int(group_b)
        self.max_draw_edges = int(max_draw_edges)
        self.min_count = int(min_count)
        self.width_min = float(width_min)
        self.width_max = float(width_max)
        self.betweenness_color = color

        self.full_option_graph = build_full_option_graph(
            config,
            include_intra=include_intra,
            skip_first_last_source=skip_first_last_source,
        )
        self._dist, self._next_hop = precompute_hop_and_next_hop(
            self.full_option_graph.graph,
            config.total_sats,
        )
        self._weighted_edges_cache: dict[int, dict[int, dict[int, int]]] = {}
        self._summary_cache: dict[int, dict] = {}
        self.betweenness_summaries: dict[int, dict] = {}

        super().__init__(group_data, config)

    def showEvent(self, event):
        super().showEvent(event)
        if self.steps:
            self.plot_satellites(int(self.slider.value()))

    @staticmethod
    def _coerce_weighted_edges(edges) -> dict[int, dict[int, int]]:
        weighted: dict[int, dict[int, int]] = {}
        for src, dsts in (edges or {}).items():
            src_i = int(src)
            if isinstance(dsts, dict):
                for dst, count in dsts.items():
                    weighted.setdefault(src_i, {})[int(dst)] = int(count)
            else:
                for dst in (dsts or []):
                    weighted.setdefault(src_i, {})[int(dst)] = 1
        return weighted

    def weighted_edges_for_step(self, step: int) -> tuple[dict[int, dict[int, int]], dict]:
        step = int(step)
        if step in self._weighted_edges_cache:
            return self._weighted_edges_cache[step], self._summary_cache[step]

        counts, summary, _ = count_step_edge_betweenness(
            step=step,
            group_data=self.group_data,
            group_a=self.group_a,
            group_b=self.group_b,
            next_hop=self._next_hop,
            edge_to_idx=self.full_option_graph.edge_to_idx,
            total_sats=self.total_sats,
            inspect=False,
        )

        weighted = _counts_to_weighted_edges(
            counts,
            self.full_option_graph.idx_to_edge,
            min_count=self.min_count,
            max_draw_edges=self.max_draw_edges,
        )

        summary = dict(summary)
        summary["drawn_edge_count"] = int(sum(len(dsts) for dsts in weighted.values()))
        summary["drawn_max_count"] = int(max(
            (count for dst_counts in weighted.values() for count in dst_counts.values()),
            default=0,
        ))

        self._weighted_edges_cache[step] = weighted
        self._summary_cache[step] = summary
        return weighted, summary

    def draw_edges(self, step, edges):
        if hasattr(self, "_edges_line_items"):
            for item in self._edges_line_items:
                self.plot_widget.removeItem(item)
        self._edges_line_items = []

        weighted_edges = self._coerce_weighted_edges(edges)
        if weighted_edges:
            summary = self.betweenness_summaries.get(int(step), {})
            self._weighted_edges_cache[int(step)] = weighted_edges
            if summary:
                self._summary_cache[int(step)] = summary
        else:
            weighted_edges, summary = self.weighted_edges_for_step(int(step))

        max_count = max(
            (count for dst_counts in weighted_edges.values() for count in dst_counts.values()),
            default=1,
        )

        for src, dst_counts in weighted_edges.items():
            for dst, count in dst_counts.items():
                x0 = self._all_cols[src]
                y0 = self._all_rows[src]
                x1 = self._all_cols[dst]
                y1 = self._all_rows[dst]

                scale = (float(count) / float(max_count)) ** 0.65 if max_count else 0.0
                width = self.width_min + (self.width_max - self.width_min) * scale

                opt = getoption(x0, y0, x1, y1, self.N)
                if opt == 2:
                    item = self.draw_curved_edge(
                        x0,
                        y0,
                        x1,
                        y1,
                        curve=0.5,
                        dash=False,
                        color=self.betweenness_color,
                        width=width,
                    )
                else:
                    item = self.draw_straight_edge(
                        x0,
                        y0,
                        x1,
                        y1,
                        dash=False,
                        color=self.betweenness_color,
                        width=width,
                    )
                self._edges_line_items.append(item)

    def plot_satellites(self, step):
        super().plot_satellites(step)
        summary = self.betweenness_summaries.get(int(step)) or self._summary_cache.get(int(step))
        if not summary:
            return
        mean_hops = summary.get("mean_hops", float("nan"))
        try:
            mean_hops_text = f"{float(mean_hops):.2f}"
        except Exception:
            mean_hops_text = "nan"

        self.label.setText(
            "Edge betweenness "
            f"group {self.group_a}->{self.group_b} | "
            f"step {int(step)} | "
            f"nodes {summary.get('group_a_node_count', '?')}x{summary.get('group_b_node_count', '?')} | "
            f"pairs {summary.get('reachable_pair_count', '?')}/{summary.get('pair_count', '?')} | "
            f"mean hops {mean_hops_text} | "
            f"drawn edges {summary.get('drawn_edge_count', '?')} | "
            f"max count {summary.get('max_edge_count', summary.get('drawn_max_count', '?'))}"
        )

    def clear_betweenness_cache(self) -> None:
        self._weighted_edges_cache.clear()
        self._summary_cache.clear()


def create_edge_betweenness_time_viewer(
    group_data,
    config: ViewerConfig = G60_CONFIG,
    *,
    group_a: int = 2,
    group_b: int = 3,
    max_draw_edges: int = 500,
    min_count: int = 1,
    include_intra: bool = True,
):
    return EdgeBetweennessTimeViewer(
        group_data,
        config,
        group_a=group_a,
        group_b=group_b,
        max_draw_edges=max_draw_edges,
        min_count=min_count,
        include_intra=include_intra,
    )
