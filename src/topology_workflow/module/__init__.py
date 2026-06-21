"""Reusable helpers for YAML-driven topology workflows."""

from .batch_shortest_hops import (
    PairStatePayload,
    RegionPairSpec,
    TopologySpec,
    all_pairs_hop_dist,
    build_pair_state_payloads,
    compute_shortest_hops_batch,
    full_link_topology_spec,
    topology_specs_from_motif_csv,
)
from .batch_shortest_delay import (
    compute_shortest_delay_batch,
    plot_shortest_delay_series,
    read_shortest_delay_series,
    write_shortest_delay_comparison,
)
from .config import (
    group_name,
    load_workflow_yaml,
    optional_path,
    time_axis_from_config,
    viewer_config_from_workflow,
)
from .edge_tables import (
    INTRA_OPTION,
    add_intra_ring_records,
    build_edge_table_from_topology_config,
    build_full_option_plus_intra_edge_table,
    build_motif_text_edge_table,
    build_single_motif_edge_table,
    make_edge_table_from_records,
    motif_text_to_matrix,
)
from .hybrid_edges import (
    EdgeRecord,
    HybridEdgeTableResult,
    InterDegreeStats,
    added_edge_count,
    build_y_band_hybrid_edge_table,
    build_y_band_hybrid_edge_tables,
    cyclic_rows,
    edge_key,
    edge_keys_from_table,
    edge_records_from_table as hybrid_edge_records_from_table,
    edge_table_mask,
    inter_degree_stats,
    node_id,
)
from .inter_edge_ports import (
    InterPortCheckResult,
    InterPortViolation,
    InvalidInterEdge,
    NormalizedInterEdge,
    check_inter_edge_port_constraints,
    dataclass_rows,
    normalize_inter_edge,
)
from .oracle_one_right import (
    ChosenRightEdge,
    build_edge_pair_index,
    choose_one_right_one_left_matching,
    edge_table_from_chosen_right_edges,
    min_cost_matching_payloads,
    normalized_pair,
    verify_one_right_one_left,
    write_chosen_right_edges_csv,
)
from .region_constraints import (
    LEFT_SIDE,
    RIGHT_SIDE,
    RegionInternalOptionConstraintStats,
    apply_region_internal_option_constraint,
    build_region_internal_option_edges,
    edge_records_from_table,
    edge_side_for_node,
    make_edge_table_subset,
    normalize_group_nodes,
    selected_group_bits,
)
from .runner import run_workflow_from_yaml
from .dynamic_schedule import (
    build_segment_cost_tables,
    compress_segments,
    infer_sample_seconds,
    optimize_min_dwell,
    per_step_oracle,
    read_compare_csv,
    write_dynamic_schedule_outputs,
    write_schedule_by_step,
)
from .shortest_delay import compute_shortest_delay_timeseries
from .shortest_hops import compute_shortest_hops_timeseries

_LINK_SETUP_EXPORTS = {
    "DynamicRegionConstraintSeries",
    "LinkSetupTimeStepStats",
    "build_dynamic_region_internal_option_constraint_series",
    "expand_group_data_for_link_setup_time",
    "write_dynamic_region_constraint_series",
}


def __getattr__(name: str):
    if name in _LINK_SETUP_EXPORTS:
        from src import link_setup_time

        return getattr(link_setup_time, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "INTRA_OPTION",
    "LEFT_SIDE",
    "PairStatePayload",
    "RIGHT_SIDE",
    "ChosenRightEdge",
    "DynamicRegionConstraintSeries",
    "EdgeRecord",
    "HybridEdgeTableResult",
    "InterDegreeStats",
    "InterPortCheckResult",
    "InterPortViolation",
    "InvalidInterEdge",
    "LinkSetupTimeStepStats",
    "NormalizedInterEdge",
    "RegionInternalOptionConstraintStats",
    "RegionPairSpec",
    "TopologySpec",
    "add_intra_ring_records",
    "added_edge_count",
    "all_pairs_hop_dist",
    "apply_region_internal_option_constraint",
    "build_edge_table_from_topology_config",
    "build_edge_pair_index",
    "build_full_option_plus_intra_edge_table",
    "build_dynamic_region_internal_option_constraint_series",
    "build_motif_text_edge_table",
    "build_pair_state_payloads",
    "build_region_internal_option_edges",
    "build_segment_cost_tables",
    "build_single_motif_edge_table",
    "build_y_band_hybrid_edge_table",
    "build_y_band_hybrid_edge_tables",
    "check_inter_edge_port_constraints",
    "choose_one_right_one_left_matching",
    "compress_segments",
    "compute_shortest_delay_batch",
    "compute_shortest_delay_timeseries",
    "compute_shortest_hops_batch",
    "compute_shortest_hops_timeseries",
    "cyclic_rows",
    "dataclass_rows",
    "edge_key",
    "edge_keys_from_table",
    "edge_records_from_table",
    "hybrid_edge_records_from_table",
    "edge_table_mask",
    "edge_side_for_node",
    "edge_table_from_chosen_right_edges",
    "expand_group_data_for_link_setup_time",
    "full_link_topology_spec",
    "group_name",
    "infer_sample_seconds",
    "inter_degree_stats",
    "load_workflow_yaml",
    "make_edge_table_from_records",
    "make_edge_table_subset",
    "min_cost_matching_payloads",
    "motif_text_to_matrix",
    "normalize_group_nodes",
    "normalize_inter_edge",
    "normalized_pair",
    "node_id",
    "optimize_min_dwell",
    "optional_path",
    "per_step_oracle",
    "plot_shortest_delay_series",
    "read_compare_csv",
    "read_shortest_delay_series",
    "selected_group_bits",
    "topology_specs_from_motif_csv",
    "run_workflow_from_yaml",
    "time_axis_from_config",
    "viewer_config_from_workflow",
    "verify_one_right_one_left",
    "write_dynamic_schedule_outputs",
    "write_dynamic_region_constraint_series",
    "write_chosen_right_edges_csv",
    "write_schedule_by_step",
    "write_shortest_delay_comparison",
]
