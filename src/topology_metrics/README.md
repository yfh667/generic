# topology_metrics

`topology_metrics` is the proposed stable home for graph-level metrics that are
independent of a specific paper, constellation, or GUI.

The current migration target is the logic prototyped in
`generic/paper1notebook/codex2`:

- build a per-step group state index from externally supplied `group_data`
- compute edge betweenness once per unique group state
- expand unique-state metric values back to the requested time axis
- write/read metric stores that viewers or later analysis scripts can consume

This module must stay parameter transparent. Code in `module/` should not hard
code G60, China/Europe, XML paths, cache directories, motif IDs, or viewer
choices. Those belong in `examples/` or caller scripts.

## Suggested Interfaces

### Group State Index

Input:

- `steps: Sequence[int]`
- `group_data: Mapping[int, ...]`
- `source_group_id: int`
- `target_group_id: int`

Output:

- `unique_states`: ordered list of `(source_nodes, target_nodes)`
- `state_ids`: `int32` array mapping each step row to a unique state

This lets expensive metrics be computed once per unique visible-station state
instead of once per second.

### Unique-State Edge Betweenness

Input:

- `edge_table: src.link_delay.module.edge_options.EdgeTable`
- `total_nodes: int`
- `source_nodes: Iterable[int]`
- `target_nodes: Iterable[int]`

Output:

- edge value vector shaped `(edge_table.num_edges,)`
- summary dictionary
- optional representative path samples

The algorithm should accept an already-built adjacency list when the caller is
looping over many states.

### Metric Store

Minimum stable files:

- `edges.csv`
- `time_indices.npy`
- `state_ids.npy`
- `state_definitions.json`
- `unique_state_values.npy`
- `state_summary.csv`
- `metric_values.npy`
- `step_summary.csv`
- `meta.json`

The store format is intentionally generic: `metric_values.npy` can represent
edge betweenness, delay, load, utilization, or any future edge metric. The
meaning should be described by `meta.json`, especially:

- `metric_name`
- `metric_unit`
- `counting_rule`
- `value_min`
- `value_max`

## Migration Plan

1. Keep the existing `codex2` scripts as prototypes until their results are
   accepted.
2. Move generic pieces into `src/topology_metrics/module`:
   - state indexing
   - shortest-path edge betweenness
   - store writer/reader helpers
3. Keep constellation-specific runnable scripts in
   `src/topology_metrics/examples`.
4. Keep Jupyter usage notes in `examples/*.ipynb` so later users can run cells
   independently.

