# codex2 Migration Map

This file records where the mature logic from
`generic/paper1notebook/codex2` has been moved.  The old `codex2` files are
left untouched for manual deletion/review.

## Main Reusable Modules

| Logic | New location |
|---|---|
| Motif exact-box enumeration, canonical primitive library, combined 102/706/808 library | `generic/src/motif_generator/module/exact_box.py`, `generic/src/motif_generator/module/library.py` |
| Motif support/YAML representation and tiling onto a full P x N grid | `generic/src/motif_generator/module/support.py`, `generic/src/motif_generator/module/tiling.py`, `generic/src/motif_generator/module/viewer_adapter.py` |
| Edge-table construction for full-link, motif, and intra-ring topologies | `generic/src/topology_workflow/module/edge_tables.py` |
| Region-internal option/grid constraint | `generic/src/topology_workflow/module/region_constraints.py` |
| Shortest-hop batch metrics | `generic/src/topology_workflow/module/batch_shortest_hops.py` |
| Shortest-delay single topology and batch comparison | `generic/src/topology_workflow/module/shortest_delay.py`, `generic/src/topology_workflow/module/batch_shortest_delay.py` |
| Unweighted edge betweenness compact stores | `generic/src/topology_metrics/module/edge_betweenness.py`, `generic/src/topology_metrics/module/edge_betweenness_store.py` |
| Weighted shortest-path edge betweenness | `generic/src/topology_metrics/module/weighted_edge_betweenness.py` |
| Dynamic topology schedule and minimum dwell post-processing | `generic/src/topology_workflow/module/dynamic_schedule.py` |
| One-right/one-left oracle matching | `generic/src/topology_workflow/module/oracle_one_right.py` |
| Full-option delay cache plus intra delay cache | `generic/src/link_delay/module/build_full_link_delay_store.py`, `generic/src/link_delay/module/plus_intra_delay_store.py` |
| 2D topology viewer and edge-value display modes | `generic/src/satellite_topology_viewer/module/base_viewer.py` and related viewer modules |

## Paper1 Pipeline Entrypoints

| Pipeline | New location |
|---|---|
| G60 motif library shortest-hop run | `generic/paper1notebook/pipeline/run_paper1_motif_shortest_hops.py` |
| G60 motif library shortest-delay run | `generic/paper1notebook/pipeline/run_paper1_motif_shortest_delay.py` |
| Paper1-specific legacy grid+ baseline adapter | `generic/paper1notebook/pipeline/paper1_edge_tables.py` |
| Shortest-hop YAML | `generic/paper1notebook/pipeline/configs/g60_w_le4_h_le3_shortest_hops.yaml` |
| Shortest-delay YAML | `generic/paper1notebook/pipeline/configs/g60_w_le4_h_le3_shortest_delay.yaml` |

## File-by-File Status

| Old codex2 file | Replacement / status |
|---|---|
| `analyze_dynamic_best_motif_delay.py` | Covered by `src/topology_workflow/module/dynamic_schedule.py` and `src/topology_workflow/examples/analyze_dynamic_schedule_from_csv.py`. |
| `analyze_dynamic_motif_schedule_delay.py` | Covered by dynamic schedule module/example. |
| `analyze_g60_w4h3_dynamic_best_motif.py` | Covered by dynamic schedule module/example; G60-specific invocation belongs in paper1 pipeline config/outputs. |
| `box_motif_enumerator_v2.py` | Covered by `src/motif_generator/module/exact_box.py` and `library.py`. |
| `build_g60_combined_102_706_shortest_hops.py` | Covered by `paper1notebook/pipeline/run_paper1_motif_shortest_hops.py`. |
| `build_g60_full_option_plus_intra_delay_store.py` | Covered by `src/link_delay/module/plus_intra_delay_store.py` and `src/link_delay/examples/build_g60_plus_intra_delay_store.py`. |
| `build_g60_group_betweenness_86164.py` | Covered by `src/topology_metrics/module/edge_betweenness_store.py`. |
| `build_g60_motif_gridplus_shortest_delay_timeseries.py` | Covered by `src/topology_workflow/module/shortest_delay.py` and `batch_shortest_delay.py`. |
| `build_g60_motif_gridplus_shortest_delay_timeseries_parallel.py` | Algorithm covered by shortest-delay modules; the old parallel worker implementation remains scratch-only. |
| `build_g60_motif_gridplus_shortest_path_timeseries.py` | Covered by shortest-hop modules and paper1 shortest-hop pipeline. |
| `build_g60_multi_region_weighted_betweenness.py` | Covered by `src/topology_metrics/module/weighted_edge_betweenness.py`. |
| `build_g60_one_right_oracle_topology_t20760.py` | Core matching covered by `src/topology_workflow/module/oracle_one_right.py`; screenshot/viewer remains an example-level concern. |
| `build_g60_selected_motifs_shortest_delay_parallel.py` | Covered by `src/topology_workflow/module/batch_shortest_delay.py` and `paper1notebook/pipeline/run_paper1_motif_shortest_delay.py`. |
| `build_small102_motif_library_up_to_w4h3.py` | Covered by `src/motif_generator/module/library.py` with `include_max_size=False`. |
| `combine_g60_shortest_delay_results.py` | Covered by `write_shortest_delay_comparison` in `src/topology_workflow/module/batch_shortest_delay.py`. |
| `draw_2x3_motifs.py` | Covered by `src/motif_generator/module/drawing.py` and examples. |
| `draw_box_motif_v2.py` | Covered by motif drawing examples. |
| `draw_g60_small102_motif_000056_viewer.py` | Covered by `src/motif_generator/examples/run_tiled_motif_2d_viewer.py` and satellite topology viewer modules. |
| `draw_motif_recursive.py` | Covered by motif drawing/enumeration modules. |
| `draw_w4h3_primitive_translation_dedup.py` | Covered by `src/motif_generator/module/library.py`; this is now a diagnostic use case. |
| `evaluate_g60_t20760_oracle_one_right_timeseries.py` | Covered by shortest-delay batch evaluation once the oracle edge table is built via `oracle_one_right.py`. |
| `group_edge_betweenness.py` | Covered by `src/topology_metrics/module/edge_betweenness.py` and store helper. |
| `motif_enumerator.py` | Covered by motif generator modules. |
| `motif_recursive.py` | Covered by motif generator modules. |
| `plot_g60_combined_102_706_region_pairs.py` | Covered by shortest-delay comparison plotting and paper1 output CSVs. |
| `postprocess_g60_primitive706_region_pair.py` | Covered by shortest-delay comparison and dynamic schedule modules. |
| `run_g60_china_europe_internal_grid_dynamic_viewer.py` | Covered by region constraint module plus satellite topology viewer modules. |
| `run_g60_china_europe_internal_grid_viewer.py` | Covered by `src/topology_workflow/examples/run_g60_motif_000056_internal_grid_constraint_viewer.py`. |
| `run_g60_dynamic_schedule_betweenness_viewer_120min.py` | Covered by dynamic schedule outputs plus viewer modules. |
| `run_g60_dynamic_schedule_topology_viewer_120min.py` | Covered by dynamic schedule outputs plus viewer modules. |
| `run_g60_group_betweenness_viewer_10s.py` | Covered by `src/satellite_topology_viewer/examples/run_g60_group_betweenness_viewer_10s.py`. |
| `run_g60_group_betweenness_viewer_full.py` | Covered by satellite topology viewer modules and metric-store inputs. |
| `run_g60_motif_000056_internal_grid_constraint_viewer.py` | Replaced by `src/topology_workflow/examples/run_g60_motif_000056_internal_grid_constraint_viewer.py`. |
| `run_g60_motif_000116_betweenness_viewer.py` | Covered by viewer modules plus edge betweenness stores. |
| `run_g60_motif_000116_dynamic_viewer.py` | Covered by motif tiling/viewer modules. |
| `run_g60_multi_region_weighted_betweenness_viewer.py` | Covered by weighted betweenness module plus satellite topology viewer modules. |
| `run_g60_region_internal_grid_betweenness_viewer.py` | Covered by region constraints, betweenness metrics, and viewer modules. |
| `run_g60_support_motif_dynamic_viewer.py` | Covered by motif support parser, tiling, and viewer adapter. |
| `run_g60_t20760_oracle_topology_viewer.py` | Core topology construction covered by oracle module; display covered by satellite topology viewer. |
| `run_g60_w5h4_motif_shortest_delay_batch.py` | Covered by generic motif library generation plus shortest-delay batch module. |
| `run_tiled_motif_g60_viewer_1s.py` | Covered by `src/motif_generator/examples/run_tiled_motif_2d_viewer.py`. |
| `show_w4h3_maximal_minus_primitive.py` | Diagnostic only; underlying counts come from motif generator library functions. |
| `tile_motif_to_grid.py` | Covered by `src/motif_generator/module/tiling.py`. |
| `topology_edges.py` | Covered by `src/satellite_topology_viewer/module/topology_edges.py` and `src/topology_workflow/module/edge_tables.py`. |

## Validation Done

Commands run after migration:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' `
  'E:\paper11\generic\paper1notebook\pipeline\run_paper1_motif_shortest_delay.py' `
  --start 0 --end 0 --stride 1 --limit-motifs 1 --pairs china_europe --skip-gridplus `
  --sample-steps 0 --sample-pairs-per-step 0 `
  --out-dir 'E:\paper11\data\satnet_experiments\_smoke\paper1_shortest_delay_t0_limit1'
```

Result: shortest-delay smoke completed for one motif plus `full_link`.

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' `
  'E:\paper11\generic\src\topology_workflow\examples\analyze_dynamic_schedule_from_csv.py' `
  --compare-csv 'E:\paper11\data\satnet_experiments\_smoke\dynamic_schedule\compare.csv' `
  --out-dir 'E:\paper11\data\satnet_experiments\_smoke\dynamic_schedule\out' `
  --min-dwell-minutes 1 2
```

Result: dynamic schedule summary, by-step CSVs, segments, and plot were generated.
