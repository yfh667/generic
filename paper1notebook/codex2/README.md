# codex2: G60 group edge-betweenness viewer

This folder is an incremental experiment copy for the China-Europe group-pair edge-betweenness view.

## What Is Here

- `topology_edges.py`: builds the full-option inter-plane edges plus intra-plane y-ring links.
- `group_edge_betweenness.py`: computes shortest-path edge betweenness between two region groups for each time step.
- `run_g60_group_betweenness_viewer_10s.py`: runnable G60 0..10s demo.
- `build_g60_group_betweenness_86164.py`: full-length `0..86164s` data builder.
- `run_g60_group_betweenness_viewer_full.py`: opens the already built full `0..86164s` GUI.
- `motif_enumerator.py`: primitive motif enumerator for arbitrary `kn x kp` cells.
- `draw_2x3_motifs.py`: enumerates and draws the primitive `2 x 3` motif set.
- `box_motif_enumerator_v2.py`: self-contained box motif enumerator with `out<=1`, `in<=1`, idle terminals, maximality, and arbitrary `m,n` input.
- `draw_box_motif_v2.py`: draws paged PNG sheets for the v2 box motif classes.
- `motif_recursive.py`: recursive-pruning exact-box motif enumerator copied from the user-provided program.
- `draw_motif_recursive.py`: draws paged PNG sheets for `motif_recursive.py` maximal/primitive exact-box motifs.
- `tile_motif_to_grid.py`: tiles a user-provided motif edge list onto a full 2D grid with overlap attempts and conflict checks.
- `run_tiled_motif_g60_viewer_1s.py`: tiles one motif onto the full G60 `P=18,N=36` grid and opens it with the 2D topology viewer at a single `1s` step.
- `build_g60_motif_gridplus_shortest_path_timeseries.py`: builds the support-motif and grid+ static topologies, precomputes all-pairs hop tables once, then exports China-Europe shortest-path time series.
- `RUN_RESULTS.md`: recorded result of the completed `0..86164s` run.

The GUI deliberately reuses `src.satellite_topology_viewer.module.base_viewer.SatelliteTopology2DViewer`.
For the viewer, edge delay and edge betweenness are both just a `time x edge` value matrix.

Edge betweenness uses a different display semantic from delay:

- `0` means the edge is unused by the current group-pair shortest paths, so it is hidden.
- Positive values are shown with one red color.
- Larger values are darker/more opaque and thicker.
- Delay still uses the default gradient mode, because every physical edge has a delay value.

## Topology

For G60, `P=18`, `N=36`.

- Inter-plane links: existing full-option links `0, 1, 2, 4`.
- Intra-plane links: `(x, y)` to `(x, (y + 1) mod N)`.
- Therefore `(x, 35)` and `(x, 0)` are linked.
- The reverse seam remains: plane `0` has no left inter-plane neighbor, and plane `17` has no right inter-plane neighbor.

Expected G60 edge count:

- Full-option inter-plane edges: `2412`
- Intra y-ring edges: `18 * 36 = 648`
- Total: `3060`

## Run

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\run_g60_group_betweenness_viewer_10s.py'
```

Check without opening GUI:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\run_g60_group_betweenness_viewer_10s.py' --check-only
```

Default 10s output:

`E:\paper11\generic\paper1notebook\codex2\outputs\g60_group_betweenness_t0_10`

Output files:

- `edge_betweenness.npy`
- `time_indices.npy`
- `edges.csv`
- `step_summary.csv`
- `path_samples.csv`
- `meta.json`

## Build Full 0..86164s Data

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_group_betweenness_86164.py'
```

Default full output:

`E:\paper11\generic\paper1notebook\codex2\outputs\g60_group_betweenness_t0_86164`

Important full-output files:

- `edge_betweenness.npy`: full `time x edge` matrix.
- `unique_state_values.npy`: cached value matrix for unique China-Europe group states.
- `state_ids.npy`: maps each time row to a unique state row.
- `state_definitions.json`: source/target node sets for every unique state.
- `edges.csv`: edge table with full-option links plus intra y-ring links.
- `step_summary.csv`: one row per time step.
- `meta.json`: shape, value range, and counting rule.

## Open Full 0..86164s GUI

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\run_g60_group_betweenness_viewer_full.py'
```

This loads the existing `edge_betweenness.npy` matrix and does not recompute shortest paths.

## Enumerate And Draw p=2,y=3 Motifs

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\draw_2x3_motifs.py' --p 2 --y 3
```

Default output:

`E:\paper11\generic\paper1notebook\codex2\outputs\motif_p2_y3`

Definition used by the enumerator:

- `p=2,y=3` means `2` orbital-plane columns and `3` y-phase rows.
- Each cell node chooses one outgoing inter direction from `A=(1,0)`, `B=(1,-1)`, `C=(1,+1)`, `D=(2,0)`.
- Periodically tiling the cell must satisfy one inter outgoing edge and one inter incoming edge per node.
- Cells equivalent by cyclic translation are counted once.
- Motifs generated by tiling a smaller rectangular cell are removed.

## Enumerate Box Motifs V2 With Arbitrary m,n

This version follows the looser terminal constraint:

- every planned node has `out<=1`
- every tiled node has `in<=1`
- idle terminals are allowed
- only maximal schemes are kept as candidates
- translation-equivalent maximal schemes are counted once

Here `m` is the box width in the plane/x direction and `n` is the box height in the y-phase direction. The planned columns are `0..m-2`; the rightmost column is only a target boundary inside the box.

Default `m=3,n=2`:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\box_motif_enumerator_v2.py' --m 3 --n 2 --out-dir 'E:\paper11\generic\paper1notebook\codex2\outputs\box_motif_v2'
```

Try another size and print only the first few classes:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\box_motif_enumerator_v2.py' --m 4 --n 2 --limit 5
```

Draw all translation-deduped maximal classes for `m=3,n=4`:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\draw_box_motif_v2.py' --m 3 --n 4 --out-dir 'E:\paper11\generic\paper1notebook\codex2\outputs\box_motif_v2_m3_n4_draw_png' --no-pdf
```

## Tile A Motif Onto A Full 2D Grid

Node IDs use the same column-major convention as the sketches:

```text
id = col * full_rows + row + 1
```

For example, `full_cols=3, full_rows=4` is displayed as:

```text
4  8  12
3  7  11
2  6  10
1  5   9
```

The tiler input is:

- full grid size: `--full-cols P --full-rows N`
- motif box size: `--motif-width W --motif-height H`
- motif edge list: `--motif-edges '3-7,4-8,8-11,7-12'`

The motif edge list may be written using any representative box in the full-grid node IDs. The script infers the motif origin, normalizes the edges to local motif coordinates, then greedily tiles them over the full grid.

Placement rule:

- horizontal origins step by `motif_width-1`
- vertical origins are tried every row by default, so overlapping placements are attempted
- a placement is accepted only if it does not introduce an outgoing or incoming conflict
- when a placement reaches the right boundary, edges whose source/target is outside the grid are clipped

User example:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\tile_motif_to_grid.py' --full-cols 3 --full-rows 4 --motif-width 3 --motif-height 2 --motif-edges '3-7,4-8,8-11,7-12' --out-dir 'E:\paper11\generic\paper1notebook\codex2\outputs\tile_motif_user_example'
```

Open a full G60 `P=18,N=36` viewer for one tiled motif at `1s`:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\run_tiled_motif_g60_viewer_1s.py'
```

The default motif edge list is `1-37,2-38,38-73,37-74`, which is the `N=36` equivalent of the small-grid sketch pattern.

## China-Europe Shortest Path Time Series For Support Motif And Grid+

This script uses the support-style motif YAML:

`E:\paper11\generic\src\motif_generator\examples\configs\dad_cxx_support.yaml`

Default run, inclusive `0..86164`:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_motif_gridplus_shortest_path_timeseries.py'
```

Small check run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_motif_gridplus_shortest_path_timeseries.py' --start 0 --end 10 --out-dir 'E:\paper11\generic\paper1notebook\codex2\outputs\g60_motif_gridplus_shortest_path_t0_10'
```

Default full output:

`E:\paper11\generic\paper1notebook\codex2\outputs\g60_motif_gridplus_shortest_path_t0_86164`

Important output files:

- `compare_step_summary.csv`: one row per time step, with support motif mean hops, grid+ mean hops, and their difference.
- `mean_shortest_path_timeseries.png`: plot of the two time series.
- `support_motif_DAD_Cxx/step_summary.csv`: per-step China-Europe shortest-path summary for the support motif topology.
- `support_motif_DAD_Cxx/path_samples.csv`: concrete shortest-path samples for manual checking.
- `support_motif_DAD_Cxx/hop_dist.npy` and `next_hop.npy`: all-pairs hop lookup tables.
- `gridplus/step_summary.csv`, `gridplus/path_samples.csv`, `gridplus/hop_dist.npy`, `gridplus/next_hop.npy`: same outputs for grid+.

For the completed `0..86164` run:

- support motif edges: `1098`
- grid+ edges: `1260`
- unique China-Europe group states: `12224`
- support motif mean-hop range: `8.0924..18.5907`
- grid+ mean-hop range: `11.0408..21.6261`

## Build Full-Option Plus Intra Delay Store

The original full-option delay store contains inter-plane options `0,1,2,4`.
This script appends the intra-plane y-ring links:

```text
(p,y) -- (p,(y+1) mod N)
```

The appended intra links use `option=-1`, and the output keeps the same query format:

- `edge_delay_ms.npy`
- `edges.csv`
- `edge_index_matrix.npy`
- `time_indices.npy`
- `times_s.npy`
- `delay_meta.json`
- `delay_store_query_meta.json`

Small check run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_full_option_plus_intra_delay_store.py' --start 0 --end 10 --out-dir 'E:\paper11\data\linshi\G60_full_options_plus_intra_t0_10_stride1' --force
```

Full `0..86164` cache:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_full_option_plus_intra_delay_store.py'
```

Default full output:

`E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1`

Query an intra edge after building:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\link_delay\module\query.py' 'E:\paper11\data\linshi\G60_full_options_plus_intra_t0_10_stride1' --time-step 0 --src 0 --dst 1
```

## China-Europe Shortest Delay Time Series

This computes true weighted shortest propagation delay, not shortest hop count.
It should read the plus-intra delay store so every topology edge is queried from one cache.

Small check run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_motif_gridplus_shortest_delay_timeseries.py' --start 0 --end 10 --delay-store-dir 'E:\paper11\data\linshi\G60_full_options_plus_intra_t0_10_stride1' --out-dir 'E:\paper11\data\linshi\g60_motif_gridplus_shortest_delay_t0_10'
```

Full run after building the default plus-intra store:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_motif_gridplus_shortest_delay_timeseries.py' --start 0 --end 86164
```

Important output files:

- `compare_step_summary.csv`: support motif vs grid+ mean shortest delay per time step.
- `mean_shortest_delay_timeseries.png`: plot of the two delay time series.
- `support_motif_DAD_Cxx/step_summary.csv`: per-step weighted shortest-delay summary.
- `support_motif_DAD_Cxx/path_samples.csv`: concrete weighted shortest-delay path samples.
- `gridplus/step_summary.csv` and `gridplus/path_samples.csv`: same outputs for grid+.

## Parallel China-Europe Shortest Delay Time Series

Use this faster test script when the full plus-intra delay cache already exists:

`E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1`

The parallel script intentionally has no position-cache fallback. Every edge weight,
including intra `option=-1`, must be found in `edge_index_matrix.npy`.

Small validation run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_motif_gridplus_shortest_delay_timeseries_parallel.py' --start 0 --end 99 --max-workers 4 --chunk-size 10 --out-dir 'E:\paper11\data\linshi\g60_motif_gridplus_shortest_delay_parallel_t0_99'
```

Full command template:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_motif_gridplus_shortest_delay_timeseries_parallel.py' --start 0 --end 86164 --out-dir 'E:\paper11\data\linshi\g60_motif_gridplus_shortest_delay_parallel_t0_86164'
```

Important output files are compatible with the slow script:

- `compare_step_summary.csv`
- `mean_shortest_delay_timeseries.png`
- `support_motif_DAD_Cxx/step_summary.csv`
- `support_motif_DAD_Cxx/path_samples.csv`
- `support_motif_DAD_Cxx/mean_shortest_delay_ms.npy`
- `gridplus/step_summary.csv`
- `gridplus/path_samples.csv`
- `gridplus/mean_shortest_delay_ms.npy`
