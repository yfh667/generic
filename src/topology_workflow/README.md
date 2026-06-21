# topology_workflow

YAML-driven experiment runner for topology metrics.

The workflow reads one YAML file and wires together:

- constellation shape and station groups
- region group XML/cache
- satellite position raw/cache metadata
- full-option delay store
- topology definition, currently `single_motif`
- metric definition, currently `shortest_delay` or `shortest_hops`
- output directory

## Example

YAML:

`E:\paper11\generic\src\topology_workflow\examples\configs\g60_motif_000056_china_europe_shortest_delay.yaml`

Smoke run for a small interval:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_workflow\examples\run_from_yaml.py' --config 'E:\paper11\generic\src\topology_workflow\examples\configs\g60_motif_000056_china_europe_shortest_delay.yaml' --end 10 --out-dir 'E:\paper11\data\linshi\workflows\smoke_g60_motif_000056_china_europe_shortest_delay_t0_10'
```

Full one-day run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_workflow\examples\run_from_yaml.py' --config 'E:\paper11\generic\src\topology_workflow\examples\configs\g60_motif_000056_china_europe_shortest_delay.yaml'
```

## Output

For the example above, the metric output lives under:

```text
<out_dir>/
  workflow_config.yaml
  workflow_meta.json
  motif_000056_DBD_xxB/
    shortest_delay/
      edges.csv
      time_indices.npy
      step_summary.csv
      path_samples.csv
      mean_shortest_delay_ms.npy
      meta.json
```

`step_summary.csv` is the main time-series table. `path_samples.csv` stores a small number of concrete source-target shortest paths for manual inspection.

For hop-count experiments, set:

```yaml
metric:
  type: shortest_hops
```

The output folder changes to `<topology>/shortest_hops/`, and the main array is `mean_shortest_hops.npy`.

## Hybrid Edge Tables

`src.topology_workflow.module.hybrid_edges` builds local hybrid topologies from
two existing `EdgeTable` objects. The common use case is:

1. keep one base topology;
2. copy selected inter-plane links from a patch topology in a y-band;
3. remove outgoing/incoming conflicts from the base topology;
4. verify each satellite has at most one right-neighbor link and at most one
   left-neighbor link.

This is parameter-transparent infrastructure: it does not know motif ids,
region groups, or paper-specific metrics. A concrete G60 bridge search example
is:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\search_local_hybrid_bridge_candidates.py' --previous-action 466 --next-action 621 --base-actions 557 --patch-actions 621 --min-width 1 --max-width 12 --row-step 1 --max-total-setup 489 --out-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\local_hybrid_bridge_candidates_466_to_621_b489'
```

## YAML Shape

```yaml
constellation:
  name: G60
  p: 18
  n: 36
  station_groups: {...}

time:
  start: 0
  end: 86164
  stride: 1

data:
  group_data:
    xml_file: ...
    cache_dir: ...
  position:
    raw_dir: ...
    cache_root: ...
    full_cache_dir: ...
    enable_fallback: false
  delay_store:
    store_dir: ...

topology:
  kind: single_motif
  add_intra_ring: true
  motif:
    w: 3
    h: 3
    support:
      - [0, 0, D]
  tiling:
    horizontal_step: null
    allow_vertical_overlap: true
    allow_clipped_right: true

metric:
  type: shortest_delay
  source_group_id: 2
  target_group_id: 3

outputs:
  out_dir: E:/paper11/data/linshi/workflows/...
```

Planned extension points:

- `topology.kind: motif_library` for `w*h` plus all smaller motif combinations
- batch execution over multiple topology definitions
