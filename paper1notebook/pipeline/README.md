# Paper1 Motif Pipeline

This folder contains paper1-specific orchestration code. Reusable algorithms
stay under `src`; this layer only binds concrete G60 paths, region pairs,
legacy baselines, and output directories.

## G60 4x3-and-Smaller Shortest Hops

Default config:

```powershell
E:\paper11\generic\paper1notebook\pipeline\configs\g60_w_le4_h_le3_shortest_hops.yaml
```

Full run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' `
  'E:\paper11\generic\paper1notebook\pipeline\run_paper1_motif_shortest_hops.py'
```

Small smoke run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' `
  'E:\paper11\generic\paper1notebook\pipeline\run_paper1_motif_shortest_hops.py' `
  --end 120 --stride 60 --limit-motifs 2 --max-workers 1 --pairs china_europe --skip-gridplus `
  --out-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3\smoke_shortest_hops_t0_120_stride60_limit2'
```

Outputs are written under:

```text
E:\paper11\data\satnet_experiments
```

The motif library is generated or reused at:

```text
E:\paper11\data\satnet_experiments\libraries\motif\exact_box\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv
```

The current YAML keeps the historical `grid+` baseline enabled, but the
referenced `motif.json` is not present in this workspace. Use `--skip-gridplus`
for runs that only need motif candidates plus `full_link`, or update
`paths.gridplus_config` when the legacy file is available.

## G60 4x3-and-Smaller Shortest Delay

Default config:

```powershell
E:\paper11\generic\paper1notebook\pipeline\configs\g60_w_le4_h_le3_shortest_delay.yaml
```

Smoke run:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' `
  'E:\paper11\generic\paper1notebook\pipeline\run_paper1_motif_shortest_delay.py' `
  --end 120 --stride 60 --limit-motifs 2 --pairs china_europe --skip-gridplus `
  --out-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3\smoke_shortest_delay_t0_120_stride60_limit2'
```

The delay pipeline reads inter-link delay from:

```text
E:\paper11\data\basic_file\G60\satellitesposition\full_option_edge_delay\G60_full_options_t0_86164_stride1
```

and uses the position cache as fallback for intra-ring links:

```text
E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s
```

For full 86160s runs, keep outputs under:

```text
E:\paper11\data\satnet_experiments\runs\paper1\G60
```

## Dynamic Schedule From Comparison CSV

Once a comparison CSV exists, build per-step oracle and minimum-dwell schedules
with the reusable example:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' `
  'E:\paper11\generic\src\topology_workflow\examples\analyze_dynamic_schedule_from_csv.py' `
  --compare-csv '<path-to-compare_csv>' `
  --out-dir '<path-to-dynamic-schedule-output>' `
  --topology-prefix motif_ `
  --min-dwell-minutes 10 30 60 120
```
