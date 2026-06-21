# topology_learning

This module prepares learning data for topology-selection experiments.

It deliberately does **not** enforce satellite-network constraints itself. Hard
constraints such as one right/left inter-plane port and link-setup-time states
should be produced by existing deterministic modules:

- `src.motif_generator`
- `src.topology_workflow`
- `src.topology_metrics`

`topology_learning` consumes their outputs and builds:

- `dataset.npz`: GNN-ready/supervised arrays.
- `teacher_by_step.csv`: one selected topology label per time step.
- `action_summary.csv`: per-action summary and motif metadata when available.
- `meta.json`: reproducibility metadata.

## Current Scope

The first implementation is dependency-free beyond the project environment:

- no PyTorch required;
- no torch-geometric required;
- no sklearn required.

It builds a teacher from one or more wide metric CSVs, where rows are time steps
and columns are topology candidates. Lower values are better.

## Experiment Convention

For the current paper1 dynamic-topology experiments:

- Region groups are endpoint sets for metrics only. They do not force
  region-internal `+grid` links.
- A topology candidate is built from a motif tiled over the full constellation,
  plus intra-plane links.
- LST is the time from issuing a link-setup command to that link becoming
  active. During this interval the edge is marked `building` and is not used in
  shortest-path metrics. The default `--setup-timing reactive` treats the
  selected topology sequence as setup commands: when a new edge appears, the
  command is issued, and the edge can route only after one LST interval.
  `--setup-timing advance` is kept only for offline known-future upper-bound
  comparisons where commands are emitted before the desired activation time.
- Edge betweenness should be interpreted as shortest-path edge usage share.
  We keep two separate views first: weighted shortest-delay usage share and
  unweighted shortest-hop usage share.
- When one scalar edge criticality is needed, the current convention is to
  combine the max-normalized shortest-delay usage share and shortest-hop usage
  share. The G60 paper1 run stores this as
  `full_link_edge_usage_share_three_pairs_t0_86160_stride60/delay_hop_combined/combined_usage_share_max_over_time.npy`.

The older `region_internal_plus_grid` YAML is kept only as a legacy example for
old runs. It is not the default convention for new dynamic-topology experiments.

## Dataset NPZ

The exported `dataset.npz` contains:

- `steps`: sampled time steps.
- `topology_names`: candidate action names.
- `action_values`: weighted objective matrix, lower is better.
- `action_valid_mask`: legal/finite action mask for each step.
- `labels`: teacher action index, i.e. `argmin(action_values[t])`.
- `label_values`: teacher objective value.
- `sample_features`: currently time-phase features, used by smoke baselines.
- `metric_<name>`: raw metric matrix for each configured metric.

## Future GNN Hook

`module.graph_state` provides `build_node_features` and `build_edge_arrays`.
These return plain numpy arrays:

- `node_features`
- `edge_index`
- `edge_features`

A later PyTorch/PyG adapter can wrap these arrays without changing the data
contract used by experiments.

## Full-Link Gap Selector

The current dynamic-topology objective is:

1. get as close as possible to `full_link` on shortest delay and shortest hops;
2. reduce dynamic link setup by penalising newly added edges when switching from one topology to the next.

The selector consumes existing `compare_*.csv` metric tables. It does not recompute
shortest paths and does not modify region groups.

Run China-Europe hop+delay selection:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\run_full_link_gap_selector.py' --config 'E:\paper11\generic\src\topology_learning\examples\configs\g60_w4h3_full_link_gap_china_europe_hop_delay.yaml'
```

Run China-Africa:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\run_full_link_gap_selector.py' --config 'E:\paper11\generic\src\topology_learning\examples\configs\g60_w4h3_full_link_gap_china_africa_hop_delay.yaml'
```

Run China-America:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\run_full_link_gap_selector.py' --config 'E:\paper11\generic\src\topology_learning\examples\configs\g60_w4h3_full_link_gap_china_america_hop_delay.yaml'
```

Each output directory contains:

- `summary.csv`: trade-off between full-link gap and switch/new-edge cost.
- `*_by_step.csv`: selected topology at each time step.
- `selector_arrays.npz`: arrays for downstream learning.
- `schedule_metric_comparison.png`: full-link vs selected schedules.

The current transition cost is `|E_next \ E_prev|`, normalised by the maximum
observed transition. This is a proxy for the number of new link-setup commands.
For critical-edge-aware schedules, transition cost can instead be weighted by
full-link shortest-path edge usage share. The combined delay/hop criticality
config is:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\run_full_link_gap_selector.py' --config 'E:\paper11\generic\src\topology_learning\examples\configs\g60_w4h3_full_link_gap_three_pairs_hop_delay_combined_critical_edges.yaml'
```

For schedules where simultaneous setup bursts are the main concern, the selector
can also use a burst profile such as `(|E_next \ E_prev| - threshold)^2`. This
does not change the LST semantics; it only changes how the candidate topology
sequence is chosen before the LST active/building simulation is applied.
For strict setup-peak studies, use `transition_caps`. A cap schedule minimises
the full-link gap while forbidding any transition where the number of newly
requested edges exceeds the configured cap. This is useful for answering "how
much quality do we lose if at most N new links may be requested in one sampled
step?"
For LST-friendly schedules, use `min_dwell_steps` with
`min_dwell_switch_penalties`. A minimum-dwell schedule only allows a topology
switch after the current topology has been held for the configured number of
sampled rows. With the current 60s metric sampling and 120s LST, `dwell=3`
means newly requested links can become active before the next switch is allowed.
The burst and dwell controls can also be combined with
`min_dwell_burst_penalties`, producing schedules such as
`dp_dwell3_burst2_penalty_10`. In the current G60 w4h3 three-pair run, this
combination removes the one-step burst-only switches while staying within 1% of
the best post-LST full-link gap.
Actual LST active/building simulation should be layered after a schedule is chosen.
By default this uses command-driven `reactive` timing:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\evaluate_lst_schedule_from_selector.py' --selector-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_china_europe_hop_delay' --schedule 'dp_new_edge_penalty_0.1' --setup-time 120 --delay-store-dir 'E:\paper11\data\basic_file\G60\satellitesposition\full_option_edge_delay\G60_full_options_t0_86164_stride1' --position-cache-dir 'E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s' --group-xml 'E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml' --group-cache-dir 'E:\paper11\data\satnet_experiments\cache\G60\group_data' --pair '2:3:china_europe'
```

Batch-evaluate every schedule in a selector output:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\evaluate_lst_sweep_from_selector.py' --selector-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_china_europe_hop_delay' --setup-time 120 --delay-store-dir 'E:\paper11\data\basic_file\G60\satellitesposition\full_option_edge_delay\G60_full_options_t0_86164_stride1' --position-cache-dir 'E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s' --group-xml 'E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml' --group-cache-dir 'E:\paper11\data\satnet_experiments\cache\G60\group_data' --pair '2:3:china_europe'
```

Write a Pareto/recommendation report from one or more post-LST summaries:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\write_post_lst_report.py' --summary 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_three_pairs_hop_delay\lst120_eval_china_europe\lst120_china_europe_sweep_summary.csv' --summary 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_three_pairs_hop_delay\lst120_eval_china_africa\lst120_china_africa_sweep_summary.csv' --summary 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_three_pairs_hop_delay\lst120_eval_china_america\lst120_china_america_sweep_summary.csv' --out-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_three_pairs_hop_delay' --prefix 'lst120_three_pair_global'
```

The report writes both a Pareto summary and a decision table. The decision table
adds explicit fields for the two current goals:

- quality loss relative to the closest-to-full-link schedule;
- total setup-command reduction relative to that same schedule;
- peak setup-command reduction relative to that same schedule.

It also writes `*_quality_tolerance_frontier.csv`. For each first-goal tolerance
band around the best full-link gap, this table reports the schedules that
minimise total setup commands and peak setup commands. This is the quickest
table for the current priority order: first stay close to full_link on
delay/hops, then reduce link setup.

Selector summaries contain two related quality notions:

- `mean_*_gap`: average of per-time relative gaps.
- `mean_*_global_gap`: relative gap after taking the full-time mean value and
  comparing it with the full-time `full_link` mean. This matches the post-LST
  report convention and should be used for the current first-goal decision
  table.

To score any stored selector schedule or by-step CSV without recomputing
shortest paths:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\score_selector_schedule.py' --selector-dir '<selector-dir>' --schedule '<schedule-name>' --quality-threshold 0.1406225823162086
```

For larger row-action candidate sets, prefer the frontier budget-DP solver. It
keeps only non-dominated `(setup budget, score)` labels for each row/action, so
it can test wider `top_k_score` candidate sets without filling a dense
`budget x action` table:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\run_budget_dp_selector.py' --source-dir '<selector-dir>' --out-dir '<out-dir>' --budget 2628 --max-transition-count 408 --score-kind meanref --top-k-score 120 --solver frontier --quality-threshold 0.1406225823162086
```

To audit whether a known good schedule can be locally spliced into a lower
setup budget, use:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\search_schedule_splices.py' --base-selector-dir '<selector-dir>' --base-schedule '<schedule-name>' --donor-dir '<donor-selector-dir>' --out-dir '<out-dir>' --quality-threshold 0.1406225823162086 --target-setup 2628
```

This writes interval candidates, feasible splices, and a JSON summary. It is a
local audit around an existing schedule, not a proof over every possible
topology sequence.

To test whether the fixed switching times of a known schedule can be improved
by choosing from all topology actions at each segment:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\run_segment_boundary_dp.py' --selector-dir '<selector-dir>' --base-schedule '<schedule-name>' --out-dir '<out-dir>' --budget 2628 --max-transition-count 408 --score-kind meanref --quality-threshold 0.1406225823162086
```

This is complementary to row-level frontier DP: it uses all topology actions
but only the base schedule's segment boundaries.

## LST-Aware Beam Selector

The DP selector above chooses a target topology sequence first and applies LST
afterward. For a more direct but heavier check, use the LST-aware beam selector.
It scores the graph that is actually active after LST. With 60s samples and
`setup_time=120`, a newly requested edge is not routable until it has appeared
continuously across the current and previous two target rows.

The search is approximate: each row keeps only the `top_k` best precomputed
full-link-gap candidates plus any supplied existing schedules, and it retains
`beam_width` histories. This is useful for local validation around switching
windows before attempting a full-day run.

Implementation invariant: for every beam history, the previous action is always
added back into the next-row candidate set. Holding the current topology must be
a legal action; otherwise a small `top_k` can force artificial switches and
inflate LST building/disconnection artifacts.

Example around the first main switching window:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\run_lst_aware_beam_selector.py' --selector-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_three_pairs_hop_delay' --out-dir 'E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning\g60_w4h3_full_link_gap_three_pairs_hop_delay\lst_aware_beam_window_r300_160' --pair '2:3:ce:china_europe' --pair '2:1:ca:china_africa' --pair '2:0:cam:china_america' --setup-time 120 --delay-store-dir 'E:\paper11\data\basic_file\G60\satellitesposition\full_option_edge_delay\G60_full_options_t0_86164_stride1' --position-cache-dir 'E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s' --group-xml 'E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml' --group-cache-dir 'E:\paper11\data\satnet_experiments\cache\G60\group_data' --top-k 8 --beam-width 5 --setup-penalty 0.03 --burst-penalty 0.08 --start-row 300 --max-rows 160 --include-schedule 'dp_new_edge_penalty_0.5' --include-schedule 'dp_dwell5_burst2_penalty_7.5' --include-schedule 'dp_burst2_penalty_10' --include-schedule 'dp_new_edge_penalty_2'
```

The output directory is a normal selector directory with `selector_arrays.npz`,
`summary.csv`, and a new `schedule_lst_aware_beam_*` entry, so it can be passed
directly to `evaluate_lst_sweep_from_selector.py`.

### Segment-Boundary Audits

When a near-threshold result differs from a better-quality result only in a few
time windows, use `examples/run_union_segment_boundary_dp.py` to test whether
another topology assignment on the union of their switch boundaries can meet the
same quality threshold with a smaller setup budget. This is a compact exact
audit over fixed time segments and all topology actions, not a learned model.

For a still stronger local audit, use `examples/run_local_window_budget_dp.py`.
It fixes the base schedule outside a selected row window, includes the
entry/exit setup costs, and lets every sampled row inside that window choose any
topology action under a local setup budget.

When the fixed motif action space is too coarse, generate local hybrid actions
first and then score them as replacements. The current useful pattern is:

1. build y-band hybrid candidates with
   `examples/search_local_hybrid_bridge_candidates.py`;
2. compute their local hop/delay metrics with
   `examples/evaluate_local_hybrid_bridge_metrics.py`;
3. replace the corresponding window of an existing selector schedule with
   `examples/apply_local_hybrid_metric_replacement.py`.

For the G60 `466 -> 558 -> 622` bridge, this produced a 2%-threshold schedule
with `1720` setup commands and peak setup `406`, compared with `1728` setup for
the original b1728 schedule. The replacement keeps region groups as metric
endpoints only; it does not impose region-internal `+grid`.

After several replacement candidates have been scored, rank them with:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\topology_learning\examples\rank_local_hybrid_replacements.py' --comparison-csv '<comparison.csv>' --quality-threshold 0.1420148851114186 --out-dir '<ranked-out-dir>'
```

The ranking report separates all schedules from replacement-only schedules, so
the original best-quality baseline can remain visible while the replacement
recommendation still answers which hybrid lowers setup under the chosen quality
threshold.
