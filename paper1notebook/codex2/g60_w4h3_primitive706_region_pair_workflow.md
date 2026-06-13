# G60 4x3 Primitive-706 Region-Pair Shortest-Delay Workflow

本流程用于在同一组 706 个 `4x3 primitive` motif 上，分别计算不同 region pair 的
China-to-target 平均最短传播延迟时序。

## 固定输入

- motif 清单：
  `E:/paper11/data/linshi/motif_w4h3_primitive_translation_dedup/w4_h3_primitive_translation_dedup.csv`
- full-option + intra delay store：
  `E:/paper11/data/linshi/G60_full_options_plus_intra_t0_86164_stride1`
- region group cache：
  `E:/paper11/data/linshi/cache/group_data_cache`
- 时间范围：
  `start=0, end=86160, stride=60`

## G60 region group id

- `0`: America
- `1`: Africa
- `2`: China
- `3`: Europe

本轮使用：

- China-Europe: `source_group=2`, `target_group=3`
- China-Africa: `source_group=2`, `target_group=1`
- China-America: `source_group=2`, `target_group=0`

## 输出目录结构

统一放在：

`E:/paper11/data/linshi/g60_w4h3_primitive706_region_pairs_t0_86160_stride60`

每个区域对一个独立子目录：

- `china_europe/`
- `china_africa/`
- `china_america/`

每个区域对内部：

- `baselines/`: 该区域对自己的 `gridplus` 和 `full_link` baseline
- `motifs706/`: 706 个 motif 的结果、总表、图和动态最优表

## 计算步骤

1. 先生成该区域对自己的 baseline。

   使用：
   `build_g60_motif_gridplus_shortest_delay_timeseries_parallel.py`

   只计算：
   `--topologies gridplus full_link`

2. 再运行 706 个 motif。

   使用：
   `build_g60_selected_motifs_shortest_delay_parallel.py`

   注意：
   `--gridplus-dir` 和 `--full-link-dir` 必须指向同一个区域对的
   `baselines/gridplus` 与 `baselines/full_link`，不能混用 China-Europe 的 baseline。

3. 最后生成区域对命名清楚的总览图和动态最优表。

   使用：
   `postprocess_g60_primitive706_region_pair.py`

   推荐加：
   `--strict-reachable`

   原因：如果只对可达 source-target pair 求有限均值，断开的 motif 会因为跳过不可达
   pair 而出现“看起来比 full_link 更低”的假象。strict reachable 口径会在某个
   step 不是所有 source-target pair 都可达时把该 motif 的该 step 标成 `NaN`，
   画图时断开，不参与动态最优选择。

   主要输出：

   - `compare_100motifs_gridplus_full_link.csv`
   - `motif_summary_sorted.csv`
   - `dynamic_best_706motifs_by_step_<pair>.csv`
   - `mean_shortest_delay_706motifs_gridplus_full_link_<pair>.png`
   - `region_pair_summary_<pair>.json`

## 本轮结果概览

| pair | best static motif | mean delay ms | dynamic unique motifs |
| --- | --- | ---: | ---: |
| China-Europe | `motif_000090` | 44.6043 | 25 |
| China-Africa | `motif_000090` | 52.2826 | 42 |
| China-America | `motif_000285` | 63.6469 | 22 |
