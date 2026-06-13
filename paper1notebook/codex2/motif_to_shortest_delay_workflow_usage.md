# Motif 到平均最短延迟的完整使用流程

本文档梳理当前已经跑通的研究流程：

```text
motif 生成
  -> motif 铺满 G60 全网，并补 intra y-ring 链路
  -> 用全链路 delay cache 查询每条边的传播延迟
  -> 对指定 region pair 计算每个时间片的平均最短传播延迟
  -> 后处理：静态最优、逐时刻动态最优、曲线图、summary
  -> 2D viewer 检查拓扑形态
```

当前这些脚本主要放在：

```text
E:\paper11\generic\paper1notebook\codex2
```

当前实验数据统一放在：

```text
E:\paper11\data\linshi
```

运行 Python 使用：

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe'
```

---

## 1. 核心概念

### 1.1 Motif 符号

inter-plane 右邻居候选用 4 个符号表示：

```text
A = (dp=1, dy=0)
B = (dp=1, dy=-1)
C = (dp=1, dy=1)
D = (dp=2, dy=0)
```

其中 `dp` 是轨道面方向偏移，`dy` 是相位方向偏移。

约束是：

```text
每个卫星最多 1 个右邻居
每个卫星最多 1 个左邻居
```

也就是：

```text
source out-degree <= 1
target in-degree <= 1
```

### 1.2 Motif box 的 `w, h`

当前代码里的 `w` 是 box width，包含最后一列 target boundary。

所以：

```text
真正可规划列数 = w - 1
h = y 方向相位高度
```

例如代码里 `w=4, h=3`，实际有：

```text
3 x 3 个可规划位置
```

### 1.3 三类 motif 数量

当前我们区分三层数量。

第一层是 `maximal_exact_box(w,h)`：

```text
固定 w x h box 内所有极大 motif。
极大表示：如果某个 source 为空，那么它所有合法 target 都已经被占用，不能再加边。
```

例如：

```text
w=4,h=3 -> maximal = 1156
```

第二层是 `primitive_exact_box(w,h)`：

```text
先枚举 maximal，再去掉 symbol matrix 能由更小周期重复得到的 motif。
```

例如：

```text
w=4,h=3 -> primitive matrix = 1148
```

第三层是我们之前实际用于 706 实验的 `function_a(w,h)`：

```text
primitive_exact_box(w,h)
  -> canon_graph 做平移/图等价去重
```

例如：

```text
w=4,h=3 -> function_a = 706
```

因此：

```text
706 不是所有 <=4x3 的 motif。
706 是 4x3 特有 primitive motif 经过平移/图等价去重后的候选库。
```

---

## 2. 当前已经跑通的两个 motif library

### 2.1 4x3 primitive706

生成脚本：

```text
E:\paper11\generic\paper1notebook\codex2\draw_w4h3_primitive_translation_dedup.py
```

运行：

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\draw_w4h3_primitive_translation_dedup.py' `
  --w 4 `
  --h 3 `
  --out-dir 'E:\paper11\data\linshi\motif_w4h3_primitive_translation_dedup'
```

主要输出：

```text
E:\paper11\data\linshi\motif_w4h3_primitive_translation_dedup\w4_h3_primitive_translation_dedup.csv
```

CSV 字段大致为：

```text
motif_id
motif
edge_count
support
edges
```

其中 `motif` 形如：

```text
DBD | --B
```

### 2.2 小尺寸 small102

这是我们后来确认的“小于 4x3 的所有 function_a 候选”：

```text
w=2..4, h=1..3，排除 w=4,h=3
```

数量为：

```text
102
```

生成脚本：

```text
E:\paper11\generic\paper1notebook\codex2\build_small102_motif_library_up_to_w4h3.py
```

运行：

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\build_small102_motif_library_up_to_w4h3.py'
```

主要输出：

```text
E:\paper11\data\linshi\motif_up_to_w4h3_small102_translation_dedup\small102_up_to_w4h3_excluding_w4h3.csv
```

这个 CSV 比 706 多了来源字段：

```text
source_w
source_h
local_motif_id
```

用于追溯某个 motif 来自哪个小 box。

---

## 3. 从 motif 铺满到全网 topology

### 3.1 实验中实际使用的铺满逻辑

批处理实验使用下面这个函数构造拓扑：

```python
edge_table_for_motif(motif_text)
```

位置：

```text
E:\paper11\generic\paper1notebook\codex2\build_g60_selected_motifs_shortest_delay_parallel.py
```

它做的事情是：

```text
1. 解析 motif 文本，例如 "DBD | --B"
2. 转为局部 EdgeRecord
3. 调用 tile_edge_records_on_grid 铺到 G60 P=18,N=36 全网
4. 横向按 motif_width - 1 重复
5. 纵向允许 overlap，用于尝试覆盖更多可规划位置
6. 允许右边界 clipped，因为最后轨道没有右邻居
7. 追加 intra y-ring 链路
8. 转成 EdgeTable，供最短路和 2D viewer 使用
```

对应的核心参数是：

```python
tile_edge_records_on_grid(
    p=18,
    n=36,
    motif_width=motif_width,
    motif_height=motif_height,
    local_edges=local_edges,
    horizontal_step=None,
    allow_vertical_overlap=True,
    allow_clipped_right=True,
)
```

### 3.2 intra y-ring 链路

所有 motif topology 都会额外补上同轨道上下链路：

```text
(p, y) -- (p, y+1 mod N)
```

也就是每条轨道内部是一个 y-ring。

2D viewer 默认隐藏 `(p, N-1) -- (p, 0)` 这种首尾 wrap 视觉线，避免画面被长线污染；但计算里这条边是存在的。

### 3.3 输出的边表

每个 topology 的边表会写成：

```text
edges.csv
```

字段包括：

```text
edge_idx
src_node
dst_node
src_sat_id
dst_sat_id
src_plane
src_y
dst_plane
dst_y
option
```

其中：

```text
option=-1 表示 intra y-ring
option=0  表示 A
option=1  表示 B
option=2  表示 D
option=4  表示 C
```

---

## 4. Delay cache

最短传播延迟实验不实时算卫星距离，而是查询已经生成好的 delay cache。

当前使用的是：

```text
E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1
```

它包含：

```text
full-option inter links
+ intra y-ring links
```

如果这个 cache 不存在，可用下面脚本从已有 full-option inter delay store 和 position cache 生成：

```text
E:\paper11\generic\paper1notebook\codex2\build_g60_full_option_plus_intra_delay_store.py
```

运行：

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\build_g60_full_option_plus_intra_delay_store.py' `
  --start 0 `
  --end 86164 `
  --stride 1
```

默认输入：

```text
inter delay store:
E:\paper11\data\basic_file\G60\satellitesposition\full_option_edge_delay\G60_full_options_t0_86164_stride1

position cache:
E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s
```

默认输出：

```text
E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1
```

---

## 5. Region group 输入

当前 region group 来自：

```text
E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml
```

group id 约定：

```text
0: America
1: Africa
2: China
3: Europe
```

常用实验：

```text
China-Europe:  source_group=2, target_group=3
China-Africa:  source_group=2, target_group=1
China-America: source_group=2, target_group=0
```

group cache 默认放在：

```text
E:\paper11\data\linshi\cache\group_data_cache
```

---

## 6. 计算某个 motif library 的平均最短传播延迟

核心批处理脚本：

```text
E:\paper11\generic\paper1notebook\codex2\build_g60_selected_motifs_shortest_delay_parallel.py
```

它的输入是一个 motif CSV，例如：

```text
motif_id,motif,edge_count,support,edges
```

也可以带额外字段，例如 small102 的：

```text
source_w,source_h,local_motif_id
```

批处理脚本只要求至少存在：

```text
motif_id
motif
```

### 6.1 跑 small102 的 China-Europe

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\build_g60_selected_motifs_shortest_delay_parallel.py' `
  --selected-motif-csv 'E:\paper11\data\linshi\motif_up_to_w4h3_small102_translation_dedup\small102_up_to_w4h3_excluding_w4h3.csv' `
  --out-dir 'E:\paper11\data\linshi\g60_small102_up_to_w4h3_china_europe_t0_86160_stride60' `
  --start 0 `
  --end 86160 `
  --stride 60 `
  --source-group 2 `
  --target-group 3 `
  --max-workers 32 `
  --chunk-size 60 `
  --progress-every-futures 100
```

### 6.2 跑 706 的 China-Europe

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\build_g60_selected_motifs_shortest_delay_parallel.py' `
  --selected-motif-csv 'E:\paper11\data\linshi\motif_w4h3_primitive_translation_dedup\w4_h3_primitive_translation_dedup.csv' `
  --out-dir 'E:\paper11\data\linshi\g60_w4h3_primitive706_china_europe_t0_86160_stride60' `
  --start 0 `
  --end 86160 `
  --stride 60 `
  --source-group 2 `
  --target-group 3 `
  --max-workers 32 `
  --chunk-size 60 `
  --progress-every-futures 100
```

### 6.3 输出目录结构

一个批处理输出目录通常包含：

```text
compare_100motifs_gridplus_full_link.csv
motif_summary_sorted.csv
meta.json
selected_motifs_original_rows.csv
_parallel_worker_inputs/
motif_000001/
motif_000002/
...
```

注意：

```text
compare_100motifs_gridplus_full_link.csv
```

这个文件名是旧脚本遗留命名。即使跑的是 102 或 706，它仍然可能叫这个名字。

每个 `motif_xxxxxx/` 目录内通常有：

```text
edges.csv
mean_shortest_delay_ms.npy
step_summary.csv
path_samples.csv
meta.json
```

---

## 7. 后处理：strict reachable、静态最优、动态最优

核心后处理脚本：

```text
E:\paper11\generic\paper1notebook\codex2\postprocess_g60_primitive706_region_pair.py
```

虽然脚本名里有 `706`，现在已经加了参数，可用于 small102 或其他 library。

强烈建议使用：

```text
--strict-reachable
```

原因是：

```text
如果某个 topology 在某个时间片不是所有 source-target pair 都可达，
只对可达 pair 平均会产生虚假的低延迟。
strict-reachable 会把这类时间片标为 NaN，不参与动态最优选择。
```

### 7.1 后处理 small102 China-Europe

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\postprocess_g60_primitive706_region_pair.py' `
  --out-dir 'E:\paper11\data\linshi\g60_small102_up_to_w4h3_china_europe_t0_86160_stride60' `
  --pair-key china_europe_small102 `
  --pair-label 'China-Europe' `
  --strict-reachable `
  --library-label '102 small-size motifs up to 4x3, excluding 4x3 primitive706' `
  --output-stem small102motifs
```

主要输出：

```text
dynamic_best_small102motifs_by_step_china_europe_small102_strict_reachable.csv
motif_summary_sorted_small102motifs_china_europe_small102_strict_reachable.csv
mean_shortest_delay_small102motifs_gridplus_full_link_china_europe_small102_strict_reachable.png
region_pair_summary_china_europe_small102_strict_reachable.json
```

### 7.2 后处理 706 China-Europe

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\postprocess_g60_primitive706_region_pair.py' `
  --out-dir 'E:\paper11\data\linshi\g60_w4h3_primitive706_region_pairs_t0_86160_stride60\china_europe\motifs706' `
  --pair-key china_europe `
  --pair-label 'China-Europe' `
  --strict-reachable `
  --library-label '706 primitive 4x3 motifs' `
  --output-stem 706motifs
```

---

## 8. 如何解释结果

### 8.1 静态最优

静态最优是：

```text
从一个 motif library 中选一个固定 topology，
它在整个时间段内的 strict mean delay 最小。
```

看：

```text
motif_summary_sorted_<library>_<pair>_strict_reachable.csv
```

### 8.2 动态最优

动态最优是：

```text
每个时间片从 motif library 中选择当前 delay 最小的 topology。
```

看：

```text
dynamic_best_<library>_by_step_<pair>_strict_reachable.csv
```

这里的动态最优不是一个物理上自动可切换的最终方案，只是用于观察：

```text
是否存在单一静态 motif 无法覆盖全时段最优的问题。
```

后续如果要做真实动态拓扑，还需要加入：

```text
最短驻留时间
切换代价
链路重构开销
稳定性约束
```

### 8.3 full_link baseline

`full_link` 是所有候选 inter option 加 intra y-ring 的上界参考。

它不是合法最终 motif，因为每个卫星可能有多个右邻居/左邻居。

它用于回答：

```text
如果完全放开 motif 限制，最短延迟的理论参考是什么？
```

---

## 9. 用 2D viewer 检查某个 motif

### 9.1 查看 small102 里的 motif_000056

我们当前 best static small102 是：

```text
motif_000056 = DBD | --B
```

绘图脚本：

```text
E:\paper11\generic\paper1notebook\codex2\draw_g60_small102_motif_000056_viewer.py
```

打开交互式 2D viewer：

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\draw_g60_small102_motif_000056_viewer.py'
```

只生成截图：

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\draw_g60_small102_motif_000056_viewer.py' `
  --offscreen
```

输出：

```text
E:\paper11\data\linshi\g60_small102_motif_000056_viewer
```

### 9.2 通用 viewer 入口

更通用的 viewer 示例在：

```text
E:\paper11\generic\src\motif_generator\examples\run_tiled_motif_2d_viewer.py
```

它更适合后续整理到正式模块。

---

## 10. 当前 small102 China-Europe 实验结果

实验参数：

```text
G60
China-Europe
start=0
end=86160
stride=60
motif_count=102
```

输出目录：

```text
E:\paper11\data\linshi\g60_small102_up_to_w4h3_china_europe_t0_86160_stride60
```

关键结果：

```text
full_time_reachable_motifs = 95
best static topology = motif_000056
best static motif = DBD | --B
best static mean delay = 42.4615 ms
dynamic best among 102 mean delay = 41.7157 ms
full_link mean delay = 37.6550 ms
gridplus mean delay = 51.3068 ms
```

这说明：

```text
小尺寸 motif 中已经有一些结构明显优于 gridplus；
但距离 full_link 仍有明显差距。
```

---

## 11. 建议后续项目化结构

当前代码已经跑通，但还比较实验化。后续可以按下面方式固化：

```text
src/
  motif_generator/
    module/
      exact_box.py
      tiling.py
      support.py
      viewer_adapter.py
      library.py              # 建议新增：library 生成、跨尺寸拼接、去重
    examples/
      enumerate_library.py
      run_tiled_motif_2d_viewer.py
      motif_generator_usage.ipynb

  link_delay/
    module/
      build_full_link_delay_store.py
      query.py
      edge_options.py
      position_cache.py
    examples/
      build_g60_delay_store.py
      query_one_edge_delay.py
      link_delay_usage.ipynb

  topology_metric/
    module/
      shortest_delay.py        # 建议新增：不绑定 G60 的最短传播延迟计算
      reachability.py
      batch_runner.py
    examples/
      run_g60_selected_motifs.py
      postprocess_region_pair.py

  satellite_topology_viewer/
    module/
      base_viewer.py
      topology_edges.py
      ...
    examples/
      run_g60_motif_viewer.py
      satellite_topology_viewer_usage.ipynb
```

项目化后的原则：

```text
module 里不硬编码星座路径和实验路径；
examples 里写 G60、Starlink、China-Europe 等具体案例；
notebook 用来记录可交互使用方式；
paper1notebook/codex2 保留临时探索脚本和中间实验。
```

---

## 12. 一个完整复现实验的最短命令链

以 small102 China-Europe 为例：

```powershell
# 1. 生成 small102 motif library
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\build_small102_motif_library_up_to_w4h3.py'

# 2. 跑 102 个 motif 的平均最短传播延迟
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\build_g60_selected_motifs_shortest_delay_parallel.py' `
  --selected-motif-csv 'E:\paper11\data\linshi\motif_up_to_w4h3_small102_translation_dedup\small102_up_to_w4h3_excluding_w4h3.csv' `
  --out-dir 'E:\paper11\data\linshi\g60_small102_up_to_w4h3_china_europe_t0_86160_stride60' `
  --start 0 `
  --end 86160 `
  --stride 60 `
  --source-group 2 `
  --target-group 3 `
  --max-workers 32 `
  --chunk-size 60 `
  --progress-every-futures 100

# 3. strict-reachable 后处理
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\postprocess_g60_primitive706_region_pair.py' `
  --out-dir 'E:\paper11\data\linshi\g60_small102_up_to_w4h3_china_europe_t0_86160_stride60' `
  --pair-key china_europe_small102 `
  --pair-label 'China-Europe' `
  --strict-reachable `
  --library-label '102 small-size motifs up to 4x3, excluding 4x3 primitive706' `
  --output-stem small102motifs

# 4. 打开 best static motif 的 2D viewer
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' -u `
  'E:\paper11\generic\paper1notebook\codex2\draw_g60_small102_motif_000056_viewer.py'
```

