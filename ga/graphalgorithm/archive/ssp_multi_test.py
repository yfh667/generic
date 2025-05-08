import ga.graphalgorithm.archive.ssp_multi as ssp

mcf = ssp.MinCostFlow(13)  # 节点0-10

# 添加原始边（同您之前的网络）
mcf.add_edge(0, 1, 18, 60)
mcf.add_edge(12, 1, 30, 80)
mcf.add_edge(12, 2, 19, 60)
mcf.add_edge(12, 3, 15, 60)

mcf.add_edge(0, 2, 19, 60)
mcf.add_edge(0, 3, 17, 60)
mcf.add_edge(1, 4, 16, 40)
mcf.add_edge(1, 5, 14, 30)
mcf.add_edge(2, 5, 16, 50)
mcf.add_edge(2, 6, 17, 30)
mcf.add_edge(3, 6, 19, 40)
mcf.add_edge(4, 7, 19, 60)
mcf.add_edge(5, 4, 15, 20)
mcf.add_edge(5, 7, 16, 30)
mcf.add_edge(5, 8, 15, 40)
mcf.add_edge(6, 5, 15, 20)
mcf.add_edge(6, 9, 13, 40)
mcf.add_edge(7, 10, 18, 60)
mcf.add_edge(7, 8, 17, 30)
mcf.add_edge(8, 10, 19, 50)
mcf.add_edge(8, 9, 14, 30)
mcf.add_edge(9, 10, 17, 50)

mcf.add_edge(8, 11, 21, 60)
mcf.add_edge(9, 11, 16, 40)

# 设置参数
sources = [0, 12]  # 两个源点
source_supplies = [60, 60]  # 每个源点供应60
sinks = [11, 10]  # 两个汇点
total_demand = 120  # 总需求

flow, cost = mcf.solve(sources, source_supplies, sinks, total_demand)


print(f"Flow: {flow}, Cost: {cost}")