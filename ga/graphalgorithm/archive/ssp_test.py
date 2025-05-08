import ga.graphalgorithm.archive.ssp as ssp


mcf = ssp.MinCostFlow(12)

# Add edges (from, to, capacity, cost)
mcf.add_edge(0, 1, 18, 60)  # 0->1 cap 10 cost 2
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


# We want to send 7 units from node 0 (source) to node 3 (sink)
flow, cost = mcf.flow(0, 10, 60)

print(f"Flow: {flow}, Cost: {cost}")