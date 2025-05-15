import ga.graphalgorithm.mcmf.ssp as ssp
import ga.graphalgorithm.mcmf.ssp as ssp
import genaric.plotgraph as plotgraph
import ga.graphalgorithm.adjact2weight as a2w
# —— 示例用法 —— #
if __name__ == "__main__":
    example_edges_multi = [
        (0, 1, 18, 60), (0, 2, 19, 60), (0, 3, 17, 60),
        (1, 4, 16, 40), (1, 5, 14, 30),
        (2, 5, 16, 50), (2, 6, 17, 30),
        (3, 6, 19, 40),
        (4, 7, 19, 60),
        (5, 4, 15, 20),
        (5, 7, 16, 30), (5, 8, 15, 40),
        (6, 5, 15, 20),
        (6, 9, 13, 40),
        (7, 10, 18, 60), (7, 8, 17, 30),
        (8, 10, 19, 50), (8, 9, 14, 30),
        (9, 10, 17, 50),
    ]



    SOURCES = 0
    SINKS   = 10
    flow_demand = 30

    result = ssp.solve_mcmf(
        edges_data=example_edges_multi,
        source_node=SOURCES,
        sink_node=SINKS,
        flow_demand=flow_demand
    )



    if result ==0:
        print("No solution found")
    else:
        # 输出结果（与原有格式兼容）
        cost =result['total_cost']
        print(f"Status: {result['status']}")
        if result['status'] == "Optimal":
            print(f"Total Fixed Cost: {result['total_cost']}")
            if result['flow_details']:
                print("\nFlow Details (原始网络中的边):")
                for detail in result['flow_details']:
                    print(f"  Edge ({detail['from']} -> {detail['to']}): "
                          f"Fixed Cost={detail['cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")

  #  plotgraph.plot_graph_with_auto_curve(full_adjacency_list, N, P, distinct)