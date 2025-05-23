import ga.graphalgorithm.mcmf.ssp as ssp
import ga.graphalgorithm.mcmf.ssp as ssp
import genaric.plotgraph as plotgraph
import ga.graphalgorithm.adjact2weight as a2w
import ga.graphalgorithm.mcmf.ssp_multi as ssp_multi
# —— 示例用法 —— #
if __name__ == "__main__":
    example_edges_multi = [
        # 原始网络边...（与用户提供的完全一致）
        (0, 1, 18, 60), (0, 2, 19, 60), (0, 3, 17, 60),
        (12, 3, 15, 60), (12, 2, 19, 60),
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
        (10, 11, 0, 100),
        (8, 11, 21, 60),
        (9, 11, 16, 40),
    ]

    SOURCES = {0: 30, 12: 30}
    SINKS = [11, 10]

    multi_result = ssp_multi.solve_multi_source_sink_with_super_nodes(
        edges_data=example_edges_multi,
        sources=SOURCES,
        sinks=SINKS
    )

    if multi_result == 0:
        print("No solution found")
    else:
        # 输出结果（与原有格式兼容）
        cost = multi_result['total_cost']

        print("\nMulti-source multi-sink solution via super nodes:")
        print(f"Status: {multi_result['status']}")
        if multi_result['status'] == "Optimal":
            print(f"Total Fixed Cost: {multi_result['total_cost']}")
            if multi_result['flow_details']:
                print("\nFlow Details (原始网络中的边):")
                for detail in multi_result['flow_details']:
                    print(f"  Edge ({detail['from']} -> {detail['to']}): "
                          f"Fixed Cost={detail['cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")