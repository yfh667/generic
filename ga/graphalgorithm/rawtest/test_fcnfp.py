import ga.graphalgorithm.fcnfp as fcnfp
import ga.graphalgorithm

if __name__ == '__main__':
    # 多源多汇测试数据（包含超级节点所需的连接边）
    example_edges_multi = [
        # 原始网络边...（与用户提供的完全一致）
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

    SOURCES =0
    SINKS =10
    flow_demand=30
    # 使用新函数求解
    multi_result = fcnfp.solve_fixed_charge_network_flow(
        edges_data=example_edges_multi,
        source_node=SOURCES,
        sink_node=SINKS,
    flow_demand=flow_demand
    )





    if multi_result ==0:
        print("No solution found")
    else:
        # 输出结果（与原有格式兼容）
        cost =multi_result['total_fixed_cost']
        print(f"Status: {multi_result['status']}")
        if multi_result['status'] == "Optimal":
            print(f"Total Fixed Cost: {multi_result['total_fixed_cost']}")
            if multi_result['flow_details']:
                print("\nFlow Details (原始网络中的边):")
                for detail in multi_result['flow_details']:
                    print(f"  Edge ({detail['from']} -> {detail['to']}): "
                          f"Fixed Cost={detail['fixed_cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")