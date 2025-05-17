
import graph.drawtd as dt
# 示例使用
if __name__ == "__main__":
    # 示例1：创建一个5x4的拓扑，3层
    print("创建一个5x4的拓扑，3层结构...")
    dt.plot_multi_layer_topology(P=5, N=4, t=10)

    # 示例2：创建一个3x3的拓扑，4层
    # print("\n创建一个3x3的拓扑，4层结构...")
    # plot_multi_layer_topology(N=3, P=3, t=4)