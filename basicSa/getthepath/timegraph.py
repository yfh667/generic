from collections import defaultdict


class TimeGraph:
    def __init__(self):
        # 使用嵌套字典存储边权重
        # 结构: {from_node: {to_node: weight}}
        self.graph = defaultdict(dict)

        # 存储节点额外属性（如abundance等）
        self.node_attributes = defaultdict(dict)

    def add_edge(self, from_node, to_node, weight):
        """添加带权边"""
        self.graph[from_node][to_node] = weight

    def add_node_attribute(self, node, **attrs):
        """添加节点属性"""
        self.node_attributes[node].update(attrs)

    def get_edge_weight(self, from_node, to_node):
        """获取边权重"""
        return self.graph[from_node].get(to_node, float('inf'))  # 不存在时返回无穷大

    def get_neighbors(self, node):
        """获取节点的所有邻居"""
        return self.graph[node].keys()

    def __str__(self):
        """可视化图结构"""
        lines = []
        for from_node in sorted(self.graph):
            for to_node in sorted(self.graph[from_node]):
                weight = self.graph[from_node][to_node]
                lines.append(f"{from_node} -> {to_node} : {weight}")
        return "\n".join(lines)


# # 初始化图实例
# graph = TimeGraph()
#
# # 添加节点属性（示例）
# graph.add_node_attribute((0, 0), abundance=0, cumulative=0)
# graph.add_node_attribute((1, 0), abundance=20, cumulative=20)
#
# # 添加边（示例）
# graph.add_edge((0, 0), (1, 0), 20)  # (0,0)到(1,0)权重20
# graph.add_edge((0, 1), (1, 1), 15)  # 带冲突惩罚的边
# graph.add_edge((0, 2), (1, 2), 20)  # 无冲突的正常边
#
# # 打印图结构
# print("当前图结构:")
# print(graph)
