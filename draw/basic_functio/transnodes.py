from typing import Dict, Tuple
import genaric2.tegnode as tegnode  # 或者按你的类名实际写

def transnodes(nodes_new: dict[tuple[int, int, int], tegnode.tegnode_new]):
    """
    nodes_new: {(x, y, step): tegnode_new}
    返回: {(x, y, step): tegnode_complete}
    """

    nodes_complete = {}

    for (x, y, step), node in nodes_new.items():
        # groupid = node.asc_nodes_region_id
        rightneighbor = node.rightneighbor
        right_state = node.state

        # 1. 先补自己节点（只设置右邻，不补左邻）
        if (x, y, step) not in nodes_complete:
            nodes_complete[(x, y, step)] = tegnode.tegnode_complete(
                asc_nodes_region_id=-1,
                rightneighbor=rightneighbor,
                leftneighbor=None,
                right_state=right_state,
                left_state=None
            )
        else:
            # 如果已存在，只补右邻和右邻状态
            nodes_complete[(x, y, step)].rightneighbor = rightneighbor
            nodes_complete[(x, y, step)].right_state = right_state

        # 2. 补右邻节点的左邻和左邻状态
        if rightneighbor is not None:
            x1, y1 = rightneighbor[0], rightneighbor[1]
            neighbor_key = (x1, y1, step)
            # 如果邻居节点还不存在，则创建，并设置leftneighbor/left_state
            if neighbor_key not in nodes_complete:
                # 从原nodes_new获取右邻的信息
                if neighbor_key in nodes_new:
                    raw_neighbor = nodes_new[neighbor_key]
                    # neighbor_groupid = raw_neighbor.asc_nodes_region_id
                    neighbor_rightneighbor = raw_neighbor.rightneighbor
                    neighbor_right_state = raw_neighbor.state
                else:
                    # 如果原nodes_new也没有，则降级为未知
                    neighbor_groupid = -1
                    neighbor_rightneighbor = None
                    neighbor_right_state = None

                nodes_complete[neighbor_key] = tegnode.tegnode_complete(
                    asc_nodes_region_id=-1,
                    rightneighbor=neighbor_rightneighbor,
                    leftneighbor=(x, y, step),
                    right_state=neighbor_right_state,
                    left_state=right_state
                )
            else:
                # 若存在，只补leftneighbor/left_state
                nodes_complete[neighbor_key].leftneighbor = (x, y, step)
                nodes_complete[neighbor_key].left_state = right_state

    return nodes_complete



        # print(f"坐标=({x},{y}), 时间={step}, 节点={node}")
        # 你可以用 node.rightneighbor, node.leftneighbor, node.state, node.importance 等属性




