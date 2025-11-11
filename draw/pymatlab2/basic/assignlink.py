UNSET = -1


def _is_triplet(val) -> bool:
    return isinstance(val, tuple) and len(val) == 3


def _ensure(nodes, key):
    node = nodes.get(key)
    if node is None:
        node = tegnode.tegnode_new(
            asc_nodes_region_id=-1,
            rightneighbor=None,
            leftneighbor=None,
            right_state=-1,
            left_state=-1, node_type=-1,

            timelast=-1,
        )  # 你的类默认字段均为 -1
        nodes[key] = node
    return node


def assign_Link(start, end, nodes, state, type, timelast):
    """
    把 start 的 rightneighbor 连接到 end，并保持双向一致。
    缺节点时自动创建默认节点；未设置邻居(-1/None)时不做清理。
    """

    # 0) 确保两端节点存在
    s = _ensure(nodes, start)
    e = _ensure(nodes, end)

    # 如果本来就连的是同一端，只更新状态即可
    if _is_triplet(s.rightneighbor) and s.rightneighbor == end:
        s.right_state = state
        s.type = type
        s.timelast = timelast
        e.leftneighbor = start
        e.left_state = state
        return

    # 1) 断开 start 原来的右邻（若存在且确实指回 start）
    rn = s.rightneighbor
    if _is_triplet(rn):
        rn_node = nodes.get(rn)
        if rn_node and rn_node.leftneighbor == start:
            rn_node.leftneighbor = None
            rn_node.left_state = -1
            # 如需一并清理，可按需解除注释：
            # rn_node.type     = UNSET
            # rn_node.timelast = UNSET

    # 2) 断开 end 原来的左邻（若存在且确实指向 end）
    ln = e.leftneighbor
    if _is_triplet(ln):
        ln_node = nodes.get(ln)
        if ln_node and ln_node.rightneighbor == end:
            ln_node.rightneighbor = None
            ln_node.right_state = UNSET
            ln_node.node_type = UNSET
            ln_node.timelast = UNSET

    # 3) 建立 start→end 与 end←start
    s.rightneighbor = end
    s.right_state = state
    s.node_type = type
    s.timelast = timelast

    e.leftneighbor = start
    e.left_state = state
