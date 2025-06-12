# genaric2/tegnode.py
class tegnode:
    def __init__(self, asc_nodes_flag, state, rightneighbor=None, leftneighbor=None):
        self.asc_nodes_flag = asc_nodes_flag
        self._rightneighbor = rightneighbor
        self._leftneighbor = leftneighbor
        self._state = state

    @property
    def state(self):
        return self._state

    @property
    def rightneighbor(self):
        return self._rightneighbor

    @property
    def leftneighbor(self):
        return self._leftneighbor

    def __repr__(self):
        return f"tegnode(asc_nodes_flag={self.asc_nodes_flag}, " \
               f"rightneighbor={self._rightneighbor}, " \
               f"leftneighbor={self._leftneighbor}, " \
               f"state={self._state})"

    # 这些方法将由ChromosomeManager调用
    def _set_rightneighbor(self, coord):
        self._rightneighbor = coord

    def _set_leftneighbor(self, coord):
        self._leftneighbor = coord

    def _set_state(self, state):
        self._state = state


# genaric2/chromosome_manager.py
class Chromosome:
    def __init__(self):
        self.chromosome = {}

    def add_node(self, coord, node):
        """添加节点到染色体"""
        self.chromosome[coord] = node

    def set_node_state(self, coord, new_state):
        """设置节点状态并自动更新邻居关系"""
        node = self.chromosome[coord]
        old_state = node.state

        # 清理旧状态的邻居关系
        if old_state == 0:  # 如果是建链状态
            # 通知原右邻居断开左邻居关系
            if node.rightneighbor:
                right_node = self.chromosome[node.rightneighbor]
                if right_node.leftneighbor == coord:
                    right_node._set_leftneighbor(None)

        # 更新状态
        node._set_state(new_state)

        # 设置新状态的邻居关系
        if new_state == 0:  # 建链状态
            if node.rightneighbor:
                # 通知新右邻居更新左邻居
                right_node = self.chromosome[node.rightneighbor]
                right_node._set_leftneighbor(coord)
        elif new_state == -1:  # 自由状态
            # 清除所有邻居关系
            if node.rightneighbor:
                # 通知右邻居清除左邻居
                right_node = self.chromosome[node.rightneighbor]
                if right_node.leftneighbor == coord:
                    right_node._set_leftneighbor(None)
            node._set_rightneighbor(None)
            node._set_leftneighbor(None)

    def set_right_neighbor(self, coord, new_right_coord):
        """设置右邻居并自动更新对称关系"""
        node = self.chromosome[coord]

        # 清理旧关系
        if node.rightneighbor:
            old_right_node = self.chromosome[node.rightneighbor]
            if old_right_node.leftneighbor == coord:
                old_right_node._set_leftneighbor(None)

        # 设置新关系
        node._set_rightneighbor(new_right_coord)

        # 更新新邻居的左邻居
        if new_right_coord:
            right_node = self.chromosome[new_right_coord]
            right_node._set_leftneighbor(coord)

    def __getitem__(self, coord):
        """通过坐标访问节点"""
        return self.chromosome.get(coord)

    def __contains__(self, coord):
        """检查坐标是否存在"""
        return coord in self.chromosome

    def items(self):
        """返回所有节点"""
        return self.chromosome.items()
