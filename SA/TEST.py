import numpy as np
import random
import math
import torch


class TopologyOptimizer:
    def __init__(self, grid_size=3, seq_length=10, temp=1000, cooling_rate=0.95, iterations=1000):
        """
        拓扑序列优化器
        :param grid_size: 拓扑网格大小 (n x n)
        :param seq_length: 拓扑序列长度
        :param temp: 初始温度
        :param cooling_rate: 退火率
        :param iterations: 迭代次数
        """
        self.n = grid_size
        self.seq_len = seq_length
        self.temp = temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations

        # 初始化拓扑序列: [seq_len, n, n]
        # 存储每个节点选择的右邻居坐标(0-2)或-1(断开)
        self.sequence = self.initialize_sequence()

        # 预计算每个节点的允许邻居
        self.neighbor_map = self.build_neighbor_map()

    def initialize_sequence(self):
        """初始化所有节点为断开状态"""
        return np.full((self.seq_len, self.n, self.n), -1, dtype=int)

    def build_neighbor_map(self):
        """构建每个节点的允许右邻居映射"""
        # 每个节点只能连接右侧三个方向之一
        # 方向编码: 0=右上, 1=正右, 2=右下
        # 返回值: (i,j) -> 可用的邻居方向列表
        neighbor_map = {}
        for i in range(self.n):
            for j in range(self.n):
                available = []
                if i > 0 and j < self.n - 1:  # 右上
                    available.append(0)
                if j < self.n - 1:  # 正右
                    available.append(1)
                if i < self.n - 1 and j < self.n - 1:  # 右下
                    available.append(2)
                neighbor_map[(i, j)] = available
        return neighbor_map

    def get_neighbor_coords(self, i, j, direction):
        """根据方向获取邻居坐标"""
        if direction == 0:  # 右上
            return (i - 1, j + 1)
        elif direction == 1:  # 正右
            return (i, j + 1)
        elif direction == 2:  # 右下
            return (i + 1, j + 1)
        return None  # 断开

    def evaluate_topology(self, topology):
        """拓扑评分函数 (示例实现)"""
        # 实际应用中替换为您的评分逻辑
        score = 0
        for i in range(self.n):
            for j in range(self.n):
                # 基本规则: 有连接加分，连接稳定性加分
                if topology[i, j] != -1:
                    score += 1  # 连接存在
                    # 获取邻居坐标
                    ni, nj = self.get_neighbor_coords(i, j, topology[i, j])
                    # 双向连接检查
                    if 0 <= ni < self.n and 0 <= nj < self.n:
                        # 对称连接额外加分
                        if topology[ni, nj] != -1:
                            nn_i, nn_j = self.get_neighbor_coords(ni, nj, topology[ni, nj])
                            if (nn_i, nn_j) == (i, j):
                                score += 2  # 双向连接奖励
        return score

    def total_score(self):
        """计算整个序列的总分"""
        return sum(self.evaluate_topology(self.sequence[t]) for t in range(self.seq_len))

    def neighbor_change_valid(self, t, i, j, new_state):
        """检查邻居变化是否满足连续约束"""
        # 检查三元组约束: (t-1, t, t+1)
        for check_t in [t - 1, t + 1]:
            if 0 <= check_t < self.seq_len:
                # 检查是否出现跳跃变化
                prev_state = self.sequence[check_t, i, j]
                if prev_state != -1 and new_state != -1 and prev_state != new_state:
                    # 需要中间状态断开
                    if self.sequence[t, i, j] != -1:
                        return False
        return True

    def generate_neighbor_solution(self):
        """生成邻居解 (模拟退火的核心操作)"""
        # 随机选择要修改的位置和节点
        t = random.randint(0, self.seq_len - 1)
        i, j = random.randint(0, self.n - 1), random.randint(0, self.n - 1)

        # 保存旧状态和受影响的拓扑索引
        old_state = self.sequence[t, i, j]
        affected_positions = {t}

        # 获取可能的邻居状态 (包括断开)
        possible_states = self.neighbor_map.get((i, j), [])
        possible_states.append(-1)  # 添加断开状态

        # 过滤掉当前状态
        possible_states = [s for s in possible_states if s != old_state]
        if not possible_states:
            return None, None, None

        # 随机选择新状态
        new_state = random.choice(possible_states)

        # 检查并处理连续约束
        if not self.neighbor_change_valid(t, i, j, new_state):
            # 强制断开中间状态以满足约束
            if t > 0:
                self.sequence[t - 1, i, j] = -1
                affected_positions.add(t - 1)
            if t < self.seq_len - 1:
                self.sequence[t + 1, i, j] = -1
                affected_positions.add(t + 1)

        # 应用新状态
        self.sequence[t, i, j] = new_state
        affected_positions.add(t)

        return old_state, new_state, affected_positions

    def revert_change(self, t, i, j, old_state, affected_positions):
        """回退更改"""
        for pos in affected_positions:
            # 只恢复被修改的位置
            if pos == t:
                self.sequence[t, i, j] = old_state
            else:
                # 恢复中间状态 (需实际实现时记录原始值)
                pass  # 简化实现

    def simulated_annealing(self):
        """执行模拟退火优化"""
        current_score = self.total_score()
        best_score = current_score
        best_sequence = self.sequence.copy()

        for iter in range(self.iterations):
            # 生成新解
            old_state, new_state, affected = self.generate_neighbor_solution()
            if old_state is None:
                continue

            # 计算新分数
            new_score = self.total_score()
            delta_score = new_score - current_score

            # 决定是否接受新解
            if delta_score > 0 or math.exp(delta_score / self.temp) > random.random():
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = self.sequence.copy()
            else:
                # 回退到之前的状态
                t, i, j = self.find_modified_position(old_state, new_state)
                if t is not None:
                    self.revert_change(t, i, j, old_state, affected)

            # 降温
            self.temp *= self.cooling_rate

            # 打印进度
            if iter % 100 == 0:
                print(f"Iteration {iter}: Temp={self.temp:.2f} Score={current_score} Best={best_score}")

        # 恢复最佳序列
        self.sequence = best_sequence
        return best_sequence, best_score

    def find_modified_position(self, old_state, new_state):
        """定位修改的位置 (简化实现)"""
        # 实际应用中需要更精确的定位
        for t in range(self.seq_len):
            for i in range(self.n):
                for j in range(self.n):
                    if self.sequence[t, i, j] == new_state:
                        return t, i, j
        return None, None, None


# 使用示例
if __name__ == "__main__":
    optimizer = TopologyOptimizer(
        grid_size=10,  # 3x3网格
        seq_length=10,  # 10个位置的序列
        temp=1000,  # 初始温度
        cooling_rate=0.95,  # 退火率
        iterations=1000  # 迭代次数
    )

    # 执行优化
    best_sequence, best_score = optimizer.simulated_annealing()

    print(f"\nOptimization Complete!")
    print(f"Best Score: {best_score}")
    print("Final Sequence:")
    for t, topology in enumerate(best_sequence):
        print(f"Position {t}:")
        print(topology)
