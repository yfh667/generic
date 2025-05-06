import random


def initialize_links(N):
    """
    初始化行连接关系，返回一个列表：
    - 列表索引代表当前行号（0到N-1）
    - 列表值代表该行的右邻居行号

    规则：
    1. 每个行的右邻居只能从其候选行号（当前行、上一行、下一行）中选择
    2. 所有右邻居必须唯一
    3. 使用环形处理（第一行的上一行是最后一行）

    参数:
        N: 总行数
        P: 总列数（当前未使用，保留参数）
    """
    # 初始化结果列表（-1表示未分配）
    links = [-1] * N

    # 定义每行的候选右邻居行号（当前行、上一行、下一行）
    candidates = [{(i - 1) % N, i % N, (i + 1) % N} for i in range(N)]

    # 随机处理顺序
    processing_order = list(range(N))
    random.shuffle(processing_order)

    used_right = set()

    for current_row in processing_order:
        # 获取可用候选（排除已使用的行号）
        available = list(candidates[current_row] - used_right)

        if not available:
            # 处理冲突：随机替换一个已分配的行号
            conflict = list(candidates[current_row] & used_right)
            if conflict:
                chosen_conflict = random.choice(conflict)
                # 找到哪个行当前使用了这个冲突行号
                for row in range(N):
                    if links[row] == chosen_conflict:
                        # 为该行重新分配
                        row_available = list(candidates[row] - (used_right - {chosen_conflict}))
                        if row_available:
                            new_right = random.choice(row_available)
                            links[row] = new_right
                            used_right.remove(chosen_conflict)
                            used_right.add(new_right)
                            break

            # 重新计算可用候选
            available = list(candidates[current_row] - used_right)

        # 如果仍然没有可用选项，从候选集中随机选择（可能违反唯一性）
        if not available:
            available = list(candidates[current_row])

        # 随机选择右邻居
        chosen = random.choice(available)
        links[current_row] = chosen
        used_right.add(chosen)

    return links


def initialize_links2(N, col, P, right_link):
    """
    初始化当前列的右邻居连接

    参数:
        N: 每列的行数
        col: 当前列号(0开始)
        P: 总列数
        right_link: 记录所有节点右连接的数组

    返回:
        当前列的右连接列表(长度为N)
    """
    links = [-1] * N  # 初始化当前列的连接

    # 定义每行可能的右邻居
    for row in range(N):
        # 可能的候选邻居
        candidates = []

        # 下一列的相邻行(上、中、下)
        next_col = col + 1
        if next_col < P:
            candidates.append(next_col * N + (row - 1) % N)  # 上一行
            candidates.append(next_col * N + row)  # 同一行
            candidates.append(next_col * N + (row + 1) % N)  # 下一行

            # 如果不是最后一列，还可以考虑下下一列的对应行
            if next_col + 1 < P:
                candidates.append((next_col + 1) * N + row)

        # 过滤掉已经被占用的候选
        available = [n for n in candidates if right_link[n] == -1]

        # available = []
        # for n in candidates:
        #     if right_link[n] == -1:
        #         available.append(n)

        # 随机选择一个可用的
        if available:
            chosen = random.choice(available)
            links[row] = chosen
            right_link[chosen] = 1  # 标记为已占用

    return links
#
# # 测试（N=4行）
# for _ in range(5):
#     result = initialize_links(4, 2)
#     print(f"连接关系: {result} 唯一性验证: {len(set(result)) == 4}")
