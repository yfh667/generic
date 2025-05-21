import random


def initialize_snap(col,N, P,T, port,setuptime,distinct_links):
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
    #T*N
    links = [[-1] * N for _ in range(T)]

  #  links = [-1] * N  # 初始化当前列的连接

    # 定义每个截面的可能的右邻居

    for t in range(T):
        for row in range(N):
            # first we need neglect the distinct
            if distinct_links[(col*N,row,t)] != -1:
                links[row][t] = distinct_links[(col*N,row,t)]
                continue
            # 可能的候选邻居
            candidates = []
            # 下一截面的相邻行(上、中、下)
            next_col = col + 1
            if t+setuptime < T:
                if next_col < P:
                    candidates.append((next_col * N , (row - 1) % N,t+setuptime))  # 下一行
                    candidates.append((next_col * N, (row ) % N, t + setuptime))  #中间
                    candidates.append((next_col * N, (row+1) % N, t + setuptime))  # 上一行
                    # 如果不是最后一列，还可以考虑下下一列的对应行
                    if next_col + 1 < P:
                        candidates.append(((next_col + 1) * N ,row, t + setuptime))

            # 过滤掉已经被占用的候选
            available = []

            for k in range(len(candidates)):
                if port[candidates[k]] == False:
                    # first it can't occupy port directly
                    available.append(candidates[k])

            if available:
                chosen = random.choice(available)
                links[(chosen[1]),chosen[2]] = chosen
                port[chosen] = 1  # 标记为已占用
               # right_link[chosen] = 1  # 标记为已占用

    return links


