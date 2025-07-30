def cyclic_width(ys, N):
    if not ys:
        return 0
    ys = sorted(set(ys))
    gaps = []
    for i in range(1, len(ys)):
        gaps.append(ys[i] - ys[i-1])
    # 跨越环首尾
    gaps.append((ys[0] + N) - ys[-1])
    maxgap = max(gaps)
    return N - maxgap + 1


def get_satellite_block_info(sats, P, N):
    xs = [sid // N for sid in sats]
    ys = [sid % N for sid in sats]

    left = min(xs)
    right = max(xs)
    block_info = [(None, None), (None, None)]

    # 判断是否两块
    if left == 0 and right == P - 1:
        # --------左侧块--------
        leftgroup = []
        x = left
        while x in xs:
            leftgroup.append(x)
            x += 1
        # 统计左块包络
        leftys = [sid % N for sid in sats if (sid // N) in leftgroup]
        left_width = cyclic_width(leftys, N)
        left_length = leftgroup[-1] - leftgroup[0] + 1 if leftgroup else 0
        block_info[0] = (left_length, left_width)

        # --------右侧块--------
        rightgroup = []
        x = right
        while x in xs:
            rightgroup.append(x)
            x -= 1
        rightys = [sid % N for sid in sats if (sid // N) in rightgroup]
        right_width = cyclic_width(rightys, N)
        right_length = rightgroup[0] - rightgroup[-1] + 1 if rightgroup else 0
        block_info[1] = (right_length, right_width)

    else:
        # 只有一块，取所有点
        w = max(xs) - min(xs) + 1
        h = cyclic_width(ys, N)
        block_info[0] = (w, h)
        block_info[1] = (None, None)

    return block_info  # [(块1长, 宽), (块2长, 宽)]


