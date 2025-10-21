def gethop(path,tempnexthop):
    tempnexthop = {}
    for i in range(len(path)):
        if i == 0:
            # 第一个元素，只关心下一个跳点
            tempnexthop[path[i]] = path[i + 1]
        elif i == len(path) - 1:
            # 最后一个元素，只关心前一个跳点
            tempnexthop[path[i]] = path[i - 1]
        else:
            # 中间的元素，关心前后两个跳点
            tempnexthop[path[i]] = [path[i - 1], path[i + 1]]
    return tempnexthop


def gethops(paths):
    tempnexthop = {}
    for path in paths:
        for i in range(len(path)):
            if i == 0:
                # 第一个元素，只关心下一个跳点
                if path[i] not in tempnexthop:
                    tempnexthop[path[i]] = set()
                tempnexthop[path[i]].add(path[i + 1])
            elif i == len(path) - 1:
                # 最后一个元素，只关心前一个跳点
                if path[i] not in tempnexthop:
                    tempnexthop[path[i]] = set()
                tempnexthop[path[i]].add(path[i - 1])
            else:
                # 中间的元素，关心前后两个跳点
                if path[i] not in tempnexthop:
                    tempnexthop[path[i]] = set()
                tempnexthop[path[i]].add(path[i - 1])
                tempnexthop[path[i]].add(path[i + 1])

    # 将集合转换为列表以便更好地显示或处理
    for key in tempnexthop:
        tempnexthop[key] = list(tempnexthop[key])

    return tempnexthop






def guinode(nodes, tempnexthop):
    data_to_send = []
    for i in range(len(nodes)):
        tempx, tempy, tempz = nodes[i]
        # 使用 get 方法，默认值为 []
        tempnexthopnode = tempnexthop.get(i, [])
        data_to_send.append({"x": tempx, "y": tempy, "z": tempz, "nexthop": tempnexthopnode})
    return data_to_send


#test
paths = []
path1 = [1,5,6,0]
path2 = [2,7,6,0]
paths.append(path1)
paths.append(path2)



print(gethops(paths))


# #
# nodes = [(1, 2, 3), (4, 5, 6), (7, 8, 9),(7, 8, 9),(7, 8, 9),(7, 8, 9),(7, 8, 9),(7, 8, 9),(7, 8, 9),(7, 8, 9)]
# data_to_send = guinode(nodes,gethop(path))
#
# print(data_to_send)
