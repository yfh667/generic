# 初始化 XML 写入类

import  basicSa.wirtetoxml.wirtetoxml as wirtetoxml
import  basicSa.utilis.Node as Node
writer = wirtetoxml.WriteToXml('/home/yfh/Desktop/Data/prefigure.xml',1,2)

# 模拟数据
for current_time in range(1, 4):  # 模拟三个时间片
    nodes = [
        Node.Node(-2178655.168, 4388872.376, 4069502.147, [-1, -1, -1, -1, -1]),
        Node.Node(-1178655.168, 3388872.376, 3069502.147, [0, 1, -1, 0, -1]),
        Node.Node(-3178655.168, 5388872.376, 5069502.147, [1, 0, -1, 1, -1]),
    ]

    # 调用写入方法


    writer.writeToXml(current_time, nodes)
