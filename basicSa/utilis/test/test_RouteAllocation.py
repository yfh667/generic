# 引入必要的模块
import basicSa.utilis.Node as Nodepy
import basicSa.utilis.RouteAllocation as routeAllocation

# 测试用例入口
if __name__ == '__main__':
    # 假设我们设置了以下路径
    path = [0, 12, 32, 2]
    srcid = 0
    dstid = 2

    # 创建一个 Node 实例作为源卫星
    srcsat = Nodepy.Node()

    # 预设 srcsat 的 matrix（路由表）
    # 假设初始情况是如下所示
    srcsat.matrix = [
        [12, 0],  # 假设第一个条目已经设置
        [-1, -1], [-1, -1], [-1, -1], [-1, -1],
        [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
        [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]
    ]

    # 打印初始路由表
    print("Initial srcsat matrix:")
    for row in srcsat.matrix:
        print(row)

    # 执行 GetLinkPort 测试函数
    print("\nRunning RouteAllocation function...")
    routeAllocation.RouteAllocation(path, srcid, dstid, srcsat)

    # 打印更新后的路由表
    print("\nUpdated srcsat matrix:")
    for row in srcsat.matrix:
        print(row)
