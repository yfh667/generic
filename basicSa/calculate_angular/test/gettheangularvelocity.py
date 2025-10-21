
import basicSa.calculate_angular.calculate_angular as calculate_angular
import basicSa.utilis.readdata as readdata
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 读取卫星初始位置
    # satpath = "/home/yfh/Desktop/Data/linsats"
    # satangle = 20
    #
    # satsnodes = readdata.readsats_multi(satpath, satangle)  # 获取卫星数据
    # # print(stationsnodes[0])
    #
    # # wo alse need add the trackangle and the RAAN for the basicSa
    # trackangle = 89
    # RAAN_deg = 10.2  # 输入为度数
    # N = 20
    # P = 2
    #


    # # # 读取卫星初始位置
    satpath = "/home/yfh/Desktop/Data/onehun"
    satangle = 20

    satsnodes = readdata.readsats_multi(satpath, satangle)  # 获取卫星数据
    # print(stationsnodes[0])

    # wo alse need add the trackangle and the RAAN for the basicSa
    trackangle = 89
    RAAN_deg = 18  # 输入为度数
    N = 10
    P = 10



    for node in satsnodes:
        if node[0].nodeid ==1:
            Node1 = node
            print("find node1")
        elif node[0].nodeid ==N+1:
            Node2 = node
            print("find node2")

    # sometimes ,the RAAN always equre to

    # for i in range(P):
    #     for j in range(N):
    #         number = i * N + j
    #         RAAN = 10.2 * (i + 1)
    #         readdata.add_number_RAAN(satsnodes[number], RAAN, trackangle, number)
            # satsnodes[number].RAAN=10.2*(i+1)
            # satsnodes[number].number = trackangle
    # print(satsnodes[2])





   # Node1 =  satsnodes[0]
  #  Node2 = satsnodes[20]
    ws = []
    for i in range(len(Node1)-1):

        xa1 = Node1[i].x
        ya1 = Node1[i].y
        za1 = Node1[i].z
        xa2 =Node1[i+1].x
        ya2 = Node1[i+1].y
        za2 = Node1[i+1].z
        xb1 = Node2[i].x
        yb1 = Node2[i].y
        zb1 = Node2[i].z
        xb2 = Node2[i+1].x
        yb2 =  Node2[i+1].y
        zb2 =  Node2[i+1].z
        timestep = 1
        #RAAN_deg = 10.2  # 输入为度数
        beta_deg = 89.0  # 输入为度数

        params = (xa1, ya1, za1, xa2, ya2, za2,
                  xb1, yb1, zb1, xb2, yb2, zb2,
                  beta_deg, timestep, RAAN_deg)
        w = calculate_angular.Get_angular_velocity(params)

        ws.append(w)

       # print(w)



    # 提取所有角速度的z分量（第三个分量）
    w_z = [w[2] if len(w)>=3 else 0 for w in ws]  # 安全处理数据长度不足的情况

    # 生成时间轴（假设每个间隔为1个时间单位）
    time_steps = list(range(len(w_z)))
    import matplotlib.pyplot as plt

    # 提取所有角速度的z分量（第三个分量）
    w_z = [w[2] if len(w)>=3 else 0 for w in ws]  # 安全处理数据长度不足的情况

    # 生成时间轴（假设每个间隔为1个时间单位）
    time_steps = list(range(len(w_z)))

    # 修改后的绘图代码段
    plt.figure(figsize=(10, 6))

    # 使用兼容性样式设置
    try:
        plt.style.use('seaborn')  # 现代版本的正确名称
    except:
        plt.style.use('classic')  # 回退到经典样式

    # 绘制曲线
    plt.plot(time_steps, w_z,
             color='#2c7bb6',
             linestyle='-',
             linewidth=2,
             marker='o',
             markersize=8,
             markerfacecolor='#d7191c',
             markeredgecolor='black',
             label='Z Component')

    # 专业图表格式设置
    plt.xlabel('Time Step', fontsize=12, fontweight='bold')
    plt.ylabel('Angular Velocity (Z) [deg/s]', fontsize=12, fontweight='bold')
    plt.title('Temporal Evolution of Angular Velocity Z Component',
              fontsize=14, fontweight='bold', pad=20)

    # 设置刻度精度
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # 添加网格和图示
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, shadow=True)

    # 自动调整坐标范围
    plt.xlim(min(time_steps)-0.5, max(time_steps)+0.5)
    plt.ylim(min(w_z)*1.1, max(w_z)*1.1)

    plt.tight_layout()
    plt.show()
