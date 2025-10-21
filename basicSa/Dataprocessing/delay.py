import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def parse_and_plot(xml_path, flow_id):
    """
    解析XML文件并绘制指定FlowId的AverageDelay随时间变化的曲线。

    参数:
    - xml_path: str，XML文件路径
    - flow_id: str，要提取的FlowId
    """
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 存储时间和对应的延迟
    times = []
    delays = []

    # 遍历TimeData块
    for time_data in root.findall("TimeData"):
        time = time_data.find("Time").text.strip(" s")  # 提取时间并去掉单位
        for flow_data in time_data.findall("FlowData"):
            if flow_data.find("FlowId").text.strip() == str(flow_id):
                delay = flow_data.find("AverageDelay").text.strip()
                if delay == "N/A":
                    delay = None  # 无效值处理
                else:
                    delay = float(delay.strip(" s"))  # 转换为浮点型秒数
                times.append(float(time))
                delays.append(delay)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(times, delays, marker="o", label=f"FlowId: {flow_id}")
    plt.title(f"Average Delay vs Time for FlowId {flow_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Delay (s)")
    plt.grid(True)
    plt.legend()
    plt.show()


# 示例调用
xml_path = "/home/yfh/Desktop/Data/test_output.xml"  # 替换为你的XML文件路径
flow_id = 2  # 指定要绘制的FlowId
parse_and_plot(xml_path, flow_id)
