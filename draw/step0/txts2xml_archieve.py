import os
from lxml import etree  # pip install lxml
import os
from concurrent.futures import ThreadPoolExecutor
from lxml import etree

def convert_txt_to_xml(txt_file_path, xml_file_path):
    """
    将每个 .txt 文件转换为 .xml 格式
    txt 文件格式：每行：time x y z
    转换为：<sat id="xxx"><p t="time" x="x" y="y" z="z" /></sat>
    """
    try:
        # 获取文件名作为卫星ID（文件名应该是数字）
        sat_id = os.path.splitext(os.path.basename(txt_file_path))[0]

        # 创建 XML 文件结构
        root = etree.Element("sat", id=sat_id)
        tree = etree.ElementTree(root)

        with open(txt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    t = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])

                    # 创建 <p> 元素
                    p = etree.Element("p", t=str(t), x=str(x), y=str(y), z=str(z))
                    root.append(p)  # 将 <p> 元素附加到 <sat> 元素
                except Exception as e:
                    print(f"跳过错误行: {line}，错误信息: {e}")

        # 写入 XML 文件
        with open(xml_file_path, "wb") as xml_file:
            tree.write(xml_file, encoding="utf-8", pretty_print=True)

        print(f"已成功转换: {txt_file_path} -> {xml_file_path}")

    except Exception as e:
        print(f"转换文件 {txt_file_path} 时发生错误：{e}")





def convert_all_txt_to_xml(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    txt_file_paths = [os.path.join(input_dir, f) for f in txt_files]
    xml_file_paths = [os.path.join(output_dir, os.path.splitext(f)[0] + ".xml") for f in txt_files]

    # 使用 ThreadPoolExecutor 来实现多线程并行处理
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        # 将文件列表和对应的输出路径打包成元组并传给 convert_txt_to_xml
        executor.map(lambda args: convert_txt_to_xml(*args), zip(txt_file_paths, xml_file_paths))


# 示例用法：将 C:/usrspace/mywork/generic/data/648qianfan 中的所有 .txt 文件转换为 .xml 文件
#C:\usrspace\mywork\data\rawposition\648qianfan1d_2
convert_all_txt_to_xml(
    input_dir="C:/usrspace/mywork/data/rawposition/648qianfan1d_2",
    output_dir="C:/usrspace/mywork/data/rawposition/648qianfan1d_2_xml"
)
