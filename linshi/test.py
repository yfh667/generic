import pandas as pd

def convert_excel_to_xyz_csv(excel_file_path, output_csv_path, date_str="2025-01-07"):
    """
    将原始GNSS Excel格式转换为标准 x,y,z,t 格式CSV。
    参数：
        excel_file_path: 原始Excel文件路径
        output_csv_path: 输出CSV文件路径
        date_str: 日期前缀（例如 '2025-01-01'），用于构造完整时间戳
    """
    df = pd.read_excel(excel_file_path, header=None)

    # 按列提取值（每5列循环一行）
    data = []
    for i in range(len(df)):
        x = df.iloc[i, 1]
        y = df.iloc[i, 3]
        z = df.iloc[i, 5]
        t = df.iloc[i, 6]
        full_time = f"{date_str}T{str(t).strip()}Z"
        data.append([full_time, x, y, z])

    # 构建新DataFrame
    new_df = pd.DataFrame(data, columns=["timestamp", "x", "y", "z"])
    new_df.to_csv(output_csv_path, index=False)
    print(f"✅ 已保存到: {output_csv_path}")
convert_excel_to_xyz_csv("E:\\研究生\\研究进展\\待整理\\千帆数据\\wang.xlsx", "E:\\研究生\\研究进展\\待整理\\千帆数据\\1.csv")
