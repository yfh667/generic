
import basicSa.utilis.matirx as  matirx
# 示例使用
# matrix = [
#     [1, 2, 3,4],
#     [4, 5, 6,7],
#     [7, 8, 9,10],
#     [10, 11, 12,13],
# ]
#
# satellitestart = 2
# # start_row = 2
# # start_col = 2
# result = matirx.extract_submatrix(matrix, satellitestart, satellitestart)
# print(result)  # 输出 [[5, 6], [8, 9]]

# 示例矩阵
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 遍历第1行（索引为0）
row_index = 1
for element in matirx.iterate_row(matrix, row_index):
    print(element)
