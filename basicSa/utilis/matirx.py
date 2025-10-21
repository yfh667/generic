import numpy as np

def extract_submatrix(matrix, start_row, start_col):
    """
    提取矩阵中从指定行和列开始的子矩阵。

    参数:
    - matrix: 二维列表或NumPy数组，表示输入的矩阵。
    - start_row: int，子矩阵的起始行索引（从0开始）。
    - start_col: int，子矩阵的起始列索引（从0开始）。

    返回:
    - 二维列表，提取的子矩阵。
    """
    if isinstance(matrix, list):
        matrix = np.array(matrix)  # 转换为NumPy数组以便于切片操作
    if start_row < 0 or start_col < 0:
        raise ValueError("起始行和列的索引不能为负数")
    if start_row >= matrix.shape[0] or start_col >= matrix.shape[1]:
        raise ValueError("起始行或列超出矩阵范围")

    # 提取从start_row和start_col开始的子矩阵
    submatrix = matrix[start_row:, start_col:]
    return submatrix.tolist()


def iterate_row(matrix, row_index):
    """
    遍历矩阵的第row_index行，返回列索引和值的元组。
    """
    if row_index < 0 or row_index >= len(matrix):
        raise ValueError("行索引超出矩阵范围")
    return enumerate(matrix[row_index])
