import pandas as pd

# 读取 CSV 文件
df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\20250603\Z_UPAR_C_HNHB_20250603000000_P_GNSSMET_PWV_HOR.csv')

# # 查看具体某一列是否有空值
# column_name = 'VAP'  # 替换成你需要检查的列名
# empty_values_count = df[column_name].isna().sum()
# print(f"列 '{column_name}' 中有 {empty_values_count} 个空值")


# 检查每一列的空值数量
empty_values_count = df.isna().sum()
# 显示每一列的空值数量
print(empty_values_count)


# 看看空值数据
# 筛选出包含空值的行
rows_with_missing_values = df[df.isna().any(axis=1)]
# 显示包含空值的所有行
print(rows_with_missing_values)


# # 计算每一列的最大值和最小值
# max_values = df.max()
# min_values = df.min()
# # 将最大值和最小值合并为一个数据框
# summary = pd.DataFrame({'Max': max_values, 'Min': min_values})
# # 展示结果
# print(summary)