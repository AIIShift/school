import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\数据处理\HNHB-2025060320.csv')  # 请替换为你的文件路径
pwv_data = df['PWV']  # 假设你的CSV文件中包含名为 'PWV' 的列

# 基于均值和标准差的异常值检测
# 计算均值和标准差
mean_pwv = pwv_data.mean()
std_pwv = pwv_data.std()

# 设置一个标准差阈值（例如 3个标准差）
threshold = 3

# 检测异常值：如果数据点超出了均值±3个标准差范围，视为异常值
outliers_mean_std = pwv_data[(pwv_data > mean_pwv + threshold * std_pwv) | (pwv_data < mean_pwv - threshold * std_pwv)]

# 基于四分位数（IQR）方法的异常值检测
# 计算四分位数（Q1, Q3）和四分位间距（IQR）
Q1 = pwv_data.quantile(0.25)
Q3 = pwv_data.quantile(0.75)
IQR = Q3 - Q1

# 设置异常值标准
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 检测异常值：如果数据点超出了Q1 - 1.5*IQR 或 Q3 + 1.5*IQR范围，视为异常值
outliers_IQR = pwv_data[(pwv_data < lower_bound) | (pwv_data > upper_bound)]

# 输出基于均值和标准差的异常值检测结果
print('基于均值和标准差的异常值检测')
print(f'均值: {mean_pwv}, 标准差: {std_pwv}')
print(f'检测到的异常值:  个数：{len(outliers_mean_std)} \n{outliers_mean_std}')

# 输出基于四分位数（IQR）方法的异常值检测结果
print('\n基于四分位数（IQR）方法的异常值检测')
print(f'第一四分位数: {Q1}, 第三四分位数: {Q3}, 四分位间距: {IQR}')
print(f'检测到的异常值:  个数： {len(outliers_IQR)}\n{outliers_IQR}')

# 取两个方法检测到的异常值的并集
outliers_combined = set(outliers_mean_std).union(set(outliers_IQR))

# 从原数据中剔除异常值
df_cleaned = df[~df['PWV'].isin(outliers_combined)]

# 输出剔除异常值后的结果
print('\n剔除异常值后的数据:')
print(df_cleaned)

# 可选：保存剔除异常值后的数据到新CSV文件
# df_cleaned.to_csv('cleaned_data.csv', index=False)
