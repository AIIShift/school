import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载CSV文件
file_path = r'D:\PyCharm\pythonProject\HN-interpolation\HNHB-2025060300.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 2. 选择特征变量和目标变量
# 假设目标变量为 'target_column'，特征变量为 'feature1', 'feature2', 'feature3', 'feature4'
features = ['PRS', 'TEM', 'RHU', 'VAP', 'Alti']
target = 'PWV'

# 3. 计算相关性系数
correlation_matrix = data[features + [target]].corr()

# 4. 提取目标变量与特征变量的相关性系数
target_correlation = correlation_matrix[target].drop(target)  # 删除目标变量本身

# 5. 对相关性系数进行排名（按绝对值排序）
ranked_correlation = target_correlation.abs().sort_values(ascending=False)

# 6. 打印排名结果
print("排名：")
print(ranked_correlation)

# 7. 可视化相关性矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
