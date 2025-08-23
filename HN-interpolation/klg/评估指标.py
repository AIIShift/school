import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 读取CSV文件
file_path = r'D:\PyCharm\pythonProject\HN-interpolation\klg\pwv_bilinear_cv_results.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 假设你的CSV文件中有两列：'predicted'（预测值列）和'target'（目标值列）
predicted = data['True_PWV'].values  # 预测值列
target = data['Predicted_PWV'].values  # 目标值列

# 计算相关系数 (COR)
correlation = np.corrcoef(predicted, target)[0, 1]
print(f"相关系数 (COR): {correlation:.4f}")

# 计算平均误差 (ME)
mean_error = np.mean(predicted - target)
print(f"平均误差 (ME): {mean_error:.4f}")

# 计算平均绝对误差 (MAE)
mae = mean_absolute_error(target, predicted)
print(f"平均绝对误差 (MAE): {mae:.4f}")

# 计算均方根误差 (RMSE)
rmse = np.sqrt(mean_squared_error(target, predicted))
print(f"均方根误差 (RMSE): {rmse:.4f}")
