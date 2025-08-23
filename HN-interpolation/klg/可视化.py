import pandas as pd
import matplotlib.pyplot as plt

# 读取数据文件
data = pd.read_csv('pwv_kriging_results4.csv')

# 假设数据包含 'latitude', 'longitude', 和 'target_value' 列
# 你可以根据你的数据修改这些列名

# 绘制散点图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['Lon'], data['Lat'], c=data['PWV'], cmap='viridis', alpha=0.7)

# 添加标题和标签
plt.title('Target Value Visualization Based on Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 添加颜色条
plt.colorbar(scatter, label='Target Value')

# 显示图形
plt.show()
