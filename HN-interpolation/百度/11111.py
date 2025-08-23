import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


lon_min, lon_max = 108.86, 114.30
lat_min, lat_max = 24.64, 30.08

# 计算按分辨率0.01生成的经纬度网格
grid_lon = np.arange(lon_min, lon_max, 0.01)
grid_lat = np.arange(lat_min, lat_max, 0.01)

file_path = r'D:\PyCharm\pythonProject\HN-interpolation\百度\train.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')
lons = df["Lon"].values
lats = df["Lat"].values
data = df["PWV"].values

OK = OrdinaryKriging(lons, lats, data, variogram_model='exponential',nlags=6)
z1, ss1 = OK.execute('grid', grid_lon, grid_lat)

#转换成网格
xgrid, ygrid = np.meshgrid(grid_lon, grid_lat)
#将插值网格数据整理
df_grid =pd.DataFrame(dict(Lon=xgrid.flatten(),Lat=ygrid.flatten()))
# 添加插值结果列（展平二维网格）
df_grid["Interpolated_PWV"] = z1.flatten()

# 现在df_grid包含三列：long, lat, Krig_gaussian
print(df_grid)
df_grid.to_csv('结果.csv', index=False, encoding='utf-8-sig')



## 空间最近邻（基于经纬度）
# 加载数据
target_df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\百度\test.csv')
source_df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\百度\结果.csv')

# 添加一列时间
source_df['time'] = '2025/6/3 0:00'

# 统一时间格式
target_df['time'] = pd.to_datetime(target_df['time'])
source_df['time'] = pd.to_datetime(source_df['time'])


# 提取源数据的特征列，不包括时间、纬度和经度
source_features = source_df.columns.difference(['time', 'Lat', 'Lon'])

# 创建一个空的DataFrame来存放插值后的数据
interpolated_df = pd.DataFrame()

# 按时间分组
for time_point in target_df['time'].unique():
    # 在目标数据和源数据中找到相同时间点的数据
    target_time_df = target_df[target_df['time'] == time_point]
    source_time_df = source_df[source_df['time'] == time_point]

    if not source_time_df.empty:
        # 提取源数据的经纬度
        source_coords = source_time_df[['Lat', 'Lon']].values

        # 构建KD树用于快速空间搜索
        tree = cKDTree(source_coords)


        # 定义函数以找到最近邻并获取相应的数据行
        def get_nearest_data(row):
            lat, lon = row['Lat'], row['Lon']
            dist, idx = tree.query([lat, lon], k=1)
            return source_time_df.iloc[idx][source_features].values


        # 应用最近邻搜索并将数据附加到目标DataFrame
        interpolated_data = target_time_df.apply(get_nearest_data, axis=1)

        # 将结果列表转换为DataFrame
        interpolated_df_time = pd.DataFrame(interpolated_data.tolist(), columns=source_features)

        # 将时间和插值数据连接起来
        interpolated_df_time = pd.concat([target_time_df.reset_index(drop=True), interpolated_df_time], axis=1)

        # 将插值后的数据添加到最终的DataFrame
        interpolated_df = pd.concat([interpolated_df, interpolated_df_time], ignore_index=True)

# 将结果保存到新的CSV文件
interpolated_df.to_csv(r'D:\PyCharm\pythonProject\HN-interpolation\百度\test结果.csv', index=False, encoding='utf-8-sig')



# 读取CSV文件
file_path = r'D:\PyCharm\pythonProject\HN-interpolation\百度\test结果.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 假设你的CSV文件中有两列：'Predicted_PWV'（预测值列）和'True_PWV'（目标值列）
predicted = data['True_PWV'].values  # 预测值列
target = data['Interpolated_PWV'].values  # 目标值列

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

# 计算R² (决定系数)
r2 = r2_score(target, predicted)
print(f"R²: {r2:.4f}")

