import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# 读取数据

data = pd.read_csv('RH.csv')

# 提取原始的经纬度、风速、风向、降水量等数据
latitudes = data['latitude'].unique()
longitudes = data['longitude'].unique()

# 生成新的经纬度网格，分辨率为0.01度
new_latitudes = np.arange(latitudes.min(), latitudes.max() + 0.05, 0.05)
new_longitudes = np.arange(longitudes.min(), longitudes.max() + 0.05, 0.05)
new_grid_lat, new_grid_lon = np.meshgrid(new_latitudes, new_longitudes)

# 获取所有时间点
times = data['time'].unique()
new_data = []

# 对每个时间点分别进行插值
for time in times:
    time_data = data[data['time'] == time]

    # 提取原始经纬度点和变量数据
    points = np.array([time_data['latitude'], time_data['longitude']]).T
    rh = time_data['rh'].values
    sp = time_data['sp'].values
    t2m = time_data['t2m'].values

    # 使用双线性插值 (linear)
    grid_rh = griddata(points, rh, (new_grid_lat, new_grid_lon), method='linear')
    grid_sp = griddata(points, sp, (new_grid_lat, new_grid_lon), method='linear')
    grid_t2m = griddata(points, t2m, (new_grid_lat, new_grid_lon), method='linear')

    # 将插值后的数据存入新的DataFrame
    for i in range(new_grid_lat.shape[0]):
        for j in range(new_grid_lon.shape[1]):
            new_data.append({
                'latitude': new_grid_lat[i, j],
                'longitude': new_grid_lon[i, j],
                'time': time,
                'rh': grid_rh[i, j],
                'sp': grid_sp[i, j],
                't2m': grid_t2m[i, j]
            })

# 转换为DataFrame
new_data_df = pd.DataFrame(new_data)

# 对插值后的数据按 latitude 和 longitude 进行排序
new_data_df_sorted = new_data_df.sort_values(by=['latitude', 'longitude'])

# 对 wind_speed, wind_direction 和 precipitation 列保留小数点后两位
new_data_df_sorted['rh'] = new_data_df_sorted['rh'].round(2)
new_data_df_sorted['sp'] = new_data_df_sorted['sp'].round(2)
new_data_df_sorted['t2m'] = new_data_df_sorted['t2m'].round(2)
new_data_df_sorted['latitude'] = new_data_df_sorted['latitude'].round(2)
new_data_df_sorted['longitude'] = new_data_df_sorted['longitude'].round(2)

# 删除任何包含空值的行
new_data_df_sorted = new_data_df_sorted.dropna()

# 保存处理后的结果到CSV文件
new_data_df_sorted.to_csv('shuj-0.05.csv', index=False)

print("风速、风向、降水的空间插值和排序处理完成")
