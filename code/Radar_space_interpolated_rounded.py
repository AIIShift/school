import numpy as np
import pandas as pd
from scipy.interpolate import griddata



def radar_space_interpolated_rounded(today_date, today_date_H, today_date_ago, today_date_ago_1):

    # 读取数据
    file_path = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '_radar_time_selected.csv'
    data = pd.read_csv(file_path)

    # 提取原始的经纬度、风速、风向、降水量等数据
    latitudes = data['latitude'].unique()
    longitudes = data['longitude'].unique()

    # 生成新的经纬度网格，分辨率为0.01度
    new_latitudes = np.arange(latitudes.min(), latitudes.max() + 0.01, 0.01)
    new_longitudes = np.arange(longitudes.min(), longitudes.max() + 0.01, 0.01)
    new_grid_lat, new_grid_lon = np.meshgrid(new_latitudes, new_longitudes)

    # 获取所有时间点
    times = data['time'].unique()
    new_data = []

    # 对每个时间点分别进行插值
    for time in times:
        time_data = data[data['time'] == time]

        # 提取原始经纬度点和变量数据
        points = np.array([time_data['latitude'], time_data['longitude']]).T
        thunderstorm = time_data['thunderstorm'].values

        # 插值并跳过错误的时间点
        try:
            # 使用双线性插值 (linear)
            grid_thunderstorm = griddata(points, thunderstorm, (new_grid_lat, new_grid_lon), method='linear')

            # 将插值后的数据存入新的DataFrame
            for i in range(new_grid_lat.shape[0]):
                for j in range(new_grid_lon.shape[1]):
                    new_data.append({
                        'latitude': new_grid_lat[i, j],
                        'longitude': new_grid_lon[i, j],
                        'time': time,
                        'thunderstorm': grid_thunderstorm[i, j],
                    })
        except Exception as e:
            print(f"跳过时间点 {time}，原因：{e}")
            continue

    # 转换为DataFrame
    new_data_df = pd.DataFrame(new_data)

    # 对插值后的数据按 latitude 和 longitude 进行排序
    new_data_df_sorted = new_data_df.sort_values(by=['latitude', 'longitude'])

    # 对 thunderstorm 列和经纬度保留小数点后两位
    new_data_df_sorted['thunderstorm'] = new_data_df_sorted['thunderstorm'].round(2)
    new_data_df_sorted['latitude'] = new_data_df_sorted['latitude'].round(2)
    new_data_df_sorted['longitude'] = new_data_df_sorted['longitude'].round(2)

    # 转换时间格式，去掉秒数
    new_data_df_sorted['time'] = pd.to_datetime(new_data_df_sorted['time']).dt.strftime('%Y-%m-%d %H:%M')

    # 保存处理后的结果到CSV文件
    output_file_path = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '_radar_space_interpolated_rounded.csv'
    new_data_df_sorted.to_csv(output_file_path, index=False)

    print("雷达数据的空间插值和排序处理完成")

