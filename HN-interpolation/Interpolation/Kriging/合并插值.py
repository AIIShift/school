import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


## 空间最近邻（基于经纬度）
# 加载数据
target_df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\HNHB-2025060300.csv')
source_df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\Interpolation\Kriging\结果\pwv_kriging_results.csv')

# 重命名列
target_df.rename(columns={'Datetime': 'time'}, inplace=True)
source_df.rename(columns={'longitude': 'Lon', 'latitude': 'Lat'}, inplace=True)
# 添加一列时间
source_df['time'] = '2025/6/3 0:00'

# 统一时间格式
target_df['time'] = pd.to_datetime(target_df['time'])
source_df['time'] = pd.to_datetime(source_df['time'], format='%Y/%m/%d %H:%M')

# 当时间格式是2023-01-01 08:00:00,
source_df['time'] = pd.to_datetime(source_df['time'], format='%Y-%m-%d %H:%M:%S')
source_df['formatted_time'] = source_df['time'].dt.strftime('%Y/%-m/%-d %-H:%M')

print(target_df)
print(source_df)

# 提取源数据的特征列，不包括时间、纬度和经度
source_features = source_df.columns.difference(['time', 'Lat', 'Lon'])

# 创建一个空的DataFrame来存放插值后的数据
interpolated_df = pd.DataFrame()

# 按时间分组
for time_point in target_df['time'].unique():
    # 在目标数据和源数据中找到相同时间点的数据
    target_time_df = target_df[target_df['time'] == time_point]
    source_time_df = source_df[source_df['time'] == time_point]

    # 如果在源数据中找不到完全匹配的时间点，则寻找最接近的时间点
    # if source_time_df.empty:
    #     time_diff = np.abs(source_df['time'] - time_point)
    #     min_diff_index = time_diff.idxmin()
    #     closest_time_point = source_df.loc[min_diff_index, 'time']
    #     source_time_df = source_df[source_df['time'] == closest_time_point]
    #
    # print(f"Processing time point: {time_point}")
    # print("Target time DataFrame Head:\n", target_time_df.head())
    # print("Source time DataFrame Head:\n", source_time_df.head())

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

# 检查最终的DataFrame
print("Interpolated DataFrame Head:\n", interpolated_df.head())

# # 重新排序列
# final_columns = ['time', 'Lat', 'Lon', 'r', 'q', 't', 'w', 'vo']
# interpolated_df = interpolated_df[final_columns]
#
# 更改时间格式
interpolated_df['time'] = interpolated_df['time'].dt.strftime('%Y/%m/%d %H:%M')

# 将结果保存到新的CSV文件
interpolated_df.to_csv(r'D:\PyCharm\pythonProject\HN-interpolation\Interpolation\Kriging\结果\站点上-湖南湖北-克里金插值结果对比.csv', index=False, encoding='utf-8-sig')
