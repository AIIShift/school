import pandas as pd



def radar_time_interpolated_rounded(today_date, today_date_H, today_date_ago, today_date_ago_1):

    # 读取CSV文件
    df = pd.read_csv('radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '_time.csv')

    # 将时间列转换为datetime格式
    df['time'] = pd.to_datetime(df['time'])

    # 获取除经纬度和时间外的其他变量列名
    value_vars = df.columns.difference(['latitude', 'longitude', 'time'])

    # 按经纬度分组
    grouped = df.groupby(['latitude', 'longitude'])

    # 存储插值后的结果
    interpolated_list = []

    # 遍历每个分组进行插值
    for name, group in grouped:
        # 将时间列设置为索引
        group = group.set_index('time')
        # 创建新的以1小时为间隔的时间索引
        new_time_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq='H')
        # 重新索引数据框以包含新的时间点
        group = group.reindex(new_time_index)
        # 对其他变量进行时间插值
        group[value_vars] = group[value_vars].interpolate(method='time')
        # 填充经纬度列
        group['latitude'] = group['latitude'].fillna(method='ffill')
        group['longitude'] = group['longitude'].fillna(method='ffill')
        # 重置索引并重命名时间列
        group = group.reset_index().rename(columns={'index': 'time'})
        # 将插值后的结果添加到列表
        interpolated_list.append(group)

    # 将所有分组的结果合并为一个DataFrame
    interpolated_df = pd.concat(interpolated_list).reset_index(drop=True)

    # 对插值后的数据按时间列进行排序
    interpolated_df = interpolated_df.sort_values(by=['time', 'latitude', 'longitude']).reset_index(drop=True)

    # 指定需要保留小数点后两位的列
    columns_to_round = ['thunderstorm']  # 替换为实际需要保留两位小数的列名

    # 对指定的列进行保留小数点后两位
    interpolated_df[columns_to_round] = interpolated_df[columns_to_round].round(2)

    # 将排序后的数据保存为 CSV 文件
    output_file = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '_radar_time_interpolated_rounded.csv'

    # 将处理后的数据保存为新的CSV文件
    interpolated_df.to_csv(output_file, index=False)

    print("雷达数据的时间插值、小数点处理和时间排序完成")


