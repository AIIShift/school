import pandas as pd


def time_3h_csv(today_date, today_date_H, today_date_ago, today_date_ago_1):

    # 加载CSV文件
    file_path = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '.csv'
    df = pd.read_csv(file_path)

    # 提取所需的列 (第2, 5, 6, 7列)
    df_selected = df.iloc[:, [1, 4, 5, 6]]

    # 添加列名
    df_selected.columns = ['time', 'longitude', 'latitude', 'thunderstorm']

    # 按time, latitude, longitude进行排序
    df_sorted = df_selected.sort_values(by=['time'])

    # 筛选第2297到41328行
    df_filtered = df_sorted.iloc[2296:41327]

    df_filtered = df_filtered.sort_values(by=['latitude', 'longitude', 'time'])


    # 保存处理后的CSV文件
    output_path = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '_time.csv'
    df_filtered.to_csv(output_path, index=False)

    print(f"山西地区grib雷达数据已转为csv")
