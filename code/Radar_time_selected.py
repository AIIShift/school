from datetime import datetime
import pandas as pd



def radar_time_selected(today_date, today_date_H, today_date_ago, today_date_ago_1):

    # 将 today_date_H 转换为 datetime 对象
    today_date_H_dt = datetime.strptime(today_date_H, "%Y%m%d%H")
    current_hour = today_date_H_dt.hour

    # 定义每个小时的行数区间（正常计数，从1开始）
    hour_ranges = {
        0: [1, 55104],
        1: [2297, 57400],
        2: [4593, 59696],
        3: [6889, 61992],
        4: [9185, 64288],
        5: [11481, 66584],
        6: [13777, 68880],
        7: [16073, 71176],
        8: [18369, 73472],
        9: [20665, 75768],
        10: [22961, 78064],
        11: [25257, 80360],
        12: [27553, 82656],
        13: [29849, 84952],
        14: [32145, 87248],
        15: [34441, 89544],
        16: [36737, 91840],
        17: [39033, 94136],
        18: [41329, 96432],
        19: [43625, 98728],
        20: [45921, 101024],
        21: [48217, 103320],
        22: [50513, 105616],
        23: [52809, 107912]
    }


    # 读取CSV文件
    csv_file = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '_radar_time_interpolated_rounded.csv'
    df = pd.read_csv(csv_file)

    # 获取当前小时的行范围，注意将行号转换为0索引
    start_row, end_row = hour_ranges[current_hour]
    start_row -= 1  # 转换为0索引
    end_row -= 1    # 转换为0索引

    # 根据当前小时数筛选数据
    filtered_df = df.iloc[start_row:end_row+1]

    # 对 'latitude' 进行升序排序，再对 'longitude' 进行升序排序
    filtered_df = filtered_df.sort_values(by=['latitude', 'longitude'], ascending=[True, True])

    # 输出或保存筛选后的数据
    output_file = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '_radar_time_selected.csv'
    filtered_df.to_csv(output_file, index=False)
    print(f"已成功筛选{today_date_H}时间的雷达数据csv")

