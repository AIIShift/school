import datetime
import time
import traceback
from data_process.Deletee import deletee
from data_process.Timees import timees
from data_process.Calculated_uv_thunderstorm_gale import calculated_uv_thunderstorm_gale
from data_process.Merge_windspeed_direction_precipitation_radar import merge_windspeed_direction_precipitation_radar
from radar.code.Radar_main import radar_main
from data_process.Nc_csv import nc_csv
from data_process.Csv_nc import csv_nc
from data_process.Space_interpolated_rounded import space_interpolated_rounded
from data_process.Calculated_wind_direction import calculated_wind_direction
from model_predicted.Precipitation_predicted import precipitation_predict
from data_process.Copy_grib import copy_grib
from data_process.Time_selected import time_selected
from data_process.Grib_nc import grib_nc
from data_process.Time_interpolated_rounded import time_interpolated_rounded
from data_process.Merge_windspeed_direction_precipitation import merge_windspeed_direction_precipitation
from data_process.Wait_for_next_hou import wait_until_five_minutes_past
from model_predicted.Windspeed_predict import windspeed_predict

def safe_execute(func, *args):
    """执行函数并捕获任何异常"""
    try:
        func(*args)
    except Exception as e:
        print(f"执行函数 {func.__name__} 时发生错误: {e}")
        traceback.print_exc()  # 打印详细的错误信息

def main():
    # 设置开始时间和结束时间
    start_date = datetime.datetime(2024, 6, 8, 0, 0)
    end_date = datetime.datetime(2024, 6, 10, 0, 0)
    current_date = start_date

    while current_date < end_date:
        # 调用 generate_time_strings 并获取自定义时间格式
        today_date, today_date_H, today_date_ago, today_date_ago_1 = timees(current_date)
        print(today_date)
        print(today_date_H)
        print(today_date_ago)
        print(today_date_ago_1)
        print("====================================================")

        # 将自定义时间传递给各处理函数并使用 safe_execute 执行
        safe_execute(deletee)
        safe_execute(copy_grib, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(grib_nc, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(nc_csv, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(time_interpolated_rounded, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(time_selected, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(windspeed_predict, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(precipitation_predict, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(calculated_wind_direction, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(merge_windspeed_direction_precipitation, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(space_interpolated_rounded, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(radar_main, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(merge_windspeed_direction_precipitation_radar, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(calculated_uv_thunderstorm_gale, today_date, today_date_H, today_date_ago, today_date_ago_1)
        safe_execute(csv_nc, today_date, today_date_H, today_date_ago, today_date_ago_1)

        # 成功执行一轮后将时间增加一小时
        current_date += datetime.timedelta(hours=1)
        time.sleep(60)  # 可选：短暂等待


if __name__ == "__main__":
    main()
    print("====================")
