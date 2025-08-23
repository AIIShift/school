from ShanXi_windspeed_predicted.radar.code.Copy_radar import copy_radar
from ShanXi_windspeed_predicted.radar.code.Radar_grib_csv import radar_grib_csv
from ShanXi_windspeed_predicted.radar.code.Radar_space_interpolated_rounded import radar_space_interpolated_rounded
from ShanXi_windspeed_predicted.radar.code.Radar_time_interpolated import radar_time_interpolated_rounded
from ShanXi_windspeed_predicted.radar.code.Radar_time_selected import radar_time_selected
from ShanXi_windspeed_predicted.radar.code.Time_3h_csv import time_3h_csv


def radar_main(today_date, today_date_H, today_date_ago, today_date_ago_1):
    copy_radar(today_date, today_date_H, today_date_ago, today_date_ago_1)
    radar_grib_csv(today_date, today_date_H, today_date_ago, today_date_ago_1)
    time_3h_csv(today_date, today_date_H, today_date_ago, today_date_ago_1)
    # dd_day()  # 部署的时候这个不需要
    radar_time_interpolated_rounded(today_date, today_date_H, today_date_ago, today_date_ago_1)
    radar_time_selected(today_date, today_date_H, today_date_ago, today_date_ago_1)
    radar_space_interpolated_rounded(today_date, today_date_H, today_date_ago, today_date_ago_1)
