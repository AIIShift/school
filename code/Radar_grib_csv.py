import os
import subprocess


def radar_grib_csv(today_date, today_date_H, today_date_ago, today_date_ago_1):
    # 定义 GRIB 文件所在的文件夹路径
    output_folder1 = 'radar/dataset/grib_radar_shanxi/' + today_date
    # 确保输出文件夹存在
    os.makedirs(output_folder1, exist_ok=True)

    output_folder2 = 'radar/dataset/csv/' + today_date + '/' + today_date_H
    # 确保输出文件夹存在
    os.makedirs(output_folder2, exist_ok=True)

    # 设置wgrib2的路径（如果wgrib2已经加入系统路径，直接使用'wgrib2'即可）
    wgrib2_path = 'D:\\wgrib2\\wgrib2.exe'  # 如果没有加入环境变量，需要使用完整路径，例如 'C:/path_to_wgrib2/wgrib2.exe'

    # 输入GRIB2文件路径
    grib2_file_Asian = 'radar/dataset/grib_radar/' + today_date + '/' + 'GRAPESGFS_RACR_1_' + today_date_ago_1 + '18_NEHE_1_2.grib2'

    # 输出山西的GRIB2文件路径
    grib2_file_shanxi = 'radar/dataset/grib_radar_shanxi/' + today_date + '/' + today_date + '.grib2'

    # 构建命令，将GRIB2文件转换为NetCDF文件
    command1 = [wgrib2_path, grib2_file_Asian, '-small_grib', '110:115', '34:41', grib2_file_shanxi]
    print(command1)
    # 使用subprocess运行命令
    try:
        result = subprocess.run(command1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Conversion successful!")
        print("Output:", result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Error during conversion!")
        print("Error message:", e.stderr.decode())
    print("雷达grib文件已处理并转换为山西地区grib。")

    # 输出csv文件路径
    csv_file = 'radar/dataset/csv/' + today_date + '/' + today_date_H + '/' + today_date_H + '.csv'

    # 构建命令，将GRIB2文件转换为NetCDF文件
    command2 = [wgrib2_path, grib2_file_shanxi, '-csv', csv_file]
    print(command2)
    # 使用subprocess运行命令
    try:
        result = subprocess.run(command2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Conversion successful!")
        print("Output:", result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Error during conversion!")
        print("Error message:", e.stderr.decode())

    print("山西雷达grib文件已处理并转换为csv文件。")




#  这部分代码主要是通过wgrib来提取grib文件，因为wgrib只能在系统终端运行，第一部分代码是将grib文件筛选到山西地区，第二部分是将山西地区的grib转为csv文件。
