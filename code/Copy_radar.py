import shutil
import os


def copy_radar(today_date, today_date_H, today_date_ago, today_date_ago_1):

    # 定义源文件夹路径
    source_folder = 'E:\\1107\\'  # 生成的两天前的文件夹名字
    # 定义目标文件夹路径
    destination_folder = 'radar/dataset/grib_radar/' + today_date  # 替换为实际的目标文件夹路径

    print(source_folder)
    print(destination_folder)
    # # 要复制的文件名列表
    file_names = ["GRAPESGFS_RACR_1_" + today_date_ago_1 + "18_NEHE_1_2.grib2"]  # 替换为实际需要复制的文件名

    # 确保目标文件夹存在，不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 复制文件
    for file_name in file_names:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)

        # 检查文件是否存在，然后复制
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            print(f"已复制 {file_name} 到 {destination_folder}")
        else:
            print(f"文件 {file_name} 不存在于 {source_folder}")
    print('雷达grib文件已复制完成')
