import os
import pandas as pd
import netCDF4 as nc
import numpy as np
from datetime import datetime

file_path = r'D:\PyCharm\pythonProject\HN-interpolation\Interpolation\Kriging\总\pwv_kriging_results.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

df['time'] = '2025-06-03 00:00:00'
df = df.rename(columns={'Lon': 'longitude', 'Lat': 'latitude', 'Interpolated_PWV': 'PWV'})

# 将 'time' 列转换为 datetime 对象
df['time'] = pd.to_datetime(df['time'])

# 获取唯一的纬度、经度和时间值
latitudes = df['latitude'].unique()
longitudes = df['longitude'].unique()
times = df['time'].unique()

# 将 numpy.datetime64 转换为 datetime 对象，并减去 15 年
times = [pd.Timestamp(t).to_pydatetime() - pd.DateOffset(years=15) for t in times]

out_file = 'D:\\PyCharm\\pythonProject\\HN-interpolation\\Interpolation\\Kriging\\NC文件\\OK-HNHB-2025060300.nc'
nc_file = nc.Dataset(out_file, 'w', format='NETCDF4')

# 定义维度
nc_file.createDimension('latitude', len(latitudes))
nc_file.createDimension('longitude', len(longitudes))
nc_file.createDimension('time', len(times))

# 创建维度变量
latitudes_var = nc_file.createVariable('latitude', 'f4', ('latitude',))
longitudes_var = nc_file.createVariable('longitude', 'f4', ('longitude',))
times_var = nc_file.createVariable('time', 'f4', ('time',))

# 将值赋给维度变量
latitudes_var[:] = latitudes
longitudes_var[:] = longitudes
times_var[:] = nc.date2num(times, units='hours since 1970-01-01', calendar='gregorian')

# 创建数据变量
PWV_var = nc_file.createVariable('PWV', 'f4', ('time', 'latitude', 'longitude'), fill_value=np.nan)

# 将数据分配给变量
for i, time_val in enumerate(times):
    time_data = df[df['time'] == pd.Timestamp(time_val) + pd.DateOffset(years=15)]  # 恢复原始时间进行匹配
    PWV_var[i, :, :] = time_data.pivot(index='latitude', columns='longitude', values='PWV').values

# 添加全局属性
nc_file.description = '从CSV转换为NetCDF格式'
nc_file.history = '创建于 ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
nc_file.source = 'Python脚本'

# 关闭NetCDF 文件
nc_file.close()

print('PWV数据已成功从CSV转换为NetCDF文件')
