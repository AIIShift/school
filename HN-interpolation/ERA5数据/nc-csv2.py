import xarray as xr
import pandas as pd
import numpy as np

# 1. 读取湿度 NetCDF 文件
ds = xr.open_dataset("shidu.nc")

# 2. 提取变量
rh = ds['r']  # 相对湿度（%）
lat = ds['latitude']
lon = ds['longitude']
time = ds['valid_time']
plev = ds['pressure_level']

# 3. 转换时间（UTC → 北京时间）
time_utc = pd.to_datetime(time.values, unit='s')
time_bj = time_utc + pd.Timedelta(hours=8)

# 4. 因为只有一个 pressure_level（1 层），可以去掉这个维度
rh = rh.squeeze(dim='pressure_level')  # 从 (time, 1, lat, lon) 变为 (time, lat, lon)

# 5. 展开为平面表格
rh_stacked = rh.stack(points=("valid_time", "latitude", "longitude"))

# 6. 创建 DataFrame
df = pd.DataFrame({
    "time_utc": time_utc.repeat(len(lat)*len(lon)),
    "time_bj": time_bj.repeat(len(lat)*len(lon)),
    "latitude": np.tile(np.repeat(lat.values, len(lon)), len(time)),
    "longitude": np.tile(lon.values, len(lat) * len(time)),
    "rh_percent": rh_stacked.values
})

# 7. 去除缺失值（NaN）
df = df.dropna(subset=["rh_percent"])

# 8. 导出 CSV
df.to_csv("era5_relative_humidity.csv", index=False, encoding='utf-8-sig')

print("✅ 相对湿度数据已保存为 'era5_relative_humidity.csv'")
