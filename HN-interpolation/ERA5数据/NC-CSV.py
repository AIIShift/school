import xarray as xr
import pandas as pd
import numpy as np

# 1. 读取 NetCDF 文件
ds = xr.open_dataset("qiya.nc")

# 2. 提取变量
t2m = ds['t2m']     # 单位：K（开尔文）
sp = ds['sp']       # 单位：Pa
lat = ds['latitude']
lon = ds['longitude']
time = ds['valid_time']

# 3. 转换时间为 pandas 时间格式（UTC → 北京时间）
time_utc = pd.to_datetime(time.values, unit='s')
time_bj = time_utc + pd.Timedelta(hours=8)

# 4. 使用 xarray 的 stack 展开为平面表格
t2m_stacked = t2m.stack(points=("valid_time", "latitude", "longitude"))
sp_stacked = sp.stack(points=("valid_time", "latitude", "longitude"))

# 5. 转成 DataFrame
df = pd.DataFrame({
    "time_utc": time_utc.repeat(len(lat)*len(lon)),
    "time_bj": time_bj.repeat(len(lat)*len(lon)),
    "latitude": np.tile(np.repeat(lat.values, len(lon)), len(time)),
    "longitude": np.tile(lon.values, len(lat) * len(time)),
    "t2m_K": t2m_stacked.values,
    "sp_Pa": sp_stacked.values
})

# 6. 可选：温度从 K 转换为 °C
df["t2m_C"] = df["t2m_K"] - 273.15

# 7. 导出 CSV 文件
df.to_csv("era5_temperature_pressure_china.csv", index=False, encoding='utf-8-sig')

print("✅ CSV 文件已保存为 'era5_temperature_pressure_china.csv'")
