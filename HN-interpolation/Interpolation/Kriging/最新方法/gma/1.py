import numpy as np
import pandas as pd
from gma import smc
# 假设你的原始数据已读入一个 DataFrame，列名分别是：
#   station_id  — 站点编号（字符串）
#   lon         — 经度 (浮点)
#   lat         — 纬度 (浮点)
#   value       — 测量值 (浮点，估算点若无观测则可填 NaN 或者跳过)
df = pd.read_csv("D:\PyCharm\pythonProject\HN-interpolation\Interpolation\Kriging\HNHB-2025060300.csv", dtype={"Station_Id_C": str})

# 标记“实际观测站”：ID 全为数字
mask_actual = df["Station_Id_C"].str.match(r"^\d+$")
actual_df = df[mask_actual].reset_index(drop=True)

results = []

for _, row in actual_df.iterrows():
    sid = row["Station_Id_C"]
    lon_pt = row["Lon"]
    lat_pt = row["Lat"]
    true_val = row["PWV"]

    # 构造训练集：排除当前这个真实站点，但保留所有估算点和其他真实站
    train_df = df[df["Station_Id_C"] != sid].dropna(subset=["PWV"])

    # 提取训练集的 XYZ
    lons_train = train_df["Lon"].values
    lats_train = train_df["Lat"].values
    vals_train = train_df["PWV"].values



    # 使用 GMA 克里金插值方法
    KD = smc.Interpolate.Kriging(np.column_stack((lons_train, lats_train)), vals_train,
                                 Resolution=0.01,  # 插值结果的空间分辨率
                                 Boundary=[108.41, 24.64, 116.13, 33.30],  # 边界
                                 VariogramModel='Spherical',  # 半变异函数模型
                                 KMethod='Ordinary',  # 克里金方法
                                 InProjection='EPSG:4326')  # 输入坐标系

    # 计算在给定点的插值结果
    pred_val = KD.Interpolate([lon_pt, lat_pt])  # 获取插值值

    # 计算方差（如果需要的话，你可以使用类似于 `KD.variance()` 的方法，具体取决于库的功能）
    var_val = np.nan  # 如果没有提供方差，设置为 NaN

    results.append({
        "station_id": sid,
        "lon": lon_pt,
        "lat": lat_pt,
        "true": true_val,
        "pred": pred_val,
        "error": pred_val - true_val,
        "variance": var_val
    })

# 汇总结果并保存
results_df = pd.DataFrame(results)
results_df.to_csv("loo_cv_results_with_GMA.csv", index=False)

# （可选）计算总体误差统计
rmse = np.sqrt(np.mean(results_df["error"] ** 2))
mae = np.mean(np.abs(results_df["error"]))
print(f"LOOCV 完成，RMSE = {rmse:.3f}，MAE = {mae:.3f}")
print("详细结果已保存为 loo_cv_results_with_GMA.csv")

# 3.1 将插值结果读取为栅格数据集
ItData = io.ReadArrayAsDataSet(KD.Data, Projection=4326, GeoTransform=KD.Transform)
