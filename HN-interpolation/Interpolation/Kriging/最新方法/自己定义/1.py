import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree


# 假设你定义了计算变差函数和高斯协方差的函数
def calculate_variogram(coords, values, max_lag=5, lag_step=0.5):
    dist_matrix = cdist(coords, coords)
    upper_tri = np.triu_indices_from(dist_matrix, k=1)
    dists = dist_matrix[upper_tri]
    value_diffs = (values[upper_tri[0]] - values[upper_tri[1]]) ** 2

    bins = np.arange(0, max_lag + lag_step, lag_step)
    bin_indices = np.digitize(dists, bins)
    semivariances = []

    for i in range(1, len(bins)):
        in_bin = (bin_indices == i)
        if np.sum(in_bin) > 0:
            semivariance = np.mean(value_diffs[in_bin]) / 2
            semivariances.append(semivariance)
        else:
            semivariances.append(np.nan)

    valid_idx = ~np.isnan(semivariances)
    if not np.any(valid_idx):
        return np.var(values), np.mean(dists) / 3

    valid_semivars = np.array(semivariances)[valid_idx]
    sill = np.max(valid_semivars)
    range_val = bins[np.argmax(valid_semivars)] if np.max(valid_semivars) > 0 else np.mean(dists) / 2

    return sill, range_val


def gaussian_cov(distance, sill, range_):
    with np.errstate(divide='ignore', invalid='ignore'):
        zero_dist = distance == 0
        cov = np.zeros_like(distance)
        cov[~zero_dist] = sill * np.exp(-(distance[~zero_dist] ** 2) / (2 * range_ ** 2))
        cov[zero_dist] = sill
    return cov


# 克里金插值方法
def ordinary_kriging(stations, values, grid_points, sill, range_, neighborhood_size):
    n = len(stations)
    m = len(grid_points)

    # 初始化结果数组
    kriged = np.zeros(m)

    # 计算站点间的距离矩阵
    station_dists = cdist(stations, stations)

    # 构建全局协方差矩阵
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = gaussian_cov(station_dists, sill, range_)

    # 添加无偏约束
    K[:n, n] = 1
    K[n, :n] = 1
    K[n, n] = 0

    # 添加正则化项
    K[:n, :n] += np.eye(n) * 1e-8

    # 构建KD树用于快速邻居搜索
    tree = cKDTree(stations)

    # 对每个网格点进行插值
    for i in range(m):
        dists, idxs = tree.query(grid_points[i], k=min(neighborhood_size, n))
        local_stations = stations[idxs]
        n_local = len(local_stations)

        # 跳过没有邻居的点
        if n_local == 0:
            kriged[i] = np.mean(values)  # 使用全局平均值
            continue

        # 创建本地数据子集
        local_values = values[idxs]

        # 计算本地距离矩阵
        local_dists = cdist(local_stations, local_stations)

        # 构建本地协方差矩阵
        local_K = np.zeros((n_local + 1, n_local + 1))
        local_K[:n_local, :n_local] = gaussian_cov(local_dists, sill, range_)

        # 添加无偏约束
        local_K[:n_local, n_local] = 1
        local_K[n_local, :n_local] = 1
        local_K[n_local, n_local] = 0

        # 添加正则化项
        local_K[:n_local, :n_local] += np.eye(n_local) * 1e-8

        # 计算当前网格点到邻居站点的距离
        grid_dists = cdist([grid_points[i]], local_stations).ravel()

        # 构建右侧向量
        k_vec = np.zeros(n_local + 1)
        k_vec[:n_local] = gaussian_cov(grid_dists, sill, range_)  # 协方差
        k_vec[n_local] = 1  # 无偏约束

        try:
            # 求解权重
            weights = lsqr(local_K, k_vec)[0]

            # 提取目标权重
            target_weights = weights[:n_local]

            # 处理负权重 - 截断并重新归一化
            target_weights = np.clip(target_weights, 0, None)
            target_weights_sum = np.sum(target_weights)

            if target_weights_sum > 1e-6:
                target_weights /= target_weights_sum
            else:
                # 如果所有权重都很小，使用反距离加权
                inv_dist = 1 / (grid_dists + 1e-6)
                target_weights = inv_dist / np.sum(inv_dist)

            # 计算估计值
            kriged[i] = np.dot(target_weights, local_values)

        except Exception as e:
            # 如果求解失败，使用反距离加权
            inv_dist = 1 / (grid_dists + 1e-6)
            weights = inv_dist / np.sum(inv_dist)
            kriged[i] = np.dot(weights, local_values)

    return kriged


# 读取数据并进行 LOOCV
df = pd.read_csv(r"D:\PyCharm\pythonProject\HN-interpolation\Interpolation\Kriging\HNHB-2025060300.csv", dtype={"Station_Id_C": str})

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

    # 计算变差函数参数
    sill, range_ = calculate_variogram(np.column_stack((lons_train, lats_train)), vals_train)

    # 在留出的点上做预测
    predicted_val = ordinary_kriging(
        np.column_stack((lons_train, lats_train)), vals_train,
        np.array([[lon_pt, lat_pt]]), sill, range_, neighborhood_size=30
    )
    pred_val = float(predicted_val[0])
    var_val = np.nan  # 对于自定义方法，方差计算可以根据需要进一步定义

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
results_df.to_csv("loo_cv_results.csv", index=False)

# （可选）计算总体误差统计
rmse = np.sqrt(np.mean(results_df["error"] ** 2))
mae = np.mean(np.abs(results_df["error"]))
print(f"LOOCV 完成，RMSE = {rmse:.3f}，MAE = {mae:.3f}")
print("详细结果已保存为 loo_cv_results.csv")
