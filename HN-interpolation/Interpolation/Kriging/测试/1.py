import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import solve, lstsq
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_variogram(coords, values, max_lag=5, lag_step=0.5, model='exponential'):
    """
    变差函数计算，使用指数模型拟合
    返回: (块金值, 基台值, 变程)
    """
    # 计算所有点对之间的距离和半方差
    dist_matrix = cdist(coords, coords)
    upper_tri = np.triu_indices_from(dist_matrix, k=1)
    dists = dist_matrix[upper_tri]
    value_diffs = (values[upper_tri[0]] - values[upper_tri[1]]) ** 2
    gammas = value_diffs / 2

    # 分组计算经验半方差
    bins = np.arange(0, max_lag + lag_step, lag_step)
    bin_indices = np.digitize(dists, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    semivariances = []
    bin_counts = []

    for i in range(1, len(bins)):
        in_bin = (bin_indices == i)
        if np.sum(in_bin) > 5:  # 需要足够点对
            semivariance = np.mean(gammas[in_bin])
            semivariances.append(semivariance)
            bin_counts.append(np.sum(in_bin))
        else:
            semivariances.append(np.nan)
            bin_counts.append(0)

    # 过滤无效值
    valid_idx = ~np.isnan(semivariances)
    bin_centers = bin_centers[valid_idx]
    semivariances = np.array(semivariances)[valid_idx]
    bin_counts = np.array(bin_counts)[valid_idx]

    if len(semivariances) == 0:
        return 0.0, np.var(values), np.mean(dists) / 3

    # 定义指数模型变差函数
    def exponential_model(h, nugget, sill, range_):
        return nugget + (sill - nugget) * (1 - np.exp(-h / range_))

    # 初始参数估计
    nugget_guess = np.min(semivariances)
    sill_guess = np.max(semivariances)
    range_guess = bin_centers[np.argmax(semivariances >= 0.95 * sill_guess)] if sill_guess > 0 else np.max(
        bin_centers) / 2

    # 拟合模型
    try:
        popt, _ = curve_fit(
            exponential_model, bin_centers, semivariances,
            p0=[nugget_guess, sill_guess, range_guess],
            bounds=([0, 0, 0], [sill_guess, np.inf, np.max(bin_centers) * 2]),
            maxfev=5000
        )
        nugget, sill, range_ = popt
    except:
        # 拟合失败时使用启发式值
        nugget = semivariances[0]
        sill = np.max(semivariances)
        range_ = bin_centers[np.argmax(semivariances)] if len(semivariances) > 0 else np.mean(dists) / 3

    return nugget, sill, range_


def covariance_function(distance, nugget, sill, range_, model='exponential'):
    """指数模型的协方差函数"""
    # 处理零距离
    zero_dist = distance == 0
    partial_sill = sill - nugget

    # 初始化协方差
    cov = np.zeros_like(distance)

    # 非零距离计算
    mask = ~zero_dist
    d = distance[mask]

    # 指数模型
    cov[mask] = partial_sill * np.exp(-d / range_)

    # 添加块金效应（零距离时协方差等于基台值）
    cov[zero_dist] = sill
    return cov


def ordinary_kriging(stations, values, grid_points, nugget, sill, range_,
                     max_distance=10, min_neighbors=5):
    """普通克里格插值函数（使用指数模型）"""
    n = len(stations)
    m = len(grid_points)
    kriged = np.zeros(m)

    # 构建KD树用于邻居搜索
    tree = cKDTree(stations)

    # 对每个网格点进行插值
    for i in range(m):
        # 动态距离内的邻居
        neighbors = tree.query_ball_point(grid_points[i], max_distance)
        if len(neighbors) < min_neighbors:
            # 扩大搜索范围
            _, neighbors = tree.query(grid_points[i], k=min_neighbors)
            neighbors = neighbors if isinstance(neighbors, np.ndarray) else np.array(neighbors)

        local_stations = stations[neighbors]
        local_values = values[neighbors]
        n_local = len(local_stations)

        if n_local == 0:
            kriged[i] = np.mean(values)
            continue

        # 计算本地距离矩阵
        local_dists = cdist(local_stations, local_stations)

        # 构建本地协方差矩阵
        local_K = np.zeros((n_local + 1, n_local + 1))
        local_K[:n_local, :n_local] = covariance_function(
            local_dists, nugget, sill, range_, 'exponential'
        )

        # 添加无偏约束
        local_K[:n_local, n_local] = 1
        local_K[n_local, :n_local] = 1
        local_K[n_local, n_local] = 0

        # 添加正则化项确保数值稳定性
        local_K[:n_local, :n_local] += np.eye(n_local) * 1e-8

        # 计算当前网格点到邻居站点的距离
        grid_dists = cdist([grid_points[i]], local_stations).ravel()

        # 构建右侧向量
        k_vec = np.zeros(n_local + 1)
        k_vec[:n_local] = covariance_function(
            grid_dists, nugget, sill, range_, 'exponential'
        )
        k_vec[n_local] = 1  # 无偏约束

        # 使用更稳健的求解方法
        try:
            weights = solve(local_K, k_vec, assume_a='sym')
        except:
            # 使用最小二乘作为备选
            weights, _, _, _ = lstsq(local_K, k_vec, lapack_driver='gelsy')

        # 提取目标权重
        target_weights = weights[:n_local]

        # 处理负权重 - 截断并重新归一化
        target_weights = np.clip(target_weights, 0, None)
        weights_sum = np.sum(target_weights)

        if weights_sum > 1e-6:
            target_weights /= weights_sum
        else:
            # 反距离加权作为备选
            inv_dist = 1 / (grid_dists + 1e-6)
            target_weights = inv_dist / np.sum(inv_dist)

        # 计算估计值
        kriged[i] = np.dot(target_weights, local_values)

    return kriged


def split_data(stations, values, test_size=0.1):
    """等距取样划分训练集和验证集"""
    n = len(stations)
    indices = np.arange(n)

    # 计算验证集大小
    test_count = max(1, int(n * test_size))

    # 等距取样验证集索引
    step = n // test_count
    test_indices = indices[::step][:test_count]

    # 训练集索引（排除验证集）
    train_indices = np.setdiff1d(indices, test_indices)

    # 划分数据
    train_stations = stations[train_indices]
    train_values = values[train_indices]
    test_stations = stations[test_indices]
    test_values = values[test_indices]

    return train_stations, train_values, test_stations, test_values


if __name__ == "__main__":
    # 1. 读取数据
    data = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\HNHB-2025060300.csv')

    # 提取站点数据
    stations = data[['Lon', 'Lat']].values
    target_pwv = data['PWV'].values

    # 检查PWV值范围
    print("PWV统计信息:")
    print(f"最小值: {np.min(target_pwv):.4f}, 最大值: {np.max(target_pwv):.4f}")
    print(f"均值: {np.mean(target_pwv):.4f}, 标准差: {np.std(target_pwv):.4f}")

    # 2. 划分训练集和验证集（90%-10%等距取样）
    print("\n划分训练集和验证集...")
    train_stations, train_values, test_stations, test_values = split_data(
        stations, target_pwv, test_size=0.1
    )

    print(f"总样本数: {len(stations)}")
    print(f"训练集大小: {len(train_stations)}")
    print(f"验证集大小: {len(test_stations)}")

    # 3. 在训练集上计算指数模型的变差函数参数
    print("\n计算指数模型变差函数（训练集）...")
    nugget, sill, range_ = calculate_variogram(train_stations, train_values)
    print(f"块金值={nugget:.4f}, 基台值={sill:.4f}, 变程={range_:.4f}")

    # 在验证集上进行预测
    print("预测验证集...")
    predictions = ordinary_kriging(
        train_stations, train_values, test_stations,
        nugget, sill, range_,
        max_distance=range_ * 1.5, min_neighbors=8
    )

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(test_values, predictions))
    mae = mean_absolute_error(test_values, predictions)
    r2 = r2_score(test_values, predictions)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    print(f"验证结果: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # 4. 使用全部数据重新计算指数模型的参数
    print("\n使用全部数据重新计算模型参数...")
    nugget_full, sill_full, range_full = calculate_variogram(
        stations, target_pwv
    )
    print(f"全部数据参数: 块金值={nugget_full:.4f}, 基台值={sill_full:.4f}, 变程={range_full:.4f}")

    # 5. 创建插值网格
    padding = 0.1
    lon_min, lon_max = data['Lon'].min() - padding, data['Lon'].max() + padding
    lat_min, lat_max = data['Lat'].min() - padding, data['Lat'].max() + padding
    resolution = 0.01

    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    print(f"\n网格点数: {len(grid_points)}")

    # 6. 使用全部数据和指数模型执行克里金插值
    print("\n执行普通克里金插值（全部数据）...")
    pwv_grid = ordinary_kriging(
        stations, target_pwv, grid_points,
        nugget_full, sill_full, range_full,
        max_distance=range_full * 1.5, min_neighbors=8
    )

    # 后处理：移除负值
    pwv_grid = np.clip(pwv_grid, 0, None)
    print("插值完成!")

    # 7. 保存结果
    # 保存网格插值结果
    grid_results = pd.DataFrame({
        'Lon': grid_points[:, 0],
        'Lat': grid_points[:, 1],
        'PWV': pwv_grid
    })

    grid_output_file = 'pwv_kriging_results.csv'
    grid_results.to_csv(grid_output_file, index=False)
    print(f"\n插值结果已保存至: {grid_output_file}")

    # 保存验证结果
    valid_results = pd.DataFrame({
        'Lon': test_stations[:, 0],
        'Lat': test_stations[:, 1],
        'True_PWV': test_values,
        'Predicted_PWV': predictions,
        'Error': test_values - predictions
    })

    valid_output_file = 'pwv_validation_results.csv'
    valid_results.to_csv(valid_output_file, index=False)
    print(f"验证结果已保存至: {valid_output_file}")

    # 打印最终统计信息
    print(f"\n插值结果统计:")
    print(f"最小值: {np.min(pwv_grid):.4f}, 最大值: {np.max(pwv_grid):.4f}")
    print(f"均值: {np.mean(pwv_grid):.4f}, 标准差: {np.std(pwv_grid):.4f}")
    print(f"验证集评估:")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")