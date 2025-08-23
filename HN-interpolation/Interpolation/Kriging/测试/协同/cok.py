import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import solve, lstsq
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time


def calculate_variogram(coords, values, max_lag=5, lag_step=0.5, model='gaussian'):
    """
    改进的变差函数计算，使用理论模型拟合
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

    # 定义理论变差函数模型
    def gaussian_model(h, nugget, sill, range_):
        return nugget + (sill - nugget) * (1 - np.exp(-(h ** 2) / (range_ ** 2)))

    def exponential_model(h, nugget, sill, range_):
        return nugget + (sill - nugget) * (1 - np.exp(-h / range_))

    def spherical_model(h, nugget, sill, range_):
        with np.errstate(invalid='ignore'):
            return np.where(
                h <= range_,
                nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
                sill
            )

    models = {
        'gaussian': gaussian_model,
        'exponential': exponential_model,
        'spherical': spherical_model
    }

    # 选择模型
    if model not in models:
        model = 'gaussian'
    model_func = models[model]

    # 初始参数估计
    nugget_guess = np.min(semivariances)
    sill_guess = np.max(semivariances)
    range_guess = bin_centers[np.argmax(semivariances >= 0.95 * sill_guess)] if sill_guess > 0 else np.max(
        bin_centers) / 2

    # 拟合模型
    try:
        popt, _ = curve_fit(
            model_func, bin_centers, semivariances,
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


def covariance_function(distance, nugget, sill, range_, model='gaussian'):
    """改进的协方差函数，支持多种模型"""
    # 处理零距离
    zero_dist = distance == 0
    partial_sill = sill - nugget

    # 初始化协方差
    cov = np.zeros_like(distance)

    # 非零距离计算
    mask = ~zero_dist
    d = distance[mask]

    if model == 'gaussian':
        cov[mask] = partial_sill * np.exp(-(d ** 2) / (range_ ** 2))
    elif model == 'exponential':
        cov[mask] = partial_sill * np.exp(-d / range_)
    elif model == 'spherical':
        # 球状模型
        with np.errstate(invalid='ignore'):
            ratio = d / range_
            cov[mask] = partial_sill * (1 - (1.5 * ratio - 0.5 * ratio ** 3))
            cov[mask] = np.where(d <= range_, cov[mask], 0)
    else:
        # 默认为高斯模型
        cov[mask] = partial_sill * np.exp(-(d ** 2) / (range_ ** 2))

    # 添加块金效应（零距离时协方差等于基台值）
    cov[zero_dist] = sill
    return cov


def ordinary_kriging(stations, values, grid_points, nugget, sill, range_,
                     model='gaussian', max_distance=10, min_neighbors=5):
    """改进的普通克里格插值函数"""
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
            neighbors = [neighbors] if not isinstance(neighbors, np.ndarray) else neighbors.tolist()

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
            local_dists, nugget, sill, range_, model
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
            grid_dists, nugget, sill, range_, model
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


def interpolate_vap(stations, vap_values, grid_points, method='rf', n_splits=5):
    """
    使用选择的插值方法在整个区域插值VAP数据
    :param stations: 已知站点坐标 (n, 2)
    :param vap_values: 已知站点VAP值 (n,)
    :param grid_points: 网格点坐标 (m, 2)
    :param method: 插值方法 ('rf' 或 'kriging')
    :param n_splits: 交叉验证折数
    :return: 网格点上的VAP值 (m,)
    """
    print(f"开始使用{method.upper()}方法插值VAP数据...")

    if method == 'rf':
        # 使用随机森林回归
        print("训练随机森林模型...")

        # 初始参数估计
        n_estimators = 100
        max_depth = None

        # 使用全部数据训练模型
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(stations, vap_values)

        # 保存模型
        joblib.dump(model, 'vap_rf_model.pkl')

        # 预测网格点VAP
        grid_vap = model.predict(grid_points)

    elif method == 'kriging':
        # 使用克里金插值
        print("计算VAP变差函数...")
        nugget, sill, range_ = calculate_variogram(stations, vap_values)
        print(f"VAP变差函数参数: 块金值={nugget:.4f}, 基台值={sill:.4f}, 变程={range_:.4f}")

        # 使用普通克里金插值
        grid_vap = ordinary_kriging(
            stations, vap_values, grid_points,
            nugget, sill, range_,
            max_distance=range_ * 1.5, min_neighbors=10
        )
    else:
        raise ValueError("不支持的插值方法，请选择 'rf' 或 'kriging'")

    return grid_vap


def cokriging(stations, pwv_values, vap_values, grid_points, grid_vap,
              pwv_nugget, pwv_sill, pwv_range, vap_nugget, vap_sill, vap_range,
              cross_nugget, cross_sill, cross_range, model='gaussian', max_distance=10, min_neighbors=8):
    """
    协同克里金插值函数
    :param stations: 已知站点坐标
    :param pwv_values: 已知站点PWV值
    :param vap_values: 已知站点VAP值
    :param grid_points: 网格点坐标
    :param grid_vap: 网格点VAP值
    :param pwv_nugget, pwv_sill, pwv_range: PWV的变差函数参数
    :param vap_nugget, vap_sill, vap_range: VAP的变差函数参数
    :param cross_nugget, cross_sill, cross_range: 交叉变差函数参数
    :param model: 协方差函数模型
    :param max_distance: 最大搜索距离
    :param min_neighbors: 最小邻居数
    :return: 网格点上的PWV值
    """
    print("开始协同克里金插值...")
    n = len(stations)
    m = len(grid_points)
    cokriged = np.zeros(m)

    # 构建KD树用于邻居搜索
    tree = cKDTree(stations)

    # 对每个网格点进行插值
    for i in range(m):
        # 动态距离内的邻居
        neighbors = tree.query_ball_point(grid_points[i], max_distance)
        if len(neighbors) < min_neighbors:
            _, neighbors = tree.query(grid_points[i], k=min_neighbors)
            neighbors = [neighbors] if not isinstance(neighbors, np.ndarray) else neighbors.tolist()

        local_stations = stations[neighbors]
        local_pwv = pwv_values[neighbors]
        local_vap = vap_values[neighbors]
        n_local = len(local_stations)

        if n_local == 0:
            # 没有邻居时使用全局平均值
            cokriged[i] = np.mean(pwv_values)
            continue

        # 创建协同克里金矩阵 (n_local*2 + 2) x (n_local*2 + 2)
        n_total = n_local * 2 + 2
        K = np.zeros((n_total, n_total))

        # 1. 填充PWV-PWV协方差
        pwv_dists = cdist(local_stations, local_stations)
        K[:n_local, :n_local] = covariance_function(
            pwv_dists, pwv_nugget, pwv_sill, pwv_range, model
        )

        # 2. 填充VAP-VAP协方差
        vap_dists = cdist(local_stations, local_stations)
        K[n_local:2 * n_local, n_local:2 * n_local] = covariance_function(
            vap_dists, vap_nugget, vap_sill, vap_range, model
        )

        # 3. 填充PWV-VAP交叉协方差
        K[:n_local, n_local:2 * n_local] = covariance_function(
            pwv_dists, cross_nugget, cross_sill, cross_range, model
        )
        K[n_local:2 * n_local, :n_local] = K[:n_local, n_local:2 * n_local].T

        # 4. 添加无偏约束
        # PWV约束
        K[:n_local, -2] = 1
        K[-2, :n_local] = 1

        # VAP约束
        K[n_local:2 * n_local, -1] = 1
        K[-1, n_local:2 * n_local] = 1

        # 右下角设为0
        K[-2:, -2:] = 0

        # 添加正则化项
        K[:2 * n_local, :2 * n_local] += np.eye(2 * n_local) * 1e-8

        # 构建右侧向量
        k_vec = np.zeros(n_total)

        # 计算到目标点的距离
        grid_dists = cdist([grid_points[i]], local_stations).ravel()

        # PWV部分
        k_vec[:n_local] = covariance_function(
            grid_dists, pwv_nugget, pwv_sill, pwv_range, model
        )

        # VAP部分 - 使用目标点的VAP值
        # 这里简化处理，实际协同克里金需要更复杂的处理
        k_vec[n_local:2 * n_local] = covariance_function(
            grid_dists, vap_nugget, vap_sill, vap_range, model
        )

        # 无偏约束部分
        k_vec[-2] = 1  # PWV约束
        k_vec[-1] = 1  # VAP约束

        # 求解权重
        try:
            weights = solve(K, k_vec, assume_a='sym')
        except:
            weights, _, _, _ = lstsq(K, k_vec, lapack_driver='gelsy')

        # 提取PWV和VAP权重
        pwv_weights = weights[:n_local]
        vap_weights = weights[n_local:2 * n_local]

        # 组合权重
        total_weights = np.concatenate([pwv_weights, vap_weights])
        total_values = np.concatenate([local_pwv, local_vap])

        # 处理负权重
        total_weights = np.clip(total_weights, 0, None)
        weights_sum = np.sum(total_weights)

        if weights_sum > 1e-6:
            total_weights /= weights_sum
        else:
            # 使用反距离加权作为备选
            inv_dist = 1 / (grid_dists + 1e-6)
            total_weights = np.concatenate([inv_dist, np.zeros_like(inv_dist)])
            total_weights /= np.sum(total_weights)

        # 计算估计值
        cokriged[i] = np.dot(total_weights, total_values)

    return cokriged


def calculate_cross_variogram(coords, pwv_values, vap_values, max_lag=5, lag_step=0.5):
    """
    计算交叉变差函数
    """
    n = len(coords)
    dist_matrix = cdist(coords, coords)
    pwv_diff = pwv_values[:, None] - pwv_values
    vap_diff = vap_values[:, None] - vap_values
    cross_gamma = 0.5 * (pwv_diff * vap_diff)

    # 仅取上三角
    upper_tri = np.triu_indices(n, k=1)
    dists = dist_matrix[upper_tri]
    cross_vals = cross_gamma[upper_tri]

    # 分组计算
    bins = np.arange(0, max_lag + lag_step, lag_step)
    bin_indices = np.digitize(dists, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cross_semivars = []

    for i in range(1, len(bins)):
        in_bin = (bin_indices == i)
        if np.sum(in_bin) > 5:
            cross_semivars.append(np.mean(cross_vals[in_bin]))
        else:
            cross_semivars.append(np.nan)

    # 过滤无效值
    valid_idx = ~np.isnan(cross_semivars)
    bin_centers = bin_centers[valid_idx]
    cross_semivars = np.array(cross_semivars)[valid_idx]

    if len(cross_semivars) == 0:
        return 0.0, np.cov(pwv_values, vap_values)[0, 1], np.mean(dists) / 3

    # 拟合线性模型简化处理
    nugget = cross_semivars[0]
    sill = np.mean(cross_semivars)
    range_ = bin_centers[np.argmax(cross_semivars)]

    return nugget, sill, range_


def cokriging_validation(stations, pwv_values, vap_values, grid_points):
    """
    使用10%的数据进行验证（等距取样）
    :return: 验证结果字典，包含指标和预测值
    """
    print("开始协同克里金验证（10%等距取样）...")
    n = len(stations)

    # 等距取样10%作为验证集
    step = max(1, n // 100)  # 确保至少取10个点
    val_indices = np.arange(0, n, step)
    train_indices = np.setdiff1d(np.arange(n), val_indices)

    print(f"总样本数: {n}, 训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")

    # 划分数据集
    train_stations = stations[train_indices]
    test_stations = stations[val_indices]
    train_pwv = pwv_values[train_indices]
    train_vap = vap_values[train_indices]
    test_pwv = pwv_values[val_indices]
    test_vap = vap_values[val_indices]  # 实际值，仅用于验证

    # 在整个区域插值VAP（包括测试点）
    print("在训练集上插值VAP...")
    grid_vap = interpolate_vap(
        train_stations, train_vap,
        np.vstack([test_stations, grid_points]),  # 包含测试点和网格点
        method='rf'
    )
    test_vap_pred = grid_vap[:len(val_indices)]  # 提取测试点预测

    # 计算变差函数参数
    print("计算PWV变差函数...")
    pwv_nugget, pwv_sill, pwv_range = calculate_variogram(train_stations, train_pwv)
    print(f"PWV: nugget={pwv_nugget:.4f}, sill={pwv_sill:.4f}, range={pwv_range:.4f}")

    print("计算VAP变差函数...")
    vap_nugget, vap_sill, vap_range = calculate_variogram(train_stations, train_vap)
    print(f"VAP: nugget={vap_nugget:.4f}, sill={vap_sill:.4f}, range={vap_range:.4f}")

    print("计算交叉变差函数...")
    cross_nugget, cross_sill, cross_range = calculate_cross_variogram(train_stations, train_pwv, train_vap)
    print(f"交叉: nugget={cross_nugget:.4f}, sill={cross_sill:.4f}, range={cross_range:.4f}")

    # 对测试点进行协同克里金预测
    test_pwv_pred = cokriging(
        train_stations, train_pwv, train_vap,
        test_stations, test_vap_pred,
        pwv_nugget, pwv_sill, pwv_range,
        vap_nugget, vap_sill, vap_range,
        cross_nugget, cross_sill, cross_range,
        max_distance=max(pwv_range, vap_range) * 1.5,
        min_neighbors=10
    )

    # 计算验证指标
    rmse = np.sqrt(mean_squared_error(test_pwv, test_pwv_pred))
    mae = mean_absolute_error(test_pwv, test_pwv_pred)
    r2 = r2_score(test_pwv, test_pwv_pred)

    print("\n===== 验证结果 =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # 保存验证结果
    val_results = pd.DataFrame({
        'Lon': test_stations[:, 0],
        'Lat': test_stations[:, 1],
        'True_PWV': test_pwv,
        'Predicted_PWV': test_pwv_pred,
        'Error': test_pwv - test_pwv_pred
    })
    val_results.to_csv('cokriging_validation_results.csv', index=False)
    print("验证结果已保存至 cokriging_validation_results.csv")

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Validation_Results': val_results
    }


if __name__ == "__main__":
    # 1. 读取数据
    data = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\CS-2025060300.csv')

    # 提取站点数据
    stations = data[['Lon', 'Lat']].values
    pwv_values = data['PWV'].values
    vap_values = data['VAP'].values  # 假设数据中有VAP列

    # 检查数据
    print("数据统计信息:")
    print(f"PWV: 最小值={np.min(pwv_values):.4f}, 最大值={np.max(pwv_values):.4f}, 均值={np.mean(pwv_values):.4f}")
    print(f"VAP: 最小值={np.min(vap_values):.4f}, 最大值={np.max(vap_values):.4f}, 均值={np.mean(vap_values):.4f}")

    # 2. 创建插值网格（湖南省范围）
    # 湖南省大致范围: 经度108.5°E-114.5°E, 纬度24.5°N-30.5°N
    lon_min, lon_max = 108.5, 114.5
    lat_min, lat_max = 24.5, 30.5
    resolution = 0.01

    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    print(f"\n网格创建完成: 点数={len(grid_points)}, 分辨率={resolution}度")

    # 3. 使用10%数据验证协同克里金
    val_metrics = cokriging_validation(
        stations, pwv_values, vap_values, grid_points
    )

    # 4. 使用全部数据计算最终PWV网格
    print("\n使用全部数据进行最终协同克里金插值...")

    # 在整个区域插值VAP（使用全部数据）
    grid_vap_full = interpolate_vap(
        stations, vap_values, grid_points,
        method='rf'
    )

    # 计算变差函数参数（使用全部数据）
    print("计算PWV变差函数...")
    pwv_nugget, pwv_sill, pwv_range = calculate_variogram(stations, pwv_values)
    print(f"PWV: nugget={pwv_nugget:.4f}, sill={pwv_sill:.4f}, range={pwv_range:.4f}")

    print("计算VAP变差函数...")
    vap_nugget, vap_sill, vap_range = calculate_variogram(stations, vap_values)
    print(f"VAP: nugget={vap_nugget:.4f}, sill={vap_sill:.4f}, range={vap_range:.4f}")

    print("计算交叉变差函数...")
    cross_nugget, cross_sill, cross_range = calculate_cross_variogram(stations, pwv_values, vap_values)
    print(f"交叉: nugget={cross_nugget:.4f}, sill={cross_sill:.4f}, range={cross_range:.4f}")

    # 执行协同克里金插值
    grid_pwv = cokriging(
        stations, pwv_values, vap_values, grid_points, grid_vap_full,
        pwv_nugget, pwv_sill, pwv_range,
        vap_nugget, vap_sill, vap_range,
        cross_nugget, cross_sill, cross_range,
        max_distance=max(pwv_range, vap_range) * 1.5,
        min_neighbors=15
    )

    # 后处理：移除负值
    grid_pwv = np.clip(grid_pwv, 0, None)

    # 6. 保存最终结果
    results = pd.DataFrame({
        'Lon': grid_points[:, 0],
        'Lat': grid_points[:, 1],
        'PWV': grid_pwv,
        'VAP': grid_vap_full
    })

    output_file = 'cokriging_final_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\n最终插值结果已保存至: {output_file}")

    # 结果统计
    print("\n插值结果统计:")
    print(f"PWV: 最小值={np.min(grid_pwv):.4f}, 最大值={np.max(grid_pwv):.4f}, 均值={np.mean(grid_pwv):.4f}")
    print(
        f"VAP: 最小值={np.min(grid_vap_full):.4f}, 最大值={np.max(grid_vap_full):.4f}, 均值={np.mean(grid_vap_full):.4f}")

    # 可视化结果示例
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=grid_pwv,
                s=1, cmap='jet', vmin=np.min(pwv_values), vmax=np.max(pwv_values))
    plt.colorbar(label='PWV')
    plt.title('协同克里金PWV插值结果')
    plt.xlabel('经度')
    plt.ylabel('纬度')

    plt.subplot(1, 2, 2)
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=grid_vap_full,
                s=1, cmap='viridis', vmin=np.min(vap_values), vmax=np.max(vap_values))
    plt.colorbar(label='VAP')
    plt.title('VAP插值结果')
    plt.xlabel('经度')
    plt.ylabel('纬度')

    plt.tight_layout()
    plt.savefig('interpolation_results.png', dpi=300)
    plt.show()