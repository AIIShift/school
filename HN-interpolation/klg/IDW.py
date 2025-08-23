import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_variogram(coords, values, max_lag=5, lag_step=0.5):
    """计算变差函数参数"""
    # 计算所有点对之间的距离
    dist_matrix = cdist(coords, coords)
    upper_tri = np.triu_indices_from(dist_matrix, k=1)
    dists = dist_matrix[upper_tri]
    value_diffs = (values[upper_tri[0]] - values[upper_tri[1]]) ** 2

    # 分组计算半方差
    bins = np.arange(0, max_lag + lag_step, lag_step)
    bin_indices = np.digitize(dists, bins)
    semivariances = []

    for i in range(1, len(bins)):
        in_bin = (bin_indices == i)
        if np.sum(in_bin) > 0:  # 确保有足够的点对
            semivariance = np.mean(value_diffs[in_bin]) / 2
            semivariances.append(semivariance)
        else:
            semivariances.append(np.nan)

    # 找到有效的变差函数参数
    valid_idx = ~np.isnan(semivariances)
    if not np.any(valid_idx):
        return np.var(values), np.mean(dists) / 3

    valid_semivars = np.array(semivariances)[valid_idx]

    # 基台值(最大半方差)
    sill = np.max(valid_semivars)
    # 变程(半方差达到基台值时的距离)
    range_val = bins[np.argmax(valid_semivars)] if np.max(valid_semivars) > 0 else np.mean(dists) / 2

    return sill, range_val


def idw_interpolation(stations, values, grid_points, power=2):
    """反距离加权插值方法"""
    n = len(stations)
    m = len(grid_points)

    # 初始化结果数组
    kriged = np.zeros(m)

    # 计算站点间的距离矩阵
    station_dists = cdist(stations, stations)

    # 对每个网格点进行插值
    for i in range(m):
        # 计算当前网格点到所有站点的距离
        grid_dists = cdist([grid_points[i]], stations).ravel()

        # 根据距离计算权重 (使用距离的倒数)
        weights = 1 / (grid_dists ** power + 1e-6)  # 加1e-6防止除零错误

        # 归一化权重
        weights /= np.sum(weights)

        # 计算插值结果
        kriged[i] = np.dot(weights, values)

    return kriged


def cross_validate_idw(stations, values, n_neighbors=30, k=5, power=2):
    """
    交叉验证反距离加权插值模型
    :param stations: 站点坐标数组
    :param values: 站点值数组
    :param n_neighbors: 邻居数量
    :param k: 交叉验证折数
    :return: 预测值数组，评估指标字典
    """
    n = len(stations)
    predictions = np.zeros(n)

    # 创建索引数组并打乱
    indices = np.arange(n)
    np.random.shuffle(indices)

    # 计算每折的大小
    fold_size = n // k
    if n % k != 0:
        fold_size += 1

    print(f"开始 {k}-折交叉验证，每折大小约 {fold_size} 个点...")

    for i in range(k):
        # 划分训练集和测试集
        start_idx = i * fold_size
        end_idx = min((i + 1) * fold_size, n)
        test_idx = indices[start_idx:end_idx]
        train_idx = np.setdiff1d(indices, test_idx)

        # 提取训练和测试数据
        train_stations = stations[train_idx]
        train_values = values[train_idx]
        test_stations = stations[test_idx]

        # 对测试点进行预测
        for j, point in enumerate(test_stations):
            # 计算当前测试点到所有训练点的距离
            dists = cdist([point], train_stations).ravel()

            # 计算距离的倒数权重
            weights = 1 / (dists ** power + 1e-6)

            # 归一化权重
            weights /= np.sum(weights)

            # 计算插值结果
            predictions[test_idx[j]] = np.dot(weights, train_values)

        print(f"完成第 {i + 1}/{k} 折验证，测试点数量: {len(test_idx)}")

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(values, predictions))
    mae = mean_absolute_error(values, predictions)
    r2 = r2_score(values, predictions)
    corr = np.corrcoef(values, predictions)[0, 1]
    bias = np.mean(predictions - values)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2,
        'Correlation': corr,
        'Bias': bias
    }

    return predictions, metrics


# 主程序
if __name__ == "__main__":
    # 1. 读取数据 (替换为您的实际文件路径)
    data = pd.read_csv('CS-2025060300.csv')  # 替换为您的文件路径

    # 提取站点数据
    stations = data[['Lon', 'Lat']].values
    target_pwv = data['PWV'].values

    # 检查PWV值范围
    print("PWV统计信息: ")
    print(f"最小值: {np.min(target_pwv):.4f}, 最大值: {np.max(target_pwv):.4f}")
    print(f"均值: {np.mean(target_pwv):.4f}, 标准差: {np.std(target_pwv):.4f}")

    # 2. 计算变差函数参数
    print("计算目标变量变差函数... ")
    t_sill, t_range = calculate_variogram(stations, target_pwv)
    print(f"目标变量: 基台值={t_sill:.4f}, 变程={t_range:.4f}")

    # 3. 执行交叉验证
    print("\n开始交叉验证IDW插值模型... ")
    cv_predictions, metrics = cross_validate_idw(
        stations, target_pwv, n_neighbors=30, k=20, power=2
    )

    # 打印评估结果
    print("\n交叉验证评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 创建交叉验证结果DataFrame
    cv_results = pd.DataFrame({
        'Lon': stations[:, 0],
        'Lat': stations[:, 1],
        'True_PWV': target_pwv,
        'Predicted_PWV': cv_predictions,
        'Error': target_pwv - cv_predictions
    })

    # 保存交叉验证结果
    cv_output_file = 'pwv_idw_cv_results.csv'
    cv_results.to_csv(cv_output_file, index=False)
    print(f"\n交叉验证结果已保存至: {cv_output_file}")

    # 4. 创建插值网格
    lon_min, lon_max = 108, 115
    lat_min, lat_max = 24, 31

    resolution = 0.01  # 网格分辨率
    print(f"\n创建网格: 经度范围({lon_min:.2f}-{lon_max:.2f}), 纬度范围({lat_min:.2f}-{lat_max:.2f}), 分辨率={resolution:.2f}")

    # 生成网格点
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    print(f"网格点数: {len(grid_points)}")

    # 5. 执行反距离加权插值
    print("\n开始反距离加权插值... ")
    pwv_grid = idw_interpolation(
        stations, target_pwv, grid_points, power=2
    )
    print("插值完成!")

    # 后处理：移除负值
    pwv_grid = np.clip(pwv_grid, 0, None)

    # 6. 创建结果DataFrame并保存为CSV
    results = pd.DataFrame({
        'Lon': grid_points[:, 0],
        'Lat': grid_points[:, 1],
        'PWV': pwv_grid
    })

    # 保存结果
    output_file = 'pwv_idw_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\n插值结果已保存至: {output_file}")
    print(f"总网格点数: {len(results)}")
    print(f"插值结果统计: 最小值={np.min(pwv_grid):.4f}, 最大值={np.max(pwv_grid):.4f}, 均值={np.mean(pwv_grid):.4f}")
