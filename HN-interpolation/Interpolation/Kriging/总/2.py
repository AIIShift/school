import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import chardet

# 全局常量
FILE_PATH = r'D:\PyCharm\pythonProject\HN-interpolation\20250603\Z_UPAR_C_HNHB_20250603000000_P_GNSSMET_PWV_HOR.csv'
COLUMN_ORDER = ['id', 'Station_Id_C', 'Datetime', 'Lat', 'Lon', 'Alti', 'City',
                'Station_Name', 'Cnty', 'VAP', 'RHU', 'PRS', 'TEM', 'PWV']


def data_process():
    """处理原始数据，分割训练集和测试集"""
    df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')

    # 数据预处理
    df = (df
          .assign(Lat=lambda x: x['Lat'].round(2),
                  Lon=lambda x: x['Lon'].round(2),
                  VAP=lambda x: x['VAP'].round(2),
                  PWV=lambda x: x['PWV'].round(4))
          .sort_values(['Lat', 'Lon'])
          )

    # 添加id列和dataset列
    df = df.assign(
        id=lambda x: range(1, len(x) + 1),
        dataset=lambda x: np.where(x['id'] % 10 == 0, 'test', 'train')
    )

    # 将id列移到第一列
    cols = ['id'] + [col for col in df.columns if col != 'id']
    df = df[cols]

    # 重新排列列顺序（现在包含dataset列）
    final_columns = COLUMN_ORDER + ['dataset']
    # 确保只选择存在的列
    existing_columns = [col for col in final_columns if col in df.columns]
    df = df[existing_columns]

    # 分割数据集
    train_df = (df[df['dataset'] == 'train']
                .rename(columns={'Datetime': 'time'})
                .drop(columns=['dataset']))
    test_df = (df[df['dataset'] == 'test']
               .rename(columns={'Datetime': 'time', 'PWV': 'True_PWV'})
               .drop(columns=['dataset']))

    # 保存数据
    train_df.to_csv('jg.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test.csv', index=False, encoding='utf-8-sig')

    return train_df, test_df


def calculate_variogram(coords, values, max_lag=5, lag_step=0.5):
    """计算变差函数参数"""
    dist_matrix = cdist(coords, coords)
    upper_tri = np.triu_indices_from(dist_matrix, k=1)
    dists = dist_matrix[upper_tri]
    value_diffs = (values[upper_tri[0]] - values[upper_tri[1]]) ** 2

    bins = np.arange(0, max_lag + lag_step, lag_step)
    bin_indices = np.digitize(dists, bins)
    semivariances = []

    for i in range(1, len(bins)):
        in_bin = (bin_indices == i)
        if np.any(in_bin):
            semivariances.append(np.mean(value_diffs[in_bin]) / 2)
        else:
            semivariances.append(np.nan)

    valid_semivars = np.array(semivariances)
    valid_semivars = valid_semivars[~np.isnan(valid_semivars)]

    if not valid_semivars.size:
        return np.var(values), np.mean(dists) / 3

    sill = np.max(valid_semivars)
    range_val = bins[np.argmax(valid_semivars)] if np.max(valid_semivars) > 0 else np.mean(dists) / 2

    return sill, range_val


def gaussian_cov(distance, sill, range_):
    """高斯协方差函数"""
    zero_dist = (distance == 0)
    cov = np.zeros_like(distance)
    cov[~zero_dist] = sill * np.exp(-(distance[~zero_dist] ** 2) / (2 * range_ ** 2))
    cov[zero_dist] = sill
    return cov


def ordinary_kriging(stations, values, grid_points, sill, range_, neighborhood_size=50):
    """普通克里格插值函数"""
    n = len(stations)
    m = len(grid_points)
    kriged = np.zeros(m)

    # 构建全局协方差矩阵
    station_dists = cdist(stations, stations)
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = gaussian_cov(station_dists, sill, range_)
    K[:n, n] = K[n, :n] = 1
    K[n, n] = 0
    K[:n, :n] += np.eye(n) * 1e-8

    tree = cKDTree(stations)
    global_mean = np.mean(values)

    for i in range(m):
        dists, idxs = tree.query(grid_points[i], k=min(neighborhood_size, n))
        if not idxs.size:
            kriged[i] = global_mean
            continue

        local_stations = stations[idxs]
        local_values = values[idxs]
        n_local = len(local_stations)

        # 构建局部协方差矩阵
        local_dists = cdist(local_stations, local_stations)
        local_K = np.zeros((n_local + 1, n_local + 1))
        local_K[:n_local, :n_local] = gaussian_cov(local_dists, sill, range_)
        local_K[:n_local, n_local] = local_K[n_local, :n_local] = 1
        local_K[n_local, n_local] = 0
        local_K[:n_local, :n_local] += np.eye(n_local) * 1e-8

        # 计算网格点到邻居的距离
        grid_dists = cdist([grid_points[i]], local_stations).ravel()
        k_vec = np.zeros(n_local + 1)
        k_vec[:n_local] = gaussian_cov(grid_dists, sill, range_)
        k_vec[n_local] = 1

        try:
            weights = solve(local_K, k_vec, assume_a='sym')[:n_local]
            weights = np.clip(weights, 0, None)
            total_weight = weights.sum()

            if total_weight > 1e-6:
                weights /= total_weight
                kriged[i] = np.dot(weights, local_values)
            else:
                # 回退到反距离加权
                inv_dist = 1 / (grid_dists + 1e-6)
                weights = inv_dist / inv_dist.sum()
                kriged[i] = np.dot(weights, local_values)

        except Exception:
            # 回退到反距离加权
            inv_dist = 1 / (grid_dists + 1e-6)
            weights = inv_dist / inv_dist.sum()
            kriged[i] = np.dot(weights, local_values)

    return np.clip(kriged, 0, None)


def merge_test_result(target_df, source_df):
    """合并测试结果和插值结果"""
    # 确保时间格式一致
    target_df['time'] = pd.to_datetime(target_df['time'])
    source_df['time'] = pd.to_datetime(source_df['time'])

    # 提取特征列
    source_features = source_df.columns.difference(['time', 'Lat', 'Lon'])

    results = []
    for time_point in target_df['time'].unique():
        target_subset = target_df[target_df['time'] == time_point]
        source_subset = source_df[source_df['time'] == time_point]

        if source_subset.empty:
            continue

        tree = cKDTree(source_subset[['Lat', 'Lon']].values)
        _, idx = tree.query(target_subset[['Lat', 'Lon']].values, k=1)
        interpolated = source_subset.iloc[idx][source_features].reset_index(drop=True)
        results.append(pd.concat([target_subset.reset_index(drop=True), interpolated], axis=1))

    if not results:
        return pd.DataFrame()

    result_df = pd.concat(results, ignore_index=True)
    result_df.to_csv('test_result.csv', index=False, encoding='utf-8-sig')
    return result_df


def evaluate_predictions(data):
    """评估预测结果"""
    y_true = data['True_PWV'].values
    y_pred = data['Interpolated_PWV'].values

    metrics = {
        'COR': np.corrcoef(y_pred, y_true)[0, 1],
        'ME': np.mean(y_pred - y_true),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

    print("评估结果:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


def main():
    """主程序流程"""
    # 1. 数据处理
    train_df, test_df = data_process()

    # 2. 准备克里金插值
    stations = train_df[['Lon', 'Lat']].values
    target_pwv = train_df['PWV'].values

    print("PWV统计信息:")
    print(f"最小值: {np.min(target_pwv):.4f}, 最大值: {np.max(target_pwv):.4f}")
    print(f"均值: {np.mean(target_pwv):.4f}, 标准差: {np.std(target_pwv):.4f}")

    # 3. 计算变差函数
    print("计算变差函数...")
    t_sill, t_range = calculate_variogram(stations, target_pwv)
    print(f"基台值={t_sill:.4f}, 变程={t_range:.4f}")

    # 4. 创建插值网格
    padding = 0.1
    lon_bounds = train_df['Lon'].min() - padding, train_df['Lon'].max() + padding
    lat_bounds = train_df['Lat'].min() - padding, train_df['Lat'].max() + padding

    # lon_min, lon_max = 108.41, 116.13
    # lat_min, lat_max = 24.64, 33.30
    # lon_bounds = 108.41, 116.13
    # lat_bounds = 24.64, 33.30

    resolution = 0.01
    lons = np.arange(*lon_bounds, resolution)
    lats = np.arange(*lat_bounds, resolution)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    print(f"网格范围: 经度({lon_bounds[0]:.2f}-{lon_bounds[1]:.2f}), "
          f"纬度({lat_bounds[0]:.2f}-{lat_bounds[1]:.2f})")
    print(f"网格点数: {len(grid_points)}")

    # 5. 执行克里金插值
    print("开始克里金插值...")
    pwv_grid = ordinary_kriging(stations, target_pwv, grid_points, t_sill, t_range)
    print("插值完成!")

    # 6. 保存插值结果
    results = pd.DataFrame({
        'Lon': grid_points[:, 0],
        'Lat': grid_points[:, 1],
        'Interpolated_PWV': pwv_grid,
        'time': train_df['time'].iloc[0]
    })

    output_file = 'pwv_kriging_results.csv'
    results.to_csv(output_file, index=False)
    print(f"结果已保存至: {output_file}")

    # 7. 评估结果
    test_result_df = merge_test_result(test_df, results)
    if not test_result_df.empty:
        evaluate_predictions(test_result_df)
    else:
        print("警告: 未找到匹配的测试结果")


if __name__ == "__main__":
    main()