import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import solve, lstsq
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加需要插值的文件路径
file_path = r'D:\PyCharm\pythonProject\HN-interpolation\20250603\Z_UPAR_C_HNHB_20250603000000_P_GNSSMET_PWV_HOR.csv'


# 一、对数据进行预处理，先保留小数点后两位，在按照经纬度进行升序，最后按照10：1划分训练集和测试集。
def data_process():
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # # 对经纬度、水汽压保存小数点后两位数据
    df['Lat'] = df['Lat'].round(2)
    df['Lon'] = df['Lon'].round(2)
    df['VAP'] = df['VAP'].round(2)
    df['PWV'] = df['PWV'].round(4)

    #  # 先对 'Lat' 升序排序，再对 'Lon' 升序排序
    df = df.sort_values(by=['Lat', 'Lon'], ascending=[True, True])

    # # 添加自增列 'id'，并将其插入到第一列
    df['id'] = range(1, len(df) + 1)
    df = df[['id'] + [col for col in df.columns if col != 'id']]  # 将 'id' 列移到第一列

    #  调整列的位置，获取当前列的顺
    new_order = ['id', 'Station_Id_C', 'Datetime', 'Lat', 'Lon', 'Alti', 'City', 'Station_Name', 'Cnty',
                 'VAP', 'RHU', 'PRS', 'TEM', 'PWV']
    df = df[new_order]

    # 添加 'dataset' 列，根据 'id' 列的值判断是否为 10 的倍数
    df['dataset'] = df['id'].apply(lambda x: 'test' if x % 10 == 0 else 'train')

    # 按照 'dataset' 列的值分成两个 DataFrame
    train_df = df[df['dataset'] == 'train']
    train_df = train_df.rename(columns={'Datetime': 'time'})
    test_df = df[df['dataset'] == 'test']
    test_df = test_df.rename(columns={'Datetime': 'time', 'PWV': 'True_PWV'})

    # 保存为两个不同的 CSV 文件，如果有需要可以保存。
    # train_df.to_csv('jg.csv', index=False, encoding='utf-8-sig')
    # test_df.to_csv('test.csv', index=False, encoding='utf-8-sig')

    # 返回 train_df 和 test_df
    return train_df, test_df


# 二、计算变差函数参数
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


# 五、合并测试集的插值结果，
def merge_test_result(target_df, source_df):
    # 统一时间格式
    target_df['time'] = pd.to_datetime(target_df['time'])
    source_df['time'] = pd.to_datetime(source_df['time'])

    # 提取源数据的特征列，不包括时间、纬度和经度
    source_features = source_df.columns.difference(['time', 'Lat', 'Lon'])

    # 创建一个空的DataFrame来存放插值后的数据
    interpolated_df = pd.DataFrame()

    # 按时间分组
    for time_point in target_df['time'].unique():
        # 在目标数据和源数据中找到相同时间点的数据
        target_time_df = target_df[target_df['time'] == time_point]
        source_time_df = source_df[source_df['time'] == time_point]

        # 如果在源数据中找不到完全匹配的时间点，则寻找最接近的时间点
        if source_time_df.empty:
            time_diff = np.abs(source_df['time'] - time_point)
            min_diff_index = time_diff.idxmin()
            closest_time_point = source_df.loc[min_diff_index, 'time']
            source_time_df = source_df[source_df['time'] == closest_time_point]

        if not source_time_df.empty:
            # 提取源数据的经纬度
            source_coords = source_time_df[['Lat', 'Lon']].values

            # 构建KD树用于快速空间搜索
            tree = cKDTree(source_coords)

            # 应用最近邻搜索并将数据附加到目标DataFrame
            interpolated_data = target_time_df.apply(
                lambda row: source_time_df.iloc[tree.query([row['Lat'], row['Lon']], k=1)[1]][source_features].values,
                axis=1
            )

            # 将结果列表转换为DataFrame
            interpolated_df_time = pd.DataFrame(interpolated_data.tolist(), columns=source_features)

            # 将时间和插值数据连接起来
            interpolated_df_time = pd.concat([target_time_df.reset_index(drop=True), interpolated_df_time], axis=1)

            # 将插值后的数据添加到最终的DataFrame
            interpolated_df = pd.concat([interpolated_df, interpolated_df_time], ignore_index=True)
            test_result_df = interpolated_df
            test_result_df.to_csv('test_result.csv', index=False, encoding='utf-8-sig')

    return test_result_df


# 六、对测试集进行评估
def evaluation(data):
    Interpolated = data['True_PWV'].values  # 插值结果列
    target = data['Interpolated_PWV'].values  # 目标值列

    # 计算相关系数 (COR)
    correlation = np.corrcoef(Interpolated, target)[0, 1]
    print(f"相关系数 (COR): {correlation:.4f}")

    # 计算平均误差 (ME)
    mean_error = np.mean(Interpolated - target)
    print(f"平均误差 (ME): {mean_error:.4f}")

    # 计算平均绝对误差 (MAE)
    mae = mean_absolute_error(target, Interpolated)
    print(f"平均绝对误差 (MAE): {mae:.4f}")

    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(mean_squared_error(target, Interpolated))
    print(f"均方根误差 (RMSE): {rmse:.4f}")


# 主程序
if __name__ == "__main__":
    train_df, test_df = data_process()
    data = train_df
    # 提取站点数据
    stations = data[['Lon', 'Lat']].values
    target_pwv = data['PWV'].values

    # 检查PWV值范围
    print("PWV统计信息:")
    print(f"最小值: {np.min(target_pwv):.4f}, 最大值: {np.max(target_pwv):.4f}")
    print(f"均值: {np.mean(target_pwv):.4f}, 标准差: {np.std(target_pwv):.4f}")

    # 计算变差函数参数
    print("计算模型参数...")
    nugget_full, sill_full, range_full = calculate_variogram(
        stations, target_pwv
    )
    print(f"全部数据参数: 块金值={nugget_full:.4f}, 基台值={sill_full:.4f}, 变程={range_full:.4f}")

    # 创建插值网格
    # 使用固定边界
    padding = 0.1
    lon_min, lon_max = data['Lon'].min() - padding, data['Lon'].max() + padding
    lat_min, lat_max = data['Lat'].min() - padding, data['Lat'].max() + padding

    resolution = 0.01  # 网格分辨率
    print(
        f"\n创建网格: 经度范围({lon_min:.2f}-{lon_max:.2f}), 纬度范围({lat_min:.2f}-{lat_max:.2f}), 分辨率={resolution:.2f}")

    # 生成网格点
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    print(f"网格点数: {len(grid_points)}")

    #  执行普通克里格插值
    print("\n开始普通克里格插值...")
    pwv_grid = ordinary_kriging(
        stations, target_pwv, grid_points,
        nugget_full, sill_full, range_full,
        max_distance=range_full * 1.5, min_neighbors=8
    )
    # [3：50]
    print("插值完成!")

    # 后处理：移除负值
    pwv_grid = np.clip(pwv_grid, 0, None)

    # 创建结果DataFrame并保存为CSV
    results = pd.DataFrame({
        'Lon': grid_points[:, 0],
        'Lat': grid_points[:, 1],
        'Interpolated_PWV': pwv_grid
    })
    results['time'] = data['time'].iloc[0]  # 重复第一个时间点

    # 保存结果
    output_file = 'pwv_kriging_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\n插值结果已保存至: {output_file}")
    print(f"总网格点数: {len(results)}")
    print(f"插值结果统计: 最小值={np.min(pwv_grid):.4f}, 最大值={np.max(pwv_grid):.4f}, 均值={np.mean(pwv_grid):.4f}")

    # 评估插值效果
    test_result_df = merge_test_result(test_df, results)
    evaluation(test_result_df)
