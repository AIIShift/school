import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 添加需要插值的文件路径
file_time = "00"
k_zhe = 10
neighborhood_size = 30

# 一、对数据进行预处理，先保留小数点后两位，在按照经纬度进行升序，最后按照10：1划分训练集和测试集。
def data_process():

    file1 = "D:\\PyCharm\\pythonProject\\HN-interpolation\\20250603\\Z_UPAR_C_BECS_20250603" + file_time + "0000_P_GNSSMET_PWV_HOR.csv"
    file2 = "D:\\PyCharm\\pythonProject\\HN-interpolation\\20250603\\Z_UPAR_C_BEWH_20250603" + file_time + "0000_P_GNSSMET_PWV_HOR.csv"
    # 读取第一个 CSV 文件
    df1 = pd.read_csv(file1)
    # 读取第二个 CSV 文件
    df2 = pd.read_csv(file2)
    # 合并两个 DataFrame（假设按行合并）
    df = pd.concat([df1, df2], ignore_index=True)
    print("合并完成")

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
    print(df)

    pwv_data = df['PWV']  # 假设你的CSV文件中包含名为 'PWV' 的列
    # 基于均值和标准差的异常值检测
    # 计算均值和标准差
    mean_pwv = pwv_data.mean()
    std_pwv = pwv_data.std()

    # 设置一个标准差阈值（例如 3个标准差）
    threshold = 3

    # 检测异常值：如果数据点超出了均值±3个标准差范围，视为异常值
    outliers_mean_std = pwv_data[
        (pwv_data > mean_pwv + threshold * std_pwv) | (pwv_data < mean_pwv - threshold * std_pwv)]

    # 基于四分位数（IQR）方法的异常值检测
    # 计算四分位数（Q1, Q3）和四分位间距（IQR）
    Q1 = pwv_data.quantile(0.25)
    Q3 = pwv_data.quantile(0.75)
    IQR = Q3 - Q1

    # 设置异常值标准
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 检测异常值：如果数据点超出了Q1 - 1.5*IQR 或 Q3 + 1.5*IQR范围，视为异常值
    outliers_IQR = pwv_data[(pwv_data < lower_bound) | (pwv_data > upper_bound)]

    # 输出基于均值和标准差的异常值检测结果
    print('基于均值和标准差的异常值检测')
    print(f'均值: {mean_pwv}, 标准差: {std_pwv}')
    print(f'检测到的异常值:  个数：{len(outliers_mean_std)} \n{outliers_mean_std}')

    # 输出基于四分位数（IQR）方法的异常值检测结果
    print('\n基于四分位数（IQR）方法的异常值检测')
    print(f'第一四分位数: {Q1}, 第三四分位数: {Q3}, 四分位间距: {IQR}')
    print(f'检测到的异常值:  个数： {len(outliers_IQR)}\n{outliers_IQR}')

    # 取两个方法检测到的异常值的并集
    outliers_combined = set(outliers_mean_std).union(set(outliers_IQR))

    # 从原数据中剔除异常值
    df = df[~df['PWV'].isin(outliers_combined)]

    # 输出剔除异常值后的结果
    print('\n剔除异常值后的数据:')
    print(df)

    # 添加 'dataset' 列，根据 'id' 列的值判断是否为 10 的倍数
    df['dataset'] = df['id'].apply(lambda x: 'test' if x % k_zhe == 0 else 'train')

    # 按照 'dataset' 列的值分成两个 DataFrame
    train_df = df[df['dataset'] == 'train']
    train_df = train_df.rename(columns={'Datetime': 'time'})
    test_df = df[df['dataset'] == 'test']
    test_df = test_df.rename(columns={'Datetime': 'time', 'PWV': 'True_PWV'})

    # 保存为两个不同的 CSV 文件，如果有需要可以保存。
    train_df.to_csv('train.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test.csv', index=False, encoding='utf-8-sig')

    # 返回 train_df 和 test_df
    return train_df, test_df


# 二、计算变差函数参数
def calculate_variogram(coords, values, max_lag=5, lag_step=0.5):
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


# 三、计算高斯协方差函数
def gaussian_cov(distance, sill, range_):
    with np.errstate(divide='ignore', invalid='ignore'):
        # 处理零距离情况
        zero_dist = distance == 0
        cov = np.zeros_like(distance)
        cov[~zero_dist] = sill * np.exp(-(distance[~zero_dist] ** 2) / (2 * range_ ** 2))
        cov[zero_dist] = sill  # 零距离时协方差等于基台值
    return cov


# 四、定义克里金插值方法
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

    # 添加正则化项确保数值稳定性
    K[:n, :n] += np.eye(n) * 1e-8

    # 构建KD树用于快速邻居搜索
    tree = cKDTree(stations)

    # 对每个网格点进行插值
    for i in range(m):
        # 找到最近的邻居站点
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
            weights = solve(local_K, k_vec, assume_a='sym')

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

    # 计算R² (决定系数)
    r2 = r2_score(target, Interpolated)
    print(f"R²: {r2:.4f}")

    evaluation_result = {
        "time": [file_time],
        "k_zhe": [k_zhe],
        "neighborhood_size": [neighborhood_size],
        "correlation": [correlation],
        "mean_error": [mean_error],
        "mae": [mae],
        "rmse": [rmse],
        "r2": [r2]
    }
    evaluation_df = pd.DataFrame(evaluation_result)
    file_name = "evaluation_df.csv"
    evaluation_df.to_csv(file_name, mode='a', header=not pd.io.common.file_exists(file_name), index=False)
    print("新内容已追加到CSV文件。")

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
    print("计算目标变量变差函数...")
    t_sill, t_range = calculate_variogram(stations, target_pwv)
    print(f"目标变量: 基台值={t_sill:.4f}, 变程={t_range:.4f}")

    # 创建插值网格
    # 使用固定边界
    padding = 0.1
    lon_min, lon_max = data['Lon'].min() - padding, data['Lon'].max() + padding
    lat_min, lat_max = data['Lat'].min() - padding, data['Lat'].max() + padding

    # lon_min, lon_max = 108.41, 116.13
    # lat_min, lat_max = 24.64, 33.30

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
        t_sill, t_range,
        neighborhood_size
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
