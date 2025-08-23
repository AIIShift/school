import numpy as np
import pandas as pd
from scipy.interpolate import griddata  # 导入双线性插值函数
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def bilinear_interpolation(stations, values, grid_points):
    """双线性插值函数"""
    # 使用griddata进行双线性插值
    interpolated_values = griddata(
        points=stations,
        values=values,
        xi=grid_points,
        method='linear',  # 双线性插值
        fill_value=np.nan  # 对于边界外的点填充为NaN
    )

    # 处理边界外的点 - 使用最近邻插值填充
    if np.isnan(interpolated_values).any():
        nearest_values = griddata(
            points=stations,
            values=values,
            xi=grid_points,
            method='nearest'  # 最近邻插值
        )
        # 用最近邻插值填充NaN
        interpolated_values = np.where(
            np.isnan(interpolated_values),
            nearest_values,
            interpolated_values
        )

    return interpolated_values


def cross_validate_bilinear(stations, values, k=5):
    """
    双线性插值的交叉验证
    :param stations: 站点坐标数组
    :param values: 站点值数组
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
        test_predictions = griddata(
            points=train_stations,
            values=train_values,
            xi=test_stations,
            method='linear',
            fill_value=np.nan
        )

        # 处理可能出现的NaN值（边界点）
        nan_mask = np.isnan(test_predictions)
        if np.any(nan_mask):
            nearest_predictions = griddata(
                points=train_stations,
                values=train_values,
                xi=test_stations[nan_mask],
                method='nearest'
            )
            test_predictions[nan_mask] = nearest_predictions

        # 存储预测结果
        predictions[test_idx] = test_predictions
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
    print("PWV统计信息:")
    print(f"最小值: {np.min(target_pwv):.4f}, 最大值: {np.max(target_pwv):.4f}")
    print(f"均值: {np.mean(target_pwv):.4f}, 标准差: {np.std(target_pwv):.4f}")

    # 2. 执行交叉验证
    print("\n开始交叉验证双线性插值模型...")
    cv_predictions, metrics = cross_validate_bilinear(
        stations, target_pwv, k=20
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
    cv_output_file = 'pwv_bilinear_cv_results.csv'
    cv_results.to_csv(cv_output_file, index=False)
    print(f"\n交叉验证结果已保存至: {cv_output_file}")

    # 3. 创建插值网格
    # 使用固定边界
    lon_min, lon_max = 108, 115
    lat_min, lat_max = 24, 31

    resolution = 0.01  # 网格分辨率
    print(
        f"\n创建网格: 经度范围({lon_min:.2f}-{lon_max:.2f}), 纬度范围({lat_min:.2f}-{lat_max:.2f}), 分辨率={resolution:.2f}")

    # 生成网格点
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    print(f"网格点数: {len(grid_points)}")

    # 4. 执行双线性插值
    print("\n开始双线性插值...")
    pwv_grid = bilinear_interpolation(stations, target_pwv, grid_points)
    print("插值完成!")

    # 后处理：移除负值
    pwv_grid = np.clip(pwv_grid, 0, None)

    # 5. 创建结果DataFrame并保存为CSV
    results = pd.DataFrame({
        'Lon': grid_points[:, 0],
        'Lat': grid_points[:, 1],
        'PWV': pwv_grid
    })

    # 保存结果
    output_file = 'pwv_bilinear_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\n插值结果已保存至: {output_file}")
    print(f"总网格点数: {len(results)}")
    print(f"插值结果统计: 最小值={np.min(pwv_grid):.4f}, 最大值={np.max(pwv_grid):.4f}, 均值={np.mean(pwv_grid):.4f}")