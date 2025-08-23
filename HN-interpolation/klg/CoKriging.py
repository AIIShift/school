import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取CSV文件
file_path = 'CS-2025060300.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 提取经纬度、协变量（RHU, VAP）和目标变量（PWV）
lons = data['Lon'].values  # 经度
lats = data['Lat'].values  # 纬度
RHU = data['RHU'].values  # 协变量RHU
VAP = data['VAP'].values  # 协变量VAP
PWV = data['PWV'].values  # 目标变量PWV

# 2. 设置插值的边界和网格分辨率
lon_min, lon_max = 108, 115
lat_min, lat_max = 24, 31
grid_resolution = 0.01  # 网格分辨率

# 创建网格
lon_grid = np.arange(lon_min, lon_max, grid_resolution)
lat_grid = np.arange(lat_min, lat_max, grid_resolution)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)


# 3. 球形变异函数
def spherical_variogram(h, sill, range_, nugget):
    """球形变异函数"""
    return np.where(h < range_, nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3), sill)


# 4. 计算目标变量和多个协变量的变异函数
def compute_variogram(data, lons, lats, sill, range_, nugget):
    n = len(data)
    distance_matrix = np.zeros((n, n))
    variogram_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # 计算经纬度之间的距离
            dist = np.sqrt((lons[i] - lons[j]) ** 2 + (lats[i] - lats[j]) ** 2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            # 计算变异函数值
            variogram_matrix[i, j] = spherical_variogram(dist, sill, range_, nugget)
            variogram_matrix[j, i] = variogram_matrix[i, j]

    return distance_matrix, variogram_matrix


# 设置变异函数的参数
sill = 1.0
range_ = 1.0
nugget = 0.0

# 计算PWV、RHU和VAP的变异函数
_, pwv_variogram = compute_variogram(PWV, lons, lats, sill, range_, nugget)
_, rhu_variogram = compute_variogram(RHU, lons, lats, sill, range_, nugget)
_, vap_variogram = compute_variogram(VAP, lons, lats, sill, range_, nugget)


# 5. 计算协方差矩阵
def compute_covariance_matrix(pwv_variogram, rhu_variogram, vap_variogram, sill, range_, nugget):
    covariance_matrix = np.zeros((len(pwv_variogram), len(pwv_variogram)))
    for i in range(len(pwv_variogram)):
        for j in range(i + 1, len(pwv_variogram)):
            # 计算目标变量和协变量之间的协方差
            covariance_matrix[i, j] = np.sqrt(pwv_variogram[i, j] * rhu_variogram[i, j] * vap_variogram[i, j])
            covariance_matrix[j, i] = covariance_matrix[i, j]
    return covariance_matrix


covariance_matrix = compute_covariance_matrix(pwv_variogram, rhu_variogram, vap_variogram, sill, range_, nugget)


# 6. 解决克里格方程进行插值
def cokriging_interpolation(cov_matrix, target_values, covariates, grid_points):
    n = len(target_values)
    interpolated_values = np.zeros(grid_points.shape[0])

    # 求解克里格方程（此处为简化版本，实际中需要更复杂的求解过程）
    for i, point in enumerate(grid_points):
        # 计算点到所有已知点的距离和协方差
        distances = np.linalg.norm(covariates - point, axis=1)
        covariances = np.exp(-distances)  # 使用指数衰减的协方差模型

        # 解决克里格方程的插值
        interpolated_values[i] = np.dot(covariances, target_values) / np.sum(covariances)

    return interpolated_values


# 假设插值网格的点
grid_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
interpolated_pwvs = cokriging_interpolation(covariance_matrix, PWV, np.column_stack([lons, lats]), grid_points)

# 7. 将插值结果保存到CSV文件
interpolated_pwvs = interpolated_pwvs.reshape(lon_mesh.shape)

# 创建一个DataFrame，包含网格点和对应的插值结果
output_data = pd.DataFrame({
    'Longitude': lon_mesh.ravel(),
    'Latitude': lat_mesh.ravel(),
    'Interpolated_PWV': interpolated_pwvs.ravel()
})

# 输出为CSV文件
output_data.to_csv('interpolated_results_with_VAP.csv', index=False)

print("插值后的数据已保存到 'interpolated_results_with_VAP.csv'")
