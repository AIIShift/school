import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

# 1. 读取点数据（CSV 中必须包含 Lon, Lat, VAP, PWV 四列）
df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\百度\train.csv')
lons = df['Lon'].values
lats = df['Lat'].values
vaps = df['VAP'].values
pwvs = df['PWV'].values

# 2. 定义插值网格（分辨率 0.01°）
res = 0.01
lon_min, lon_max = lons.min()+0.1, lons.max()+0.1
lat_min, lat_max = lats.min()+0.1, lats.max()+0.1
grid_lon = np.arange(lon_min, lon_max + res, res)
grid_lat = np.arange(lat_min, lat_max + res, res)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

# ——————————————————————————————————————
# 3. 普通克里金插值 VAP（协变量）
#    使用 PyKrige 的 OrdinaryKriging（自动拟合 variogram）
# ——————————————————————————————————————
from pykrige.ok import O

OK_vap = OrdinaryKriging(
    lons, lats, vaps,
    variogram_model='exponential',  # 半变异函数可选 'linear','spherical','gaussian' 等
    verbose=False, enable_plotting=False
)
vap_grid, vap_ss = OK_vap.execute('grid', grid_lon, grid_lat)
# vap_grid.shape == (n_lat, n_lon) :contentReference[oaicite:0]{index=0}

# ——————————————————————————————————————
# 4. 共克里金插值 PWV
#    使用 pyKriging 的 coKriging 接口
# ——————————————————————————————————————
from pykrige.ck import ClassificationKriging

# 训练时：X1=z1 为主变量（PWV），X2=z2 为次变量（VAP）
coords = np.column_stack([lons, lats])
ck = coKriging.coKriging(coords, pwvs, coords, vaps)
ck.train(covModel='exponential')  # 同样使用指数模型 :contentReference[oaicite:1]{index=1}

# 预测时：对网格中心点做预测
grid_points = np.column_stack([grid_lon_mesh.ravel(), grid_lat_mesh.ravel()])
pwv_pred, pwv_var = ck.predict(grid_points)

# 重塑为栅格
pwv_grid = pwv_pred.reshape(grid_lon_mesh.shape)
pwv_var_grid = pwv_var.reshape(grid_lon_mesh.shape)

# ——————————————————————————————————————
# 5. 保存结果到 CSV
# ——————————————————————————————————————
out = pd.DataFrame({
    'Lon': grid_lon_mesh.ravel(),
    'Lat': grid_lat_mesh.ravel(),
    'VAP_interp': vap_grid.ravel(),
    'PWV_coKrig': pwv_pred,
    'PWV_var': pwv_var
})
out.to_csv('PWV_coKrig_results.csv', index=False)
print("已保存：PWV_coKrig_results.csv")