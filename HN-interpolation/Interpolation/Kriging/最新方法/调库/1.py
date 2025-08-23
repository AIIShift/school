import re
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging


# 假设你的原始数据已读入一个 DataFrame，列名分别是：
#   station_id  — 站点编号（字符串）
#   lon         — 经度 (浮点)
#   lat         — 纬度 (浮点)
#   value       — 测量值 (浮点，估算点若无观测则可填 NaN 或者跳过)
file_time = "01"
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

    # 拟合克里金模型
    OK = OrdinaryKriging(
        lons_train, lats_train, vals_train,
        variogram_model="exponential",
        nlags=6,
        verbose=False,
        enable_plotting=False
    )

    # 在留出的点上做预测
    z_pred, ss = OK.execute("points", np.array([lon_pt]), np.array([lat_pt]))

    # # 使用泛化克里金
    # UK = UniversalKriging(
    #     lons_train, lats_train, vals_train,
    #     variogram_model="exponential",  # 选择变异函数
    #     variogram_parameters=None,  # 可选择定义变异函数参数，或使用默认设置
    #     nlags=6,
    #     verbose=False,
    #     enable_plotting=False
    # )
    #
    # # 执行预测
    # z_pred, ss = UK.execute("points", np.array([lon_pt]), np.array([lat_pt]))

    pred_val = float(z_pred[0])
    var_val = float(ss[0])

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
results_df.to_csv("loo_cv_results-泛化.csv", index=False)

# （可选）计算总体误差统计
rmse = np.sqrt(np.mean(results_df["error"] ** 2))
mae = np.mean(np.abs(results_df["error"]))
print(f"LOOCV 完成，RMSE = {rmse:.3f}，MAE = {mae:.3f}")
print("详细结果已保存为 loo_cv_results.csv")
