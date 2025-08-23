import pandas as pd
import chardet

# 自动检测文件编码
with open(r'D:\PyCharm\pythonProject\HN-interpolation\20250603\Z_UPAR_C_HNHB_20250603200000_P_GNSSMET_PWV_HOR.csv', 'rb') as file:
    result = chardet.detect(file.read())
    print(result)

df = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\20250603\Z_UPAR_C_HNHB_20250603200000_P_GNSSMET_PWV_HOR.csv', encoding='utf-8-sig')

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

# #  #  调整列的位置，获取当前列的顺
new_order = ['id', 'Station_Id_C', 'Datetime', 'Lat', 'Lon', 'Alti', 'City', 'Station_Name', 'Cnty',
             'VAP', 'RHU', 'PRS', 'TEM', 'PWV']
df = df[new_order]

# # 添加 'dataset' 列，根据 'id' 列的值判断是否为 5 的倍数
df['dataset'] = df['id'].apply(lambda x: 'test' if x % 5 == 0 else 'train')


df.to_csv('HNHB-2025060320.csv', encoding='utf-8-sig', index=False)
