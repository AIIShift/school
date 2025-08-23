import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv("e.csv")  # 或替换为你的具体路径

# 2. 删除任何包含空值的行
df_clean = df.dropna()

# 3. 保存清洗后的数据
df_clean.to_csv("era5_relative_humidity_clean.csv", index=False, encoding='utf-8-sig')

print("✅ 已删除包含空值的行，保存为 'era5_relative_humidity_clean.csv'")
