import pandas as pd
import numpy as np
import math

# 定义常量，外半径为0.15米
r_outer = 0.15 # m
# 定义体积转换因子，将升转换为立方米
liter_to_m3 = 0.001 # 1 L = 0.001 m^3
# 定义体积转换因子，将微升转换为立方米
volume_conversion_factor = 1e-6 * liter_to_m3 # 1 microliter = 1e-6 L = 1e-9 m^3

# 定义体积范围，从125微升到200微升，步长为5微升
volumes_ul = np.arange(125, 201, 5) # in microliters
# 定义x值范围，从0.003米到0.010米，步长为0.001米
x_values_m = np.arange(0.003, 0.011, 0.001) # in meters

# 创建一个空列表用于存储计算结果
results = []

# 遍历每一个体积值
for vol_ul in volumes_ul:
    # 将微升体积转换为立方米
    vol_m3 = vol_ul * volume_conversion_factor 

    # 对于每一个体积，遍历每一个x值
    for x in x_values_m:
        # 计算内半径
        r_inner = r_outer - x
        # 根据公式计算面积（单位：平方米）
        area_m2 = math.pi * (r_outer**2 - r_inner**2)
        # 计算长度（单位：米）
        length_m = vol_m3 / area_m2

        # 将结果以字典形式添加到列表中
        results.append({
            '体积 (微升)': vol_ul,
            'x 值 (米)': x,
            '面积 (平方米)': area_m2,
            '长度 (米)': length_m
        })

# 使用pandas库将结果列表转换为DataFrame表格
df = pd.DataFrame(results)

# 将DataFrame保存为名为 'calculation_results_fixed.csv' 的CSV文件
df.to_csv('calculation_results_fixed.csv', index=False, encoding='utf-8-sig')