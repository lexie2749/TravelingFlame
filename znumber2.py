import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 定义一个函数来计算泽尔多维奇数
def calculate_zeldovich_number(Ea, Tb, Tu):
    """
    计算泽尔多维奇数 (β).
    
    参数:
    Ea (float or array): 活化能 (J/mol)
    Tb (float or array): 已燃气体温度 (K)
    Tu (float): 未燃气体温度 (K)
    
    返回:
    float or array: 无量纲的泽尔多维奇数.
    """
    Ru = 8.314  # 普适气体常数 (J/mol·K)
    # 避免当Tb=0时除法错误
    if np.any(Tb == 0):
        return np.inf
    
    beta = (Ea * (Tb - Tu)) / (Ru * Tb**2)
    return beta

# --- 1. 设定基准物理参数 ---
# 我们将固定未燃气体温度，变化另外两个参数
baseline_Tu = 300       # K (室温)

# --- 2. 为3D绘图创建二维参数网格 ---
# 定义活化能 (Ea) 和已燃气体温度 (Tb) 的范围
Ea_range = np.linspace(80000, 200000, 100)  # 80 to 200 kJ/mol
Tb_range = np.linspace(1500, 2500, 100)     # 1500 to 2500 K

# 使用 meshgrid 创建二维网格
Ea_grid, Tb_grid = np.meshgrid(Ea_range, Tb_range)

# --- 3. 在整个网格上计算泽尔多维奇数 ---
beta_grid = calculate_zeldovich_number(Ea_grid, Tb_grid, baseline_Tu)

# --- 4. 创建并显示3D图表 ---

print("Generating 3D Surface Diagram...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图. 将Ea转换为kJ/mol以便于阅读
# plot_surface的x, y, z分别是Ea_grid, Tb_grid, beta_grid
surf = ax.plot_surface(Ea_grid / 1000, Tb_grid, beta_grid, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

# 添加一个颜色条，它将数值映射到颜色
cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label(r'Zel\'dovich Number, $\beta$', fontsize=12)

# 设置所有三个坐标轴的标题和标签
ax.set_title(r'3D Diagram of Zel\'dovich Number ($\beta$)', fontsize=16)
ax.set_xlabel(r'Activation Energy, $E_a$ (kJ/mol)', fontsize=12, labelpad=10)
ax.set_ylabel(r'Burned Gas Temperature, $T_b$ (K)', fontsize=12, labelpad=10)
ax.set_zlabel(r'Zel\'dovich Number, $\beta$', fontsize=12, labelpad=10)

# 调整视角
ax.view_init(elev=25, azim=-120)

# 显示图形
plt.tight_layout()
plt.show()