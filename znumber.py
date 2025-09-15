import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数来计算泽尔多维奇数
def calculate_zeldovich_number(Ea, Tb, Tu):
    """
    计算泽尔多维奇数 (β).
    
    参数:
    Ea (float): 活化能 (J/mol)
    Tb (float): 已燃气体温度 (K)
    Tu (float): 未燃气体温度 (K)
    
    返回:
    float: 无量纲的泽尔多维奇数.
    """
    Ru = 8.314  # 普适气体常数 (J/mol·K)
    # 避免当Tb=0时除法错误
    if np.any(Tb == 0):
        return np.inf
    
    beta = (Ea * (Tb - Tu)) / (Ru * Tb**2)
    return beta

# --- 1. 设定基准物理参数 ---
# 这些是典型的碳氢燃料燃烧的代表值
baseline_Ea = 140000    # 140 kJ/mol
baseline_Tb = 2200      # K
baseline_Tu = 300       # K (室温)

# 计算基准泽尔多维奇数
baseline_beta = calculate_zeldovich_number(baseline_Ea, baseline_Tb, baseline_Tu)
print(f"基准泽尔多维奇数 (β): {baseline_beta:.2f}")
print("一个大的 β 值 (通常 > 5) 意味着反应对温度高度敏感，火焰锋面很薄。")

# --- 2. 为绘图创建参数范围 ---
# 我们将每个参数在其基准值的±50%范围内变动
Ea_range = np.linspace(0.5 * baseline_Ea, 1.5 * baseline_Ea, 200)
Tb_range = np.linspace(0.5 * baseline_Tb, 1.5 * baseline_Tb, 200)
Tu_range = np.linspace(0.5 * baseline_Tu, 1.5 * baseline_Tu, 200)

# --- 3. 计算每个范围内的泽尔多维奇数 ---
beta_vs_Ea = calculate_zeldovich_number(Ea_range, baseline_Tb, baseline_Tu)
beta_vs_Tb = calculate_zeldovich_number(baseline_Ea, Tb_range, baseline_Tu)
beta_vs_Tu = calculate_zeldovich_number(baseline_Ea, baseline_Tb, Tu_range)

# --- 4. 依次创建并显示图表 ---

# 图 1: β vs. 活化能 (Ea)
plt.figure(figsize=(8, 6))
plt.plot(Ea_range / 1000, beta_vs_Ea, color='crimson', linewidth=2.5)
plt.title(r'Zel\'dovich Number vs. Activation Energy', fontsize=14)
plt.xlabel(r'Activation Energy, $E_a$ (kJ/mol)', fontsize=12)
plt.ylabel(r'Zel\'dovich Number, $\beta$', fontsize=12)
plt.axvline(baseline_Ea / 1000, color='black', linestyle='--', 
           label=f'Baseline Ea = {baseline_Ea/1000:.0f} kJ/mol')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()

# 图 2: β vs. 已燃气体温度 (Tb)
plt.figure(figsize=(8, 6))
plt.plot(Tb_range, beta_vs_Tb, color='darkorange', linewidth=2.5)
plt.title(r'Zel\'dovich Number vs. Burned Gas Temperature', fontsize=14)
plt.xlabel(r'Burned Gas Temperature, $T_b$ (K)', fontsize=12)
plt.ylabel(r'Zel\'dovich Number, $\beta$', fontsize=12)
plt.axvline(baseline_Tb, color='black', linestyle='--', 
           label=f'Baseline Tb = {baseline_Tb:.0f} K')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()

# 图 3: β vs. 未燃气体温度 (Tu)
plt.figure(figsize=(8, 6))
plt.plot(Tu_range, beta_vs_Tu, color='forestgreen', linewidth=2.5)
plt.title(r'Zel\'dovich Number vs. Unburned Gas Temperature', fontsize=14)
plt.xlabel(r'Unburned Gas Temperature, $T_u$ (K)', fontsize=12)
plt.ylabel(r'Zel\'dovich Number, $\beta$', fontsize=12)
plt.axvline(baseline_Tu, color='black', linestyle='--', 
           label=f'Baseline Tu = {baseline_Tu:.0f} K')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()