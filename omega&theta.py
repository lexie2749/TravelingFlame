import numpy as np
import matplotlib.pyplot as plt

# 定义omega函数
def omega(theta, beta):
  """
  根据theta和beta计算omega的值
  """
  return (beta**2 / 2) * (1 - theta) * np.exp(-beta * (1 - theta))

# 设定beta的值
beta = 5.95

# 创建一个从0到1的theta值数组
theta = np.linspace(0, 1, 500)

# 计算对应的omega值
omega_vals = omega(theta, beta)

# 创建图像
plt.figure(figsize=(10, 6))

# 绘制omega vs. theta曲线
plt.plot(theta, omega_vals, label=f'$\\beta = {beta}$')

# 添加标签、标题、图例和网格以提高可读性
plt.xlabel('temperature', fontsize=14)
plt.ylabel('reaction rate', fontsize=14)
plt.legend()
plt.grid(False)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# 将图像保存到文件
plt.savefig('omega_vs_theta_beta_5_95.png')

print("图像已生成并保存为 'omega_vs_theta_beta_5_95.png'")