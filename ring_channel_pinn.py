"""
简化稳定版 PINN - 环形通道火焰传播
避免版本兼容性问题，专注于核心功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib import cm

# 设置设备和随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")


class SimpleRingPINN(nn.Module):
    """
    简化的PINN网络
    """
    
    def __init__(self, n_hidden=64, n_layers=6):
        super(SimpleRingPINN, self).__init__()
        
        # 构建网络层
        layers = []
        
        # 输入层 (4个特征)
        layers.append(nn.Linear(4, n_hidden))
        layers.append(nn.Tanh())
        
        # 隐藏层
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        
        # 输出层 (2个输出: T和Y)
        layers.append(nn.Linear(n_hidden, 2))
        
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, theta, t):
        """
        前向传播
        """
        # 创建输入特征（处理周期性）
        x1 = torch.cos(theta)
        x2 = torch.sin(theta)
        x3 = t
        x4 = t * t  # 时间的二次项
        
        inputs = torch.cat([x1, x2, x3, x4], dim=1)
        output = self.net(inputs)
        
        # 分离温度和燃料，确保在物理范围内
        T = 1.0 + torch.relu(output[:, 0:1]) * 2.0  # T ∈ [1.0, 3.0]
        Y = torch.sigmoid(output[:, 1:2])            # Y ∈ [0, 1]
        
        return T, Y


def train_pinn(model, n_epochs=3000, lr=1e-3):
    """
    训练PINN模型
    """
    # 物理参数
    alpha = 0.02    # 热扩散
    D = 0.01        # 质量扩散
    Da = 50.0       # Damköhler数
    Q = 6.0         # 热释放
    beta = 4.0      # 活化能
    r_mid = 0.9     # 环中心半径
    t_max = 1.0     # 最大时间
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 损失历史
    losses = []
    
    print("\n" + "="*60)
    print("开始训练 PINN...")
    print("="*60)
    
    for epoch in range(n_epochs):
        # ========== 1. 生成训练点 ==========
        n_col = 1000  # 搭配点数量
        n_ic = 500    # 初始条件点数量
        n_bc = 200    # 边界条件点数量
        
        # PDE搭配点
        theta_pde = torch.rand((n_col, 1), device=device) * 2 * np.pi
        t_pde = torch.rand((n_col, 1), device=device) * t_max
        theta_pde.requires_grad = True
        t_pde.requires_grad = True
        
        # 初始条件点 (t=0)
        theta_ic = torch.rand((n_ic, 1), device=device) * 2 * np.pi
        t_ic = torch.zeros((n_ic, 1), device=device)
        
        # 周期边界点
        t_bc = torch.rand((n_bc, 1), device=device) * t_max
        
        # ========== 2. 计算PDE残差 ==========
        # 获取预测值和导数
        T, Y = model(theta_pde, t_pde)
        
        # 计算导数
        T_t = torch.autograd.grad(T.sum(), t_pde, create_graph=True)[0]
        T_theta = torch.autograd.grad(T.sum(), theta_pde, create_graph=True)[0]
        T_theta_theta = torch.autograd.grad(T_theta.sum(), theta_pde, create_graph=True)[0]
        
        Y_t = torch.autograd.grad(Y.sum(), t_pde, create_graph=True)[0]
        Y_theta = torch.autograd.grad(Y.sum(), theta_pde, create_graph=True)[0]
        Y_theta_theta = torch.autograd.grad(Y_theta.sum(), theta_pde, create_graph=True)[0]
        
        # 反应速率 (简化的Arrhenius)
        T_norm = (T - 1.0) / 2.0
        omega = Da * Y * torch.sigmoid(beta * (T_norm - 0.2))
        
        # PDE残差
        res_T = T_t - alpha / (r_mid**2) * T_theta_theta - Q * omega
        res_Y = Y_t - D / (r_mid**2) * Y_theta_theta + omega
        
        loss_pde = torch.mean(res_T**2) + torch.mean(res_Y**2)
        
        # ========== 3. 初始条件损失 ==========
        T_ic_pred, Y_ic_pred = model(theta_ic, t_ic)
        
        # 初始点火脉冲 (在θ=π处)
        ignition_center = np.pi
        ignition_width = 0.3
        
        # 计算角度距离
        dtheta = torch.abs(theta_ic - ignition_center)
        dtheta = torch.minimum(dtheta, 2*np.pi - dtheta)
        
        # 初始温度和燃料分布
        T_ic_true = 1.0 + 1.0 * torch.exp(-10 * (dtheta/ignition_width)**2)
        Y_ic_true = 1.0 - 0.5 * torch.exp(-10 * (dtheta/ignition_width)**2)
        
        loss_ic = torch.mean((T_ic_pred - T_ic_true)**2) + \
                  torch.mean((Y_ic_pred - Y_ic_true)**2)
        
        # ========== 4. 周期边界条件损失 ==========
        theta_0 = torch.zeros((n_bc, 1), device=device)
        theta_2pi = torch.full((n_bc, 1), 2*np.pi, device=device)
        
        T_0, Y_0 = model(theta_0, t_bc)
        T_2pi, Y_2pi = model(theta_2pi, t_bc)
        
        loss_bc = torch.mean((T_0 - T_2pi)**2) + torch.mean((Y_0 - Y_2pi)**2)
        
        # ========== 5. 总损失（动态权重）==========
        if epoch < 500:
            # 早期强调初始条件
            w_pde, w_ic, w_bc = 0.1, 100.0, 10.0
        elif epoch < 1500:
            # 中期平衡
            w_pde, w_ic, w_bc = 1.0, 10.0, 10.0
        else:
            # 后期强调PDE
            w_pde, w_ic, w_bc = 10.0, 1.0, 10.0
        
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        
        # ========== 6. 优化步骤 ==========
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # ========== 7. 打印进度 ==========
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:4d}/{n_epochs} | Loss: {loss.item():.6f}")
            print(f"  ├─ PDE: {loss_pde.item():.6f}")
            print(f"  ├─ IC:  {loss_ic.item():.6f}")
            print(f"  └─ BC:  {loss_bc.item():.6f}")
    
    print("\n✅ 训练完成!")
    return losses


def visualize_results(model, t_max=1.0):
    """
    可视化结果
    """
    # 环形通道参数
    r_inner = 0.8
    r_outer = 1.0
    
    # 时间点
    t_vals = np.linspace(0, t_max, 6)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 角度网格
    n_theta = 200
    theta = np.linspace(0, 2*np.pi, n_theta)
    
    model.eval()
    with torch.no_grad():
        for idx, t in enumerate(t_vals):
            ax = axes[idx]
            ax.set_aspect('equal')
            
            # 预测温度
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device).reshape(-1, 1)
            t_tensor = torch.full_like(theta_tensor, t)
            
            T_pred, Y_pred = model(theta_tensor, t_tensor)
            T_pred = T_pred.cpu().numpy().flatten()
            Y_pred = Y_pred.cpu().numpy().flatten()
            
            # 绘制环形通道
            for i in range(len(theta)-1):
                # 温度归一化到颜色
                T_norm = (T_pred[i] - 1.0) / 1.5
                T_norm = np.clip(T_norm, 0, 1)
                color = cm.hot(T_norm)
                
                # 创建楔形
                wedge = Wedge((0, 0), r_outer,
                            np.degrees(theta[i]),
                            np.degrees(theta[i+1]),
                            width=r_outer-r_inner,
                            facecolor=color,
                            edgecolor='none')
                ax.add_patch(wedge)
            
            # 添加边界圆
            circle_out = Circle((0, 0), r_outer, fill=False, 
                               edgecolor='black', linewidth=2)
            circle_in = Circle((0, 0), r_inner, fill=False, 
                              edgecolor='black', linewidth=2)
            ax.add_patch(circle_out)
            ax.add_patch(circle_in)
            
            # 标记最高温度位置
            max_idx = np.argmax(T_pred)
            max_theta = theta[max_idx]
            r_mid = (r_inner + r_outer) / 2
            ax.plot(r_mid * np.cos(max_theta), 
                   r_mid * np.sin(max_theta), 
                   'w*', markersize=12, markeredgecolor='yellow', 
                   markeredgewidth=2, label=f'Max T={T_pred[max_idx]:.2f}')
            
            # 设置坐标轴
            ax.set_xlim(-1.2*r_outer, 1.2*r_outer)
            ax.set_ylim(-1.2*r_outer, 1.2*r_outer)
            ax.set_title(f't = {t:.3f}')
            ax.axis('off')
            
            # 在第一个子图添加图例
            if idx == 0:
                ax.legend(loc='upper right')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cm.hot, 
                              norm=plt.Normalize(vmin=1.0, vmax=2.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                       fraction=0.046, pad=0.08)
    cbar.set_label('Temperature', fontsize=12)
    
    plt.suptitle('🔥 Ring Channel Flame Propagation (PINN)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_profiles(model, t_max=1.0):
    """
    绘制温度和燃料剖面
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 角度网格
    theta = np.linspace(0, 2*np.pi, 200)
    theta_deg = np.degrees(theta)
    
    # 不同时刻
    times = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(times)))
    
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(times):
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device).reshape(-1, 1)
            t_tensor = torch.full_like(theta_tensor, t)
            
            T_pred, Y_pred = model(theta_tensor, t_tensor)
            T_pred = T_pred.cpu().numpy().flatten()
            Y_pred = Y_pred.cpu().numpy().flatten()
            
            ax1.plot(theta_deg, T_pred, color=colors[i], 
                    label=f't={t:.1f}', linewidth=2)
            ax2.plot(theta_deg, Y_pred, color=colors[i], 
                    label=f't={t:.1f}', linewidth=2)
    
    ax1.set_xlabel('Angle θ (degrees)')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Temperature Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    
    ax2.set_xlabel('Angle θ (degrees)')
    ax2.set_ylabel('Fuel Fraction')
    ax2.set_title('Fuel Concentration Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    
    plt.suptitle('Flame Profiles Evolution')
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("🔥 简化稳定版 PINN - 环形通道火焰传播")
    print("="*70)
    
    # 创建模型
    model = SimpleRingPINN(n_hidden=64, n_layers=6).to(device)
    
    # 训练模型
    losses = train_pinn(model, n_epochs=3000, lr=5e-4)
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 可视化结果
    print("\n📊 生成可视化...")
    visualize_results(model)
    plot_profiles(model)
    
    # 测量火焰速度
    print("\n📏 分析火焰传播...")
    theta = np.linspace(0, 2*np.pi, 100)
    times = [0.2, 0.4, 0.6, 0.8]
    positions = []
    
    model.eval()
    with torch.no_grad():
        for t in times:
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device).reshape(-1, 1)
            t_tensor = torch.full_like(theta_tensor, t)
            T_pred, _ = model(theta_tensor, t_tensor)
            T_pred = T_pred.cpu().numpy().flatten()
            
            max_idx = np.argmax(T_pred)
            max_pos = theta[max_idx]
            positions.append(max_pos)
            print(f"  t={t:.2f}: θ = {np.degrees(max_pos):6.1f}°, T_max = {T_pred[max_idx]:.3f}")
    
    # 估算速度
    if len(positions) > 1:
        # 处理周期性边界
        positions = np.array(positions)
        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] < -np.pi:
                positions[i] += 2*np.pi
        
        # 线性拟合
        coeffs = np.polyfit(times, positions, 1)
        speed = coeffs[0]
        
        print(f"\n  估算火焰速度: {speed:.3f} rad/s")
        print(f"  物理速度: ~{abs(speed) * 0.9 * 100:.1f} cm/s (在r=0.9处)")
    
    print("\n" + "="*70)
    print("✅ 模拟完成!")
    print("="*70)
    
    return model


if __name__ == "__main__":
    model = main()