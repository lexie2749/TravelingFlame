import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Wedge
import matplotlib.animation as animation

class RingFlamePropagation:
    """
    使用有限差分法直接求解环形槽中的火焰传播
    这是确定性的方法，不依赖于神经网络优化
    """
    
    def __init__(self, n_theta=200, n_t=500, dt=0.0002):
        """
        初始化参数
        n_theta: 空间网格点数
        n_t: 时间步数
        dt: 时间步长
        """
        # 网格参数
        self.n_theta = n_theta
        self.n_t = n_t
        self.dt = dt
        
        # 空间网格
        self.theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        self.dtheta = self.theta[1] - self.theta[0]
        
        # 物理参数（无量纲）
        self.alpha = 0.01  # 热扩散系数
        self.D = 0.008     # 质量扩散系数
        self.Le = self.D / self.alpha  # Lewis数
        self.Da = 50.0     # Damköhler数
        self.Q = 6.0       # 热释放参数
        self.beta = 6.0    # Zeldovich数（活化能）
        
        # 初始化场变量
        self.T = np.ones(n_theta)  # 温度
        self.Y = np.ones(n_theta)  # 燃料质量分数
        
        # 存储历史
        self.T_history = []
        self.Y_history = []
        self.omega_history = []
        
        # 设置初始条件
        self.set_initial_condition()
        
    def set_initial_condition(self):
        """设置初始条件 - 局部点火"""
        # 在θ=π处点火（局部高温）
        ignition_center = np.pi
        ignition_width = 0.3
        
        for i, theta in enumerate(self.theta):
            # 计算到点火中心的距离
            dist = np.abs(theta - ignition_center)
            dist = min(dist, 2*np.pi - dist)  # 考虑周期性
            
            if dist < ignition_width:
                # 点火区域：高温，部分燃料消耗
                self.T[i] = 1.0 + 0.8 * np.exp(-10 * (dist/ignition_width)**2)
                self.Y[i] = 1.0 - 0.5 * np.exp(-10 * (dist/ignition_width)**2)
            else:
                # 未燃区域：低温，满燃料
                self.T[i] = 1.0
                self.Y[i] = 1.0
    
    def reaction_rate(self, T, Y):
        """计算反应速率（Arrhenius）"""
        # 避免低温反应
        T_eff = np.maximum(T - 1.0, 0)
        
        # Arrhenius速率
        omega = self.Da * Y * np.exp(self.beta * T_eff / (T + 0.1))
        
        # 限制最大速率
        omega = np.minimum(omega, 100.0)
        
        return omega
    
    def laplacian_periodic(self, f):
        """计算周期边界条件下的拉普拉斯算子"""
        laplacian = np.zeros_like(f)
        
        for i in range(len(f)):
            i_plus = (i + 1) % len(f)
            i_minus = (i - 1) % len(f)
            
            laplacian[i] = (f[i_plus] - 2*f[i] + f[i_minus]) / (self.dtheta**2)
        
        return laplacian
    
    def gradient_periodic(self, f):
        """计算周期边界条件下的梯度（中心差分）"""
        gradient = np.zeros_like(f)
        
        for i in range(len(f)):
            i_plus = (i + 1) % len(f)
            i_minus = (i - 1) % len(f)
            
            gradient[i] = (f[i_plus] - f[i_minus]) / (2 * self.dtheta)
        
        return gradient
    
    def step(self):
        """时间步进（显式Euler）"""
        # 计算反应速率
        omega = self.reaction_rate(self.T, self.Y)
        
        # 计算拉普拉斯算子
        laplacian_T = self.laplacian_periodic(self.T)
        laplacian_Y = self.laplacian_periodic(self.Y)
        
        # 更新方程
        # ∂T/∂t = α∇²T + Q·ω
        dT_dt = self.alpha * laplacian_T + self.Q * omega
        
        # ∂Y/∂t = D∇²Y - ω
        dY_dt = self.D * laplacian_Y - omega
        
        # 时间步进（显式Euler）
        self.T = self.T + self.dt * dT_dt
        self.Y = self.Y + self.dt * dY_dt
        
        # 确保物理边界
        self.T = np.maximum(self.T, 1.0)  # 最低温度
        self.T = np.minimum(self.T, 3.0)  # 最高温度
        self.Y = np.maximum(self.Y, 0.0)  # 燃料不能为负
        self.Y = np.minimum(self.Y, 1.0)  # 燃料不能超过1
        
        # 存储历史
        self.T_history.append(self.T.copy())
        self.Y_history.append(self.Y.copy())
        self.omega_history.append(omega.copy())
    
    def simulate(self):
        """运行完整模拟"""
        print("🔥 开始有限差分模拟...")
        print(f"   网格: {self.n_theta} × {self.n_t}")
        print(f"   时间步长: {self.dt}")
        print(f"   总时间: {self.n_t * self.dt:.3f}")
        print("="*50)
        
        # 存储初始状态
        self.T_history.append(self.T.copy())
        self.Y_history.append(self.Y.copy())
        self.omega_history.append(self.reaction_rate(self.T, self.Y))
        
        # 时间步进
        for step in range(self.n_t):
            self.step()
            
            if (step + 1) % 100 == 0:
                # 计算火焰位置
                flame_idx = np.argmax(self.T)
                flame_pos = self.theta[flame_idx]
                T_max = np.max(self.T)
                T_min = np.min(self.T)
                omega_mean = np.mean(self.reaction_rate(self.T, self.Y))
                
                print(f"Step {step+1:4d}: θ_flame={np.degrees(flame_pos):6.1f}°, "
                      f"T∈[{T_min:.3f}, {T_max:.3f}], "
                      f"<ω>={omega_mean:.3f}")
        
        print("="*50)
        print("✅ 模拟完成！")
        
        # 转换为数组
        self.T_history = np.array(self.T_history)
        self.Y_history = np.array(self.Y_history)
        self.omega_history = np.array(self.omega_history)
    
    def measure_speed(self):
        """测量火焰传播速度"""
        if len(self.T_history) < 10:
            return 0.0
        
        # 追踪火焰前沿位置
        positions = []
        times = []
        
        for i in range(0, len(self.T_history), 50):
            T = self.T_history[i]
            flame_idx = np.argmax(T)
            positions.append(self.theta[flame_idx])
            times.append(i * self.dt)
        
        if len(positions) > 2:
            # 处理周期性跳变
            positions = np.array(positions)
            for i in range(1, len(positions)):
                if positions[i] - positions[i-1] > np.pi:
                    positions[i:] -= 2*np.pi
                elif positions[i] - positions[i-1] < -np.pi:
                    positions[i:] += 2*np.pi
            
            # 线性拟合
            times = np.array(times)
            p = np.polyfit(times, positions, 1)
            speed = p[0]  # rad/s
            
            return speed
        
        return 0.0
    
    def visualize(self):
        """可视化结果"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 温度时空图
        ax1 = plt.subplot(2, 3, 1)
        extent = [0, 360, 0, self.n_t * self.dt]
        im1 = ax1.imshow(self.T_history, aspect='auto', origin='lower', 
                        cmap='hot', extent=extent)
        ax1.set_title('Temperature Space-Time Evolution', fontsize=12)
        ax1.set_xlabel('Angle θ (degrees)')
        ax1.set_ylabel('Time (s)')
        plt.colorbar(im1, ax=ax1)
        
        # 测量并绘制火焰轨迹
        speed = self.measure_speed()
        if speed != 0:
            times = np.linspace(0, self.n_t * self.dt, 100)
            trajectory = 180 + np.degrees(speed * times)
            ax1.plot(trajectory % 360, times, 'w--', linewidth=2,
                    label=f'Speed: {speed:.3f} rad/s')
            ax1.legend(loc='upper left')
        
        # 2. 燃料时空图
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(self.Y_history, aspect='auto', origin='lower',
                        cmap='YlGn_r', extent=extent)
        ax2.set_title('Fuel Concentration', fontsize=12)
        ax2.set_xlabel('Angle θ (degrees)')
        ax2.set_ylabel('Time (s)')
        plt.colorbar(im2, ax=ax2)
        
        # 3. 反应速率时空图
        ax3 = plt.subplot(2, 3, 3)
        im3 = ax3.imshow(self.omega_history, aspect='auto', origin='lower',
                        cmap='YlOrRd', extent=extent, vmax=10)
        ax3.set_title('Reaction Rate ω', fontsize=12)
        ax3.set_xlabel('Angle θ (degrees)')
        ax3.set_ylabel('Time (s)')
        plt.colorbar(im3, ax=ax3)
        
        # 4. 温度剖面（不同时刻）
        ax4 = plt.subplot(2, 3, 4)
        n_profiles = 5
        step_interval = len(self.T_history) // n_profiles
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_profiles))
        
        for i in range(n_profiles):
            step = i * step_interval
            t = step * self.dt
            T = self.T_history[step]
            ax4.plot(np.degrees(self.theta), T, color=colors[i],
                    label=f't={t:.3f}s', linewidth=2)
        
        ax4.set_title('Temperature Profiles', fontsize=12)
        ax4.set_xlabel('Angle θ (degrees)')
        ax4.set_ylabel('Temperature')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. 燃料剖面
        ax5 = plt.subplot(2, 3, 5)
        for i in range(n_profiles):
            step = i * step_interval
            t = step * self.dt
            Y = self.Y_history[step]
            ax5.plot(np.degrees(self.theta), Y, color=colors[i],
                    label=f't={t:.3f}s', linewidth=2)
        
        ax5.set_title('Fuel Profiles', fontsize=12)
        ax5.set_xlabel('Angle θ (degrees)')
        ax5.set_ylabel('Fuel Fraction')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 6. 参数信息
        ax6 = plt.subplot(2, 3, 6)
        info_text = f"""
Physical Parameters:
• Thermal Diffusivity α = {self.alpha:.3f}
• Mass Diffusivity D = {self.D:.3f}
• Lewis Number Le = {self.Le:.3f}
• Damköhler Da = {self.Da:.1f}
• Heat Release Q = {self.Q:.1f}
• Activation Energy β = {self.beta:.1f}

Results:
• Flame Speed = {speed:.4f} rad/s
• Physical Speed ≈ {speed*5:.2f} cm/s
  (assuming R = 5 cm)
• Direction: {"Forward" if speed > 0 else "Backward"}
        """
        ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='center',
                fontfamily='monospace')
        ax6.axis('off')
        
        plt.suptitle('🔥 Ring Flame Propagation - Finite Difference Solution',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_animation(self, filename='flame_animation.gif', skip=10):
        """创建动画"""
        print("\n📹 创建动画...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        def animate(frame):
            # 清空
            ax1.clear()
            ax2.clear()
            
            # 获取当前状态
            step = frame * skip
            if step >= len(self.T_history):
                return
            
            T = self.T_history[step]
            Y = self.Y_history[step]
            t = step * self.dt
            
            # 左图：剖面
            ax1.plot(np.degrees(self.theta), T, 'r-', linewidth=2, label='Temperature')
            ax1.plot(np.degrees(self.theta), Y, 'g-', linewidth=2, label='Fuel')
            ax1.set_xlabel('Angle θ (degrees)')
            ax1.set_ylabel('Value')
            ax1.set_title(f'Profiles at t={t:.3f}s')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 2.5])
            
            # 右图：环形可视化
            ax2.set_aspect('equal')
            
            # 绘制环
            circle_out = Circle((0, 0), 1.0, fill=False, edgecolor='k', linewidth=2)
            circle_in = Circle((0, 0), 0.8, fill=False, edgecolor='k', linewidth=2)
            ax2.add_patch(circle_out)
            ax2.add_patch(circle_in)
            
            # 温度分布
            for i in range(len(self.theta)-1):
                temp_norm = (T[i] - 1.0) / 1.5
                temp_norm = np.clip(temp_norm, 0, 1)
                color = cm.hot(temp_norm)
                
                wedge = Wedge((0, 0), 1.0,
                            np.degrees(self.theta[i]),
                            np.degrees(self.theta[i+1]),
                            width=0.2,
                            facecolor=color,
                            edgecolor='none')
                ax2.add_patch(wedge)
            
            ax2.set_xlim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            ax2.set_title(f'Ring View at t={t:.3f}s')
            ax2.axis('off')
        
        # 创建动画
        n_frames = len(self.T_history) // skip
        anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                      interval=50, blit=False)
        
        # 保存
        try:
            anim.save(filename, writer='pillow', fps=20)
            print(f"✓ 动画保存至: {filename}")
        except:
            print("无法保存动画，显示静态图")
        
        plt.show()


def main():
    """主程序"""
    print("="*70)
    print("🔥 环形槽火焰传播 - 有限差分法")
    print("="*70)
    print("\n优势:")
    print("  ✓ 确定性方法，不依赖优化")
    print("  ✓ 直接求解物理方程")
    print("  ✓ 稳定可靠的结果")
    print("="*70)
    
    # 创建模拟器
    sim = RingFlamePropagation(
        n_theta=200,  # 空间分辨率
        n_t=800,      # 时间步数
        dt=0.0002     # 时间步长
    )
    
    # 运行模拟
    sim.simulate()
    
    # 测量速度
    speed = sim.measure_speed()
    print(f"\n📏 测量结果:")
    print(f"   火焰速度: {speed:.4f} rad/s")
    print(f"   物理速度: {speed*5:.2f} cm/s (R=5cm)")
    print(f"   方向: {'顺时针' if speed > 0 else '逆时针'}")
    
    # 理论估计
    S_theory = np.sqrt(sim.alpha * sim.Da) * 0.1
    print(f"   理论估计: ~{S_theory:.4f} rad/s")
    
    # 可视化
    sim.visualize()
    
    # 创建动画（可选）
    # sim.create_animation()
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()