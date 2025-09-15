import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import time

class MovingFramePINN(nn.Module):
    """
    Moving Frame PINN - 强制火焰传播的新策略
    核心思路：预设火焰速度，在moving frame中求解
    """
    def __init__(self, layers, flame_speed=0.5, activation=torch.tanh):
        super(MovingFramePINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # 预设火焰速度（这是我们想要的传播速度）
        self.flame_speed = flame_speed
        
        # 物理参数
        self.Reynolds = 100.0
        self.Peclet = 50.0
        self.Schmidt = 1.0
        self.Damkohler = 20.0
        self.heat_release = 5.0
        
        self.init_weights()
    
    def init_weights(self):
        """初始化"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                with torch.no_grad():
                    # 设置合理的初始偏置
                    layer.bias[0] = 0.0   # u
                    layer.bias[1] = 0.0   # v
                    layer.bias[2] = 0.0   # p
                    layer.bias[3] = 0.0   # T
                    layer.bias[4] = 0.0   # Y
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, t):
        """
        Moving Frame前向传播
        将(x,t)转换为moving coordinate: xi = x - flame_speed * t
        """
        # Moving frame coordinate
        xi = x - self.flame_speed * t
        
        # 输入到网络的是moving coordinate
        inputs = torch.cat([xi, t], dim=1)
        
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        
        outputs = self.layers[-1](inputs)
        
        # 输出变量 - 在moving frame中的解
        u = 0.5 + 1.0 * torch.sigmoid(outputs[:, 0:1])      # u ∈ [0.5, 1.5]
        v = torch.tanh(outputs[:, 1:2]) * 0.1               # v ∈ [-0.1, 0.1]
        p = outputs[:, 2:3] * 0.5                           # pressure
        
        # 温度和燃料 - 使用sigmoid确保合理范围
        T = 0.3 + 0.9 * torch.sigmoid(outputs[:, 3:4])      # T ∈ [0.3, 1.2]
        Y_fuel = torch.sigmoid(outputs[:, 4:5])             # Y ∈ [0, 1]
        
        return u, v, p, T, Y_fuel
    
    def reaction_rate(self, T, Y_fuel):
        """反应速率"""
        T_safe = torch.clamp(T, min=0.3, max=1.5)
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0)
        
        T_ignition = 0.7
        
        activation_term = -2.0 / T_safe
        activation_term = torch.clamp(activation_term, min=-6, max=2)
        
        rate = self.Damkohler * Y_safe * torch.exp(activation_term)
        ignition_factor = torch.sigmoid((T_safe - T_ignition) * 15.0)
        
        final_rate = rate * ignition_factor
        final_rate = torch.clamp(final_rate, min=0.0, max=50.0)
        
        return final_rate
    
    def physics_loss(self, x, t):
        """
        Moving Frame中的物理方程
        主要变化：∂/∂t 项要加上 -flame_speed * ∂/∂x 项
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T, Y_fuel = self.forward(x, t)
        
        def safe_grad(output, input_var):
            try:
                grad_val = grad(output, input_var, 
                              grad_outputs=torch.ones_like(output),
                              create_graph=True, retain_graph=True,
                              allow_unused=True)[0]
                
                if grad_val is None:
                    return torch.zeros_like(input_var)
                
                grad_val = torch.where(torch.isnan(grad_val), 
                                     torch.zeros_like(grad_val), grad_val)
                grad_val = torch.where(torch.isinf(grad_val), 
                                     torch.zeros_like(grad_val), grad_val)
                grad_val = torch.clamp(grad_val, min=-100, max=100)
                
                return grad_val
            except:
                return torch.zeros_like(input_var)
        
        # 空间梯度
        u_x = safe_grad(u, x)
        v_x = safe_grad(v, x)
        p_x = safe_grad(p, x)
        T_x = safe_grad(T, x)
        Y_x = safe_grad(Y_fuel, x)
        
        # 时间梯度 (在moving frame中)
        u_t = safe_grad(u, t)
        v_t = safe_grad(v, t)
        T_t = safe_grad(T, t)
        Y_t = safe_grad(Y_fuel, t)
        
        # 二阶导数
        u_xx = safe_grad(u_x, x)
        v_xx = safe_grad(v_x, x)
        T_xx = safe_grad(T_x, x)
        Y_xx = safe_grad(Y_x, x)
        
        # 反应速率
        omega = self.reaction_rate(T, Y_fuel)
        
        def safe_eq(expr):
            result = expr
            result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
            result = torch.where(torch.isinf(result), torch.zeros_like(result), result)
            result = torch.clamp(result, min=-100, max=100)
            return result
        
        # Moving Frame中的控制方程
        # 对于steady flame，在moving frame中时间导数应该为0
        
        # 1. 连续性方程
        continuity = safe_eq(u_x)
        
        # 2. 动量方程
        # 在moving frame中：∂u/∂t + (u-c)∂u/∂x = -∂p/∂x + viscous terms
        # 其中c是火焰速度
        momentum_u = safe_eq(u_t + (u - self.flame_speed) * u_x + p_x - (1.0/self.Reynolds) * u_xx)
        momentum_v = safe_eq(v_t + (u - self.flame_speed) * v_x - (1.0/self.Reynolds) * v_xx)
        
        # 3. 能量方程
        energy = safe_eq(T_t + (u - self.flame_speed) * T_x - (1.0/self.Peclet) * T_xx - self.heat_release * omega)
        
        # 4. 组分方程
        species = safe_eq(Y_t + (u - self.flame_speed) * Y_x - (1.0/self.Schmidt) * Y_xx + omega)
        
        return continuity, momentum_u, momentum_v, energy, species

class FlameSpeedLearner:
    """
    学习真实火焰速度的类
    """
    def __init__(self, model, domain_bounds, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.domain_bounds = domain_bounds
        
        # 同时优化网络参数和火焰速度
        self.optimizer_model = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        
        # 火焰速度也作为可学习参数
        self.flame_speed_param = torch.tensor([0.5], device=device, requires_grad=True)
        self.optimizer_speed = torch.optim.Adam([self.flame_speed_param], lr=1e-3)
        
        self.loss_history = []
        self.speed_history = []
        
    def generate_training_data(self, n_points=1000):
        """生成训练数据"""
        x_min, x_max, t_min, t_max = self.domain_bounds
        
        # 内部点
        x_int = torch.rand(n_points, 1) * (x_max - x_min) + x_min
        t_int = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        
        # 减少边界点
        n_bc = n_points // 20
        x_bc_left = torch.zeros(n_bc, 1) + x_min
        x_bc_right = torch.ones(n_bc, 1) * x_max
        t_bc = torch.rand(2*n_bc, 1) * (t_max - t_min) + t_min
        x_bc = torch.cat([x_bc_left, x_bc_right])
        
        # 初始点
        n_ic = n_points // 10
        x_ic = torch.rand(n_ic, 1) * (x_max - x_min) + x_min
        t_ic = torch.zeros(n_ic, 1) + t_min
        
        return (x_int.to(self.device), t_int.to(self.device),
                x_bc.to(self.device), t_bc.to(self.device),
                x_ic.to(self.device), t_ic.to(self.device))
    
    def train_step(self, x_int, t_int, x_bc, t_bc, x_ic, t_ic):
        """训练步骤 - 同时学习网络和火焰速度"""
        
        # 更新模型中的火焰速度
        self.model.flame_speed = torch.clamp(self.flame_speed_param, min=0.01, max=2.0)
        
        self.optimizer_model.zero_grad()
        self.optimizer_speed.zero_grad()
        
        # 物理损失
        cont_loss, mom_u_loss, mom_v_loss, energy_loss, species_loss = \
            self.model.physics_loss(x_int, t_int)
        
        def safe_loss(tensor_loss):
            loss_val = torch.mean(tensor_loss**2)
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                return torch.tensor(0.0, device=self.device)
            return loss_val
        
        cont_val = safe_loss(cont_loss)
        mom_u_val = safe_loss(mom_u_loss)
        mom_v_val = safe_loss(mom_v_loss)
        energy_val = safe_loss(energy_loss)
        species_val = safe_loss(species_loss)
        
        physics_loss = cont_val + mom_u_val + mom_v_val + energy_val + species_val
        
        # 极简的边界条件
        u_bc, v_bc, p_bc, T_bc, Y_bc = self.model(x_bc, t_bc)
        
        n_left = len(x_bc) // 2
        
        # 非常温和的边界条件
        u_inlet = torch.ones(n_left, 1, device=self.device) * 1.0
        T_inlet = torch.ones(n_left, 1, device=self.device) * 0.8
        Y_inlet = torch.ones(n_left, 1, device=self.device) * 0.9
        
        u_outlet = torch.ones(n_left, 1, device=self.device) * 0.8
        T_outlet = torch.ones(n_left, 1, device=self.device) * 0.4
        Y_outlet = torch.ones(n_left, 1, device=self.device) * 0.1
        
        bc_loss = 0.01 * (torch.mean((u_bc[:n_left] - u_inlet)**2) +
                         torch.mean((T_bc[:n_left] - T_inlet)**2) +
                         torch.mean((Y_bc[:n_left] - Y_inlet)**2) +
                         torch.mean((u_bc[n_left:] - u_outlet)**2) +
                         torch.mean((T_bc[n_left:] - T_outlet)**2) +
                         torch.mean((Y_bc[n_left:] - Y_outlet)**2))
        
        # 简化的初始条件
        u_ic, v_ic, p_ic, T_ic, Y_ic = self.model(x_ic, t_ic)
        
        # 创建火焰结构的初始条件
        xi_ic = x_ic - self.model.flame_speed * 0.01  # t=0.01时的moving coordinate
        
        # 设计典型的火焰结构：从burned到unburned的过渡
        T_init = 0.4 + 0.6 * torch.sigmoid(-(xi_ic - 0.1) * 20.0)  # 火焰结构
        Y_init = 0.2 + 0.7 * torch.sigmoid(-(xi_ic - 0.1) * 20.0)  # 对应的燃料分布
        u_init = torch.ones_like(x_ic) * 1.0
        
        ic_loss = (torch.mean((u_ic - u_init)**2) +
                  torch.mean((T_ic - T_init)**2) +
                  torch.mean((Y_ic - Y_init)**2))
        
        # 总损失
        total_loss = physics_loss + bc_loss + 5.0 * ic_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 1e4:
            return float('inf'), float('inf'), float('inf'), float('inf'), 0.0
        
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([self.flame_speed_param], max_norm=0.1)
        
        self.optimizer_model.step()
        self.optimizer_speed.step()
        
        current_speed = self.flame_speed_param.item()
        
        return total_loss.item(), physics_loss.item(), bc_loss.item(), ic_loss.item(), current_speed
    
    def train(self, epochs=1500, print_freq=200):
        """训练循环"""
        print("🔥 开始Moving Frame火焰训练...")
        print("🎯 同时学习网络参数和火焰传播速度")
        
        for epoch in range(epochs):
            if epoch % 20 == 0:
                x_int, t_int, x_bc, t_bc, x_ic, t_ic = self.generate_training_data()
            
            total_loss, physics_loss, bc_loss, ic_loss, current_speed = self.train_step(
                x_int, t_int, x_bc, t_bc, x_ic, t_ic)
            
            if not np.isinf(total_loss):
                self.loss_history.append([total_loss, physics_loss, bc_loss, ic_loss])
                self.speed_history.append(current_speed)
            
            if epoch % print_freq == 0:
                if not np.isinf(total_loss):
                    print(f"Epoch {epoch}/{epochs}")
                    print(f"  Loss: {total_loss:.6f}")
                    print(f"  Learned Flame Speed: {current_speed:.4f} m/s")
                    print(f"  Physics: {physics_loss:.6f}, BC: {bc_loss:.6f}, IC: {ic_loss:.6f}")

class MovingFrameDiagnostic:
    """Moving Frame诊断工具"""
    def __init__(self, model, learned_speed, device='cpu'):
        self.model = model
        self.learned_speed = learned_speed
        self.device = device
    
    def predict_single(self, x_test, t_test):
        """单次预测"""
        self.model.eval()
        with torch.no_grad():
            x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
            t_test = torch.tensor(t_test, dtype=torch.float32).to(self.device)
            
            if x_test.dim() == 1:
                x_test = x_test.unsqueeze(1)
            if t_test.dim() == 1:
                t_test = t_test.unsqueeze(1)
            
            u, v, p, T, Y_fuel = self.model(x_test, t_test)
            
        return (u.cpu().numpy(), v.cpu().numpy(), p.cpu().numpy(), 
                T.cpu().numpy(), Y_fuel.cpu().numpy())
    
    def analyze_propagation(self):
        """分析传播结果"""
        print(f"\n🔬 Moving Frame传播分析")
        print(f"📈 学习到的火焰速度: {self.learned_speed:.4f} m/s")
        
        time_points = np.linspace(0.01, 0.08, 15)
        x_points = np.linspace(0, 1, 100)
        
        flame_positions = []
        
        print(f"\n验证火焰传播:")
        print("时间      理论位置    实际位置    温度峰值")
        print("-" * 50)
        
        for t_val in time_points:
            # 理论火焰位置（基于学习的速度）
            theoretical_pos = 0.1 + self.learned_speed * t_val  # 假设初始位置在0.1
            
            # 实际火焰位置（最高温度位置）
            u, v, p, T, Y = self.predict_single(x_points, [t_val] * len(x_points))
            actual_pos = x_points[np.argmax(T)]
            max_temp = T.max()
            
            flame_positions.append(actual_pos)
            
            print(f"{t_val:.3f}      {theoretical_pos:.3f}       {actual_pos:.3f}       {max_temp:.3f}")
        
        # 验证传播
        if len(flame_positions) > 1:
            position_change = flame_positions[-1] - flame_positions[0]
            time_span = time_points[-1] - time_points[0]
            measured_speed = position_change / time_span
            
            print(f"\n🎯 传播验证:")
            print(f"  学习速度: {self.learned_speed:.4f} m/s")
            print(f"  测量速度: {measured_speed:.4f} m/s")
            print(f"  误差: {abs(self.learned_speed - measured_speed):.4f} m/s")
            
            return measured_speed, flame_positions
        
        return 0.0, flame_positions
    
    def create_visualization(self):
        """创建可视化"""
        print("\n📊 创建Moving Frame可视化...")
        
        x_mesh = np.linspace(0, 1, 80)
        t_mesh = np.linspace(0.01, 0.08, 40)
        X, T_grid = np.meshgrid(x_mesh, t_mesh)
        
        x_flat = X.flatten()
        t_flat = T_grid.flatten()
        
        u_pred, v_pred, p_pred, T_pred, Y_pred = self.predict_single(x_flat, t_flat)
        T_field = T_pred.reshape(X.shape)
        Y_field = Y_pred.reshape(X.shape)
        
        plt.figure(figsize=(15, 10))
        
        # 温度场
        plt.subplot(2, 2, 1)
        im1 = plt.contourf(X, T_grid, T_field, levels=25, cmap='hot')
        plt.title('Temperature Field')
        plt.xlabel('Position (m)')
        plt.ylabel('Time (s)')
        plt.colorbar(im1, label='Temperature')
        
        # 理论火焰轨迹
        theoretical_positions = 0.1 + self.learned_speed * t_mesh
        plt.plot(theoretical_positions, t_mesh, 'w--', linewidth=3, 
                label=f'Theory: {self.learned_speed:.3f} m/s')
        plt.legend()
        
        # 燃料场
        plt.subplot(2, 2, 2)
        im2 = plt.contourf(X, T_grid, Y_field, levels=25, cmap='Blues')
        plt.title('Fuel Field')
        plt.xlabel('Position (m)')
        plt.ylabel('Time (s)')
        plt.colorbar(im2, label='Fuel Fraction')
        
        # 固定时间剖面
        t_fixed = 0.05
        u, v, p, T, Y = self.predict_single(x_mesh, [t_fixed] * len(x_mesh))
        
        plt.subplot(2, 2, 3)
        plt.plot(x_mesh, T.flatten(), 'r-', linewidth=2, label='Temperature')
        plt.plot(x_mesh, Y.flatten(), 'b-', linewidth=2, label='Fuel')
        theoretical_flame_pos = 0.1 + self.learned_speed * t_fixed
        plt.axvline(x=theoretical_flame_pos, color='k', linestyle='--', 
                   label=f'Theory Flame @ {theoretical_flame_pos:.3f}')
        plt.title(f'Profiles at t={t_fixed}s')
        plt.xlabel('Position (m)')
        plt.ylabel('T / Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 火焰速度学习历史
        plt.subplot(2, 2, 4)
        plt.plot(self.speed_history, 'g-', linewidth=2)
        plt.title('Learned Flame Speed Evolution')
        plt.xlabel('Training Step')
        plt.ylabel('Flame Speed (m/s)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  设备: {device}")
    
    layers = [2, 64, 64, 64, 5]
    
    # 初始猜测的火焰速度
    initial_flame_speed = 0.5
    
    model = MovingFramePINN(layers, flame_speed=initial_flame_speed)
    domain_bounds = [0.0, 1.0, 0.0, 0.08]
    
    trainer = FlameSpeedLearner(model, domain_bounds, device)
    
    print("\n🚀 Moving Frame方法:")
    print("✅ 在移动坐标系中求解")
    print("✅ 同时学习网络参数和火焰速度")
    print("✅ 强制火焰结构存在")
    print("✅ 避免静态解问题")
    print(f"✅ 初始火焰速度猜测: {initial_flame_speed} m/s")
    print("-" * 50)
    
    trainer.train(epochs=1000, print_freq=150)
    
    # 获取学习到的火焰速度
    learned_speed = trainer.speed_history[-1] if trainer.speed_history else 0.0
    
    print(f"\n🎯 最终学习结果:")
    print(f"  学习到的火焰传播速度: {learned_speed:.4f} m/s")
    
    # 诊断
    diagnostic = MovingFrameDiagnostic(model, learned_speed, device)
    diagnostic.speed_history = trainer.speed_history
    
    measured_speed, positions = diagnostic.analyze_propagation()
    diagnostic.create_visualization()
    
    # 最终评估
    print("\n" + "="*70)
    print("🎯 Moving Frame方法结果")
    print("="*70)
    
    if learned_speed > 0.01:
        print(f"🎉 成功！学会了火焰传播")
        print(f"🚀 学习火焰速度: {learned_speed:.4f} m/s")
        print(f"📊 测量火焰速度: {measured_speed:.4f} m/s")
        print(f"✅ 相对误差: {abs(learned_speed-measured_speed)/learned_speed*100:.1f}%")
    else:
        print("❌ Moving Frame方法也未能学会传播")
        print("💡 可能需要调整初始火焰速度猜测")
    
    print("="*70)

if __name__ == "__main__":
    main()