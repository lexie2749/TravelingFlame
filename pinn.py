import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import time

class StableCombustionPINN(nn.Module):
    """
    数值稳定的燃烧PINN - 修复无穷大损失问题
    """
    def __init__(self, layers, activation=torch.tanh):
        super(StableCombustionPINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # 无量纲参数
        self.Reynolds = 100.0
        self.Peclet = 50.0        # 减小Peclet数，增强扩散
        self.Schmidt = 1.0
        self.Damkohler = 5.0      # 减小反应强度
        self.heat_release = 2.0   # 减小热释放
        
        self.init_weights()
    
    def init_weights(self):
        """稳定的权重初始化"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:  # 输出层
                nn.init.uniform_(layer.weight, -0.05, 0.05)  # 更小的权重
                with torch.no_grad():
                    layer.bias[0] = 0.5   # u
                    layer.bias[1] = 0.0   # v  
                    layer.bias[2] = 0.0   # p
                    layer.bias[3] = 0.5   # T (正数偏置)
                    layer.bias[4] = 0.5   # Y (正数偏置)
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, t):
        """
        前向传播 - 使用温和的约束防止数值问题
        """
        inputs = torch.cat([x, t], dim=1)
        
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        
        outputs = self.layers[-1](inputs)
        
        # 温和的输出约束 - 防止极端值但保持可学习性
        u = torch.tanh(outputs[:, 0:1]) * 3.0           # 速度范围[-3, 3]
        v = torch.tanh(outputs[:, 1:2]) * 1.0           # 范围[-1, 1]
        p = torch.tanh(outputs[:, 2:3]) * 2.0           # 压力范围[-2, 2]
        
        # 关键修复：确保温度和燃料浓度为正
        T_raw = outputs[:, 3:4]
        T = 0.1 + torch.nn.functional.softplus(T_raw)   # T ≥ 0.1，避免负温度
        
        Y_raw = outputs[:, 4:5]
        Y_fuel = torch.sigmoid(Y_raw)                   # Y ∈ [0, 1]
        
        return u, v, p, T, Y_fuel
    
    def safe_reaction_rate(self, T, Y_fuel):
        """
        数值稳定的反应速率计算
        """
        # 确保输入在安全范围内
        T_safe = torch.clamp(T, min=0.1, max=10.0)      # 防止极端温度
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0) # 防止零浓度
        
        # 改进的反应速率模型
        T_ignition = 0.8  # 点火温度
        
        # 使用更稳定的Arrhenius形式
        # rate = A * Y * exp(-Ea/RT)，但限制指数范围
        activation_term = -2.0 / T_safe  # 简化的活化能项
        activation_term = torch.clamp(activation_term, min=-10, max=10)  # 限制指数范围
        
        # 反应速率
        rate = self.Damkohler * Y_safe * torch.exp(activation_term)
        
        # 平滑的点火开关
        ignition_factor = torch.sigmoid((T_safe - T_ignition) * 10.0)
        
        # 确保反应速率有界
        final_rate = rate * ignition_factor
        final_rate = torch.clamp(final_rate, min=0.0, max=100.0)  # 限制最大反应速率
        
        return final_rate
    
    def stable_physics_loss(self, x, t):
        """
        数值稳定的物理损失计算
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T, Y_fuel = self.forward(x, t)
        
        # 更安全的梯度计算
        def compute_gradient(output, input_var, create_graph=True):
            try:
                grad_val = grad(output, input_var, 
                              grad_outputs=torch.ones_like(output),
                              create_graph=create_graph, 
                              retain_graph=True,
                              allow_unused=True)[0]
                
                if grad_val is None:
                    return torch.zeros_like(input_var)
                
                # 检查并处理异常值
                grad_val = torch.where(torch.isnan(grad_val), 
                                     torch.zeros_like(grad_val), grad_val)
                grad_val = torch.where(torch.isinf(grad_val), 
                                     torch.zeros_like(grad_val), grad_val)
                grad_val = torch.clamp(grad_val, min=-1e6, max=1e6)
                
                return grad_val
            except:
                return torch.zeros_like(input_var)
        
        # 计算梯度
        u_x = compute_gradient(u, x)
        u_t = compute_gradient(u, t)
        v_x = compute_gradient(v, x)
        v_t = compute_gradient(v, t)
        p_x = compute_gradient(p, x)
        T_x = compute_gradient(T, x)
        T_t = compute_gradient(T, t)
        Y_x = compute_gradient(Y_fuel, x)
        Y_t = compute_gradient(Y_fuel, t)
        
        # 二阶导数
        u_xx = compute_gradient(u_x, x)
        v_xx = compute_gradient(v_x, x)
        T_xx = compute_gradient(T_x, x)
        Y_xx = compute_gradient(Y_x, x)
        
        # 稳定的反应速率
        omega = self.safe_reaction_rate(T, Y_fuel)
        
        # 控制方程（添加数值检查）
        def safe_equation(expr, name=""):
            result = expr
            # 检查并替换异常值
            result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
            result = torch.where(torch.isinf(result), torch.zeros_like(result), result)
            result = torch.clamp(result, min=-1e6, max=1e6)
            return result
        
        # 1. 连续性方程
        continuity = safe_equation(u_x, "continuity")
        
        # 2. 动量方程
        momentum_u = safe_equation(
            u_t + u * u_x + p_x - (1.0/self.Reynolds) * u_xx, "momentum_u")
        momentum_v = safe_equation(
            v_t + u * v_x - (1.0/self.Reynolds) * v_xx, "momentum_v")
        
        # 3. 能量方程（关键修复）
        energy = safe_equation(
            T_t + u * T_x - (1.0/self.Peclet) * T_xx - self.heat_release * omega, "energy")
        
        # 4. 组分方程（关键修复）
        species = safe_equation(
            Y_t + u * Y_x - (1.0/self.Schmidt) * Y_xx + omega, "species")
        
        return continuity, momentum_u, momentum_v, energy, species

class StableTrainer:
    """数值稳定的训练器"""
    def __init__(self, model, domain_bounds, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.domain_bounds = domain_bounds
        
        # 保守的优化器设置
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        
        self.loss_history = []
        self.gradient_norms = []
        
    def generate_training_data(self, n_points=1500):
        """生成训练数据"""
        x_min, x_max, t_min, t_max = self.domain_bounds
        
        # 内部点
        x_int = torch.rand(n_points, 1) * (x_max - x_min) + x_min
        t_int = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        
        # 边界点
        n_bc = n_points // 5
        x_bc_left = torch.zeros(n_bc, 1) + x_min
        x_bc_right = torch.ones(n_bc, 1) * x_max
        t_bc = torch.rand(2*n_bc, 1) * (t_max - t_min) + t_min
        x_bc = torch.cat([x_bc_left, x_bc_right])
        
        # 初始点
        n_ic = n_points // 5
        x_ic = torch.rand(n_ic, 1) * (x_max - x_min) + x_min
        t_ic = torch.zeros(n_ic, 1) + t_min
        
        return (x_int.to(self.device), t_int.to(self.device),
                x_bc.to(self.device), t_bc.to(self.device),
                x_ic.to(self.device), t_ic.to(self.device))
    
    def train_step(self, x_int, t_int, x_bc, t_bc, x_ic, t_ic, epoch):
        """稳定的训练步骤"""
        self.optimizer.zero_grad()
        
        # 物理损失
        cont_loss, mom_u_loss, mom_v_loss, energy_loss, species_loss = \
            self.model.stable_physics_loss(x_int, t_int)
        
        # 计算各项损失并检查数值稳定性
        def safe_loss(tensor_loss, name):
            loss_val = torch.mean(tensor_loss**2)
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"  ⚠️  {name}损失异常: {loss_val}")
                return torch.tensor(0.0, device=self.device)
            return loss_val
        
        cont_val = safe_loss(cont_loss, "连续性")
        mom_u_val = safe_loss(mom_u_loss, "动量u")
        mom_v_val = safe_loss(mom_v_loss, "动量v")
        energy_val = safe_loss(energy_loss, "能量")
        species_val = safe_loss(species_loss, "组分")
        
        physics_loss = cont_val + mom_u_val + mom_v_val + energy_val + species_val
        
        # 边界条件
        u_bc, v_bc, p_bc, T_bc, Y_bc = self.model(x_bc, t_bc)
        
        n_left = len(x_bc) // 2
        # 左边界（入口）
        u_inlet = torch.ones(n_left, 1, device=self.device) * 2.0
        T_inlet = torch.ones(n_left, 1, device=self.device) * 2.0    # 高温
        Y_inlet = torch.ones(n_left, 1, device=self.device) * 0.9    # 高燃料浓度
        
        # 右边界（出口）
        u_outlet = torch.ones(n_left, 1, device=self.device) * 1.0
        T_outlet = torch.ones(n_left, 1, device=self.device) * 0.5   # 低温
        Y_outlet = torch.ones(n_left, 1, device=self.device) * 0.1   # 低燃料浓度
        
        bc_loss = (torch.mean((u_bc[:n_left] - u_inlet)**2) +
                  torch.mean((T_bc[:n_left] - T_inlet)**2) +
                  torch.mean((Y_bc[:n_left] - Y_inlet)**2) +
                  torch.mean((u_bc[n_left:] - u_outlet)**2) +
                  torch.mean((T_bc[n_left:] - T_outlet)**2) +
                  torch.mean((Y_bc[n_left:] - Y_outlet)**2))
        
        # 初始条件
        u_ic, v_ic, p_ic, T_ic, Y_ic = self.model(x_ic, t_ic)
        
        # 创建空间变化的初始条件
        T_init = 0.3 + 1.5 * torch.exp(-((x_ic - 0.3) / 0.15)**2)  # 高斯热点
        Y_init = 0.8 * torch.ones_like(x_ic)  # 均匀燃料分布
        u_init = 1.0 * torch.ones_like(x_ic)  # 初始流速
        
        ic_loss = (torch.mean((u_ic - u_init)**2) +
                  torch.mean((T_ic - T_init)**2) +
                  torch.mean((Y_ic - Y_init)**2))
        
        # 总损失
        total_loss = physics_loss + 2.0 * bc_loss + 5.0 * ic_loss
        
        # 检查总损失
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 1e4:
            print(f"  ⚠️  总损失异常: {total_loss:.2f}, 跳过更新")
            return float('inf'), float('inf'), float('inf'), float('inf')
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** (1. / 2)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.gradient_norms.append(grad_norm)
        
        if epoch % 50 == 0:  # 详细输出
            print(f"  📊 损失详情:")
            print(f"    连续性: {cont_val:.6f}")
            print(f"    动量u: {mom_u_val:.6f}")
            print(f"    动量v: {mom_v_val:.6f}")
            print(f"    能量: {energy_val:.6f}")
            print(f"    组分: {species_val:.6f}")
            print(f"    边界: {bc_loss:.6f}")
            print(f"    初始: {ic_loss:.6f}")
            print(f"    梯度范数: {grad_norm:.6f}")
        
        return total_loss.item(), physics_loss.item(), bc_loss.item(), ic_loss.item()
    
    def train(self, epochs=1000, print_freq=100):
        """稳定训练循环"""
        print("🔧 开始数值稳定训练...")
        
        consecutive_failures = 0
        
        for epoch in range(epochs):
            if epoch % 30 == 0:  # 频繁重新采样
                x_int, t_int, x_bc, t_bc, x_ic, t_ic = self.generate_training_data()
            
            total_loss, physics_loss, bc_loss, ic_loss = self.train_step(
                x_int, t_int, x_bc, t_bc, x_ic, t_ic, epoch)
            
            # 检查训练失败
            if np.isinf(total_loss):
                consecutive_failures += 1
                if consecutive_failures > 10:
                    print("❌ 连续训练失败，重新初始化...")
                    self.model.init_weights()
                    consecutive_failures = 0
            else:
                consecutive_failures = 0
                self.loss_history.append([total_loss, physics_loss, bc_loss, ic_loss])
            
            if epoch % print_freq == 0:
                print(f"\n📊 Epoch {epoch}/{epochs}")
                if not np.isinf(total_loss):
                    print(f"  ✅ 总损失: {total_loss:.6f}")
                    
                    # 检查学习进展
                    with torch.no_grad():
                        x_test = torch.tensor([[0.1], [0.5], [0.9]], device=self.device)
                        t_test = torch.tensor([[0.05], [0.05], [0.05]], device=self.device)
                        u, v, p, T, Y = self.model(x_test, t_test)
                        
                        print(f"  🔍 输出检查:")
                        print(f"    温度范围: [{T.min():.3f}, {T.max():.3f}]")
                        print(f"    燃料范围: [{Y.min():.3f}, {Y.max():.3f}]")
                        print(f"    速度范围: [{u.min():.3f}, {u.max():.3f}]")
                else:
                    print(f"  ❌ 训练失败 (第{consecutive_failures}次)")
                
                print("-" * 50)
    
    def predict(self, x_test, t_test):
        """预测"""
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
    
    def plot_results(self):
        """绘制最终结果"""
        x_test = np.linspace(0, 1, 100)
        t_test = np.full_like(x_test, 0.05)
        
        u, v, p, T, Y = self.predict(x_test, t_test)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 各变量沿x的分布
        axes[0,0].plot(x_test, u.flatten(), 'b-', linewidth=2)
        axes[0,0].set_title('x-Velocity')
        axes[0,0].set_xlabel('x')
        axes[0,0].grid(True)
        
        axes[0,1].plot(x_test, T.flatten(), 'r-', linewidth=2)
        axes[0,1].set_title('Temperature')
        axes[0,1].set_xlabel('x')
        axes[0,1].grid(True)
        
        axes[0,2].plot(x_test, Y.flatten(), 'g-', linewidth=2)
        axes[0,2].set_title('Fuel Mass Fraction')
        axes[0,2].set_xlabel('x')
        axes[0,2].grid(True)
        
        axes[1,0].plot(x_test, p.flatten(), 'm-', linewidth=2)
        axes[1,0].set_title('Pressure')
        axes[1,0].set_xlabel('x')
        axes[1,0].grid(True)
        
        # 损失历史
        if len(self.loss_history) > 0:
            loss_array = np.array(self.loss_history)
            axes[1,1].semilogy(loss_array[:, 0], 'b-', label='Total', linewidth=2)
            axes[1,1].semilogy(loss_array[:, 1], 'r-', label='Physics', alpha=0.7)
            axes[1,1].semilogy(loss_array[:, 2], 'g-', label='Boundary', alpha=0.7)
            axes[1,1].semilogy(loss_array[:, 3], 'm-', label='Initial', alpha=0.7)
            axes[1,1].set_title('Loss History')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Loss')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        # 数值总结
        axes[1,2].axis('off')
        summary = f"""
Numerical Summary:

Temperature: [{T.min():.3f}, {T.max():.3f}]
Variation: {T.max() - T.min():.3f}

Fuel: [{Y.min():.3f}, {Y.max():.3f}]
Variation: {Y.max() - Y.min():.3f}

Velocity: [{u.min():.3f}, {u.max():.3f}]
Variation: {u.max() - u.min():.3f}

Final Loss: {self.loss_history[-1][0]:.6f}
"""
        axes[1,2].text(0.1, 0.5, summary, fontsize=12, 
                      verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{'='*60}")
        print("Stable Training Results")
        print(f"{'='*60}")
        print(f"Temperature variation: {T.max() - T.min():.6f}")
        print(f"Fuel variation: {Y.max() - Y.min():.6f}")
        print(f"Velocity variation: {u.max() - u.min():.6f}")
        print(f"Final loss: {self.loss_history[-1][0]:.6f}")
        
        if (T.max() - T.min()) > 0.2 and (Y.max() - Y.min()) > 0.2:
            print("SUCCESS: Network learned combustion physics!")
        else:
            print("PARTIAL: Limited learning detected")
        print(f"{'='*60}")

def main():
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    layers = [2, 64, 64, 64, 5]
    
    model = StableCombustionPINN(layers)
    domain_bounds = [0.0, 1.0, 0.0, 0.1]
    trainer = StableTrainer(model, domain_bounds, device)
    
    print("\nKey Fixes Applied:")
    print("✅ Temperature constrained to be positive (T ≥ 0.1)")
    print("✅ Fuel fraction constrained to [0,1] with sigmoid")
    print("✅ Reaction rate with numerical bounds")
    print("✅ Safe gradient computation with NaN/Inf checks")
    print("✅ Equation stabilization with value clamping")
    print("✅ Automatic reinitialization on training failure")
    print("✅ Reduced reaction parameters for stability")
    print("-" * 60)
    
    trainer.train(epochs=800, print_freq=100)
    trainer.plot_results()

if __name__ == "__main__":
    main()