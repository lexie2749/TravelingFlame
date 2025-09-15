import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import matplotlib.patches as patches

class TJunctionCombustionPINN(nn.Module):
    """
    Physics-Informed Neural Network for T-Junction Combustion
    T-Junction geometry:
    - Main channel: horizontal from x=0 to x=10, y in [-0.5, 0.5]
    - Branch channel: vertical from y=0.5 to y=3, x in [4, 6]
    """
    def __init__(self, layers, activation=torch.tanh):
        super(TJunctionCombustionPINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # Physical Parameters (Dimensionless)
        self.Re = 100.0    # Reynolds number
        self.Pe = 50.0     # Peclet number  
        self.Sc = 1.0      # Schmidt number
        self.Da = 8.0      # Damkohler number
        self.beta = 3.0    # Heat release parameter
        
        # T-Junction geometry parameters
        self.x_branch_start = 4.0
        self.x_branch_end = 6.0
        self.y_branch_top = 3.0
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with proper scaling"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:  # Output layer
                nn.init.uniform_(layer.weight, -0.05, 0.05)
                with torch.no_grad():
                    layer.bias[0] = 0.5   # u
                    layer.bias[1] = 0.0   # v  
                    layer.bias[2] = 0.0   # p
                    layer.bias[3] = 0.5   # T
                    layer.bias[4] = 0.5   # Y
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def is_in_tjunction(self, x, y):
        """Check if point (x,y) is inside T-junction geometry"""
        # Main horizontal channel
        in_main = (x >= 0) & (x <= 10.0) & (y >= -0.5) & (y <= 0.5)
        
        # Vertical branch channel
        in_branch = (x >= self.x_branch_start) & (x <= self.x_branch_end) & \
                   (y >= 0.5) & (y <= self.y_branch_top)
        
        return in_main | in_branch
    
    def forward(self, x, y, t):
        """Forward pass with geometry-aware constraints"""
        inputs = torch.cat([x, y, t], dim=1)
        
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        
        outputs = self.layers[-1](inputs)
        
        # Apply geometry mask
        mask = self.is_in_tjunction(x, y).float()
        
        u = torch.tanh(outputs[:, 0:1]) * 3.0 * mask
        v = torch.tanh(outputs[:, 1:2]) * 2.0 * mask
        p = torch.tanh(outputs[:, 2:3]) * 2.0 * mask
        T = (0.1 + torch.nn.functional.softplus(outputs[:, 3:4])) * mask
        Y_fuel = torch.sigmoid(outputs[:, 4:5]) * mask
        
        return u, v, p, T, Y_fuel
    
    def reaction_rate(self, T, Y_fuel):
        """Enhanced reaction rate for T-junction"""
        T_safe = torch.clamp(T, min=0.1, max=10.0)
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0)
        
        # Modified activation energy for better flame propagation
        activation_term = -1.5 / T_safe
        activation_term = torch.clamp(activation_term, min=-10, max=10)
        
        rate = self.Da * Y_safe * torch.exp(activation_term)
        
        # Ignition temperature threshold
        ignition_factor = torch.sigmoid((T_safe - 0.7) * 15.0)
        
        final_rate = rate * ignition_factor
        return torch.clamp(final_rate, min=0.0, max=100.0)
    
    def physics_loss(self, x, y, t):
        """Physics loss for T-junction geometry"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T, Y = self.forward(x, y, t)
        
        def compute_gradient(output, input_var):
            grad_val = grad(output, input_var, 
                          grad_outputs=torch.ones_like(output),
                          create_graph=True, retain_graph=True,
                          allow_unused=True)[0]
            if grad_val is None:
                return torch.zeros_like(input_var)
            grad_val = torch.where(torch.isnan(grad_val) | torch.isinf(grad_val),
                                  torch.zeros_like(grad_val), grad_val)
            return torch.clamp(grad_val, -1e6, 1e6)
        
        # First derivatives
        u_x = compute_gradient(u, x)
        u_y = compute_gradient(u, y)
        u_t = compute_gradient(u, t)
        v_x = compute_gradient(v, x)
        v_y = compute_gradient(v, y)
        v_t = compute_gradient(v, t)
        p_x = compute_gradient(p, x)
        p_y = compute_gradient(p, y)
        T_x = compute_gradient(T, x)
        T_y = compute_gradient(T, y)
        T_t = compute_gradient(T, t)
        Y_x = compute_gradient(Y, x)
        Y_y = compute_gradient(Y, y)
        Y_t = compute_gradient(Y, t)
        
        # Second derivatives
        u_xx = compute_gradient(u_x, x)
        u_yy = compute_gradient(u_y, y)
        v_xx = compute_gradient(v_x, x)
        v_yy = compute_gradient(v_y, y)
        T_xx = compute_gradient(T_x, x)
        T_yy = compute_gradient(T_y, y)
        Y_xx = compute_gradient(Y_x, x)
        Y_yy = compute_gradient(Y_y, y)
        
        omega = self.reaction_rate(T, Y)
        
        # Governing equations
        f_continuity = u_x + v_y
        f_momentum_x = u_t + u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)
        f_momentum_y = v_t + u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)
        f_energy = T_t + u * T_x + v * T_y - (1/self.Pe) * (T_xx + T_yy) - self.beta * omega
        f_species = Y_t + u * Y_x + v * Y_y - (1/self.Sc) * (Y_xx + Y_yy) + omega
        
        return f_continuity, f_momentum_x, f_momentum_y, f_energy, f_species


class TJunctionTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        self.loss_history = []
    
    def generate_tjunction_points(self, n_points=5000):
        """Generate training points for T-junction geometry"""
        points = []
        
        # Main channel points
        n_main = int(n_points * 0.7)
        x_main = torch.rand(n_main, 1) * 10.0
        y_main = torch.rand(n_main, 1) * 1.0 - 0.5
        t_main = torch.rand(n_main, 1) * 0.2
        
        # Branch channel points
        n_branch = int(n_points * 0.3)
        x_branch = torch.rand(n_branch, 1) * 2.0 + 4.0
        y_branch = torch.rand(n_branch, 1) * 2.5 + 0.5
        t_branch = torch.rand(n_branch, 1) * 0.2
        
        # Combine all points
        x = torch.cat([x_main, x_branch])
        y = torch.cat([y_main, y_branch])
        t = torch.cat([t_main, t_branch])
        
        return x.to(self.device), y.to(self.device), t.to(self.device)
    
    def generate_boundary_points(self, n_points=1000):
        """Generate boundary points for T-junction"""
        data = {}
        
        # Inlet (left side of main channel)
        n_inlet = n_points // 4
        x_inlet = torch.zeros(n_inlet, 1)
        y_inlet = torch.rand(n_inlet, 1) * 1.0 - 0.5
        t_inlet = torch.rand(n_inlet, 1) * 0.2
        data['inlet'] = (x_inlet.to(self.device), 
                        y_inlet.to(self.device),
                        t_inlet.to(self.device))
        
        # Outlet (right side of main channel)
        n_outlet = n_points // 4
        x_outlet = torch.full((n_outlet, 1), 10.0)
        y_outlet = torch.rand(n_outlet, 1) * 1.0 - 0.5
        t_outlet = torch.rand(n_outlet, 1) * 0.2
        data['outlet'] = (x_outlet.to(self.device),
                         y_outlet.to(self.device),
                         t_outlet.to(self.device))
        
        # Branch outlet (top of branch)
        n_branch_out = n_points // 4
        x_branch_out = torch.rand(n_branch_out, 1) * 2.0 + 4.0
        y_branch_out = torch.full((n_branch_out, 1), 3.0)
        t_branch_out = torch.rand(n_branch_out, 1) * 0.2
        data['branch_outlet'] = (x_branch_out.to(self.device),
                                y_branch_out.to(self.device),
                                t_branch_out.to(self.device))
        
        # Initial condition points
        n_ic = n_points
        x_ic, y_ic, _ = self.generate_tjunction_points(n_ic)
        t_ic = torch.zeros(n_ic, 1).to(self.device)
        data['initial'] = (x_ic, y_ic, t_ic)
        
        return data
    
    def train_step(self, x_int, y_int, t_int, boundary_data):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # 1. Physics loss
        f_c, f_mx, f_my, f_e, f_s = self.model.physics_loss(x_int, y_int, t_int)
        loss_pde = torch.mean(f_c**2) + torch.mean(f_mx**2) + torch.mean(f_my**2) + \
                  torch.mean(f_e**2) + torch.mean(f_s**2)
        
        # 2. Inlet boundary condition
        x_in, y_in, t_in = boundary_data['inlet']
        u_in, v_in, _, T_in, Y_in = self.model(x_in, y_in, t_in)
        loss_inlet = torch.mean((u_in - 2.0)**2) + torch.mean(v_in**2) + \
                    torch.mean((T_in - 2.5)**2) + torch.mean((Y_in - 0.9)**2)
        
        # 3. Initial condition
        x_ic, y_ic, t_ic = boundary_data['initial']
        u_ic, v_ic, _, T_ic, Y_ic = self.model(x_ic, y_ic, t_ic)
        
        # Hot spot near junction entrance
        T_init = 0.3 + 2.0 * torch.exp(-((x_ic - 1.0)**2 + y_ic**2) / 0.3)
        Y_init = 0.8 * torch.ones_like(x_ic)
        
        loss_ic = torch.mean((u_ic - 0.5)**2) + torch.mean(v_ic**2) + \
                 torch.mean((T_ic - T_init)**2) + torch.mean((Y_ic - Y_init)**2)
        
        # Total loss with weights
        total_loss = loss_pde + 3.0 * loss_inlet + 5.0 * loss_ic
        
        if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, epochs=2000, print_freq=100):
        """Training loop"""
        print("🔥 Starting T-Junction Combustion Training...")
        print("=" * 50)
        
        for epoch in range(epochs):
            # Generate training data
            if epoch % 50 == 0:
                x_int, y_int, t_int = self.generate_tjunction_points(5000)
                boundary_data = self.generate_boundary_points(1000)
            
            loss = self.train_step(x_int, y_int, t_int, boundary_data)
            self.loss_history.append(loss)
            self.scheduler.step()
            
            if epoch % print_freq == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f} | LR: {current_lr:.6f}")
        
        print("=" * 50)
        print("✅ Training Complete!")
    
    def visualize_results(self, time_points=[0.0, 0.05, 0.1, 0.15]):
        """Visualize T-junction flame propagation"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create meshgrid for main channel
        x_main = np.linspace(0, 10, 200)
        y_main = np.linspace(-0.5, 0.5, 40)
        X_main, Y_main = np.meshgrid(x_main, y_main)
        
        # Create meshgrid for branch channel
        x_branch = np.linspace(4, 6, 40)
        y_branch = np.linspace(0.5, 3, 50)
        X_branch, Y_branch = np.meshgrid(x_branch, y_branch)
        
        for idx, t_val in enumerate(time_points):
            # Predict for main channel
            x_flat_main = X_main.flatten()[:, None]
            y_flat_main = Y_main.flatten()[:, None]
            t_flat_main = np.full_like(x_flat_main, t_val)
            
            self.model.eval()
            with torch.no_grad():
                x_torch_main = torch.tensor(x_flat_main, dtype=torch.float32).to(self.device)
                y_torch_main = torch.tensor(y_flat_main, dtype=torch.float32).to(self.device)
                t_torch_main = torch.tensor(t_flat_main, dtype=torch.float32).to(self.device)
                _, _, _, T_main, Y_main_fuel = self.model(x_torch_main, y_torch_main, t_torch_main)
            
            T_grid_main = T_main.cpu().numpy().reshape(40, 200)
            Y_grid_main = Y_main_fuel.cpu().numpy().reshape(40, 200)
            
            # Predict for branch channel
            x_flat_branch = X_branch.flatten()[:, None]
            y_flat_branch = Y_branch.flatten()[:, None]
            t_flat_branch = np.full_like(x_flat_branch, t_val)
            
            with torch.no_grad():
                x_torch_branch = torch.tensor(x_flat_branch, dtype=torch.float32).to(self.device)
                y_torch_branch = torch.tensor(y_flat_branch, dtype=torch.float32).to(self.device)
                t_torch_branch = torch.tensor(t_flat_branch, dtype=torch.float32).to(self.device)
                _, _, _, T_branch, Y_branch_fuel = self.model(x_torch_branch, y_torch_branch, t_torch_branch)
            
            T_grid_branch = T_branch.cpu().numpy().reshape(50, 40)
            Y_grid_branch = Y_branch_fuel.cpu().numpy().reshape(50, 40)
            
            # Plot Temperature
            ax = fig.add_subplot(len(time_points), 2, 2*idx + 1)
            
            # Plot main channel
            c1 = ax.contourf(X_main, Y_main, T_grid_main, levels=30, cmap='hot', vmin=0, vmax=3)
            # Plot branch channel
            c2 = ax.contourf(X_branch, Y_branch, T_grid_branch, levels=30, cmap='hot', vmin=0, vmax=3)
            
            # Draw T-junction outline
            rect_main = patches.Rectangle((0, -0.5), 10, 1, linewidth=2, 
                                         edgecolor='cyan', facecolor='none')
            rect_branch = patches.Rectangle((4, 0.5), 2, 2.5, linewidth=2,
                                           edgecolor='cyan', facecolor='none')
            ax.add_patch(rect_main)
            ax.add_patch(rect_branch)
            
            ax.set_title(f'Temperature at t={t_val:.2f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-1, 3.5)
            plt.colorbar(c1, ax=ax, label='T')
            
            # Plot Fuel Mass Fraction
            ax = fig.add_subplot(len(time_points), 2, 2*idx + 2)
            
            c3 = ax.contourf(X_main, Y_main, Y_grid_main, levels=30, cmap='viridis', vmin=0, vmax=1)
            c4 = ax.contourf(X_branch, Y_branch, Y_grid_branch, levels=30, cmap='viridis', vmin=0, vmax=1)
            
            # Draw T-junction outline
            rect_main = patches.Rectangle((0, -0.5), 10, 1, linewidth=2,
                                         edgecolor='white', facecolor='none')
            rect_branch = patches.Rectangle((4, 0.5), 2, 2.5, linewidth=2,
                                           edgecolor='white', facecolor='none')
            ax.add_patch(rect_main)
            ax.add_patch(rect_branch)
            
            ax.set_title(f'Fuel Fraction at t={t_val:.2f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-1, 3.5)
            plt.colorbar(c3, ax=ax, label='Y')
        
        plt.suptitle('T-Junction Flame Propagation Simulation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Plot loss history
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(self.loss_history, 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title('Training Loss History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    print("=" * 50)
    
    # Network architecture
    layers = [3, 128, 128, 128, 128, 5]  # Increased network capacity
    
    # Create model
    model = TJunctionCombustionPINN(layers)
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50)
    
    # Create trainer
    trainer = TJunctionTrainer(model, device)
    
    # Train model
    trainer.train(epochs=2000, print_freq=200)
    
    # Visualize results
    trainer.visualize_results(time_points=[0.0, 0.05, 0.1, 0.15])


if __name__ == "__main__":
    main()