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
        self.Re = 60.0     # Lower Reynolds for smoother flow
        self.Pe = 30.0     # Lower Peclet for better heat diffusion
        self.Sc = 0.7      # Lower Schmidt for better species mixing
        self.Da = 10.0     # Damkohler number for moderate reaction rate
        self.beta = 3.5    # Heat release parameter
        
        # T-Junction geometry parameters
        self.x_branch_start = 0.5   # Branch starts at x=0.5
        self.x_branch_end = 1.5     # Branch ends at x=1.5
        self.y_branch_start = 1.0   # Branch starts at y=1.0
        self.y_branch_top = 3.5     # Branch extends to y=3.5
        
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
        # Main horizontal channel (now extends to y=1.0)
        in_main = (x >= 0) & (x <= 10.0) & (y >= -0.5) & (y <= 1.0)
        
        # Vertical branch channel (starts at y=1.0)
        in_branch = (x >= self.x_branch_start) & (x <= self.x_branch_end) & \
                   (y >= self.y_branch_start) & (y <= self.y_branch_top)
        
        return in_main | in_branch
    
    def forward(self, x, y, t):
        """Forward pass with geometry-aware constraints and natural flow splitting"""
        inputs = torch.cat([x, y, t], dim=1)
        
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        
        outputs = self.layers[-1](inputs)
        
        # Apply geometry mask
        mask = self.is_in_tjunction(x, y).float()
        
        # Base velocity field
        u_base = torch.tanh(outputs[:, 0:1]) * 2.5
        v_base = torch.tanh(outputs[:, 1:2]) * 1.5
        
        # Add gentle flow splitting near junction (x=1.0, y=1.0)
        # Smaller factor for more natural flow
        junction_factor = torch.exp(-((x - 1.0)**2 / 0.3 + (y - 1.0)**2 / 0.3))
        v_split = v_base + junction_factor * 0.8  # Gentler upward component at junction
        
        u = u_base * mask
        v = v_split * mask
        p = torch.tanh(outputs[:, 2:3]) * 2.0 * mask
        T = (0.1 + torch.nn.functional.softplus(outputs[:, 3:4])) * mask
        Y_fuel = torch.sigmoid(outputs[:, 4:5]) * mask
        
        return u, v, p, T, Y_fuel
    
    def reaction_rate(self, T, Y_fuel):
        """Enhanced reaction rate for natural flame propagation"""
        T_safe = torch.clamp(T, min=0.1, max=10.0)
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0)
        
        # Moderate activation energy for steady propagation
        activation_term = -1.2 / T_safe
        activation_term = torch.clamp(activation_term, min=-10, max=10)
        
        rate = self.Da * Y_safe * torch.exp(activation_term)
        
        # Smooth ignition transition
        ignition_factor = torch.sigmoid((T_safe - 0.6) * 15.0)
        
        final_rate = rate * ignition_factor
        return torch.clamp(final_rate, min=0.0, max=120.0)
    
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=0.85)
        self.loss_history = []
    
    def generate_tjunction_points(self, n_points=5000):
        """Generate training points for T-junction geometry with focus on junction"""
        points = []
        
        # Main channel points (now extends to y=1.0)
        n_main = int(n_points * 0.5)
        x_main = torch.rand(n_main, 1) * 10.0
        y_main = torch.rand(n_main, 1) * 1.5 - 0.5  # y in [-0.5, 1.0]
        t_main = torch.rand(n_main, 1) * 0.5  # Extended time range
        
        # Branch channel points (now at x=0.5-1.5)
        n_branch = int(n_points * 0.25)
        x_branch = torch.rand(n_branch, 1) * 1.0 + 0.5  # x in [0.5, 1.5]
        y_branch = torch.rand(n_branch, 1) * 2.5 + 1.0  # y in [1.0, 3.5]
        t_branch = torch.rand(n_branch, 1) * 0.5  # Extended time range
        
        # Junction area points (concentrated sampling around x=1.0, y=1.0)
        n_junction = int(n_points * 0.25)
        x_junction = torch.rand(n_junction, 1) * 2.0 + 0.0  # x in [0.0, 2.0]
        y_junction = torch.rand(n_junction, 1) * 1.0 + 0.5  # y in [0.5, 1.5]
        t_junction = torch.rand(n_junction, 1) * 0.5  # Extended time range
        
        # Filter junction points to be in valid regions
        mask_main_junction = (y_junction <= 1.0).squeeze()
        mask_branch_junction = ((x_junction >= 0.5) & (x_junction <= 1.5) & (y_junction >= 1.0)).squeeze()
        mask_junction = mask_main_junction | mask_branch_junction
        
        # Ensure we maintain 2D shape after filtering
        if mask_junction.any():
            x_junction = x_junction[mask_junction].reshape(-1, 1)
            y_junction = y_junction[mask_junction].reshape(-1, 1)
            t_junction = t_junction[mask_junction].reshape(-1, 1)
        else:
            # If no points pass the filter, create empty tensors with correct shape
            x_junction = torch.empty(0, 1)
            y_junction = torch.empty(0, 1)
            t_junction = torch.empty(0, 1)
        
        # Combine all points
        x = torch.cat([x_main, x_branch, x_junction])
        y = torch.cat([y_main, y_branch, y_junction])
        t = torch.cat([t_main, t_branch, t_junction])
        
        return x.to(self.device), y.to(self.device), t.to(self.device)
    
    def generate_boundary_points(self, n_points=1000):
        """Generate boundary points for T-junction"""
        data = {}
        
        # Inlet (left side of main channel)
        n_inlet = n_points // 4
        x_inlet = torch.zeros(n_inlet, 1)
        y_inlet = torch.rand(n_inlet, 1) * 1.5 - 0.5  # y in [-0.5, 1.0]
        t_inlet = torch.rand(n_inlet, 1) * 0.5  # Extended time range
        data['inlet'] = (x_inlet.to(self.device), 
                        y_inlet.to(self.device),
                        t_inlet.to(self.device))
        
        # Outlet (right side of main channel)
        n_outlet = n_points // 4
        x_outlet = torch.full((n_outlet, 1), 10.0)
        y_outlet = torch.rand(n_outlet, 1) * 1.5 - 0.5  # y in [-0.5, 1.0]
        t_outlet = torch.rand(n_outlet, 1) * 0.5  # Extended time range
        data['outlet'] = (x_outlet.to(self.device),
                         y_outlet.to(self.device),
                         t_outlet.to(self.device))
        
        # Branch outlet (top of branch at x=1.0)
        n_branch_out = n_points // 4
        x_branch_out = torch.rand(n_branch_out, 1) * 1.0 + 0.5  # x in [0.5, 1.5]
        y_branch_out = torch.full((n_branch_out, 1), 3.5)  # Top at y=3.5
        t_branch_out = torch.rand(n_branch_out, 1) * 0.5  # Extended time range
        data['branch_outlet'] = (x_branch_out.to(self.device),
                                y_branch_out.to(self.device),
                                t_branch_out.to(self.device))
        
        # Initial condition points - generate separately to ensure correct size
        n_ic = n_points
        
        # Main channel initial points
        n_ic_main = int(n_ic * 0.7)
        x_ic_main = torch.rand(n_ic_main, 1) * 10.0
        y_ic_main = torch.rand(n_ic_main, 1) * 1.5 - 0.5  # y in [-0.5, 1.0]
        
        # Branch channel initial points (at x=1.0)
        n_ic_branch = n_ic - n_ic_main
        x_ic_branch = torch.rand(n_ic_branch, 1) * 1.0 + 0.5  # x in [0.5, 1.5]
        y_ic_branch = torch.rand(n_ic_branch, 1) * 2.5 + 1.0  # y in [1.0, 3.5]
        
        x_ic = torch.cat([x_ic_main, x_ic_branch])
        y_ic = torch.cat([y_ic_main, y_ic_branch])
        t_ic = torch.zeros(n_ic, 1)
        
        data['initial'] = (x_ic.to(self.device), 
                          y_ic.to(self.device), 
                          t_ic.to(self.device))
        
        return data
    
    def train_step(self, x_int, y_int, t_int, boundary_data):
        """Single training step with enhanced flow conditions"""
        self.optimizer.zero_grad()
        
        # 1. Physics loss
        f_c, f_mx, f_my, f_e, f_s = self.model.physics_loss(x_int, y_int, t_int)
        loss_pde = torch.mean(f_c**2) + torch.mean(f_mx**2) + torch.mean(f_my**2) + \
                  torch.mean(f_e**2) + torch.mean(f_s**2)
        
        # 2. Inlet boundary condition with stronger flow
        x_in, y_in, t_in = boundary_data['inlet']
        u_in, v_in, _, T_in, Y_in = self.model(x_in, y_in, t_in)
        
        # Add slight vertical velocity component near junction level
        v_target = torch.where((y_in > 0.5) & (y_in < 0.9), 
                              0.3 * torch.ones_like(v_in),  # Slight upward flow
                              torch.zeros_like(v_in))
        
        loss_inlet = torch.mean((u_in - 2.5)**2) + torch.mean((v_in - v_target)**2) + \
                    torch.mean((T_in - 3.5)**2) + torch.mean((Y_in - 0.95)**2)
        
        # 3. Initial condition - flame starts from inlet, not at junction
        x_ic, y_ic, t_ic = boundary_data['initial']
        u_ic, v_ic, _, T_ic, Y_ic = self.model(x_ic, y_ic, t_ic)
        
        # Place initial hot spot near inlet to start natural propagation
        T_init = 0.3 + 3.0 * torch.exp(-((x_ic - 0.2)**2 / 0.3 + (y_ic - 0.25)**2 / 0.4))
        # Small temperature gradient to guide flow
        T_init += 0.5 * torch.exp(-((x_ic - 0.5)**2 / 0.5 + (y_ic - 0.5)**2 / 0.5))
        
        Y_init = torch.where(x_ic < 0.5, 
                            0.95 * torch.ones_like(x_ic),  # High fuel at inlet
                            0.85 * torch.ones_like(x_ic))  # Lower fuel elsewhere
        
        # Initial velocity field - mainly horizontal flow
        u_init = torch.where(x_ic < 2.0, 
                            2.5 * torch.ones_like(u_ic),  # Strong flow at inlet
                            1.5 * torch.ones_like(u_ic))  # Moderate flow elsewhere
        
        # Small upward component only near junction
        v_init = torch.where((x_ic > 0.8) & (x_ic < 1.2) & (y_ic > 0.6) & (y_ic < 1.1),
                            0.3 * torch.ones_like(v_ic),  # Small upward flow at junction
                            torch.zeros_like(v_ic))
        
        loss_ic = torch.mean((u_ic - u_init)**2) + torch.mean((v_ic - v_init)**2) + \
                 torch.mean((T_ic - T_init)**2) + torch.mean((Y_ic - Y_init)**2)
        
        # Total loss with weights
        total_loss = loss_pde + 3.0 * loss_inlet + 5.0 * loss_ic
        
        if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, epochs=3000, print_freq=100):
        """Training loop with adaptive sampling"""
        print("🔥 Starting T-Junction Natural Flame Propagation Training...")
        print("=" * 50)
        
        for epoch in range(epochs):
            # Generate training data more frequently for better convergence
            if epoch % 30 == 0:
                x_int, y_int, t_int = self.generate_tjunction_points(7000)
                boundary_data = self.generate_boundary_points(1500)
            
            loss = self.train_step(x_int, y_int, t_int, boundary_data)
            self.loss_history.append(loss)
            self.scheduler.step()
            
            if epoch % print_freq == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f} | LR: {current_lr:.6f}")
        
        print("=" * 50)
        print("✅ Training Complete!")
    
    def visualize_results(self, time_points=[0.0, 0.15, 0.3, 0.45]):
        """Visualize T-junction flame propagation with natural progression"""
        fig = plt.figure(figsize=(16, 14))
        
        # Create finer meshgrid for main channel (now extends to y=1.0)
        x_main = np.linspace(0, 10, 250)
        y_main = np.linspace(-0.5, 1.0, 75)  # Extended to y=1.0
        X_main, Y_main = np.meshgrid(x_main, y_main)
        
        # Create finer meshgrid for branch channel (now at x=0.5-1.5)
        x_branch = np.linspace(0.5, 1.5, 30)  # Branch at x=1.0
        y_branch = np.linspace(1.0, 3.5, 60)  # From y=1.0 to y=3.5
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
                u_main, v_main, _, T_main, Y_main_fuel = self.model(x_torch_main, y_torch_main, t_torch_main)
            
            T_grid_main = T_main.cpu().numpy().reshape(75, 250)
            Y_grid_main = Y_main_fuel.cpu().numpy().reshape(75, 250)
            u_grid_main = u_main.cpu().numpy().reshape(75, 250)
            v_grid_main = v_main.cpu().numpy().reshape(75, 250)
            
            # Predict for branch channel
            x_flat_branch = X_branch.flatten()[:, None]
            y_flat_branch = Y_branch.flatten()[:, None]
            t_flat_branch = np.full_like(x_flat_branch, t_val)
            
            with torch.no_grad():
                x_torch_branch = torch.tensor(x_flat_branch, dtype=torch.float32).to(self.device)
                y_torch_branch = torch.tensor(y_flat_branch, dtype=torch.float32).to(self.device)
                t_torch_branch = torch.tensor(t_flat_branch, dtype=torch.float32).to(self.device)
                u_branch, v_branch, _, T_branch, Y_branch_fuel = self.model(x_torch_branch, y_torch_branch, t_torch_branch)
            
            T_grid_branch = T_branch.cpu().numpy().reshape(60, 30)
            Y_grid_branch = Y_branch_fuel.cpu().numpy().reshape(60, 30)
            u_grid_branch = u_branch.cpu().numpy().reshape(60, 30)
            v_grid_branch = v_branch.cpu().numpy().reshape(60, 30)
            
            # Plot Temperature
            ax = fig.add_subplot(len(time_points), 2, 2*idx + 1)
            
            # Plot main channel
            c1 = ax.contourf(X_main, Y_main, T_grid_main, levels=40, cmap='hot', vmin=0, vmax=4)
            # Plot branch channel
            c2 = ax.contourf(X_branch, Y_branch, T_grid_branch, levels=40, cmap='hot', vmin=0, vmax=4)
            
            # Add velocity vectors to show flow splitting (less frequent for clarity)
            skip = 20
            ax.quiver(X_main[::skip, ::skip], Y_main[::skip, ::skip], 
                     u_grid_main[::skip, ::skip], v_grid_main[::skip, ::skip],
                     alpha=0.4, color='cyan', scale=25, width=0.003)
            
            # Draw T-junction outline
            rect_main = patches.Rectangle((0, -0.5), 10, 1.5, linewidth=2, 
                                         edgecolor='cyan', facecolor='none')
            rect_branch = patches.Rectangle((0.5, 1.0), 1, 2.5, linewidth=2,
                                           edgecolor='cyan', facecolor='none')
            ax.add_patch(rect_main)
            ax.add_patch(rect_branch)
            
            # Add arrow to show flow direction
            ax.annotate('', xy=(9.5, 0.25), xytext=(0.5, 0.25),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))
            ax.annotate('', xy=(1, 3.3), xytext=(1, 1.2),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))
            
            ax.set_title(f'Temperature & Flow at t={t_val:.2f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-1, 4)
            plt.colorbar(c1, ax=ax, label='T')
            
            # Plot Fuel Mass Fraction
            ax = fig.add_subplot(len(time_points), 2, 2*idx + 2)
            
            c3 = ax.contourf(X_main, Y_main, Y_grid_main, levels=40, cmap='viridis', vmin=0, vmax=1)
            c4 = ax.contourf(X_branch, Y_branch, Y_grid_branch, levels=40, cmap='viridis', vmin=0, vmax=1)
            
            # Draw T-junction outline
            rect_main = patches.Rectangle((0, -0.5), 10, 1.5, linewidth=2,
                                         edgecolor='white', facecolor='none')
            rect_branch = patches.Rectangle((0.5, 1.0), 1, 2.5, linewidth=2,
                                           edgecolor='white', facecolor='none')
            ax.add_patch(rect_main)
            ax.add_patch(rect_branch)
            
            ax.set_title(f'Fuel Fraction at t={t_val:.2f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-1, 4)
            plt.colorbar(c3, ax=ax, label='Y')
        
        plt.suptitle('Natural Flame Propagation in T-Junction (Branch at x=1.0)', fontsize=16, fontweight='bold')
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
    
    # Network architecture - Larger network for complex flow
    layers = [3, 150, 150, 150, 150, 150, 5]  
    
    # Create model
    model = TJunctionCombustionPINN(layers)
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50)
    
    # Create trainer
    trainer = TJunctionTrainer(model, device)
    
    # Train model with more epochs for better convergence
    trainer.train(epochs=3000, print_freq=300)
    
    # Visualize results at longer time points for natural propagation
    trainer.visualize_results(time_points=[0.0, 0.15, 0.3, 0.45])


if __name__ == "__main__":
    main()