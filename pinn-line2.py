import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import time

class CombustionPINN_2D(nn.Module):
    """
    Physics-Informed Neural Network for 2D Channel Combustion
    - Input: (x, y, t)
    - Output: (u, v, p, T, Y)
    """
    def __init__(self, layers, activation=torch.tanh):
        super(CombustionPINN_2D, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        self.Re = 100.0
        self.Pe = 50.0
        self.Sc = 1.0
        self.Da = 5.0
        self.beta = 2.0
        
        self.init_weights()
    
    def init_weights(self):
        """Stable weight initialization"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                nn.init.uniform_(layer.weight, -0.05, 0.05)
                with torch.no_grad():
                    layer.bias[0] = 0.5; layer.bias[1] = 0.0; layer.bias[2] = 0.0
                    layer.bias[3] = 0.5; layer.bias[4] = 0.5
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, y, t):
        """Forward pass"""
        inputs = torch.cat([x, y, t], dim=1)
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        outputs = self.layers[-1](inputs)
        
        u = torch.tanh(outputs[:, 0:1]) * 3.0
        v = torch.tanh(outputs[:, 1:2]) * 1.0
        p = torch.tanh(outputs[:, 2:3]) * 2.0
        T = 0.1 + torch.nn.functional.softplus(outputs[:, 3:4])
        Y_fuel = torch.sigmoid(outputs[:, 4:5])
        
        return u, v, p, T, Y_fuel
    
    def safe_reaction_rate(self, T, Y_fuel):
        """Numerically stable reaction rate"""
        T_safe = torch.clamp(T, min=0.1, max=10.0)
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0)
        activation_term = torch.clamp(-2.0 / T_safe, min=-10, max=10)
        rate = self.Da * Y_safe * torch.exp(activation_term)
        ignition_factor = torch.sigmoid((T_safe - 0.8) * 10.0)
        return torch.clamp(rate * ignition_factor, min=0.0, max=100.0)
    
    def physics_loss(self, x, y, t):
        """Calculates the physics loss from 2D governing equations"""
        x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
        u, v, p, T, Y = self.forward(x, y, t)
        
        def compute_gradient(output, input_var):
            grad_val = grad(output, input_var, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, allow_unused=True)[0]
            if grad_val is None: return torch.zeros_like(input_var)
            grad_val = torch.where(torch.isnan(grad_val) | torch.isinf(grad_val), torch.zeros_like(grad_val), grad_val)
            return torch.clamp(grad_val, -1e6, 1e6)

        u_x, u_y, u_t = compute_gradient(u, x), compute_gradient(u, y), compute_gradient(u, t)
        v_x, v_y, v_t = compute_gradient(v, x), compute_gradient(v, y), compute_gradient(v, t)
        p_x, p_y = compute_gradient(p, x), compute_gradient(p, y)
        T_x, T_y, T_t = compute_gradient(T, x), compute_gradient(T, y), compute_gradient(T, t)
        Y_x, Y_y, Y_t = compute_gradient(Y, x), compute_gradient(Y, y), compute_gradient(Y, t)
        
        u_xx, u_yy = compute_gradient(u_x, x), compute_gradient(u_y, y)
        v_xx, v_yy = compute_gradient(v_x, x), compute_gradient(v_y, y)
        T_xx, T_yy = compute_gradient(T_x, x), compute_gradient(T_y, y)
        Y_xx, Y_yy = compute_gradient(Y_x, x), compute_gradient(Y_y, y)

        omega = self.safe_reaction_rate(T, Y)
        
        f_continuity = u_x + v_y
        f_momentum_x = u_t + u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)
        f_momentum_y = v_t + u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)
        f_energy = T_t + u * T_x + v * T_y - (1/self.Pe) * (T_xx + T_yy) - self.beta * omega
        f_species = Y_t + u * Y_x + v * Y_y - (1/self.Sc) * (Y_xx + Y_yy) + omega
        
        return f_continuity, f_momentum_x, f_momentum_y, f_energy, f_species, T_y

class Trainer_2D:
    def __init__(self, model, domain_bounds, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.domain_bounds = domain_bounds
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss_history = []

    def generate_training_data(self, n_points=4000):
        x_min, x_max, y_min, y_max, t_min, t_max = self.domain_bounds
        
        x_int = torch.rand(n_points, 1) * (x_max - x_min) + x_min
        y_int = torch.rand(n_points, 1) * (y_max - y_min) + y_min
        t_int = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        
        n_bc = n_points // 4
        x_inlet = torch.full((n_bc, 1), x_min)
        y_inlet = torch.rand(n_bc, 1) * (y_max - y_min) + y_min
        t_inlet = torch.rand(n_bc, 1) * (t_max - t_min) + t_min

        x_wall = torch.rand(n_bc, 1) * (x_max - x_min) + x_min
        y_wall_top = torch.full((n_bc//2, 1), y_max)
        y_wall_bottom = torch.full((n_bc//2, 1), y_min)
        y_wall = torch.cat([y_wall_top, y_wall_bottom])
        t_wall = torch.rand(n_bc, 1) * (t_max - t_min) + t_min
        
        n_ic = n_points // 2
        x_ic = torch.rand(n_ic, 1) * (x_max - x_min) + x_min
        y_ic = torch.rand(n_ic, 1) * (y_max - y_min) + y_min
        t_ic = torch.full((n_ic, 1), t_min)
        
        return {
            'internal': (x_int.to(self.device), y_int.to(self.device), t_int.to(self.device)),
            'inlet': (x_inlet.to(self.device), y_inlet.to(self.device), t_inlet.to(self.device)),
            'wall': (x_wall.to(self.device), y_wall.to(self.device), t_wall.to(self.device)),
            'initial': (x_ic.to(self.device), y_ic.to(self.device), t_ic.to(self.device))
        }

    def train_step(self, data):
        self.optimizer.zero_grad()
        
        # Physics Loss
        x_int, y_int, t_int = data['internal']
        f_c, f_mx, f_my, f_e, f_s, _ = self.model.physics_loss(x_int, y_int, t_int)
        loss_pde = torch.mean(f_c**2) + torch.mean(f_mx**2) + torch.mean(f_my**2) + \
                   torch.mean(f_e**2) + torch.mean(f_s**2)
        
        # Boundary Condition Loss
        # Inlet with time-dependent ignition
        x_in, y_in, t_in = data['inlet']
        u_in, v_in, _, T_in, Y_in = self.model(x_in, y_in, t_in)
        
        ignition_on = (t_in <= 0.01)
        T_inlet_target = torch.where(ignition_on, 2.0, 0.3) # Hot then cool
        Y_inlet_target = torch.where(ignition_on, 0.9, 0.1) # Fuel-rich then lean
        
        loss_inlet = torch.mean((u_in - 1.0)**2) + torch.mean(v_in**2) + \
                     torch.mean((T_in - T_inlet_target)**2) + torch.mean((Y_in - Y_inlet_target)**2)
        
        # Wall Loss
        x_w, y_w, t_w = data['wall']
        u_w, v_w, _, _, _ = self.model(x_w, y_w, t_w)
        _, _, _, _, _, T_y_w = self.model.physics_loss(x_w, y_w, t_w)
        loss_wall = torch.mean(u_w**2) + torch.mean(v_w**2) + torch.mean(T_y_w**2)
        
        # Initial Condition Loss
        x_ic, y_ic, t_ic = data['initial']
        u_ic, v_ic, _, T_ic, Y_ic = self.model(x_ic, y_ic, t_ic)
        T_init = 0.3 + 1.8 * torch.exp(-((x_ic - 0.5) / 0.2)**2) # Centered hot spot
        Y_init = 0.9 * torch.ones_like(x_ic)
        loss_ic = torch.mean((u_ic - 1.0)**2) + torch.mean(v_ic**2) + \
                  torch.mean((T_ic - T_init)**2) + torch.mean((Y_ic - Y_init)**2)
        
        total_loss = loss_pde + 3.0 * loss_inlet + 3.0 * loss_wall + 3.0 * loss_ic
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return float('inf')

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def train(self, epochs=3000, print_freq=100):
        print("🚀 Starting 2D Channel Combustion Training...")
        for epoch in range(epochs):
            if epoch % 50 == 0:
                training_data = self.generate_training_data()
            
            total_loss = self.train_step(training_data)
            self.loss_history.append(total_loss)
            
            if epoch % print_freq == 0:
                print(f"Epoch {epoch}/{epochs} | Total Loss: {total_loss:.6f}")
        print("✅ Training complete!")
    
    def plot_results(self):
        """Plots the flame evolution at multiple time steps."""
        x_min, x_max, y_min, y_max, _, _ = self.domain_bounds
        time_steps = np.linspace(0.1, 0.8, 8)
        
        # --- Figure 1: Time Evolution of Temperature Field ---
        fig1, axes1 = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
        axes1 = axes1.flatten()
        fig1.suptitle('Flame Propagation and Dissipation Over Time', fontsize=16)

        # --- Figure 2: Combined Flame Front Progression ---
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
        ax2.set_title('Combined Flame Fronts (T=1.5 Contour)')
        ax2.set_xlabel('x (dimensionless)')
        ax2.set_ylabel('y (dimensionless)')
        ax2.set_aspect('equal')
        ax2.grid(True, linestyle='--', alpha=0.6)
        colors = plt.cm.jet(np.linspace(0, 1, len(time_steps)))

        self.model.eval()
        with torch.no_grad():
            for i, t_val in enumerate(time_steps):
                print(f"Plotting for t = {t_val:.2f}...")
                nx, ny = 150, 75
                x = np.linspace(x_min, x_max, nx)
                y = np.linspace(y_min, y_max, ny)
                X, Y = np.meshgrid(x, y)
                
                x_flat = X.flatten()[:, None]
                y_flat = Y.flatten()[:, None]
                t_flat = np.full_like(x_flat, t_val)
                
                x_torch = torch.tensor(x_flat, dtype=torch.float32).to(self.device)
                y_torch = torch.tensor(y_flat, dtype=torch.float32).to(self.device)
                t_torch = torch.tensor(t_flat, dtype=torch.float32).to(self.device)
                _, _, _, T, _ = self.model(x_torch, y_torch, t_torch)
                
                T_grid = T.cpu().numpy().reshape(ny, nx)
                
                # Plot individual temperature field
                ax = axes1[i]
                c = ax.contourf(X, Y, T_grid, levels=50, cmap='inferno', vmin=0, vmax=2.5)
                ax.set_title(f't = {t_val:.2f} s')
                ax.set_aspect('equal')
                
                # Plot combined flame front
                ax2.contour(X, Y, T_grid, levels=[1.5], colors=[colors[i]], linewidths=2)

        # Add a single colorbar for the temperature plots
        fig1.colorbar(c, ax=axes1.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Create a custom legend for the combined plot
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f't={t_val:.2f}s') for i, t_val in enumerate(time_steps)]
        ax2.legend(handles=legend_elements, loc='upper right')
        fig2.tight_layout()

        # --- Figure 3: Loss History ---
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.semilogy(self.loss_history, 'b-', label='Total Loss')
        ax3.set_title('Loss History')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (log scale)')
        ax3.grid(True)
        ax3.legend()
        fig3.tight_layout()

        plt.show()

def main():
    """Main function"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    layers = [3, 64, 64, 64, 64, 5]
    model = CombustionPINN_2D(layers)
    
    # Extend time domain to capture the full evolution
    domain_bounds = [0.0, 10.0, -0.5, 0.5, 0.0, 1.0]
    
    trainer = Trainer_2D(model, domain_bounds, device)
    
    trainer.train(epochs=3000, print_freq=100)
    trainer.plot_results()

if __name__ == "__main__":
    main()
