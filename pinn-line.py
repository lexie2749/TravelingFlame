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
        # Input layer is 3 (x, y, t)
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # --- Physical Parameters (Dimensionless) ---
        # Based on characteristic length L = 5mm (channel width)
        self.Re = 100.0  # Reynolds number
        self.Pe = 50.0   # Peclet number
        self.Sc = 1.0    # Schmidt number
        self.Da = 5.0    # Damkohler number (reaction rate)
        self.beta = 2.0  # Heat release parameter
        
        self.init_weights()
    
    def init_weights(self):
        """Stable weight initialization"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1: # Output layer
                nn.init.uniform_(layer.weight, -0.05, 0.05)
                with torch.no_grad():
                    layer.bias[0] = 0.5   # u
                    layer.bias[1] = 0.0   # v
                    layer.bias[2] = 0.0   # p
                    layer.bias[3] = 0.5   # T (positive bias)
                    layer.bias[4] = 0.5   # Y (positive bias)
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, y, t):
        """
        Forward pass with soft constraints to prevent numerical issues.
        """
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
        """Numerically stable reaction rate calculation"""
        T_safe = torch.clamp(T, min=0.1, max=10.0)
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0)
        
        activation_term = -2.0 / T_safe
        activation_term = torch.clamp(activation_term, min=-10, max=10)
        
        rate = self.Da * Y_safe * torch.exp(activation_term)
        ignition_factor = torch.sigmoid((T_safe - 0.8) * 10.0)
        
        final_rate = rate * ignition_factor
        return torch.clamp(final_rate, min=0.0, max=100.0)
    
    def physics_loss(self, x, y, t):
        """
        Calculates the physics loss defined by the 2D governing equations.
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T, Y = self.forward(x, y, t)
        
        def compute_gradient(output, input_var):
            grad_val = grad(output, input_var, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, allow_unused=True)[0]
            if grad_val is None: return torch.zeros_like(input_var)
            grad_val = torch.where(torch.isnan(grad_val) | torch.isinf(grad_val), torch.zeros_like(grad_val), grad_val)
            return torch.clamp(grad_val, -1e6, 1e6)

        # First derivatives
        u_x, u_y, u_t = compute_gradient(u, x), compute_gradient(u, y), compute_gradient(u, t)
        v_x, v_y, v_t = compute_gradient(v, x), compute_gradient(v, y), compute_gradient(v, t)
        p_x, p_y = compute_gradient(p, x), compute_gradient(p, y)
        T_x, T_y, T_t = compute_gradient(T, x), compute_gradient(T, y), compute_gradient(T, t)
        Y_x, Y_y, Y_t = compute_gradient(Y, x), compute_gradient(Y, y), compute_gradient(Y, t)
        
        # Second derivatives
        u_xx = compute_gradient(u_x, x)
        u_yy = compute_gradient(u_y, y)
        v_xx = compute_gradient(v_x, x)
        v_yy = compute_gradient(v_y, y)
        T_xx = compute_gradient(T_x, x)
        T_yy = compute_gradient(T_y, y)
        Y_xx = compute_gradient(Y_x, x)
        Y_yy = compute_gradient(Y_y, y)

        omega = self.safe_reaction_rate(T, Y)
        
        # 2D Governing Equations
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
        
        # Internal points
        x_int = torch.rand(n_points, 1) * (x_max - x_min) + x_min
        y_int = torch.rand(n_points, 1) * (y_max - y_min) + y_min
        t_int = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        
        n_bc = n_points // 4
        # Boundary points: inlet, outlet, top/bottom walls
        x_inlet = torch.full((n_bc, 1), x_min)
        y_inlet = torch.rand(n_bc, 1) * (y_max - y_min) + y_min
        t_inlet = torch.rand(n_bc, 1) * (t_max - t_min) + t_min

        x_outlet = torch.full((n_bc, 1), x_max)
        y_outlet = torch.rand(n_bc, 1) * (y_max - y_min) + y_min
        t_outlet = torch.rand(n_bc, 1) * (t_max - t_min) + t_min

        x_wall = torch.rand(n_bc, 1) * (x_max - x_min) + x_min
        y_wall_top = torch.full((n_bc//2, 1), y_max)
        y_wall_bottom = torch.full((n_bc//2, 1), y_min)
        y_wall = torch.cat([y_wall_top, y_wall_bottom])
        t_wall = torch.rand(n_bc, 1) * (t_max - t_min) + t_min
        
        # Initial points
        n_ic = n_points // 2
        x_ic = torch.rand(n_ic, 1) * (x_max - x_min) + x_min
        y_ic = torch.rand(n_ic, 1) * (y_max - y_min) + y_min
        t_ic = torch.full((n_ic, 1), t_min)
        
        return {
            'internal': (x_int.to(self.device), y_int.to(self.device), t_int.to(self.device)),
            'inlet': (x_inlet.to(self.device), y_inlet.to(self.device), t_inlet.to(self.device)),
            'outlet': (x_outlet.to(self.device), y_outlet.to(self.device), t_outlet.to(self.device)),
            'wall': (x_wall.to(self.device), y_wall.to(self.device), t_wall.to(self.device)),
            'initial': (x_ic.to(self.device), y_ic.to(self.device), t_ic.to(self.device))
        }

    def train_step(self, data):
        self.optimizer.zero_grad()
        
        # 1. Physics Loss (internal points)
        x_int, y_int, t_int = data['internal']
        f_c, f_mx, f_my, f_e, f_s, _ = self.model.physics_loss(x_int, y_int, t_int)
        loss_pde = torch.mean(f_c**2) + torch.mean(f_mx**2) + torch.mean(f_my**2) + \
                   torch.mean(f_e**2) + torch.mean(f_s**2)
        
        # 2. Boundary Condition Loss
        # Inlet (x=0)
        x_in, y_in, t_in = data['inlet']
        u_in, v_in, p_in, T_in, Y_in = self.model(x_in, y_in, t_in)
        loss_inlet = torch.mean((u_in - 2.0)**2) + torch.mean(v_in**2) + \
                     torch.mean((T_in - 2.0)**2) + torch.mean((Y_in - 0.9)**2)
        
        # Walls (y = +/- 0.5) - No-slip & Adiabatic
        x_w, y_w, t_w = data['wall']
        u_w, v_w, _, _, _ = self.model(x_w, y_w, t_w)
        _, _, _, _, _, T_y_w = self.model.physics_loss(x_w, y_w, t_w) # Get temperature gradient
        loss_wall = torch.mean(u_w**2) + torch.mean(v_w**2) + torch.mean(T_y_w**2)
        
        # 3. Initial Condition Loss
        x_ic, y_ic, t_ic = data['initial']
        u_ic, v_ic, p_ic, T_ic, Y_ic = self.model(x_ic, y_ic, t_ic)
        T_init = 0.3 + 1.5 * torch.exp(-((x_ic - 1.0) / 0.5)**2) # Gaussian hot spot
        Y_init = 0.8 * torch.ones_like(x_ic)
        loss_ic = torch.mean((u_ic - 1.0)**2) + torch.mean(v_ic**2) + \
                  torch.mean((T_ic - T_init)**2) + torch.mean((Y_ic - Y_init)**2)
        
        # Total Loss
        total_loss = loss_pde + 2.0 * loss_inlet + 2.0 * loss_wall + 5.0 * loss_ic
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("  ⚠️  Loss is NaN or Inf, skipping update.")
            return float('inf')

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def train(self, epochs=1500, print_freq=100):
        print("🚀 Starting 2D Channel Combustion Training...")
        for epoch in range(epochs):
            if epoch % 50 == 0: # Resample data periodically
                training_data = self.generate_training_data()
            
            total_loss = self.train_step(training_data)
            self.loss_history.append(total_loss)
            
            if epoch % print_freq == 0:
                print(f"Epoch {epoch}/{epochs} | Total Loss: {total_loss:.6f}")
        print("✅ Training complete!")
    
    def plot_results(self, t_final=0.1):
        """Plots the 2D result fields and loss history."""
        x_min, x_max, y_min, y_max, _, _ = self.domain_bounds
        
        # Create meshgrid
        nx, ny = 100, 50
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        x_flat = X.flatten()[:, None]
        y_flat = Y.flatten()[:, None]
        t_flat = np.full_like(x_flat, t_final)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            x_torch = torch.tensor(x_flat, dtype=torch.float32).to(self.device)
            y_torch = torch.tensor(y_flat, dtype=torch.float32).to(self.device)
            t_torch = torch.tensor(t_flat, dtype=torch.float32).to(self.device)
            u, v, p, T, Y_fuel = self.model(x_torch, y_torch, t_torch)
        
        # Reshape to 2D grid
        u_grid = u.cpu().numpy().reshape(ny, nx)
        v_grid = v.cpu().numpy().reshape(ny, nx)
        T_grid = T.cpu().numpy().reshape(ny, nx)
        Y_grid = Y_fuel.cpu().numpy().reshape(ny, nx)

        # Plotting
        fig = plt.figure(figsize=(18, 10))
        
        # Temperature Field
        ax1 = fig.add_subplot(2, 2, 1)
        c1 = ax1.contourf(X, Y, T_grid, levels=50, cmap='inferno')
        fig.colorbar(c1, ax=ax1)
        ax1.set_title(f'Temperature Field T (t={t_final})')
        ax1.set_xlabel('x (dimensionless)')
        ax1.set_ylabel('y (dimensionless)')
        ax1.set_aspect('equal')

        # Fuel Mass Fraction Field
        ax2 = fig.add_subplot(2, 2, 2)
        c2 = ax2.contourf(X, Y, Y_grid, levels=50, cmap='viridis')
        fig.colorbar(c2, ax=ax2)
        ax2.set_title(f'Fuel Mass Fraction Y (t={t_final})')
        ax2.set_xlabel('x (dimensionless)')
        ax2.set_ylabel('y (dimensionless)')
        ax2.set_aspect('equal')
        
        # Velocity Field (Streamlines)
        ax3 = fig.add_subplot(2, 2, 3)
        speed = np.sqrt(u_grid**2 + v_grid**2)
        ax3.streamplot(X, Y, u_grid, v_grid, color=speed, cmap='coolwarm', density=1.5)
        ax3.set_title(f'Velocity Streamlines (t={t_final})')
        ax3.set_xlabel('x (dimensionless)')
        ax3.set_ylabel('y (dimensionless)')
        ax3.set_aspect('equal')

        # Loss History
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.semilogy(self.loss_history, 'b-', label='Total Loss')
        ax4.set_title('Loss History')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss (log scale)')
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        plt.show()

def main():
    """Main function"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Network architecture: Input layer 3 (x,y,t), 4 hidden layers, Output layer 5 (u,v,p,T,Y)
    layers = [3, 64, 64, 64, 64, 5]
    
    model = CombustionPINN_2D(layers)
    
    # Define computational domain: x-direction is 10x width, y-direction is 1x width (5mm)
    # [x_min, x_max, y_min, y_max, t_min, t_max]
    # y is normalized from -0.5 to 0.5 to represent the channel
    domain_bounds = [0.0, 10.0, -0.5, 0.5, 0.0, 0.1]
    
    trainer = Trainer_2D(model, domain_bounds, device)
    
    trainer.train(epochs=1500, print_freq=100)
    trainer.plot_results(t_final=0.1)

if __name__ == "__main__":
    main()
