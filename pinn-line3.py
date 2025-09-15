import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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
        
        # REVISED PARAMETERS FOR FASTER FLAME SPEED
        self.Re = 100.0      # Reynolds number
        self.Pe = 20.0       # Reduced Péclet number (faster heat diffusion)
        self.Sc = 0.8        # Reduced Schmidt number (faster mass diffusion)
        self.Da = 15.0       # Increased Damköhler number (faster reaction)
        self.beta = 3.0      # Increased heat release (more energy)
        self.Le = 0.5        # Lewis number < 1 for faster flames
        
        self.init_weights()
    
    def init_weights(self):
        """Stable weight initialization"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                nn.init.uniform_(layer.weight, -0.05, 0.05)
                with torch.no_grad():
                    layer.bias[0] = 0.5  # u velocity
                    layer.bias[1] = 0.0  # v velocity
                    layer.bias[2] = 0.0  # pressure
                    layer.bias[3] = 0.8  # temperature (higher initial)
                    layer.bias[4] = 0.5  # fuel fraction
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, y, t):
        """Forward pass with enhanced outputs for faster propagation"""
        inputs = torch.cat([x, y, t], dim=1)
        
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        
        outputs = self.layers[-1](inputs)
        
        # Increased velocity range for faster convection
        u = torch.tanh(outputs[:, 0:1]) * 5.0  # Increased from 3.0
        v = torch.tanh(outputs[:, 1:2]) * 1.5  # Increased from 1.0
        p = torch.tanh(outputs[:, 2:3]) * 2.0
        T = 0.2 + torch.nn.functional.softplus(outputs[:, 3:4]) * 1.5  # Enhanced temperature
        Y_fuel = torch.sigmoid(outputs[:, 4:5])
        
        return u, v, p, T, Y_fuel
    
    def safe_reaction_rate(self, T, Y_fuel):
        """Enhanced reaction rate for faster flame propagation"""
        T_safe = torch.clamp(T, min=0.1, max=10.0)
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0)
        
        # Reduced activation energy for easier ignition
        activation_term = torch.clamp(-1.0 / T_safe, min=-10, max=10)  # Reduced from -2.0
        rate = self.Da * Y_safe * torch.exp(activation_term)
        
        # Lower ignition temperature threshold
        ignition_factor = torch.sigmoid((T_safe - 0.5) * 15.0)  # Lowered from 0.8, steeper transition
        
        return torch.clamp(rate * ignition_factor, min=0.0, max=200.0)  # Increased max from 100.0
    
    def physics_loss(self, x, y, t):
        """Calculates the physics loss from 2D governing equations"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T, Y = self.forward(x, y, t)
        
        def compute_gradient(output, input_var):
            grad_val = grad(output, input_var, 
                          grad_outputs=torch.ones_like(output),
                          create_graph=True, retain_graph=True, allow_unused=True)[0]
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
        
        # Enhanced reaction rate
        omega = self.safe_reaction_rate(T, Y)
        
        # Governing equations with modified coefficients
        f_continuity = u_x + v_y
        
        f_momentum_x = u_t + u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)
        f_momentum_y = v_t + u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)
        
        # Modified energy equation with enhanced heat release
        f_energy = T_t + u * T_x + v * T_y - (1/self.Pe) * (T_xx + T_yy) - self.beta * omega
        
        # Modified species equation with Lewis number effect
        f_species = Y_t + u * Y_x + v * Y_y - (1/(self.Sc * self.Le)) * (Y_xx + Y_yy) + omega
        
        return f_continuity, f_momentum_x, f_momentum_y, f_energy, f_species, T_y

class Trainer_2D:
    def __init__(self, model, domain_bounds, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.domain_bounds = domain_bounds
        
        # Adaptive learning rate for better convergence
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Check PyTorch version for compatibility
        try:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=200, verbose=True
            )
        except TypeError:
            # For newer PyTorch versions without verbose parameter
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=200
            )
        self.loss_history = []
    
    def generate_training_data(self, n_points=5000):  # Increased points
        x_min, x_max, y_min, y_max, t_min, t_max = self.domain_bounds
        
        # Interior points
        x_int = torch.rand(n_points, 1) * (x_max - x_min) + x_min
        y_int = torch.rand(n_points, 1) * (y_max - y_min) + y_min
        t_int = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        
        # Boundary conditions
        n_bc = n_points // 4
        
        # Inlet (left boundary)
        x_inlet = torch.full((n_bc, 1), x_min)
        y_inlet = torch.rand(n_bc, 1) * (y_max - y_min) + y_min
        t_inlet = torch.rand(n_bc, 1) * (t_max - t_min) + t_min
        
        # Walls (top and bottom)
        x_wall = torch.rand(n_bc, 1) * (x_max - x_min) + x_min
        y_wall_top = torch.full((n_bc//2, 1), y_max)
        y_wall_bottom = torch.full((n_bc//2, 1), y_min)
        y_wall = torch.cat([y_wall_top, y_wall_bottom])
        t_wall = torch.rand(n_bc, 1) * (t_max - t_min) + t_min
        
        # Initial condition
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
        
        loss_pde = (torch.mean(f_c**2) + 
                   torch.mean(f_mx**2) + 
                   torch.mean(f_my**2) + 
                   torch.mean(f_e**2) * 2.0 +  # Emphasis on energy
                   torch.mean(f_s**2) * 2.0)    # Emphasis on species
        
        # Inlet Boundary Condition - Enhanced for faster ignition
        x_in, y_in, t_in = data['inlet']
        u_in, v_in, _, T_in, Y_in = self.model(x_in, y_in, t_in)
        
        # Stronger and longer ignition pulse
        ignition_on = (t_in <= 0.05)  # Extended from 0.01
        T_inlet_target = torch.where(ignition_on, 3.0, 0.5)  # Higher temperature
        Y_inlet_target = torch.where(ignition_on, 0.95, 0.2)  # More fuel
        u_inlet_target = torch.where(ignition_on, 2.0, 1.5)  # Higher inlet velocity
        
        loss_inlet = (torch.mean((u_in - u_inlet_target)**2) + 
                     torch.mean(v_in**2) + 
                     torch.mean((T_in - T_inlet_target)**2) * 2.0 +  # Emphasis on temperature
                     torch.mean((Y_in - Y_inlet_target)**2))
        
        # Wall Boundary Conditions
        x_w, y_w, t_w = data['wall']
        u_w, v_w, _, _, _ = self.model(x_w, y_w, t_w)
        _, _, _, _, _, T_y_w = self.model.physics_loss(x_w, y_w, t_w)
        
        # Slip condition for faster flow
        loss_wall = torch.mean(v_w**2) + torch.mean(T_y_w**2) * 0.5  # Reduced u_w constraint
        
        # Initial Condition - Wider ignition zone
        x_ic, y_ic, t_ic = data['initial']
        u_ic, v_ic, _, T_ic, Y_ic = self.model(x_ic, y_ic, t_ic)
        
        # Broader initial hot zone for faster ignition
        T_init = 0.5 + 2.5 * torch.exp(-((x_ic - 0.3) / 0.4)**2)  # Wider, hotter zone
        Y_init = 0.95 * torch.ones_like(x_ic)
        u_init = 1.5 * torch.ones_like(x_ic)  # Initial velocity
        
        loss_ic = (torch.mean((u_ic - u_init)**2) + 
                  torch.mean(v_ic**2) + 
                  torch.mean((T_ic - T_init)**2) * 2.0 +  # Emphasis on temperature
                  torch.mean((Y_ic - Y_init)**2))
        
        # Total loss with adjusted weights
        total_loss = loss_pde + 5.0 * loss_inlet + 2.0 * loss_wall + 5.0 * loss_ic
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return float('inf')
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, epochs=3000, print_freq=100):
        print("🚀 Starting Enhanced 2D Channel Combustion Training...")
        print(f"   Da = {self.model.Da:.1f} (Damköhler number)")
        print(f"   Pe = {self.model.Pe:.1f} (Péclet number)")
        print(f"   β = {self.model.beta:.1f} (Heat release)")
        print("-" * 50)
        
        best_loss = float('inf')
        patience_counter = 0
        last_lr = self.optimizer.param_groups[0]['lr']
        
        for epoch in range(epochs):
            if epoch % 50 == 0:
                training_data = self.generate_training_data()
            
            total_loss = self.train_step(training_data)
            self.loss_history.append(total_loss)
            
            # Learning rate scheduling
            if epoch > 500:
                self.scheduler.step(total_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr != last_lr:
                    print(f"  → Learning rate changed: {last_lr:.6f} → {current_lr:.6f}")
                    last_lr = current_lr
            
            # Early stopping check
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{epochs} | Loss: {total_loss:.6f} | LR: {current_lr:.6f}")
                
                if patience_counter > 500:
                    print("Early stopping due to no improvement")
                    break
        
        print("✅ Training complete!")
        print(f"   Final loss: {total_loss:.6f}")
        print(f"   Best loss: {best_loss:.6f}")
    
    def plot_results(self):
        """Plots the flame evolution with improved aesthetics."""
        x_min, x_max, y_min, y_max, _, _ = self.domain_bounds
        time_steps = np.linspace(0.05, 0.8, 8)  # Start earlier to catch fast flame
        
        # Figure 1: Time Evolution of Temperature Field
        fig1, axes1 = plt.subplots(2, 4, figsize=(20, 7), sharex=True, sharey=True)
        axes1 = axes1.flatten()
        fig1.suptitle('Enhanced Flame Propagation (Faster Speed)', fontsize=20, fontweight='bold')
        
        # Figure 2: Combined Flame Front Progression
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 7))
        ax2.set_title('Flame Front Progression (T=1.5 Contour) - Enhanced Speed', fontsize=16)
        ax2.set_xlabel('x (dimensionless)', fontsize=12)
        ax2.set_ylabel('y (dimensionless)', fontsize=12)
        ax2.set_aspect('equal')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        cmap = cm.get_cmap('plasma')  # Changed colormap
        norm = Normalize(vmin=time_steps.min(), vmax=time_steps.max())
        
        self.model.eval()
        with torch.no_grad():
            # Get final temperature field for background
            print("Generating enhanced flame propagation visualization...")
            nx, ny = 150, 75
            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)
            X, Y = np.meshgrid(x, y)
            
            x_flat = X.flatten()[:, None]
            y_flat = Y.flatten()[:, None]
            
            # Final time background
            t_final = torch.full_like(torch.from_numpy(x_flat), time_steps[-1])
            _, _, _, T_final, _ = self.model(
                torch.from_numpy(x_flat).float().to(self.device),
                torch.from_numpy(y_flat).float().to(self.device),
                t_final.float().to(self.device)
            )
            T_final_grid = T_final.cpu().numpy().reshape(ny, nx)
            ax2.contourf(X, Y, T_final_grid, levels=50, cmap='bone', alpha=0.3)
            
            # Loop through time steps
            for i, t_val in enumerate(time_steps):
                print(f"  Plotting t = {t_val:.2f}s (faster flame)...")
                
                t_flat = np.full_like(x_flat, t_val)
                x_torch = torch.tensor(x_flat, dtype=torch.float32).to(self.device)
                y_torch = torch.tensor(y_flat, dtype=torch.float32).to(self.device)
                t_torch = torch.tensor(t_flat, dtype=torch.float32).to(self.device)
                
                _, _, _, T, _ = self.model(x_torch, y_torch, t_torch)
                T_grid = T.cpu().numpy().reshape(ny, nx)
                
                # Plot individual temperature field
                ax = axes1[i]
                c = ax.contourf(X, Y, T_grid, levels=50, cmap='hot', vmin=0, vmax=3.0)
                ax.set_title(f't = {t_val:.2f} s', fontsize=14, fontweight='bold')
                ax.set_aspect('equal')
                
                # Add flame speed annotation
                flame_pos = np.where(T_grid > 1.5)[1]
                if len(flame_pos) > 0:
                    flame_front = np.max(flame_pos) * (x_max - x_min) / nx
                    ax.text(0.02, 0.95, f'Front: {flame_front:.1f}', 
                           transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # Add axis labels
                if i >= 4:
                    ax.set_xlabel('x', fontsize=12)
                if i % 4 == 0:
                    ax.set_ylabel('y', fontsize=12)
                
                # Plot combined flame front
                contour = ax2.contour(X, Y, T_grid, levels=[1.5], 
                                     colors=[cmap(norm(t_val))], linewidths=3.0)
                
                # Add arrow to show propagation direction
                if i == 3:  # Middle of animation
                    if len(flame_pos) > 0:
                        flame_x = np.max(flame_pos) * (x_max - x_min) / nx
                        ax2.annotate('', xy=(flame_x + 1, 0), xytext=(flame_x, 0),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
                        ax2.text(flame_x + 0.5, 0.2, 'Propagation', fontsize=12, color='red')
        
        # Adjust Figure 1 layout and add colorbar
        fig1.subplots_adjust(right=0.85, hspace=0.3, wspace=0.15)
        cbar_ax = fig1.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig1.colorbar(c, cax=cbar_ax)
        cbar.set_label('Temperature', rotation=270, labelpad=20, fontsize=12)
        
        # Add parameter info
        fig1.text(0.02, 0.98, f'Da={self.model.Da:.1f}, Pe={self.model.Pe:.1f}, β={self.model.beta:.1f}',
                 transform=fig1.transFigure, fontsize=12, 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Finalize Figure 2 with time colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar2 = fig2.colorbar(sm, ax=ax2)
        cbar2.set_label('Time (s)', rotation=270, labelpad=15, fontsize=12)
        fig2.tight_layout()
        
        # Figure 3: Loss History
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.semilogy(self.loss_history, 'b-', linewidth=2, label='Total Loss')
        ax3.set_title('Training Loss History', fontsize=16)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss (log scale)', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=12)
        
        # Add convergence rate annotation
        if len(self.loss_history) > 100:
            initial_loss = self.loss_history[100]
            final_loss = self.loss_history[-1]
            reduction = (1 - final_loss/initial_loss) * 100
            ax3.text(0.6, 0.9, f'Loss reduction: {reduction:.1f}%',
                    transform=ax3.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        fig3.tight_layout()
        
        # Figure 4: Flame Speed Analysis
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Compute flame speed
        print("\nComputing flame propagation speed...")
        flame_positions = []
        times = np.linspace(0.1, 0.8, 15)
        
        with torch.no_grad():
            for t_val in times:
                t_flat = np.full_like(x_flat, t_val)
                x_torch = torch.tensor(x_flat, dtype=torch.float32).to(self.device)
                y_torch = torch.tensor(y_flat, dtype=torch.float32).to(self.device)
                t_torch = torch.tensor(t_flat, dtype=torch.float32).to(self.device)
                
                _, _, _, T, _ = self.model(x_torch, y_torch, t_torch)
                T_grid = T.cpu().numpy().reshape(ny, nx)
                
                # Find flame front position (T = 1.5 contour)
                flame_pos = np.where(T_grid > 1.5)[1]
                if len(flame_pos) > 0:
                    flame_positions.append(np.max(flame_pos) * (x_max - x_min) / nx)
                else:
                    flame_positions.append(0)
        
        flame_positions = np.array(flame_positions)
        flame_speed = np.gradient(flame_positions, times)
        
        # Plot flame position
        ax4a.plot(times, flame_positions, 'b-', linewidth=2, marker='o', markersize=6)
        ax4a.set_xlabel('Time (s)', fontsize=12)
        ax4a.set_ylabel('Flame Position (x)', fontsize=12)
        ax4a.set_title('Flame Front Position vs Time', fontsize=14)
        ax4a.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(times[5:], flame_positions[5:], 1)
        p = np.poly1d(z)
        ax4a.plot(times[5:], p(times[5:]), "r--", alpha=0.8, 
                 label=f'Linear fit: speed ≈ {z[0]:.2f}')
        ax4a.legend()
        
        # Plot flame speed
        ax4b.plot(times, flame_speed, 'r-', linewidth=2, marker='s', markersize=6)
        ax4b.set_xlabel('Time (s)', fontsize=12)
        ax4b.set_ylabel('Flame Speed (dx/dt)', fontsize=12)
        ax4b.set_title('Instantaneous Flame Speed', fontsize=14)
        ax4b.grid(True, alpha=0.3)
        
        # Add average speed
        avg_speed = np.mean(flame_speed[5:-2])
        ax4b.axhline(y=avg_speed, color='g', linestyle='--', 
                    label=f'Average: {avg_speed:.2f}')
        ax4b.legend()
        
        # Add speed statistics
        max_speed = np.max(flame_speed)
        fig4.text(0.5, 0.02, f'Max Speed: {max_speed:.2f} | Avg Speed: {avg_speed:.2f}',
                 ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        fig4.suptitle('Flame Propagation Speed Analysis', fontsize=16, fontweight='bold')
        fig4.tight_layout()
        
        plt.show()
        
        print(f"\n📊 Flame Speed Statistics:")
        print(f"   Average speed: {avg_speed:.3f}")
        print(f"   Maximum speed: {max_speed:.3f}")
        print(f"   Final position: {flame_positions[-1]:.2f}")

def main():
    """Main function"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    print("=" * 50)
    
    # Network architecture
    layers = [3, 64, 64, 64, 64, 5]
    model = CombustionPINN_2D(layers)
    
    # Domain bounds [x_min, x_max, y_min, y_max, t_min, t_max]
    domain_bounds = [0.0, 10.0, -0.5, 0.5, 0.0, 1.0]
    
    # Create trainer
    trainer = Trainer_2D(model, domain_bounds, device)
    
    # Train the model
    trainer.train(epochs=3000, print_freq=100)
    
    # Visualize results
    trainer.plot_results()

if __name__ == "__main__":
    main()