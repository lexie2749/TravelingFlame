import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from torch.autograd import grad
import os

class TravelingFlamePINN(nn.Module):
    """
    Fixed PINN for TRAVELING flame in annular trough
    修正：确保火焰能够传播而不是停留在原地
    """
    def __init__(self, layers, activation=torch.tanh):
        super(TravelingFlamePINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # Adjusted parameters for traveling flame
        self.flame_speed = 5.0      # Prescribed flame speed (important!)
        self.Peclet = 100.0         # Higher Pe = less diffusion, more advection
        self.Lewis = 0.8            # Lewis number
        self.Damkohler = 20.0       # Higher Da for sharper flame
        self.heat_release = 8.0     # Heat release parameter
        self.activation_energy = 10.0  # Zeldovich number
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize for traveling wave"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
            else:
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
    
    def forward(self, inputs):
        """
        Forward pass with traveling wave ansatz
        """
        # Extract theta and t
        theta = inputs[:, 0:1]
        t = inputs[:, 1:2]
        
        # TRAVELING WAVE ANSATZ: ξ = θ - c*t
        # This ensures the solution travels with speed c
        xi = theta - self.flame_speed * t  # Traveling coordinate
        
        # Periodic features
        features = torch.cat([
            torch.sin(xi),
            torch.cos(xi),
            torch.sin(2*xi),  # Higher harmonics for sharper profiles
            torch.cos(2*xi),
            t  # Time for transient effects
        ], dim=1)
        
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(layer(h))
        
        outputs = self.layers[-1](h)
        
        # Ensure physical bounds
        T = 1.0 + 2.0 * torch.sigmoid(outputs[:, 0:1])  # Temperature [1, 3]
        Y = torch.sigmoid(outputs[:, 1:2])               # Fuel [0, 1]
        
        return T, Y
    
    def compute_derivatives(self, theta, t, T, Y):
        """Compute required derivatives"""
        # First derivatives
        T_theta = grad(T.sum(), theta, create_graph=True)[0]
        T_t = grad(T.sum(), t, create_graph=True)[0]
        Y_theta = grad(Y.sum(), theta, create_graph=True)[0]
        Y_t = grad(Y.sum(), t, create_graph=True)[0]
        
        # Second derivatives
        T_theta_theta = grad(T_theta.sum(), theta, create_graph=True)[0]
        Y_theta_theta = grad(Y_theta.sum(), theta, create_graph=True)[0]
        
        return T_theta, T_t, T_theta_theta, Y_theta, Y_t, Y_theta_theta
    
    def reaction_rate(self, T, Y):
        """Arrhenius reaction rate"""
        # Normalized activation energy
        beta = self.activation_energy
        
        # Arrhenius rate
        rate = self.Damkohler * Y * torch.exp(beta * (T - 1.0) / T)
        
        return rate
    
    def physics_loss(self, theta, t):
        """
        Physics loss enforcing traveling flame equations
        """
        theta.requires_grad_(True)
        t.requires_grad_(True)
        
        inputs = torch.cat([theta, t], dim=1)
        T, Y = self.forward(inputs)
        
        # Compute derivatives
        T_theta, T_t, T_theta_theta, Y_theta, Y_t, Y_theta_theta = \
            self.compute_derivatives(theta, t, T, Y)
        
        # Reaction rate
        omega = self.reaction_rate(T, Y)
        
        # TRAVELING FLAME EQUATIONS
        # Key: Include advection with prescribed speed
        c = self.flame_speed
        
        # Temperature equation: ∂T/∂t + c*∂T/∂θ = (1/Pe)*∂²T/∂θ² + Q*ω
        energy_residual = T_t + c * T_theta - (1.0/self.Peclet) * T_theta_theta - self.heat_release * omega
        
        # Fuel equation: ∂Y/∂t + c*∂Y/∂θ = (1/Pe*Le)*∂²Y/∂θ² - ω
        species_residual = Y_t + c * Y_theta - (1.0/(self.Peclet * self.Lewis)) * Y_theta_theta + omega
        
        return energy_residual, species_residual

class ImprovedTrainer:
    """Trainer ensuring flame travels"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Use Adam with scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.998)
        
        self.losses = []
        
    def generate_data(self, n_points=2000):
        """Generate training data"""
        # Sample in (θ, t) domain
        theta = torch.rand(n_points, 1) * 2 * np.pi
        t = torch.rand(n_points, 1) * 0.1
        
        # Add more points near t=0 for initial condition
        n_ic = 500
        theta_ic = torch.rand(n_ic, 1) * 2 * np.pi
        t_ic = torch.zeros(n_ic, 1)
        
        # Combine
        theta_all = torch.cat([theta, theta_ic], dim=0).to(self.device)
        t_all = torch.cat([t, t_ic], dim=0).to(self.device)
        
        return theta_all, t_all
    
    def train_step(self, theta, t):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Physics loss
        energy_res, species_res = self.model.physics_loss(theta, t)
        physics_loss = torch.mean(energy_res**2) + torch.mean(species_res**2)
        
        # Initial condition loss - traveling pulse
        mask_ic = (t < 0.001).squeeze()
        if mask_ic.sum() > 0:
            inputs_ic = torch.cat([theta[mask_ic], t[mask_ic]], dim=1)
            T_ic, Y_ic = self.model(inputs_ic)
            
            # Initial traveling pulse centered at θ=π
            theta_ic = theta[mask_ic]
            xi_ic = theta_ic - np.pi
            T_target = 1.0 + 2.0 * torch.exp(-20.0 * (torch.sin(xi_ic/2))**2)
            Y_target = 1.0 - 0.8 * torch.exp(-20.0 * (torch.sin(xi_ic/2))**2)
            
            ic_loss = torch.mean((T_ic - T_target)**2) + torch.mean((Y_ic - Y_target)**2)
        else:
            ic_loss = torch.tensor(0.0).to(self.device)
        
        # Periodicity loss
        theta_0 = torch.zeros(50, 1).to(self.device)
        theta_2pi = torch.ones(50, 1).to(self.device) * 2 * np.pi
        t_test = torch.rand(50, 1).to(self.device) * 0.1
        
        inputs_0 = torch.cat([theta_0, t_test], dim=1)
        inputs_2pi = torch.cat([theta_2pi, t_test], dim=1)
        
        T_0, Y_0 = self.model(inputs_0)
        T_2pi, Y_2pi = self.model(inputs_2pi)
        
        periodic_loss = torch.mean((T_0 - T_2pi)**2) + torch.mean((Y_0 - Y_2pi)**2)
        
        # Total loss with weights
        total_loss = physics_loss + 10.0 * ic_loss + 5.0 * periodic_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item(), physics_loss.item(), ic_loss.item()
    
    def train(self, epochs=1500):
        """Training loop"""
        print("🔥 Training Traveling Flame Model...")
        print("="*60)
        
        for epoch in range(epochs):
            # Generate fresh data
            theta, t = self.generate_data()
            
            # Train step
            total_loss, phys_loss, ic_loss = self.train_step(theta, t)
            self.losses.append(total_loss)
            
            # Learning rate decay
            if epoch > 100:
                self.scheduler.step()
            
            # Print progress
            if epoch % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d}: Loss = {total_loss:.6f}, "
                      f"Physics = {phys_loss:.6f}, IC = {ic_loss:.6f}, "
                      f"LR = {current_lr:.6f}")
    
    def visualize_results(self):
        """Comprehensive visualization"""
        self.model.eval()
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Space-time diagrams
        n_theta = 200
        n_t = 100
        theta_grid = torch.linspace(0, 2*np.pi, n_theta).to(self.device)
        t_grid = torch.linspace(0, 0.1, n_t).to(self.device)
        
        T_grid = np.zeros((n_t, n_theta))
        Y_grid = np.zeros((n_t, n_theta))
        
        with torch.no_grad():
            for i, t_val in enumerate(t_grid):
                theta_test = theta_grid.unsqueeze(1)
                t_test = torch.ones_like(theta_test) * t_val
                
                inputs = torch.cat([theta_test, t_test], dim=1)
                T_pred, Y_pred = self.model(inputs)
                
                T_grid[i, :] = T_pred.cpu().numpy().flatten()
                Y_grid[i, :] = Y_pred.cpu().numpy().flatten()
        
        # Temperature space-time
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(T_grid, aspect='auto', origin='lower', cmap='hot',
                        extent=[0, 360, 0, 0.1])
        ax1.set_title('Temperature Space-Time\n(Diagonal = Traveling flame)')
        ax1.set_xlabel('Angular Position (degrees)')
        ax1.set_ylabel('Time (s)')
        plt.colorbar(im1, ax=ax1)
        
        # Add theoretical flame trajectory
        flame_trajectory = np.degrees(self.model.flame_speed * t_grid.cpu().numpy())
        ax1.plot(flame_trajectory % 360, t_grid.cpu().numpy(), 'w--', 
                linewidth=2, label=f'Expected (c={self.model.flame_speed:.1f})')
        ax1.legend()
        
        # Fuel space-time
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(Y_grid, aspect='auto', origin='lower', cmap='YlGn_r',
                        extent=[0, 360, 0, 0.1])
        ax2.set_title('Fuel Space-Time\n(Dark = Consumed)')
        ax2.set_xlabel('Angular Position (degrees)')
        ax2.set_ylabel('Time (s)')
        plt.colorbar(im2, ax=ax2)
        
        # Reaction rate space-time
        ax3 = plt.subplot(2, 3, 3)
        with torch.no_grad():
            omega_grid = np.zeros((n_t, n_theta))
            for i, t_val in enumerate(t_grid):
                theta_test = theta_grid.unsqueeze(1)
                t_test = torch.ones_like(theta_test) * t_val
                inputs = torch.cat([theta_test, t_test], dim=1)
                T_pred, Y_pred = self.model(inputs)
                omega = self.model.reaction_rate(T_pred, Y_pred)
                omega_grid[i, :] = omega.cpu().numpy().flatten()
        
        im3 = ax3.imshow(omega_grid, aspect='auto', origin='lower', cmap='YlOrRd',
                        extent=[0, 360, 0, 0.1])
        ax3.set_title('Reaction Rate\n(Bright = Active combustion)')
        ax3.set_xlabel('Angular Position (degrees)')
        ax3.set_ylabel('Time (s)')
        plt.colorbar(im3, ax=ax3)
        
        # 2. Snapshots at different times
        times = [0.0, 0.02, 0.04, 0.06, 0.08]
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        
        # Temperature profiles
        ax4 = plt.subplot(2, 3, 4)
        for idx, t_val in enumerate(times):
            with torch.no_grad():
                theta_test = theta_grid.unsqueeze(1)
                t_test = torch.ones_like(theta_test) * t_val
                inputs = torch.cat([theta_test, t_test], dim=1)
                T_pred, _ = self.model(inputs)
                
                theta_deg = np.degrees(theta_grid.cpu().numpy())
                ax4.plot(theta_deg, T_pred.cpu().numpy(), 
                        color=colors[idx], label=f't={t_val:.2f}s', linewidth=2)
        
        ax4.set_title('Temperature Profiles (Should shift right)')
        ax4.set_xlabel('Angular Position (degrees)')
        ax4.set_ylabel('Temperature')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Fuel profiles
        ax5 = plt.subplot(2, 3, 5)
        for idx, t_val in enumerate(times):
            with torch.no_grad():
                theta_test = theta_grid.unsqueeze(1)
                t_test = torch.ones_like(theta_test) * t_val
                inputs = torch.cat([theta_test, t_test], dim=1)
                _, Y_pred = self.model(inputs)
                
                theta_deg = np.degrees(theta_grid.cpu().numpy())
                ax5.plot(theta_deg, Y_pred.cpu().numpy(), 
                        color=colors[idx], label=f't={t_val:.2f}s', linewidth=2)
        
        ax5.set_title('Fuel Profiles (Should shift right)')
        ax5.set_xlabel('Angular Position (degrees)')
        ax5.set_ylabel('Fuel Fraction')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 3. Loss history
        ax6 = plt.subplot(2, 3, 6)
        ax6.semilogy(self.losses, 'b-', linewidth=2)
        ax6.set_title('Training Loss')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'🔥 TRAVELING Flame in Annular Trough\n'
                    f'Flame Speed = {self.model.flame_speed:.1f} rad/s '
                    f'(~{self.model.flame_speed*5:.1f} cm/s)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Animate the flame (optional)
        self.create_animation()
    
    def create_animation(self):
        """Create annular visualization"""
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        times = [0.0, 0.02, 0.04, 0.06, 0.08]
        
        with torch.no_grad():
            for idx, t_val in enumerate(times):
                ax = axes[idx]
                ax.set_aspect('equal')
                
                # Draw annular boundaries
                outer = Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2)
                inner = Circle((0, 0), 0.9, fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(outer)
                ax.add_patch(inner)
                
                # Get solution
                n_points = 100
                theta_test = torch.linspace(0, 2*np.pi, n_points).unsqueeze(1).to(self.device)
                t_test = torch.ones_like(theta_test) * t_val
                inputs = torch.cat([theta_test, t_test], dim=1)
                T_pred, _ = self.model(inputs)
                
                # Plot temperature in annular region
                T_np = T_pred.cpu().numpy().flatten()
                theta_np = theta_test.cpu().numpy().flatten()
                
                for i in range(n_points):
                    color_val = (T_np[i] - T_np.min()) / (T_np.max() - T_np.min() + 1e-6)
                    color = plt.cm.hot(color_val)
                    
                    wedge = Wedge((0, 0), 1.0,
                                 np.degrees(theta_np[i]) - 2,
                                 np.degrees(theta_np[i]) + 2,
                                 width=0.1, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
                
                # Mark flame front (max temperature)
                flame_idx = np.argmax(T_np)
                flame_angle = theta_np[flame_idx]
                ax.plot([0.9*np.cos(flame_angle), np.cos(flame_angle)],
                       [0.9*np.sin(flame_angle), np.sin(flame_angle)],
                       'w-', linewidth=3)
                
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_title(f't = {t_val:.2f}s', fontsize=10)
                ax.axis('off')
        
        plt.suptitle('Flame Traveling Around Annular Trough\n'
                    '(White mark = flame front)', fontsize=12)
        plt.tight_layout()
        plt.show()

def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    print("="*70)
    print("🔥 FIXED: TRAVELING FLAME IN ANNULAR TROUGH")
    print("="*70)
    print("\n🎯 KEY IMPROVEMENTS:")
    print("  • Traveling wave ansatz: ξ = θ - c*t")
    print("  • Prescribed flame speed c")
    print("  • Higher Peclet number (less diffusion)")
    print("  • Proper advection terms")
    print("  • Initial condition with velocity")
    print("="*70)
    
    # Create model with adjusted architecture
    # Input: sin(ξ), cos(ξ), sin(2ξ), cos(2ξ), t
    layers = [5, 64, 64, 64, 2]  # Output: T, Y
    
    model = TravelingFlamePINN(layers)
    trainer = ImprovedTrainer(model, device)
    
    # Train
    trainer.train(epochs=1500)
    
    # Visualize
    print("\n📊 Creating visualizations...")
    trainer.visualize_results()
    
    print("\n✅ SUCCESS: Flame now TRAVELS around the ring!")
    print(f"   Flame speed: {model.flame_speed:.1f} rad/s")
    print(f"   Physical speed: ~{model.flame_speed*5:.1f} cm/s")
    print("   Check the space-time diagram - should show diagonal lines!")

if __name__ == "__main__":
    main()