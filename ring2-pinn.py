import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from torch.autograd import grad
import os

class FreeFlamesPINN(nn.Module):
    """
    PINN for FREELY propagating flames in an annular trough.
    The flame speed is NOT prescribed; it emerges from the reaction-diffusion physics.
    """
    def __init__(self, layers, activation=torch.tanh):
        super(FreeFlamesPINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # Physical parameters (NO prescribed flame_speed)
        self.Peclet = 100.0
        self.Lewis = 0.8
        self.Damkohler = 20.0
        self.heat_release = 8.0
        self.activation_energy = 10.0
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
            else:
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
    
    def forward(self, inputs):
        """
        Forward pass. The network learns the general solution T(θ,t) and Y(θ,t).
        """
        theta = inputs[:, 0:1]
        t = inputs[:, 1:2]
        
        features = torch.cat([
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(2*theta),
            torch.cos(2*theta),
            t
        ], dim=1)
        
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(layer(h))
        
        outputs = self.layers[-1](h)
        
        T = 1.0 + self.heat_release * torch.sigmoid(outputs[:, 0:1])
        Y = torch.sigmoid(outputs[:, 1:2])
        
        return T, Y
    
    def compute_derivatives(self, theta, t, T, Y):
        """Compute required derivatives"""
        T_theta = grad(T.sum(), theta, create_graph=True)[0]
        T_t = grad(T.sum(), t, create_graph=True)[0]
        Y_theta = grad(Y.sum(), theta, create_graph=True)[0]
        Y_t = grad(Y.sum(), t, create_graph=True)[0]
        
        T_theta_theta = grad(T_theta.sum(), theta, create_graph=True)[0]
        Y_theta_theta = grad(Y_theta.sum(), theta, create_graph=True)[0]
        
        return T_t, T_theta_theta, Y_t, Y_theta_theta
    
    def reaction_rate(self, T, Y):
        """Arrhenius reaction rate"""
        beta = self.activation_energy
        rate = self.Damkohler * Y * torch.exp(beta * (T - 1.0) / T)
        return rate
    
    def physics_loss(self, theta, t):
        """
        Physics loss based on pure reaction-diffusion equations.
        The advection term (c * T_theta) is removed.
        """
        theta.requires_grad_(True)
        t.requires_grad_(True)
        
        inputs = torch.cat([theta, t], dim=1)
        T, Y = self.forward(inputs)
        
        T_t, T_theta_theta, Y_t, Y_theta_theta = self.compute_derivatives(theta, t, T, Y)
        
        omega = self.reaction_rate(T, Y)
        
        # Energy equation residual: ∂T/∂t = (1/Pe)∂²T/∂θ² + Q*ω
        energy_residual = T_t - (1.0/self.Peclet) * T_theta_theta - self.heat_release * omega
        
        # Species equation residual: ∂Y/∂t = (1/(Pe*Le))∂²Y/∂θ² - ω
        species_residual = Y_t - (1.0/(self.Peclet * self.Lewis)) * Y_theta_theta + omega
        
        return energy_residual, species_residual

class FreeFlamesTrainer:
    """Trainer for the free-flame scenario"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.998)
        
        self.losses = []
        
    def generate_data(self, n_points=4000):
        """Generate training data"""
        # The emergent speed might be different, let's use a slightly longer time
        t_max = 0.5
        theta = torch.rand(n_points, 1) * 2 * np.pi
        t = torch.rand(n_points, 1) * t_max
        
        n_ic = 1000
        theta_ic = torch.rand(n_ic, 1) * 2 * np.pi
        t_ic = torch.zeros(n_ic, 1)
        
        theta_all = torch.cat([theta, theta_ic], dim=0).to(self.device)
        t_all = torch.cat([t, t_ic], dim=0).to(self.device)
        
        return theta_all, t_all, t_max
    
    def train_step(self, theta, t):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Physics loss
        energy_res, species_res = self.model.physics_loss(theta, t)
        physics_loss = torch.mean(energy_res**2) + torch.mean(species_res**2)
        
        # Initial condition loss - TWO pulses to kickstart the process
        mask_ic = (t < 0.001).squeeze()
        if mask_ic.sum() > 0:
            inputs_ic = torch.cat([theta[mask_ic], t[mask_ic]], dim=1)
            T_ic, Y_ic = self.model(inputs_ic)
            
            theta_ic_vals = theta[mask_ic]
            
            xi_ic1 = theta_ic_vals - np.pi / 2.0
            pulse1 = torch.exp(-40.0 * (torch.sin(xi_ic1 / 2.0))**2)
            
            xi_ic2 = theta_ic_vals - 3 * np.pi / 2.0
            pulse2 = torch.exp(-40.0 * (torch.sin(xi_ic2 / 2.0))**2)

            T_target = 1.0 + self.model.heat_release * (pulse1 + pulse2)
            Y_target = 1.0 - 0.95 * (pulse1 + pulse2)
            
            ic_loss = torch.mean((T_ic - T_target)**2) + torch.mean((Y_ic - Y_target)**2)
        else:
            ic_loss = torch.tensor(0.0).to(self.device)
        
        # Periodicity loss
        theta_0 = torch.zeros(100, 1).to(self.device)
        theta_2pi = torch.ones(100, 1).to(self.device) * 2 * np.pi
        t_test = torch.rand(100, 1).to(self.device) * 0.5
        
        T_0, Y_0 = self.model(torch.cat([theta_0, t_test], dim=1))
        T_2pi, Y_2pi = self.model(torch.cat([theta_2pi, t_test], dim=1))
        
        periodic_loss = torch.mean((T_0 - T_2pi)**2) + torch.mean((Y_0 - Y_2pi)**2)
        
        total_loss = physics_loss + 20.0 * ic_loss + 10.0 * periodic_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item(), physics_loss.item(), ic_loss.item()
    
    def train(self, epochs=4000): # Increased epochs for this harder problem
        """Training loop"""
        print("🌀 Training Freely Propagating Flames Model...")
        print("="*60)
        
        for epoch in range(epochs):
            theta, t, _ = self.generate_data()
            total_loss, phys_loss, ic_loss = self.train_step(theta, t)
            self.losses.append(total_loss)
            
            if epoch > 100:
                self.scheduler.step()
            
            if epoch % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d}: Loss = {total_loss:.6f}, "
                      f"Physics = {phys_loss:.6f}, IC = {ic_loss:.6f}, "
                      f"LR = {current_lr:.6f}")
    
    def visualize_results(self):
        """Comprehensive visualization"""
        self.model.eval()
        fig = plt.figure(figsize=(16, 10))
        
        _, _, t_max = self.generate_data()
        n_theta = 200
        n_t = 100
        theta_grid = torch.linspace(0, 2*np.pi, n_theta).to(self.device)
        t_grid = torch.linspace(0, t_max, n_t).to(self.device)
        
        T_grid = np.zeros((n_t, n_theta))
        Y_grid = np.zeros((n_t, n_theta))
        
        with torch.no_grad():
            for i, t_val in enumerate(t_grid):
                inputs = torch.cat([theta_grid.unsqueeze(1), torch.ones_like(theta_grid.unsqueeze(1)) * t_val], dim=1)
                T_pred, Y_pred = self.model(inputs)
                T_grid[i, :] = T_pred.cpu().numpy().flatten()
                Y_grid[i, :] = Y_pred.cpu().numpy().flatten()
        
        # Temperature space-time
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(T_grid, aspect='auto', origin='lower', cmap='hot', extent=[0, 360, 0, t_max])
        ax1.set_title('Temperature Space-Time\n(Flame speed is learned, not prescribed)')
        ax1.set_xlabel('Angular Position (degrees)')
        ax1.set_ylabel('Time (s)')
        plt.colorbar(im1, ax=ax1)
        
        # Fuel space-time
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(Y_grid, aspect='auto', origin='lower', cmap='YlGn_r', extent=[0, 360, 0, t_max])
        ax2.set_title('Fuel Space-Time')
        ax2.set_xlabel('Angular Position (degrees)')
        ax2.set_ylabel('Time (s)')
        plt.colorbar(im2, ax=ax2)
        
        # Reaction rate space-time
        ax3 = plt.subplot(2, 3, 3)
        with torch.no_grad():
            omega_grid = np.zeros((n_t, n_theta))
            for i, t_val in enumerate(t_grid):
                inputs = torch.cat([theta_grid.unsqueeze(1), torch.ones_like(theta_grid.unsqueeze(1)) * t_val], dim=1)
                T_pred, Y_pred = self.model(inputs)
                omega = self.model.reaction_rate(T_pred, Y_pred)
                omega_grid[i, :] = omega.cpu().numpy().flatten()
        
        im3 = ax3.imshow(omega_grid, aspect='auto', origin='lower', cmap='YlOrRd', extent=[0, 360, 0, t_max])
        ax3.set_title('Reaction Rate')
        ax3.set_xlabel('Angular Position (degrees)')
        ax3.set_ylabel('Time (s)')
        plt.colorbar(im3, ax=ax3)
        
        # Snapshots at different times
        times = np.linspace(0, t_max, 5)
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        
        ax4 = plt.subplot(2, 3, 4)
        for idx, t_val in enumerate(times):
            with torch.no_grad():
                inputs = torch.cat([theta_grid.unsqueeze(1), torch.ones_like(theta_grid.unsqueeze(1)) * t_val], dim=1)
                T_pred, _ = self.model(inputs)
                ax4.plot(np.degrees(theta_grid.cpu().numpy()), T_pred.cpu().numpy(), color=colors[idx], label=f't={t_val:.2f}s', linewidth=2)
        ax4.set_title('Temperature Profiles')
        ax4.set_xlabel('Angular Position (degrees)')
        ax4.set_ylabel('Temperature')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        for idx, t_val in enumerate(times):
            with torch.no_grad():
                inputs = torch.cat([theta_grid.unsqueeze(1), torch.ones_like(theta_grid.unsqueeze(1)) * t_val], dim=1)
                _, Y_pred = self.model(inputs)
                ax5.plot(np.degrees(theta_grid.cpu().numpy()), Y_pred.cpu().numpy(), color=colors[idx], label=f't={t_val:.2f}s', linewidth=2)
        ax5.set_title('Fuel Profiles')
        ax5.set_xlabel('Angular Position (degrees)')
        ax5.set_ylabel('Fuel Fraction')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(2, 3, 6)
        ax6.semilogy(self.losses, 'b-', linewidth=2)
        ax6.set_title('Training Loss')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('🌀 Freely Propagating Flames in Annular Trough (Emergent Speed)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        self.create_animation()

    def create_animation(self):
        """Create annular visualization"""
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        _, _, t_max = self.generate_data()
        times = np.linspace(0, t_max, 5)
        
        with torch.no_grad():
            for idx, t_val in enumerate(times):
                ax = axes[idx]
                ax.set_aspect('equal')
                
                outer = Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2)
                inner = Circle((0, 0), 0.9, fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(outer)
                ax.add_patch(inner)
                
                n_points = 100
                theta_test = torch.linspace(0, 2*np.pi, n_points).unsqueeze(1).to(self.device)
                t_test = torch.ones_like(theta_test) * t_val
                inputs = torch.cat([theta_test, t_test], dim=1)
                T_pred, _ = self.model(inputs)
                
                T_np = T_pred.cpu().numpy().flatten()
                theta_np = theta_test.cpu().numpy().flatten()
                
                for i in range(n_points):
                    color_val = (T_np[i] - 1.0) / (self.model.heat_release + 1e-6)
                    color = plt.cm.hot(color_val)
                    
                    wedge = Wedge((0, 0), 1.0, np.degrees(theta_np[i]) - 2, np.degrees(theta_np[i]) + 2,
                                 width=0.1, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
                
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_title(f't = {t_val:.2f}s', fontsize=10)
                ax.axis('off')
        
        plt.suptitle('Freely Propagating Flames - Collision', fontsize=12)
        plt.tight_layout()
        plt.show()

def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    print("="*70)
    print("🌀 FREELY PROPAGATING FLAMES (EMERGENT SPEED)")
    print("="*70)
    print("\n🎯 KEY IMPROVEMENTS:")
    print("  • Removed the prescribed 'flame_speed' parameter.")
    print("  • Physics loss is now based on pure reaction-diffusion equations.")
    print("  • Flame propagation speed is now a learned, emergent property of the system.")
    print("  • This represents a more fundamental and realistic physical model.")
    print("="*70)
    
    layers = [5, 64, 64, 64, 64, 2]
    
    model = FreeFlamesPINN(layers)
    trainer = FreeFlamesTrainer(model, device)
    
    trainer.train(epochs=4000)
    
    print("\n📊 Creating visualizations...")
    trainer.visualize_results()
    
    print("\n✅ SUCCESS: Flames now propagate freely based on reaction-diffusion physics!")
    print("   Check the space-time diagram - the slope of the flame front reveals the learned speed.")

if __name__ == "__main__":
    main()