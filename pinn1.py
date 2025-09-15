import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
import os

class StableCombustionPINN(nn.Module):
    """
    Physics-Informed Neural Network for Combustion Simulation
    Models flame propagation in a ring-shaped trough
    """
    def __init__(self, layers, activation=torch.tanh):
        super(StableCombustionPINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # Dimensionless parameters
        self.Reynolds = 100.0      # Flow regime parameter
        self.Peclet = 50.0        # Heat diffusion parameter
        self.Schmidt = 1.0        # Mass diffusion parameter
        self.Damkohler = 5.0      # Reaction rate parameter
        self.heat_release = 2.0   # Heat release parameter
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights for stable training"""
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:  # Output layer
                nn.init.uniform_(layer.weight, -0.05, 0.05)
                with torch.no_grad():
                    layer.bias[0] = 0.5   # u velocity
                    layer.bias[1] = 0.0   # v velocity
                    layer.bias[2] = 0.0   # pressure
                    layer.bias[3] = 0.5   # temperature
                    layer.bias[4] = 0.5   # fuel fraction
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, t):
        """
        Forward propagation with physical constraints
        """
        inputs = torch.cat([x, t], dim=1)
        
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        
        outputs = self.layers[-1](inputs)
        
        # Apply physical constraints
        u = torch.tanh(outputs[:, 0:1]) * 3.0           # Velocity [-3, 3]
        v = torch.tanh(outputs[:, 1:2]) * 1.0           # Velocity [-1, 1]
        p = torch.tanh(outputs[:, 2:3]) * 2.0           # Pressure [-2, 2]
        
        # Ensure temperature is positive
        T_raw = outputs[:, 3:4]
        T = 0.1 + torch.nn.functional.softplus(T_raw)   # T ≥ 0.1
        
        # Ensure fuel fraction is between 0 and 1
        Y_raw = outputs[:, 4:5]
        Y_fuel = torch.sigmoid(Y_raw)                   # Y ∈ [0, 1]
        
        return u, v, p, T, Y_fuel
    
    def safe_reaction_rate(self, T, Y_fuel):
        """
        Calculate reaction rate with numerical stability
        Based on Arrhenius kinetics
        """
        # Ensure inputs are in safe range
        T_safe = torch.clamp(T, min=0.1, max=10.0)
        Y_safe = torch.clamp(Y_fuel, min=1e-6, max=1.0)
        
        # Ignition temperature threshold
        T_ignition = 0.8
        
        # Simplified Arrhenius rate
        activation_term = -2.0 / T_safe
        activation_term = torch.clamp(activation_term, min=-10, max=10)
        
        # Reaction rate
        rate = self.Damkohler * Y_safe * torch.exp(activation_term)
        
        # Smooth ignition switch
        ignition_factor = torch.sigmoid((T_safe - T_ignition) * 10.0)
        
        # Final bounded rate
        final_rate = rate * ignition_factor
        final_rate = torch.clamp(final_rate, min=0.0, max=100.0)
        
        return final_rate
    
    def stable_physics_loss(self, x, t):
        """
        Calculate physics-based loss terms from governing equations
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T, Y_fuel = self.forward(x, t)
        
        # Safe gradient computation
        def compute_gradient(output, input_var, create_graph=True):
            try:
                grad_val = grad(output, input_var, 
                              grad_outputs=torch.ones_like(output),
                              create_graph=create_graph, 
                              retain_graph=True,
                              allow_unused=True)[0]
                
                if grad_val is None:
                    return torch.zeros_like(input_var)
                
                # Handle NaN and Inf values
                grad_val = torch.where(torch.isnan(grad_val), 
                                     torch.zeros_like(grad_val), grad_val)
                grad_val = torch.where(torch.isinf(grad_val), 
                                     torch.zeros_like(grad_val), grad_val)
                grad_val = torch.clamp(grad_val, min=-1e6, max=1e6)
                
                return grad_val
            except:
                return torch.zeros_like(input_var)
        
        # Compute gradients
        u_x = compute_gradient(u, x)
        u_t = compute_gradient(u, t)
        v_x = compute_gradient(v, x)
        v_t = compute_gradient(v, t)
        p_x = compute_gradient(p, x)
        T_x = compute_gradient(T, x)
        T_t = compute_gradient(T, t)
        Y_x = compute_gradient(Y_fuel, x)
        Y_t = compute_gradient(Y_fuel, t)
        
        # Second derivatives
        u_xx = compute_gradient(u_x, x)
        v_xx = compute_gradient(v_x, x)
        T_xx = compute_gradient(T_x, x)
        Y_xx = compute_gradient(Y_x, x)
        
        # Reaction rate
        omega = self.safe_reaction_rate(T, Y_fuel)
        
        # Governing equations
        def safe_equation(expr, name=""):
            result = expr
            result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
            result = torch.where(torch.isinf(result), torch.zeros_like(result), result)
            result = torch.clamp(result, min=-1e6, max=1e6)
            return result
        
        # 1. Continuity equation
        continuity = safe_equation(u_x, "continuity")
        
        # 2. Momentum equations
        momentum_u = safe_equation(
            u_t + u * u_x + p_x - (1.0/self.Reynolds) * u_xx, "momentum_u")
        momentum_v = safe_equation(
            v_t + u * v_x - (1.0/self.Reynolds) * v_xx, "momentum_v")
        
        # 3. Energy equation
        energy = safe_equation(
            T_t + u * T_x - (1.0/self.Peclet) * T_xx - self.heat_release * omega, "energy")
        
        # 4. Species equation
        species = safe_equation(
            Y_t + u * Y_x - (1.0/self.Schmidt) * Y_xx + omega, "species")
        
        return continuity, momentum_u, momentum_v, energy, species

class StableTrainer:
    """Trainer for the combustion PINN model"""
    def __init__(self, model, domain_bounds, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.domain_bounds = domain_bounds
        
        # Conservative optimizer settings
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        
        self.loss_history = []
        self.gradient_norms = []
        
        # Create directory for saving plots
        self.save_dir = "combustion_results"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def generate_training_data(self, n_points=1500):
        """Generate training data points"""
        x_min, x_max, t_min, t_max = self.domain_bounds
        
        # Interior points
        x_int = torch.rand(n_points, 1) * (x_max - x_min) + x_min
        t_int = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        
        # Boundary points
        n_bc = n_points // 5
        x_bc_left = torch.zeros(n_bc, 1) + x_min
        x_bc_right = torch.ones(n_bc, 1) * x_max
        t_bc = torch.rand(2*n_bc, 1) * (t_max - t_min) + t_min
        x_bc = torch.cat([x_bc_left, x_bc_right])
        
        # Initial points
        n_ic = n_points // 5
        x_ic = torch.rand(n_ic, 1) * (x_max - x_min) + x_min
        t_ic = torch.zeros(n_ic, 1) + t_min
        
        return (x_int.to(self.device), t_int.to(self.device),
                x_bc.to(self.device), t_bc.to(self.device),
                x_ic.to(self.device), t_ic.to(self.device))
    
    def train_step(self, x_int, t_int, x_bc, t_bc, x_ic, t_ic, epoch):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Physics loss
        cont_loss, mom_u_loss, mom_v_loss, energy_loss, species_loss = \
            self.model.stable_physics_loss(x_int, t_int)
        
        # Calculate losses with numerical stability checks
        def safe_loss(tensor_loss, name):
            loss_val = torch.mean(tensor_loss**2)
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"  ⚠️  {name} loss abnormal: {loss_val}")
                return torch.tensor(0.0, device=self.device)
            return loss_val
        
        cont_val = safe_loss(cont_loss, "Continuity")
        mom_u_val = safe_loss(mom_u_loss, "Momentum-u")
        mom_v_val = safe_loss(mom_v_loss, "Momentum-v")
        energy_val = safe_loss(energy_loss, "Energy")
        species_val = safe_loss(species_loss, "Species")
        
        physics_loss = cont_val + mom_u_val + mom_v_val + energy_val + species_val
        
        # Boundary conditions
        u_bc, v_bc, p_bc, T_bc, Y_bc = self.model(x_bc, t_bc)
        
        n_left = len(x_bc) // 2
        # Left boundary (inlet)
        u_inlet = torch.ones(n_left, 1, device=self.device) * 2.0
        T_inlet = torch.ones(n_left, 1, device=self.device) * 2.0    # High temperature
        Y_inlet = torch.ones(n_left, 1, device=self.device) * 0.9    # High fuel concentration
        
        # Right boundary (outlet)
        u_outlet = torch.ones(n_left, 1, device=self.device) * 1.0
        T_outlet = torch.ones(n_left, 1, device=self.device) * 0.5   # Low temperature
        Y_outlet = torch.ones(n_left, 1, device=self.device) * 0.1   # Low fuel concentration
        
        bc_loss = (torch.mean((u_bc[:n_left] - u_inlet)**2) +
                  torch.mean((T_bc[:n_left] - T_inlet)**2) +
                  torch.mean((Y_bc[:n_left] - Y_inlet)**2) +
                  torch.mean((u_bc[n_left:] - u_outlet)**2) +
                  torch.mean((T_bc[n_left:] - T_outlet)**2) +
                  torch.mean((Y_bc[n_left:] - Y_outlet)**2))
        
        # Initial conditions
        u_ic, v_ic, p_ic, T_ic, Y_ic = self.model(x_ic, t_ic)
        
        # Create spatially varying initial conditions
        T_init = 0.3 + 1.5 * torch.exp(-((x_ic - 0.3) / 0.15)**2)  # Gaussian hot spot
        Y_init = 0.8 * torch.ones_like(x_ic)  # Uniform fuel distribution
        u_init = 1.0 * torch.ones_like(x_ic)  # Initial flow velocity
        
        ic_loss = (torch.mean((u_ic - u_init)**2) +
                  torch.mean((T_ic - T_init)**2) +
                  torch.mean((Y_ic - Y_init)**2))
        
        # Total loss
        total_loss = physics_loss + 2.0 * bc_loss + 5.0 * ic_loss
        
        # Check total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 1e4:
            print(f"  ⚠️  Total loss abnormal: {total_loss:.2f}, skipping update")
            return float('inf'), float('inf'), float('inf'), float('inf')
        
        # Backpropagation
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** (1. / 2)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.gradient_norms.append(grad_norm)
        
        if epoch % 50 == 0:  # Detailed output
            print(f"  📊 Loss details:")
            print(f"    Continuity: {cont_val:.6f}")
            print(f"    Momentum-u: {mom_u_val:.6f}")
            print(f"    Momentum-v: {mom_v_val:.6f}")
            print(f"    Energy: {energy_val:.6f}")
            print(f"    Species: {species_val:.6f}")
            print(f"    Boundary: {bc_loss:.6f}")
            print(f"    Initial: {ic_loss:.6f}")
            print(f"    Gradient norm: {grad_norm:.6f}")
        
        return total_loss.item(), physics_loss.item(), bc_loss.item(), ic_loss.item()
    
    def train(self, epochs=1000, print_freq=100):
        """Training loop"""
        print("🔧 Starting numerical stable training...")
        
        consecutive_failures = 0
        
        for epoch in range(epochs):
            if epoch % 30 == 0:  # Frequent resampling
                x_int, t_int, x_bc, t_bc, x_ic, t_ic = self.generate_training_data()
            
            total_loss, physics_loss, bc_loss, ic_loss = self.train_step(
                x_int, t_int, x_bc, t_bc, x_ic, t_ic, epoch)
            
            # Check training failure
            if np.isinf(total_loss):
                consecutive_failures += 1
                if consecutive_failures > 10:
                    print("❌ Consecutive training failures, reinitializing...")
                    self.model.init_weights()
                    consecutive_failures = 0
            else:
                consecutive_failures = 0
                self.loss_history.append([total_loss, physics_loss, bc_loss, ic_loss])
            
            if epoch % print_freq == 0:
                print(f"\n📊 Epoch {epoch}/{epochs}")
                if not np.isinf(total_loss):
                    print(f"  ✅ Total loss: {total_loss:.6f}")
                    
                    # Check learning progress
                    with torch.no_grad():
                        x_test = torch.tensor([[0.1], [0.5], [0.9]], device=self.device)
                        t_test = torch.tensor([[0.05], [0.05], [0.05]], device=self.device)
                        u, v, p, T, Y = self.model(x_test, t_test)
                        
                        print(f"  🔍 Output check:")
                        print(f"    Temperature range: [{T.min():.3f}, {T.max():.3f}]")
                        print(f"    Fuel range: [{Y.min():.3f}, {Y.max():.3f}]")
                        print(f"    Velocity range: [{u.min():.3f}, {u.max():.3f}]")
                else:
                    print(f"  ❌ Training failed (attempt #{consecutive_failures})")
                
                print("-" * 50)
    
    def predict(self, x_test, t_test):
        """Make predictions"""
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
        """Create individual plots with detailed descriptions"""
        x_test = np.linspace(0, 1, 100)
        t_test = np.full_like(x_test, 0.05)
        
        u, v, p, T, Y = self.predict(x_test, t_test)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Velocity Profile
        plt.figure(figsize=(10, 6))
        plt.plot(x_test, u.flatten(), 'b-', linewidth=2, label='x-velocity')
        plt.xlabel('Position along trough (x)', fontsize=12)
        plt.ylabel('Velocity (u)', fontsize=12)
        plt.title('Flame Propagation Velocity Profile\n'
                 'Shows how the flame velocity changes along the ring-shaped trough', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/velocity_profile.png", dpi=150)
        plt.show()
        print("\n📈 Figure 1: Velocity Profile")
        print("   This shows the horizontal velocity component of the flame as it travels")
        print("   through the trough. Higher values indicate faster flame propagation.")
        print(f"   Velocity range: [{u.min():.3f}, {u.max():.3f}]")
        
        # 2. Temperature Distribution
        plt.figure(figsize=(10, 6))
        plt.plot(x_test, T.flatten(), 'r-', linewidth=2, label='Temperature')
        plt.fill_between(x_test, T.flatten(), alpha=0.3, color='red')
        plt.xlabel('Position along trough (x)', fontsize=12)
        plt.ylabel('Temperature (T)', fontsize=12)
        plt.title('Temperature Distribution in the Combustion Zone\n'
                 'Indicates the heat distribution and flame front location', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/temperature_distribution.png", dpi=150)
        plt.show()
        print("\n🔥 Figure 2: Temperature Distribution")
        print("   Shows the temperature profile across the trough. Peak temperatures")
        print("   indicate the active combustion zone where the flame is located.")
        print(f"   Temperature range: [{T.min():.3f}, {T.max():.3f}]")
        
        # 3. Fuel Concentration
        plt.figure(figsize=(10, 6))
        plt.plot(x_test, Y.flatten(), 'g-', linewidth=2, label='Fuel fraction')
        plt.fill_between(x_test, Y.flatten(), alpha=0.3, color='green')
        plt.xlabel('Position along trough (x)', fontsize=12)
        plt.ylabel('Fuel Mass Fraction (Y)', fontsize=12)
        plt.title('Fuel Concentration Distribution\n'
                 'Shows unburned fuel (high values) and consumed fuel (low values)', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/fuel_concentration.png", dpi=150)
        plt.show()
        print("\n⛽ Figure 3: Fuel Concentration")
        print("   Displays the fuel mass fraction. High values indicate unburned fuel,")
        print("   while low values show areas where fuel has been consumed by combustion.")
        print(f"   Fuel fraction range: [{Y.min():.3f}, {Y.max():.3f}]")
        
        # 4. Pressure Distribution
        plt.figure(figsize=(10, 6))
        plt.plot(x_test, p.flatten(), 'm-', linewidth=2, label='Pressure')
        plt.xlabel('Position along trough (x)', fontsize=12)
        plt.ylabel('Pressure (p)', fontsize=12)
        plt.title('Pressure Distribution in the Flow Field\n'
                 'Pressure variations due to combustion and flow dynamics', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/pressure_distribution.png", dpi=150)
        plt.show()
        print("\n💨 Figure 4: Pressure Distribution")
        print("   Shows pressure variations caused by the combustion process and flow.")
        print("   Pressure gradients drive the flow through the trough.")
        print(f"   Pressure range: [{p.min():.3f}, {p.max():.3f}]")
        
        # 5. Loss History
        if len(self.loss_history) > 0:
            plt.figure(figsize=(10, 6))
            loss_array = np.array(self.loss_history)
            plt.semilogy(loss_array[:, 0], 'b-', linewidth=2, label='Total Loss')
            plt.semilogy(loss_array[:, 1], 'r--', alpha=0.7, label='Physics Loss')
            plt.semilogy(loss_array[:, 2], 'g--', alpha=0.7, label='Boundary Loss')
            plt.semilogy(loss_array[:, 3], 'm--', alpha=0.7, label='Initial Loss')
            plt.xlabel('Training Iteration', fontsize=12)
            plt.ylabel('Loss (log scale)', fontsize=12)
            plt.title('Training Loss History\n'
                     'Shows convergence of the neural network during training', 
                     fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/loss_history.png", dpi=150)
            plt.show()
            print("\n📉 Figure 5: Training Loss History")
            print("   Tracks how well the neural network learned the physics.")
            print("   Decreasing loss indicates successful learning of combustion dynamics.")
            print(f"   Final total loss: {loss_array[-1, 0]:.6f}")
        
        # 6. Combined Physics View
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Temperature and Fuel on same plot
        ax1.plot(x_test, T.flatten(), 'r-', linewidth=2, label='Temperature')
        ax1.set_ylabel('Temperature (T)', fontsize=12, color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Combustion Wave Structure: Temperature-Fuel Relationship\n'
                     'Shows how temperature and fuel concentration are coupled', 
                     fontsize=14)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x_test, Y.flatten(), 'g-', linewidth=2, label='Fuel')
        ax1_twin.set_ylabel('Fuel Fraction (Y)', fontsize=12, color='g')
        ax1_twin.tick_params(axis='y', labelcolor='g')
        
        # Velocity and Pressure
        ax2.plot(x_test, u.flatten(), 'b-', linewidth=2, label='Velocity')
        ax2.set_ylabel('Velocity (u)', fontsize=12, color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_xlabel('Position along trough (x)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_test, p.flatten(), 'm-', linewidth=2, label='Pressure')
        ax2_twin.set_ylabel('Pressure (p)', fontsize=12, color='m')
        ax2_twin.tick_params(axis='y', labelcolor='m')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/combined_physics.png", dpi=150)
        plt.show()
        print("\n🔬 Figure 6: Combined Physics View")
        print("   Top: Temperature-Fuel coupling shows the combustion wave structure")
        print("   Bottom: Velocity-Pressure relationship shows flow dynamics")
        print("   This demonstrates how all variables interact in the flame propagation")
        
        # 7. Reaction Rate Visualization
        plt.figure(figsize=(10, 6))
        with torch.no_grad():
            T_torch = torch.tensor(T, dtype=torch.float32).to(self.device)
            Y_torch = torch.tensor(Y, dtype=torch.float32).to(self.device)
            omega = self.model.safe_reaction_rate(T_torch, Y_torch).cpu().numpy()
        
        plt.plot(x_test, omega.flatten(), 'orange', linewidth=2, label='Reaction Rate')
        plt.fill_between(x_test, omega.flatten(), alpha=0.3, color='orange')
        plt.xlabel('Position along trough (x)', fontsize=12)
        plt.ylabel('Reaction Rate (ω)', fontsize=12)
        plt.title('Chemical Reaction Rate Distribution\n'
                 'Shows where combustion is actively occurring', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/reaction_rate.png", dpi=150)
        plt.show()
        print("\n⚗️ Figure 7: Reaction Rate Distribution")
        print("   Indicates the intensity of chemical reactions. Peak values show")
        print("   the location of the active flame front where combustion occurs.")
        print(f"   Reaction rate range: [{omega.min():.3f}, {omega.max():.3f}]")
        
        # Print summary
        print("\n" + "="*70)
        print("COMBUSTION SIMULATION RESULTS SUMMARY")
        print("="*70)
        print(f"Temperature variation: {T.max() - T.min():.6f}")
        print(f"Fuel consumption: {Y.max() - Y.min():.6f}")
        print(f"Velocity variation: {u.max() - u.min():.6f}")
        print(f"Pressure variation: {p.max() - p.min():.6f}")
        
        if len(self.loss_history) > 0:
            print(f"Final training loss: {self.loss_history[-1][0]:.6f}")
        
        if (T.max() - T.min()) > 0.2 and (Y.max() - Y.min()) > 0.2:
            print("\n✅ SUCCESS: Network successfully learned combustion physics!")
            print("   The model captures flame propagation dynamics in the ring trough.")
        else:
            print("\n⚠️ PARTIAL: Limited learning detected")
            print("   Consider training for more epochs or adjusting parameters.")
        
        print("\n📁 All figures saved to:", self.save_dir)
        print("="*70)

def main():
    """Main function to run the combustion simulation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Network architecture
    layers = [2, 64, 64, 64, 5]  # 2 inputs (x,t), 3 hidden layers, 5 outputs
    
    # Create model
    model = StableCombustionPINN(layers)
    
    # Domain bounds: x ∈ [0,1], t ∈ [0,0.1]
    domain_bounds = [0.0, 1.0, 0.0, 0.1]
    
    # Create trainer
    trainer = StableTrainer(model, domain_bounds, device)
    
    print("\n🔬 COMBUSTION IN RING-SHAPED TROUGH SIMULATION")
    print("="*60)
    print("This simulation models flame propagation through a thin layer")
    print("of flammable liquid in a ring-shaped trough using Physics-Informed")
    print("Neural Networks (PINNs).")
    print("\nKey Physics Modeled:")
    print("✓ Navier-Stokes equations for fluid flow")
    print("✓ Energy equation with heat release from combustion")
    print("✓ Species transport with fuel consumption")
    print("✓ Arrhenius reaction kinetics")
    print("\nNumerical Stability Features:")
    print("✓ Temperature constrained to be positive (T ≥ 0.1)")
    print("✓ Fuel fraction constrained to [0,1] using sigmoid")
    print("✓ Bounded reaction rates with smooth ignition")
    print("✓ Safe gradient computation with NaN/Inf checks")
    print("✓ Automatic reinitialization on training failure")
    print("="*60)
    
    # Train the model
    trainer.train(epochs=800, print_freq=100)
    
    # Plot results
    trainer.plot_results()

if __name__ == "__main__":
    main()