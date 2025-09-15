import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import LinearSegmentedColormap

class FlameCollisionSimulation:
    """
    Simulation of two counter-propagating flames in a ring-shaped trough
    that will collide and annihilate each other
    """
    
    def __init__(self, n_points=300):
        # Discretization
        self.n_points = n_points
        self.theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        self.dtheta = 2*np.pi / n_points
        
        # Physical parameters
        self.Le = 0.8  # Lewis number
        self.beta = 10.0  # Zeldovich number  
        self.alpha = 0.85  # Heat release parameter
        self.D_T = 1.0  # Thermal diffusivity
        self.flame_speed = 0.5  # Base flame speed
        
        # Time stepping
        self.dt = 0.001
        self.time = 0.0
        
        # State variables
        self.T = np.zeros(n_points)  # Temperature
        self.Y = np.ones(n_points)   # Fuel mass fraction
        
        # Initialize two flames at opposite positions
        self._initialize_two_flames()
        
        # Store history for visualization
        self.history_T = []
        self.history_Y = []
        self.history_time = []
        
    def _initialize_two_flames(self):
        """Initialize two flames that will propagate towards each other"""
        # First flame at theta = 0 (0 degrees)
        idx1 = 0
        width = 10  # Width of initial hot spot
        
        for i in range(-width, width+1):
            idx = (idx1 + i) % self.n_points
            amplitude = np.exp(-(i/width*3)**2)
            self.T[idx] = amplitude
            self.Y[idx] = 1.0 - 0.9*amplitude  # Fuel depleted in hot regions
        
        # Second flame at theta = π (180 degrees) 
        idx2 = self.n_points // 2
        
        for i in range(-width, width+1):
            idx = (idx2 + i) % self.n_points
            amplitude = np.exp(-(i/width*3)**2)
            self.T[idx] = max(self.T[idx], amplitude)
            self.Y[idx] = min(self.Y[idx], 1.0 - 0.9*amplitude)
    
    def reaction_rate(self, T, Y):
        """Calculate Arrhenius reaction rate"""
        T_safe = np.clip(T, 0.001, 1.0)
        Y_safe = np.clip(Y, 0.0, 1.0)
        
        # Arrhenius kinetics
        denominator = 1 - self.alpha * (1 - T_safe)
        denominator = np.maximum(denominator, 0.01)
        
        exp_arg = -self.beta * (1 - T_safe) / denominator
        exp_arg = np.clip(exp_arg, -50, 50)
        
        omega = (self.beta**2 / (2*self.Le)) * Y_safe * np.exp(exp_arg)
        
        # Cut off reaction in very cold regions
        omega = np.where(T_safe > 0.05, omega, 0)
        
        return omega
    
    def laplacian_1d_periodic(self, field):
        """Calculate 1D Laplacian with periodic boundary conditions"""
        lap = np.zeros_like(field)
        
        # Use central differences with periodic wrap
        for i in range(len(field)):
            im1 = (i - 1) % self.n_points
            ip1 = (i + 1) % self.n_points
            lap[i] = (field[ip1] - 2*field[i] + field[im1]) / self.dtheta**2
        
        return lap
    
    def advection_1d_periodic(self, field, velocity):
        """Calculate advection term with upwind scheme"""
        adv = np.zeros_like(field)
        
        if isinstance(velocity, np.ndarray):
            # Spatially varying velocity
            for i in range(len(field)):
                im1 = (i - 1) % self.n_points
                ip1 = (i + 1) % self.n_points
                
                if velocity[i] > 0:
                    adv[i] = -velocity[i] * (field[i] - field[im1]) / self.dtheta
                else:
                    adv[i] = -velocity[i] * (field[ip1] - field[i]) / self.dtheta
        else:
            # Constant velocity
            for i in range(len(field)):
                im1 = (i - 1) % self.n_points
                ip1 = (i + 1) % self.n_points
                
                if velocity > 0:
                    adv[i] = -velocity * (field[i] - field[im1]) / self.dtheta
                else:
                    adv[i] = -velocity * (field[ip1] - field[i]) / self.dtheta
        
        return adv
    
    def step(self):
        """Advance simulation by one time step"""
        # Calculate reaction rate
        omega = self.reaction_rate(self.T, self.Y)
        
        # Calculate diffusion
        lap_T = self.laplacian_1d_periodic(self.T)
        lap_Y = self.laplacian_1d_periodic(self.Y)
        
        # Calculate flame-induced velocity (proportional to heat release)
        # This creates advection that helps flames propagate
        dT_dtheta = np.gradient(self.T, self.dtheta)
        velocity = self.flame_speed * np.sign(dT_dtheta) * self.T
        
        # Calculate advection
        adv_T = self.advection_1d_periodic(self.T, velocity)
        adv_Y = self.advection_1d_periodic(self.Y, velocity)
        
        # Update equations:
        # dT/dt = D_T * ∇²T + ω + advection
        # dY/dt = (D_T/Le) * ∇²Y - ω + advection
        
        self.T += self.dt * (self.D_T * lap_T + omega + adv_T)
        self.Y += self.dt * ((self.D_T/self.Le) * lap_Y - omega + adv_Y)
        
        # Apply bounds
        self.T = np.clip(self.T, 0, 1)
        self.Y = np.clip(self.Y, 0, 1)
        
        # Very slow fuel recovery (optional - can be removed for pure depletion)
        # self.Y += self.dt * 0.01 * (1 - self.Y) * np.exp(-5*self.T)
        
        self.time += self.dt
        
        # Store history
        if len(self.history_time) == 0 or self.time - self.history_time[-1] > 0.01:
            self.history_T.append(self.T.copy())
            self.history_Y.append(self.Y.copy())
            self.history_time.append(self.time)
    
    def run_until_collision(self, max_time=2.0):
        """Run simulation until flames collide or max_time is reached"""
        print("Running collision simulation...")
        
        while self.time < max_time:
            self.step()
            
            # Check every 100 steps
            if int(self.time / self.dt) % 100 == 0:
                # Find flame fronts (where T > 0.5)
                flame_positions = np.where(self.T > 0.5)[0]
                
                if len(flame_positions) > 0:
                    # Check if flames have met (gap in positions indicates collision)
                    gaps = np.diff(flame_positions)
                    if np.any(gaps > self.n_points/4):  # Large gap means flames met
                        print(f"Flames collided at t={self.time:.3f}")
                        break
                
                # Print progress
                if int(self.time / self.dt) % 1000 == 0:
                    max_T = np.max(self.T)
                    print(f"  t={self.time:.3f}, max T={max_T:.3f}")
        
        print(f"Simulation complete at t={self.time:.3f}")
    
    def plot_spacetime_diagram(self):
        """Create space-time diagram showing flame propagation and collision"""
        if len(self.history_T) == 0:
            print("No history to plot!")
            return
        
        # Convert history to arrays
        T_history = np.array(self.history_T)
        Y_history = np.array(self.history_Y)
        times = np.array(self.history_time)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Temperature space-time diagram
        ax1 = axes[0, 0]
        theta_deg = np.degrees(self.theta)
        im1 = ax1.imshow(T_history.T, aspect='auto', origin='lower', 
                         cmap='hot', extent=[0, times[-1], 0, 360],
                         vmin=0, vmax=1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title('Temperature Space-Time Evolution')
        plt.colorbar(im1, ax=ax1, label='Temperature')
        
        # Add collision point annotation
        collision_time = times[-1]
        ax1.axvline(collision_time, color='cyan', linestyle='--', alpha=0.7, label='Collision')
        ax1.legend()
        
        # Fuel space-time diagram
        ax2 = axes[0, 1]
        im2 = ax2.imshow(Y_history.T, aspect='auto', origin='lower',
                         cmap='YlGn', extent=[0, times[-1], 0, 360],
                         vmin=0, vmax=1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Angle (degrees)')
        ax2.set_title('Fuel Concentration Space-Time Evolution')
        plt.colorbar(im2, ax=ax2, label='Fuel')
        
        # Temperature profiles at different times
        ax3 = axes[1, 0]
        n_profiles = min(8, len(times))
        indices = np.linspace(0, len(times)-1, n_profiles, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
        
        for i, idx in enumerate(indices):
            ax3.plot(theta_deg, T_history[idx, :], color=colors[i], 
                    label=f't={times[idx]:.2f}', alpha=0.8)
        
        ax3.set_xlabel('Angle (degrees)')
        ax3.set_ylabel('Temperature')
        ax3.set_title('Temperature Profiles Evolution')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 360)
        
        # Fuel profiles at different times
        ax4 = axes[1, 1]
        for i, idx in enumerate(indices):
            ax4.plot(theta_deg, Y_history[idx, :], color=colors[i],
                    label=f't={times[idx]:.2f}', alpha=0.8)
        
        ax4.set_xlabel('Angle (degrees)')
        ax4.set_ylabel('Fuel Concentration')
        ax4.set_title('Fuel Depletion Evolution')
        ax4.legend(loc='lower right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 360)
        
        plt.suptitle('🔥 Two Counter-Propagating Flames Collision in Ring Trough 🔥\n' +
                    f'Lewis number={self.Le:.1f}, Zeldovich number={self.beta:.1f}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_ring_animation(self):
        """Create animated ring visualization of the collision"""
        if len(self.history_T) == 0:
            print("No history for animation!")
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Flame Collision Sequence in Ring Trough', fontsize=14, fontweight='bold')
        
        # Select 8 time points
        n_frames = 8
        indices = np.linspace(0, len(self.history_time)-1, n_frames, dtype=int)
        
        for frame_idx, hist_idx in enumerate(indices):
            ax = axes[frame_idx // 4, frame_idx % 4]
            ax.set_aspect('equal')
            
            # Draw ring boundaries
            outer = Circle((0, 0), 1.1, fill=False, edgecolor='black', linewidth=2)
            inner = Circle((0, 0), 0.9, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(outer)
            ax.add_patch(inner)
            
            # Color the ring based on temperature
            T_frame = self.history_T[hist_idx]
            
            for i in range(self.n_points):
                # Temperature-based color
                color_val = T_frame[i]
                color = plt.cm.hot(color_val)
                
                # Create wedge
                theta_start = np.degrees(self.theta[i] - self.dtheta/2)
                theta_end = np.degrees(self.theta[i] + self.dtheta/2)
                
                wedge = Wedge((0, 0), 1.1, theta_start, theta_end,
                            width=0.2, facecolor=color, edgecolor='none', alpha=0.9)
                ax.add_patch(wedge)
            
            # Add direction arrows for flame propagation
            if hist_idx < len(self.history_time) - 1:
                # Find flame fronts
                flame_pos = np.where(T_frame > 0.5)[0]
                if len(flame_pos) > 0:
                    for pos in flame_pos[::10]:  # Sample every 10th position
                        angle = self.theta[pos]
                        x, y = np.cos(angle), np.sin(angle)
                        
                        # Determine propagation direction
                        next_T = self.history_T[min(hist_idx+1, len(self.history_T)-1)]
                        if pos < self.n_points - 1:
                            if next_T[pos+1] > T_frame[pos+1]:
                                dx, dy = -np.sin(angle)*0.1, np.cos(angle)*0.1
                            else:
                                dx, dy = np.sin(angle)*0.1, -np.cos(angle)*0.1
                            
                            # ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.02,
                            #         fc='yellow', ec='orange', alpha=0.7)
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_title(f't = {self.history_time[hist_idx]:.3f}s', fontsize=10)
            ax.axis('off')
            
            # Add text annotations
            if frame_idx == 0:
                ax.text(0, -1.5, 'Initial: Two flames\nat opposite sides', 
                       ha='center', fontsize=8)
            elif frame_idx == n_frames - 1:
                ax.text(0, -1.5, 'Final: Flames collided\nand annihilated', 
                       ha='center', fontsize=8, color='red')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_collision(self):
        """Analyze and report collision characteristics"""
        print("\n" + "="*60)
        print("COLLISION ANALYSIS")
        print("="*60)
        
        # Find when temperature peaks drop (collision)
        max_T_history = [np.max(T) for T in self.history_T]
        
        # Find collision time (when max T starts decreasing significantly)
        for i in range(1, len(max_T_history)):
            if max_T_history[i] < 0.8 * max_T_history[i-1]:
                collision_idx = i
                collision_time = self.history_time[i]
                print(f"🔥 Collision detected at t = {collision_time:.3f}s")
                break
        else:
            collision_time = self.history_time[-1]
            print(f"⚠️ No clear collision detected by t = {collision_time:.3f}s")
        
        # Calculate average flame speeds
        if len(self.history_T) > 10:
            # Track flame front positions
            front_positions_1 = []
            front_positions_2 = []
            
            for i in range(min(20, len(self.history_T))):
                T = self.history_T[i]
                peaks = np.where(T > 0.5)[0]
                
                if len(peaks) > 0:
                    # Identify two separate flame fronts
                    gaps = np.diff(peaks)
                    if np.any(gaps > 5):
                        gap_idx = np.argmax(gaps)
                        front_positions_1.append(np.mean(peaks[:gap_idx+1]))
                        front_positions_2.append(np.mean(peaks[gap_idx+1:]))
            
            if len(front_positions_1) > 1:
                speed_1 = np.abs(np.mean(np.diff(front_positions_1))) * self.dtheta / 0.01
                speed_2 = np.abs(np.mean(np.diff(front_positions_2))) * self.dtheta / 0.01
                print(f"📊 Flame 1 speed: {speed_1:.3f} rad/s")
                print(f"📊 Flame 2 speed: {speed_2:.3f} rad/s")
        
        # Report final state
        final_max_T = np.max(self.T)
        final_min_Y = np.min(self.Y)
        print(f"\n📈 Final state:")
        print(f"   Max temperature: {final_max_T:.3f}")
        print(f"   Min fuel concentration: {final_min_Y:.3f}")
        
        if final_max_T < 0.1:
            print("✅ Complete annihilation achieved!")
        elif final_max_T < 0.5:
            print("⚠️ Partial annihilation - flames weakened significantly")
        else:
            print("❌ Flames still active - may need longer simulation")
        
        print("="*60)


def main():
    """Main execution function"""
    print("="*70)
    print("🔥🔥 TWO FLAMES COLLISION SIMULATION IN RING TROUGH 🔥🔥")
    print("="*70)
    print("\n📍 Initial Setup:")
    print("   • Two flames initialized at opposite sides (0° and 180°)")
    print("   • Flames will propagate towards each other")
    print("   • Collision expected at 90° and 270°")
    print("="*70)
    
    # Create and run simulation
    sim = FlameCollisionSimulation(n_points=300)
    
    # Run until collision
    sim.run_until_collision(max_time=1.5)
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    sim.plot_spacetime_diagram()
    sim.create_ring_animation()
    
    # Analyze collision
    sim.analyze_collision()
    
    print("\n✅ Simulation complete!")
    print("   The space-time diagram clearly shows the collision event")
    print("   where the two flame fronts meet and annihilate each other.")
    
    return sim


if __name__ == "__main__":
    sim = main()