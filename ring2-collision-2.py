import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import LinearSegmentedColormap

class SimpleFlameCollision:
    """
    Simplified flame collision with guaranteed propagation and collision
    """
    
    def __init__(self, n_points=400):
        # Grid
        self.n_points = n_points
        self.theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        self.dtheta = 2*np.pi / n_points
        
        # Simple physical parameters
        self.D = 0.01  # Diffusion coefficient (small)
        self.c = 5.0   # Propagation speed (large)
        self.dt = 0.0001  # Time step
        self.time = 0.0
        
        # State: Temperature only (simplified)
        self.T = np.zeros(n_points)
        
        # Initial flame positions (close together)
        self.pos1_init = 150  # degrees
        self.pos2_init = 210  # degrees
        
        # Initialize
        self._initialize_flames()
        
        # History
        self.history_T = []
        self.history_time = []
        
        print("="*60)
        print("SIMPLE FLAME COLLISION SETUP")
        print(f"  Flame 1: {self.pos1_init}°")
        print(f"  Flame 2: {self.pos2_init}°")
        print(f"  Expected collision: 180°")
        print(f"  Propagation speed: {self.c} rad/s")
        print("="*60)
    
    def _initialize_flames(self):
        """Create two narrow flame pulses"""
        # Convert positions to radians
        pos1_rad = np.radians(self.pos1_init)
        pos2_rad = np.radians(self.pos2_init)
        
        # Create Gaussian pulses
        for i in range(self.n_points):
            # Distance to flame 1
            dist1 = np.abs(self.theta[i] - pos1_rad)
            dist1 = min(dist1, 2*np.pi - dist1)
            
            # Distance to flame 2
            dist2 = np.abs(self.theta[i] - pos2_rad)
            dist2 = min(dist2, 2*np.pi - dist2)
            
            # Narrow Gaussians
            width = 0.1
            self.T[i] = np.exp(-(dist1/width)**2) + np.exp(-(dist2/width)**2)
        
        # Normalize
        self.T = self.T / np.max(self.T)
    
    def step_simple(self):
        """Simple propagation step"""
        T_new = self.T.copy()
        
        # For each point, determine if it's part of a flame
        for i in range(self.n_points):
            if self.T[i] > 0.1:  # Part of a flame
                angle_deg = np.degrees(self.theta[i])
                
                # Determine propagation direction based on initial position
                # Flame from 150° propagates clockwise toward 180°
                if 120 < angle_deg < 180:
                    # Propagate right (increase angle)
                    ip1 = (i + 1) % self.n_points
                    T_new[ip1] = max(T_new[ip1], self.T[i] * 0.98)
                
                # Flame from 210° propagates counter-clockwise toward 180°
                elif 180 < angle_deg < 240:
                    # Propagate left (decrease angle)
                    im1 = (i - 1) % self.n_points
                    T_new[im1] = max(T_new[im1], self.T[i] * 0.98)
        
        # Add small diffusion for smoothness
        for i in range(self.n_points):
            im1 = (i - 1) % self.n_points
            ip1 = (i + 1) % self.n_points
            diffusion = self.D * (self.T[ip1] - 2*self.T[i] + self.T[im1]) / self.dtheta**2
            T_new[i] += self.dt * diffusion
        
        # Decay (fuel consumption)
        T_new *= 0.999
        
        # Check for collision and enforce annihilation
        collision_zone = (170 < np.degrees(self.theta)) & (np.degrees(self.theta) < 190)
        if np.sum(self.T[collision_zone]) > 1.5:  # Two flames overlapping
            print(f"  COLLISION at t={self.time:.4f}s!")
            T_new[collision_zone] *= 0.5  # Rapid annihilation
        
        # Update
        self.T = np.clip(T_new, 0, 1)
        self.time += self.dt
        
        # Store history
        if len(self.history_time) == 0 or self.time - self.history_time[-1] > 0.001:
            self.history_T.append(self.T.copy())
            self.history_time.append(self.time)
    
    def run(self, max_time=0.1):
        """Run simulation"""
        print("\nRunning simulation...")
        
        steps = int(max_time / self.dt)
        report_every = steps // 10
        
        for step in range(steps):
            self.step_simple()
            
            if step % report_every == 0:
                max_T = np.max(self.T)
                # Find flame positions
                flame_mask = self.T > 0.2
                if np.any(flame_mask):
                    angles = np.degrees(self.theta[flame_mask])
                    print(f"  t={self.time:.4f}s: max T={max_T:.3f}, "
                          f"flames at {angles[0]:.0f}°-{angles[-1]:.0f}°")
                else:
                    print(f"  t={self.time:.4f}s: flames extinguished")
        
        print(f"Simulation complete at t={self.time:.4f}s")
    
    def plot_results(self):
        """Visualize results"""
        if len(self.history_T) == 0:
            return
        
        T_history = np.array(self.history_T)
        times = np.array(self.history_time)
        theta_deg = np.degrees(self.theta)
        
        fig = plt.figure(figsize=(14, 8))
        
        # 1. Space-time diagram
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(T_history.T, aspect='auto', origin='lower',
                        cmap='hot', extent=[0, times[-1], 0, 360],
                        vmin=0, vmax=1)
        ax1.axhline(180, color='cyan', linestyle='--', alpha=0.5, label='Collision point')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title('Temperature Evolution')
        ax1.legend()
        plt.colorbar(im1, ax=ax1)
        
        # 2. Temperature profiles
        ax2 = plt.subplot(2, 3, 2)
        n_profiles = min(8, len(times))
        indices = np.linspace(0, len(times)-1, n_profiles, dtype=int)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_profiles))
        
        for i, idx in enumerate(indices):
            ax2.plot(theta_deg, T_history[idx], color=colors[i],
                    label=f't={times[idx]:.3f}s', linewidth=2)
        
        ax2.axvline(180, color='black', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Profiles')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Maximum temperature over time
        ax3 = plt.subplot(2, 3, 3)
        max_T = [np.max(T) for T in T_history]
        ax3.plot(times, max_T, 'r-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Max Temperature')
        ax3.set_title('Peak Temperature Evolution')
        ax3.grid(True, alpha=0.3)
        
        # 4-6. Ring snapshots
        snapshot_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
        
        for plot_idx, time_idx in enumerate(snapshot_indices[:3]):
            ax = plt.subplot(2, 3, 4 + plot_idx)
            ax.set_aspect('equal')
            
            # Ring
            outer = Circle((0, 0), 1.1, fill=False, edgecolor='black', linewidth=2)
            inner = Circle((0, 0), 0.9, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(outer)
            ax.add_patch(inner)
            
            # Temperature
            T_ring = T_history[time_idx]
            
            # Wedges
            for i in range(0, self.n_points, 2):
                if T_ring[i] > 0.05:
                    color_val = T_ring[i]
                    color = plt.cm.hot(color_val)
                    
                    theta_start = np.degrees(self.theta[i] - self.dtheta)
                    theta_end = np.degrees(self.theta[i] + self.dtheta)
                    
                    wedge = Wedge((0, 0), 1.1, theta_start, theta_end,
                                width=0.2, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
            
            # Mark 180°
            ax.plot([0, 0], [0.85, 1.15], 'c--', alpha=0.5)
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_title(f't={times[time_idx]:.3f}s', fontsize=10)
            ax.axis('off')
        
        plt.suptitle('SIMPLIFIED FLAME COLLISION DEMONSTRATION\n' +
                    f'Initial: {self.pos1_init}° and {self.pos2_init}°, Collision at 180°',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Alternative: Frame-by-frame animation approach
class AnimatedFlameCollision:
    """
    Direct animation of flame collision for clear visualization
    """
    
    def __init__(self):
        self.n_points = 360  # One point per degree
        self.angles = np.arange(360)
        
        # Initialize two flame positions
        self.flame1_center = 150
        self.flame2_center = 210
        self.flame_width = 20
        
        # Speed (degrees per frame)
        self.speed = 2
        
        self.frames = []
        self.generate_collision_sequence()
    
    def generate_collision_sequence(self):
        """Generate frame-by-frame collision"""
        frame_count = 0
        
        while self.flame1_center < 180 and self.flame2_center > 180:
            # Create temperature field
            T = np.zeros(self.n_points)
            
            # Flame 1 (moving right)
            for i in range(self.n_points):
                dist1 = np.abs(i - self.flame1_center)
                if dist1 < self.flame_width:
                    T[i] = np.exp(-(dist1/10)**2)
            
            # Flame 2 (moving left)
            for i in range(self.n_points):
                dist2 = np.abs(i - self.flame2_center)
                if dist2 < self.flame_width:
                    T[i] = max(T[i], np.exp(-(dist2/10)**2))
            
            self.frames.append(T.copy())
            
            # Move flames
            self.flame1_center += self.speed
            self.flame2_center -= self.speed
            frame_count += 1
            
            # Check for collision
            if abs(self.flame1_center - self.flame2_center) < self.flame_width:
                print(f"Collision at frame {frame_count}!")
                # Add annihilation frames
                for decay in range(5):
                    T *= 0.5
                    self.frames.append(T.copy())
                break
    
    def plot_animation(self):
        """Show the collision sequence"""
        n_frames = len(self.frames)
        n_show = min(8, n_frames)
        indices = np.linspace(0, n_frames-1, n_show, dtype=int)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for plot_idx, frame_idx in enumerate(indices):
            ax = axes[plot_idx]
            ax.set_aspect('equal')
            
            # Draw ring
            outer = Circle((0, 0), 1.1, fill=False, edgecolor='black', linewidth=1)
            inner = Circle((0, 0), 0.9, fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(outer)
            ax.add_patch(inner)
            
            # Temperature
            T = self.frames[frame_idx]
            
            # Create wedges
            for i in range(0, 360, 3):
                if T[i] > 0.05:
                    color = plt.cm.hot(T[i])
                    wedge = Wedge((0, 0), 1.1, i-1.5, i+1.5,
                                width=0.2, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
            
            # Mark 180°
            ax.plot([0, 0], [-1.2, 1.2], 'c--', alpha=0.3)
            ax.text(0, 1.25, '180°', ha='center', fontsize=8)
            
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.set_title(f'Frame {frame_idx+1}', fontsize=9)
            ax.axis('off')
        
        plt.suptitle('ANIMATED FLAME COLLISION SEQUENCE', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """Run demonstrations"""
    print("="*70)
    print("FLAME COLLISION DEMONSTRATIONS")
    print("="*70)
    
    # Method 1: Simplified physics
    print("\n1. SIMPLIFIED PHYSICS APPROACH:")
    print("-"*40)
    sim1 = SimpleFlameCollision(n_points=400)
    sim1.run(max_time=0.05)
    sim1.plot_results()
    
    # Method 2: Direct animation
    print("\n2. DIRECT ANIMATION APPROACH:")
    print("-"*40)
    sim2 = AnimatedFlameCollision()
    sim2.plot_animation()
    
    print("\n" + "="*70)
    print("Both methods should show clear collision at 180°")
    print("="*70)


if __name__ == "__main__":
    main()