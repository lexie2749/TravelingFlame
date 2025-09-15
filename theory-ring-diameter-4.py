import numpy as np
import matplotlib.pyplot as plt

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Physical Parameters ==============
# Base parameters for liquid fuel
s_L = 0.4  # Base laminar flame speed (m/s) for liquid fuel
Ma = -0.5  # Markstein number
delta_L = 0.3e-3  # Flame thickness (m)

# ============== Acceleration Model ==============
def calculate_acceleration_factor(BR):
    """
    Calculate acceleration factor based on blockage ratio
    Using a physically reasonable model for liquid fuel
    """
    if BR < 1:
        # For BR < 1, slight deceleration
        return BR**0.7
    else:
        # For BR >= 1, moderate acceleration with saturation
        return 1 + 0.6 * (1 - np.exp(-0.8 * (BR - 1)))

def calculate_flame_speed(ring_radius, BR):
    """
    Calculate flame speed with curvature and blockage effects
    """
    # Curvature effect
    curvature_factor = 1 - Ma * delta_L / ring_radius
    curvature_factor = np.clip(curvature_factor, 0.8, 1.2)
    
    # Acceleration from blockage
    acceleration_factor = calculate_acceleration_factor(BR)
    
    # Combined flame speed
    v = s_L * curvature_factor * acceleration_factor
    return v

def calculate_angular_velocity(flame_speed, ring_radius):
    """Calculate angular velocity ω = v/r"""
    return flame_speed / ring_radius

# ============== Plotting with Uncertainty ==============
# Create a single plot instead of two
fig, ax1 = plt.subplots(figsize=(10, 7))

# Define ring diameter range
ring_diameters_mm = np.linspace(30, 200, 200)
ring_radii_m = ring_diameters_mm * 1e-3 / 2

# Define BR values to plot
BR_values = [2.0]
# 统一颜色方案
theoretical_color = '#2E86AB'
experimental_color = 'purple'

# Uncertainty factor (±20% for theoretical variation)
uncertainty = 0.2

# ---- Plot: Angular Velocity vs Ring Diameter with BR effect ----
for i, BR in enumerate(BR_values):
    angular_velocities = []
    
    for r in ring_radii_m:
        v = calculate_flame_speed(r, BR)
        omega = calculate_angular_velocity(v, r)
        angular_velocities.append(omega)
        
    # Plot main line
    ax1.plot(ring_diameters_mm, angular_velocities, 
             linewidth=2.5, color=theoretical_color,
             label=f'Theoretical Model (BR = {BR:.1f})', alpha=0.9)
    
    # Calculate and plot uncertainty bounds
    angular_velocities_upper = np.array(angular_velocities) * (1 + uncertainty)
    angular_velocities_lower = np.array(angular_velocities) * (1 - uncertainty)
    
    ax1.fill_between(ring_diameters_mm, 
                     angular_velocities_lower, 
                     angular_velocities_upper,
                     color=theoretical_color, alpha=0.15, label='Theoretical Uncertainty (±20%)')
    
    # Add dashed lines for bounds
    ax1.plot(ring_diameters_mm, angular_velocities_upper, 
            '--', color=theoretical_color, linewidth=1, alpha=0.5)
    ax1.plot(ring_diameters_mm, angular_velocities_lower, 
            '--', color=theoretical_color, linewidth=1, alpha=0.5)

# ============== Add experimental data ==============
exp_radii_mm = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
exp_diameters_mm = exp_radii_mm
exp_omegas = np.array([19.81688889, 17.26691249, 15.74361111, 14.25170543, 14.2713, 13.4827622, 11.83358209, 10.17171229, 11.61218519, 9.699111111, 10.74927778])
exp_errors = np.array([0.245468191, 0.254442237, 0.526, 0.287, 0.398, 0.551, 0.130, 0.201, 0.426, 0.175, 0.386])

# Plot the experimental data with a label that explicitly mentions the error bars
ax1.errorbar(exp_diameters_mm, exp_omegas, yerr=exp_errors, 
             fmt='o', color=experimental_color, capsize=4, label='Experimental Data with Error Bars', zorder=5)

ax1.set_xlabel('Ring Diameter (mm)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=11)
ax1.grid(False)
ax1.set_xlim(30, 200)

ax1.set_ylim(0, 25)


plt.tight_layout()
plt.show()