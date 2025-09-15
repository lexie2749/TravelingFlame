import numpy as np
import matplotlib.pyplot as plt

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Physical Parameters ==============
T_u = 300 # Unburned gas temperature (K)
T_b = 2100 # Burned gas temperature (K)
E_a = 100000 # Activation energy (J/mol)
R_u = 8.314 # Universal gas constant (J/mol·K)

# Transport properties
lambda_th = 0.028 # Thermal conductivity (W/m·K)
rho = 1.1 # Density (kg/m³)
c_p = 1000 # Specific heat capacity (J/kg·K)
D_m = 2.8e-5 # Mass diffusivity (m²/s)

# Calculate derived parameters
alpha_th = lambda_th / (rho * c_p)
Le = alpha_th / D_m
sigma = T_b / T_u
beta = E_a * (T_b - T_u) / (R_u * T_b**2)

# Calculate Markstein number
Ma = ((sigma - 1) / (2 * sigma)) * np.log(sigma) + (beta * (Le - 1)) / (2 * sigma)

# Set laminar flame speed
s_L = 0.33 # m/s
delta_L = 0.35e-3 # Flame thickness (m)

def calculate_acceleration_factor(BR):
    """Calculate acceleration factor based on blockage ratio"""
    if BR < 1:
        return BR**0.85
    else:
        return 1 + 1.2 * (1 - np.exp(-0.6 * (BR - 1)))

def calculate_flame_speed(ring_radius, channel_diameter, fuel_height):
    """Calculate flame speed with all effects"""
    curvature_factor = 1 - Ma * delta_L / ring_radius
    curvature_factor = np.clip(curvature_factor, 0.95, 1.35)
    BR = channel_diameter / fuel_height
    acceleration_factor = calculate_acceleration_factor(BR)
    v = s_L * curvature_factor * acceleration_factor
    return v

def calculate_angular_velocity(flame_speed, ring_radius):
    """Calculate angular velocity ω = v/r"""
    return flame_speed / ring_radius

# ============== Create Figure ==============
fig, ax = plt.subplots(figsize=(10, 7))

# 统一颜色方案
theoretical_color = '#2E86AB'
experimental_color = 'purple'

# Channel diameter range for theoretical model (3-10 mm)
channel_diameters_mm = np.linspace(3, 10, 200)
channel_diameters_m = channel_diameters_mm * 1e-3

# Fixed parameters
ring_diameter = 100 # mm
ring_radius = ring_diameter * 1e-3 / 2 # Convert to radius in meters
fuel_height = 2e-3 # 2 mm

# Calculate theoretical angular velocities
angular_velocities = []
flame_speeds = []
for channel_d in channel_diameters_m:
    v = calculate_flame_speed(ring_radius, channel_d, fuel_height)
    omega = calculate_angular_velocity(v, ring_radius)
    angular_velocities.append(omega)
    flame_speeds.append(v)

# Plot theoretical model line
ax.plot(channel_diameters_mm, angular_velocities,
        linewidth=3, color=theoretical_color, alpha=0.9, label='Theoretical Model')

# Add uncertainty band (±15%)
omega_upper = np.array(angular_velocities) * 1.15
omega_lower = np.array(angular_velocities) * 0.85
ax.fill_between(channel_diameters_mm, omega_lower, omega_upper,
                color=theoretical_color, alpha=0.15, label='Theoretical Uncertainty ($\pm$15%)')
ax.plot(channel_diameters_mm, omega_upper, '--', color=theoretical_color, linewidth=1, alpha=0.5)
ax.plot(channel_diameters_mm, omega_lower, '--', color=theoretical_color, linewidth=1, alpha=0.5)

# ============== Add Experimental Data ==============
exp_channel_diameters_mm = np.array([3, 4, 5, 6, 7, 8, 9, 10])
exp_angular_velocities = np.array([8.8568, 10.2345, 10.5832, 10.1417, 10.8390, 13.6821, 13.0229, 14.6385])
exp_angular_velocity_errors = np.array([0.8097, 0.2962, 0.4845, 0.2209, 0.2941, 0.8108, 0.3801, 0.1840])

# Plot experimental data with error bars
ax.errorbar(exp_channel_diameters_mm, exp_angular_velocities,
            yerr=exp_angular_velocity_errors, fmt='D', color=experimental_color,
            capsize=5, zorder=5, label='Experimental Data with Error Bars')

# ============== Formatting and Labels ==============
ax.set_xlabel('Channel Diameter (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')

ax.set_xlim(2.5, 10.5)
ax.set_ylim(0, 20)

# Add legend and save figure
ax.legend(frameon=True)
plt.grid(False)
plt.show()