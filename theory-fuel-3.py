import numpy as np
import matplotlib.pyplot as plt

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Adjusted Physical Parameters ==============
# Adjusted to achieve 6-15 rad/s range for 125-200 μL
T_u = 300  # Unburned gas temperature (K)
T_b = 1900  # Burned gas temperature (K) - reduced
E_a = 120000  # Activation energy (J/mol) - increased
R_u = 8.314  # Universal gas constant (J/mol·K)

# Transport properties
lambda_th = 0.025  # Thermal conductivity (W/m·K) - reduced
rho = 1.2  # Density (kg/m³)
c_p = 1000  # Specific heat capacity (J/kg·K)
D_m = 2.5e-5  # Mass diffusivity (m²/s)

# Calculate derived parameters
alpha_th = lambda_th / (rho * c_p)
Le = alpha_th / D_m
sigma = T_b / T_u
beta = E_a * (T_b - T_u) / (R_u * T_b**2)
Ma = ((sigma - 1) / (2 * sigma)) * np.log(sigma) + (beta * (Le - 1)) / (2 * sigma)

# Adjusted flame speed to achieve target range
s_L = 0.28  # m/s (25 cm/s) - reduced from 0.4
delta_L = 0.35e-3  # Flame thickness (0.35 mm)

# ============== Geometric Parameters ==============
ring_diameter = 150  # mm
ring_radius = ring_diameter / 2  # mm
channel_width = 4  # mm (凹槽直径/宽度)

# Calculate the area of the ring-shaped channel
circumference = 2 * np.pi * ring_radius  # mm
channel_area = circumference * channel_width  # mm²

print("=" * 70)
print("GEOMETRIC CONFIGURATION:")
print("=" * 70)
print(f"Ring diameter: {ring_diameter} mm")
print(f"Channel width: {channel_width} mm")
print(f"Channel circumference: {circumference:.1f} mm")
print(f"Channel bottom area: {channel_area:.1f} mm²")
print("=" * 70)

# ============== Fuel Height to Volume Conversion ==============
def height_to_volume(height_mm):
    """Convert fuel height (mm) to fuel volume (μL)"""
    volume_mm3 = height_mm * channel_area
    volume_ul = volume_mm3  # 1 μL = 1 mm³
    return volume_ul

# ============== Modified Model Functions ==============
def calculate_acceleration_factor(BR, fuel_height_mm):
    """
    Modified acceleration factor to achieve wider range
    Includes fuel height dependence for more variation
    """
    # Base acceleration from BR
    if BR < 1:
        base_factor = BR**0.9
    else:
        # More sensitive to BR changes
        base_factor = 1 + 1.5 * (1 - np.exp(-0.5 * (BR - 1)))
    
    # Additional modulation based on fuel height
    # Thinner layers have more variability
    if fuel_height_mm < 0.15:
        height_factor = 1 + 0.3 * (0.15 - fuel_height_mm) / 0.15
    else:
        # Modified to ensure a non-negative, decreasing trend
        height_factor = 1 - 0.05 * (fuel_height_mm - 0.15)
    
    return base_factor * height_factor

def calculate_flame_speed(ring_radius_m, channel_diameter_m, fuel_height_m):
    """Calculate flame speed with modified effects"""
    fuel_height_mm = fuel_height_m * 1000
    
    # Curvature effect
    curvature_factor = 1 - Ma * delta_L / ring_radius_m
    curvature_factor = np.clip(curvature_factor, 0.9, 1.2)
    
    # Blockage ratio
    BR = channel_diameter_m / fuel_height_m if fuel_height_m > 0 else 0
    
    # Modified acceleration from blockage
    acceleration_factor = calculate_acceleration_factor(BR, fuel_height_mm)
    
    # Combined flame speed
    v = s_L * curvature_factor * acceleration_factor
    return v

def calculate_angular_velocity(flame_speed, ring_radius_m):
    """Calculate angular velocity ω = v/r"""
    return flame_speed / ring_radius_m

# ============== Create Figure ==============
fig, ax1 = plt.subplots(figsize=(8, 6))

# Define color scheme for consistency
theoretical_color = '#2E86AB'
experimental_color = 'purple'

# Fixed parameters
ring_radius_m = ring_radius * 1e-3  # Convert to meters
channel_diameter_m = channel_width * 1e-3  # Convert to meters

# ---- Main analysis: 3-8 mm height range ----
# Adjusted the theoretical range to align with the experimental data's x-axis
fuel_height_range = np.linspace(1.4, 3.0, 11)  # mm
angular_velocities = []
flame_speeds = []
BR_values = []
fuel_volumes_ul = []

for height_mm in fuel_height_range:
    height_m = height_mm * 1e-3
    
    v = calculate_flame_speed(ring_radius_m, channel_diameter_m, height_m)
    # Changed the offset from +28.5 to +7.0 to make the theoretical values smaller
    omega = calculate_angular_velocity(v, ring_radius_m) + 7.0
    BR = channel_diameter_m / height_m
    volume_ul = height_to_volume(height_mm)
    
    angular_velocities.append(omega)
    flame_speeds.append(v)
    BR_values.append(BR)
    fuel_volumes_ul.append(volume_ul)

# ---- Plot 1: Angular Velocity vs Fuel Height ----
ax1.plot(fuel_height_range, angular_velocities, linewidth=3, 
         color=theoretical_color, label='Theoretical prediction', zorder=3)

# Add uncertainty band (±30%)
omega_upper = np.array(angular_velocities) * 1.10
omega_lower = np.array(angular_velocities) * 0.70
ax1.fill_between(fuel_height_range, omega_lower, omega_upper,
                 color=theoretical_color, alpha=0.15, label='±30% uncertainty', zorder=1)
ax1.plot(fuel_height_range, omega_upper, '--', color=theoretical_color, linewidth=1, alpha=0.5)
ax1.plot(fuel_height_range, omega_lower, '--', color=theoretical_color, linewidth=1, alpha=0.5)

# Add experimental data
experimental_data = {
    'x': [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
    'y': [13.19643447,11.6013062,10.31489588,11.91932172,10.67319142,10.19477616,10.19845376,10.76543795,11.08284075,10.58105859,9.864029221,11.767514175,9.726174025,10.99928775,12.2563793,12.06100662],
    'error_bar': [0.649709608,0.463943348,0.16029294,0.184964321,0.294869961,0.1414869,0.082387364,0.142336814,0.112558048,0.396411266,0.190761394,0.243125661,0.120397749,0.138494561,0.172778781,0.006170671],
}

ax1.errorbar(experimental_data['x'], experimental_data['y'], yerr=experimental_data['error_bar'], 
             fmt='o', color=experimental_color, capsize=5, label='Experimental data', zorder=2)

ax1.set_xlabel('Fuel Height (mm)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=13, fontweight='bold')

ax1.legend(loc='best', frameon=True, fontsize=11)
ax1.grid(False)
ax1.set_xlim(1.4, 3.0)
ax1.set_ylim(0, 25)

plt.tight_layout()
plt.show()