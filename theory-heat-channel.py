import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Adjusted Physical Parameters (Theoretical Model) ==============
T_u = 300  # Unburned gas temperature (K)
T_b = 1900  # Burned gas temperature (K)
E_a = 120000  # Activation energy (J/mol)
R_u = 8.314  # Universal gas constant (J/mol·K)
lambda_th = 0.025  # Thermal conductivity (W/m·K)
rho = 1.2  # Density (kg/m³)
c_p = 1000  # Specific heat capacity (J/kg·K)
D_m = 2.5e-5  # Mass diffusivity (m²/s)

alpha_th = lambda_th / (rho * c_p)
Le = alpha_th / D_m
sigma = T_b / T_u
beta = E_a * (T_b - T_u) / (R_u * T_b**2)
Ma = ((sigma - 1) / (2 * sigma)) * np.log(sigma) + (beta * (Le - 1)) / (2 * sigma)

s_L = 0.25  # m/s (25 cm/s)
delta_L = 0.35e-3  # Flame thickness (0.35 mm)

ring_diameter = 100  # mm
ring_radius = ring_diameter / 2  # mm
ring_radius_m = ring_radius * 1e-3  # Convert to meters

def volume_to_height(volume_ul, channel_width):
    volume_mm3 = volume_ul
    circumference = 2 * np.pi * ring_radius
    channel_area = circumference * channel_width
    height_mm = volume_mm3 / channel_area
    return height_mm

def calculate_acceleration_factor(BR, fuel_height_mm):
    if BR < 1:
        base_factor = BR**0.9
    else:
        base_factor = 1 + 1.5 * (1 - np.exp(-0.5 * (BR - 1)))
    
    if fuel_height_mm < 0.15:
        height_factor = 1 + 0.3 * (0.15 - fuel_height_mm) / 0.15
    else:
        height_factor = 1 - 0.2 * (fuel_height_mm - 0.15) / 0.15
    
    return base_factor * height_factor

def calculate_flame_speed(channel_diameter_m, fuel_height_m):
    fuel_height_mm = fuel_height_m * 1000
    curvature_factor = 1 - Ma * delta_L / ring_radius_m
    curvature_factor = np.clip(curvature_factor, 0.9, 1.2)
    BR = channel_diameter_m / fuel_height_m if fuel_height_m > 0 else 0
    acceleration_factor = calculate_acceleration_factor(BR, fuel_height_mm)
    v = s_L * curvature_factor * acceleration_factor
    return v

def calculate_angular_velocity(flame_speed, ring_radius_m):
    return flame_speed / ring_radius_m

# ============== Create 2D Analysis Grid (Theoretical Model) ==============
# Define the range for channel width and fuel volume to match experimental data
channel_width_range = np.linspace(3, 10, 200)
fuel_volume_range = np.linspace(125, 200, 200)

X, Y = np.meshgrid(channel_width_range, fuel_volume_range)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        channel_width_mm = X[i, j]
        fuel_volume_ul = Y[i, j]
        channel_width_m = channel_width_mm * 1e-3
        height_mm = volume_to_height(fuel_volume_ul, channel_width_mm)
        height_m = height_mm * 1e-3
        v = calculate_flame_speed(channel_width_m, height_m)
        omega = calculate_angular_velocity(v, ring_radius_m)
        Z[i, j] = omega

# ============== Create Combined Plot ==============
fig, ax = plt.subplots(figsize=(10, 8))

# Define custom levels for the theoretical contour plot
levels = np.linspace(Z.min(), Z.max(), 500)
contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', extend='both')
cbar = fig.colorbar(contourf, ax=ax)
cbar.set_label('Theoretical Angular Velocity (rad/s)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Set labels and title
ax.set_xlabel('Ring Radius (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Fuel Volume (µL)', fontsize=14, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(False)

plt.tight_layout()
plt.show()
