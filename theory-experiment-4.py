import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Adjusted Physical Parameters ==============
T_u = 300  # Unburned gas temperature (K)
T_b = 1900  # Burned gas temperature (K)
E_a = 120000  # Activation energy (J/mol)
R_u = 8.314  # Universal gas constant (J/mol·K)

# Transport properties
lambda_th = 0.025  # Thermal conductivity (W/m·K)
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
s_L = 0.25  # m/s (25 cm/s)
delta_L = 0.35e-3  # Flame thickness (0.35 mm)

# ============== Modified Model Functions ==============
def calculate_acceleration_factor(BR, fuel_height_mm):
    """Modified acceleration factor"""
    if BR < 1:
        base_factor = BR**0.9
    else:
        base_factor = 1 + 1.5 * (1 - np.exp(-0.5 * (BR - 1)))
    
    if fuel_height_mm < 0.15:
        height_factor = 1 + 0.3 * (0.15 - fuel_height_mm) / 0.15
    else:
        height_factor = 1 - 0.2 * (fuel_height_mm - 0.15) / 0.15
    
    return base_factor * height_factor

def calculate_flame_speed(ring_radius_m, channel_diameter_m, fuel_height_m):
    """Calculate flame speed with modified effects"""
    fuel_height_mm = fuel_height_m * 1000
    
    curvature_factor = 1 - Ma * delta_L / ring_radius_m
    curvature_factor = np.clip(curvature_factor, 0.9, 1.2)
    
    BR = channel_diameter_m / fuel_height_m if fuel_height_m > 0 else 0
    
    acceleration_factor = calculate_acceleration_factor(BR, fuel_height_mm)
    
    v = s_L * curvature_factor * acceleration_factor
    return v

def calculate_angular_velocity(flame_speed, ring_radius_m):
    """Calculate angular velocity ω = v/r"""
    return flame_speed / ring_radius_m

# ============== Create 3D Analysis Grid ==============
# Define the range for each variable
channel_width_range = np.linspace(3, 10, 20)  # mm
fuel_height_range = np.linspace(1.4, 3, 20)  # mm, NEW RANGE
ring_diameter_range = np.linspace(50, 160, 20)  # mm

# Create a 3D meshgrid
CW, FH, RD = np.meshgrid(channel_width_range, fuel_height_range, ring_diameter_range)

# Calculate the angular velocity for each combination
W = np.zeros_like(CW)
for i in range(CW.shape[0]):
    for j in range(CW.shape[1]):
        for k in range(CW.shape[2]):
            channel_width_mm = CW[i, j, k]
            fuel_height_mm = FH[i, j, k] # Use new fuel height
            ring_diameter_mm = RD[i, j, k]
            
            ring_radius_mm = ring_diameter_mm / 2
            
            # Convert units for calculations
            ring_radius_m = ring_radius_mm * 1e-3
            channel_width_m = channel_width_mm * 1e-3
            height_m = fuel_height_mm * 1e-3
            
            v = calculate_flame_speed(ring_radius_m, channel_width_m, height_m)
            omega = calculate_angular_velocity(v, ring_radius_m)
            
            W[i, j, k] = omega

# ============== Create 3D Scatter Plot ==============
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Flatten the data for plotting
x = CW.flatten()
y = FH.flatten() # Use new fuel height for y-axis
z = RD.flatten()
c = W.flatten()

# Create the scatter plot
scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=20)

# Add color bar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Angular Velocity (rad/s)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Set labels and title
ax.set_xlabel('Channel Diameter (mm)', fontsize=12, fontweight='bold', labelpad=15)
ax.set_ylabel('Fuel Height (mm)', fontsize=12, fontweight='bold', labelpad=15) # Updated label
ax.set_zlabel('Ring Diameter (mm)', fontsize=12, fontweight='bold', labelpad=15)
ax.set_title('Angular Velocity as a Function of 3 Key Parameters', fontsize=16, fontweight='bold', pad=20)

# Adjust plot viewing angle for better clarity
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.show()