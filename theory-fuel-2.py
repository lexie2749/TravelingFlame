import numpy as np
import matplotlib.pyplot as plt

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Physical Parameters for 10-15 rad/s range ==============
T_u = 300  # Unburned gas temperature (K)
T_b = 1900  # Burned gas temperature (K)
E_a = 110000  # Activation energy (J/mol)
R_u = 8.314  # Universal gas constant (J/mol·K)

# Transport properties
lambda_th = 0.026  # Thermal conductivity (W/m·K)
rho = 1.2  # Density (kg/m³)
c_p = 1000  # Specific heat capacity (J/kg·K)
D_m = 2.5e-5  # Mass diffusivity (m²/s)

# Calculate derived parameters
alpha_th = lambda_th / (rho * c_p)
Le = alpha_th / D_m
sigma = T_b / T_u
beta = E_a * (T_b - T_u) / (R_u * T_b**2)
Ma = ((sigma - 1) / (2 * sigma)) * np.log(sigma) + (beta * (Le - 1)) / (2 * sigma)

# ADJUSTED for proper angular velocity range
s_L = 0.1  # m/s (32 cm/s)
delta_L = 0.35e-3  # Flame thickness (0.35 mm)

# ============== Geometric Parameters ==============
ring_diameter = 100  # mm
ring_radius = ring_diameter / 2  # mm

# ============== Model Functions for 10-15 rad/s ==============
def volume_to_height(volume_ul, channel_area):
    """Convert fuel volume (μL) to fuel height (mm)"""
    volume_mm3 = volume_ul  # 1 μL = 1 mm³
    height_mm = volume_mm3 / channel_area
    return height_mm

def calculate_acceleration_factor(BR, fuel_height_mm):
    """
    Acceleration factor calibrated for 10-15 rad/s range
    """
    if BR < 1:
        base_factor = BR**0.85
    else:
        # Increased coefficient to make the slope steeper
        base_factor = 1 + 4.5 * (1 - np.exp(-0.4 * (BR - 1)))
    
    if fuel_height_mm < 0.15:
        height_factor = 1 + 0.3 * (0.15 - fuel_height_mm) / 0.15
    else:
        # Adjusted for thicker layers
        height_factor = 1 - 0.8 * (fuel_height_mm - 0.15) / 0.15
    
    return base_factor * height_factor

def calculate_flame_speed(ring_radius_m, channel_diameter_m, fuel_height_m, s_L, delta_L, Ma):
    """Calculate flame speed for target angular velocity range"""
    fuel_height_mm = fuel_height_m * 1000
    
    curvature_factor = 1 - Ma * delta_L / ring_radius_m
    curvature_factor = np.clip(curvature_factor, 0.95, 1.1)
    
    BR = channel_diameter_m / fuel_height_m if fuel_height_m > 0 else 0
    
    acceleration_factor = calculate_acceleration_factor(BR, fuel_height_mm)
    
    v = s_L * curvature_factor * acceleration_factor
    return v

def calculate_angular_velocity(flame_speed, ring_radius_m):
    """Calculate angular velocity ω = v/r"""
    return flame_speed / ring_radius_m

# ============== Experimental Data ==============
# The data is manually extracted from the provided image
experimental_data = {
    3: [
        {'volume': 125, 'omega': 12.35, 'error': 2.04}, {'volume': 130, 'omega': 10.55, 'error': 1.23},
        {'volume': 135, 'omega': 17.40, 'error': 5.62}, {'volume': 140, 'omega': 7.84, 'error': 9.27},
        {'volume': 150, 'omega': 15.44, 'error': 9.58}, {'volume': 155, 'omega': 12.60, 'error': 0.99},
        {'volume': 160, 'omega': 16.25, 'error': 2.96}, {'volume': 165, 'omega': 12.17, 'error': 1.12},
        {'volume': 170, 'omega': 8.18, 'error': 2.02}, {'volume': 175, 'omega': 13.25, 'error': 4.50},
        {'volume': 180, 'omega': 13.01, 'error': 1.10}, {'volume': 185, 'omega': 13.14, 'error': 1.10},
        {'volume': 190, 'omega': 14.15, 'error': 1.41}, {'volume': 195, 'omega': 6.94, 'error': 1.41},
        {'volume': 200, 'omega': 15.36, 'error': 0.43}
    ],
    4: [
        {'volume': 125, 'omega': 16.63, 'error': 7.42}, {'volume': 130, 'omega': 14.19, 'error': 5.31},
        {'volume': 135, 'omega': 12.93, 'error': 1.83}, {'volume': 140, 'omega': 13.42, 'error': 2.12},
        {'volume': 145, 'omega': 11.73, 'error': 3.38}, {'volume': 150, 'omega': 12.20, 'error': 1.62},
        {'volume': 155, 'omega': 11.57, 'error': 0.94}, {'volume': 160, 'omega': 12.10, 'error': 1.63},
        {'volume': 165, 'omega': 12.51, 'error': 1.29}, {'volume': 170, 'omega': 12.12, 'error': 4.54},
        {'volume': 175, 'omega': 1.24, 'error': 9.92}, {'volume': 180, 'omega': 8.45, 'error': 2.79},
        {'volume': 185, 'omega': 11.20, 'error': 1.38}, {'volume': 190, 'omega': 13.21, 'error': 1.59},
        {'volume': 195, 'omega': 15.14, 'error': 1.97}, {'volume': 200, 'omega': 15.54, 'error': 0.07}
    ],
    5: [
        {'volume': 125, 'omega': 10.27, 'error': 1.26}, {'volume': 130, 'omega': 1.08, 'error': 9.22},
        {'volume': 140, 'omega': 13.66, 'error': 0.59}, {'volume': 145, 'omega': 9.85, 'error': 0.99},
        {'volume': 150, 'omega': 12.13, 'error': 1.79}, {'volume': 155, 'omega': 9.84, 'error': 0.55},
        {'volume': 160, 'omega': 14.05, 'error': 3.17}, {'volume': 165, 'omega': 16.31, 'error': 0.41},
        {'volume': 170, 'omega': 10.86, 'error': 1.92}, {'volume': 180, 'omega': 9.26, 'error': 1.39},
        {'volume': 185, 'omega': 12.13, 'error': 1.49}, {'volume': 190, 'omega': 11.54, 'error': 1.56},
        {'volume': 195, 'omega': 13.18, 'error': 2.44}, {'volume': 200, 'omega': 13.12, 'error': 1.41}
    ],
    6: [
        {'volume': 125, 'omega': 10.84, 'error': 1.55}, {'volume': 130, 'omega': 12.27, 'error': 2.08},
        {'volume': 135, 'omega': 11.95, 'error': 1.61}, {'volume': 140, 'omega': 10.17, 'error': 0.72},
        {'volume': 145, 'omega': 12.15, 'error': 1.19}, {'volume': 150, 'omega': 11.08, 'error': 1.90},
        {'volume': 155, 'omega': 11.62, 'error': 2.53}, {'volume': 160, 'omega': 11.85, 'error': 1.55},
        {'volume': 165, 'omega': 10.40, 'error': 3.73}, {'volume': 170, 'omega': 8.99, 'error': 6.85},
        {'volume': 175, 'omega': 11.10, 'error': 3.88}, {'volume': 180, 'omega': 8.40, 'error': 1.71},
        {'volume': 185, 'omega': 8.40, 'error': 1.71}, {'volume': 190, 'omega': 12.88, 'error': 0.98},
        {'volume': 195, 'omega': 18.99, 'error': 2.01}, {'volume': 200, 'omega': 11.07, 'error': 2.30}
    ],
    7: [
        {'volume': 125, 'omega': 13.38, 'error': 0.16}, {'volume': 130, 'omega': 10.89, 'error': 1.86},
        {'volume': 135, 'omega': 12.27, 'error': 3.54}, {'volume': 140, 'omega': 7.86, 'error': 0.70},
        {'volume': 145, 'omega': 9.75, 'error': 2.42}, {'volume': 150, 'omega': 13.22, 'error': 1.09},
        {'volume': 155, 'omega': 9.35, 'error': 1.13}, {'volume': 160, 'omega': 6.42, 'error': 3.74},
        {'volume': 165, 'omega': 16.62, 'error': 0.69}, {'volume': 170, 'omega': 11.41, 'error': 2.08},
        {'volume': 175, 'omega': 12.99, 'error': 1.68}, {'volume': 180, 'omega': 10.02, 'error': 1.85},
        {'volume': 185, 'omega': 13.04, 'error': 2.98}, {'volume': 190, 'omega': 14.32, 'error': 2.64},
        {'volume': 200, 'omega': 10.99, 'error': 3.73}
    ],
    8: [
        {'volume': 130, 'omega': 13.08, 'error': 2.34}, {'volume': 135, 'omega': 11.59, 'error': 6.96},
        {'volume': 140, 'omega': 11.44, 'error': 5.63}, {'volume': 145, 'omega': 22.34, 'error': 0.18},
        {'volume': 155, 'omega': 9.77, 'error': 1.83}, {'volume': 160, 'omega': 18.99, 'error': 0.89},
        {'volume': 175, 'omega': 8.76, 'error': 0.61}, {'volume': 180, 'omega': 18.29, 'error': 2.41},
        {'volume': 195, 'omega': 14.31, 'error': 2.36}, {'volume': 200, 'omega': 14.31, 'error': 2.36}
    ],
    9: [
        {'volume': 130, 'omega': 11.05, 'error': 1.83}, {'volume': 135, 'omega': 10.95, 'error': 1.28},
        {'volume': 140, 'omega': 7.26, 'error': 1.46}, {'volume': 145, 'omega': 15.55, 'error': 1.29},
        {'volume': 150, 'omega': 15.47, 'error': 1.34}, {'volume': 155, 'omega': 7.05, 'error': 2.25},
        {'volume': 160, 'omega': 14.83, 'error': 2.52}, {'volume': 175, 'omega': 12.82, 'error': 4.27},
        {'volume': 180, 'omega': 15.39, 'error': 1.66}, {'volume': 185, 'omega': 15.75, 'error': 0.16},
        {'volume': 200, 'omega': 16.82, 'error': 1.87}
    ],
    10: [
        {'volume': 135, 'omega': 8.09, 'error': 0.44}, {'volume': 140, 'omega': 15.46, 'error': 1.06},
        {'volume': 145, 'omega': 7.92, 'error': 1.94}, {'volume': 150, 'omega': 18.23, 'error': 1.20},
        {'volume': 160, 'omega': 14.18, 'error': 2.91}, {'volume': 165, 'omega': 9.93, 'error': 2.25},
        {'volume': 175, 'omega': 8.94, 'error': 0.20}, {'volume': 190, 'omega': 18.42, 'error': 3.62},
        {'volume': 200, 'omega': 11.74, 'error': 0.70}
    ]
}

# Define the range of channel widths to plot
channel_widths_mm = np.arange(3, 11, 1)

# ---- Create a plot for each channel width ----
for channel_width in channel_widths_mm:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate the area and dimensions
    circumference = 2 * np.pi * ring_radius  # mm
    channel_area = circumference * channel_width  # mm²
    
    # Convert to meters
    ring_radius_m = ring_radius * 1e-3
    channel_diameter_m = channel_width * 1e-3

    # Define the fuel volume range for the theoretical plot
    fuel_volume_range = np.linspace(125, 200, 100)
    angular_velocities = []

    for vol in fuel_volume_range:
        height_mm = volume_to_height(vol, channel_area)
        height_m = height_mm * 1e-3
        
        v = calculate_flame_speed(ring_radius_m, channel_diameter_m, height_m, s_L, delta_L, Ma)
        omega = calculate_angular_velocity(v, ring_radius_m)
        angular_velocities.append(omega)

    # Plot the theoretical prediction
    ax.plot(fuel_volume_range, angular_velocities, 'b-', linewidth=3,
            label='Theoretical prediction', zorder=3)

    # Add uncertainty band (±10%)
    omega_upper = np.array(angular_velocities) * 1.10
    omega_lower = np.array(angular_velocities) * 0.90
    ax.fill_between(fuel_volume_range, omega_lower, omega_upper,
                    color='blue', alpha=0.15, zorder=1)
    
    # Plot experimental data if available for the current channel width
    if channel_width in experimental_data:
        exp_data = experimental_data[channel_width]
        exp_volumes = [d['volume'] for d in exp_data]
        exp_omegas = [d['omega'] for d in exp_data]
        exp_errors = [d['error'] for d in exp_data]
        
        ax.errorbar(exp_volumes, exp_omegas, yerr=exp_errors, fmt='o', capsize=5,
                    label='Experimental data', color='red', zorder=4)

    # Set plot labels and title
    ax.set_xlabel('Fuel Volume (μL)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=13, fontweight='bold')
    ax.set_title(f'Angular Velocity vs Fuel Volume\n(Channel Width: {channel_width} mm)',
                 fontsize=14, fontweight='bold')
    
    # Add horizontal reference lines for the target range
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Target Range')
    ax.axhline(y=15, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Re-add legend to include all labels
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(120, 205)
    ax.set_ylim(5, 25)

# Display all plots
plt.tight_layout()
plt.show()
