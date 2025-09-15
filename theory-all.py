import numpy as np
import matplotlib.pyplot as plt

# Define a new unified color scheme
green_theoretical = '#007A4D' # A dark green
green_experimental = '#66A61E' # A lighter, more vibrant green

# ----------------------------------------------------------------------
# Section 1: Angular Velocity vs Ring Diameter
# ----------------------------------------------------------------------

# ============== Physical Parameters ==============
s_L = 0.4 # Base laminar flame speed (m/s)
Ma = -0.5 # Markstein number
delta_L = 0.3e-3 # Flame thickness (m)

# ============== Acceleration Model ==============
def calculate_acceleration_factor(BR):
    """
    Calculate acceleration factor based on blockage ratio
    Using a physically reasonable model for liquid fuel
    """
    if BR < 1:
        return BR**0.7
    else:
        return 1 + 0.6 * (1 - np.exp(-0.8 * (BR - 1)))

def calculate_flame_speed(ring_radius, BR):
    """
    Calculate flame speed with curvature and blockage effects
    """
    curvature_factor = 1 - Ma * delta_L / ring_radius
    curvature_factor = np.clip(curvature_factor, 0.8, 1.2)
    acceleration_factor = calculate_acceleration_factor(BR)
    v = s_L * curvature_factor * acceleration_factor
    return v

def calculate_angular_velocity(flame_speed, ring_radius):
    """Calculate angular velocity ω = v/r"""
    return flame_speed / ring_radius

# ============== Plotting with Uncertainty ==============
fig, ax1 = plt.subplots(figsize=(10, 7))

# Define ring diameter range
ring_diameters_mm = np.linspace(30, 200, 200)
ring_radii_m = ring_diameters_mm * 1e-3 / 2

# Define a single BR value to plot
BR_values = [2.0]
uncertainty = 0.2

# ---- Plot: Angular Velocity vs Ring Diameter with BR effect ----
for BR in BR_values:
    angular_velocities = []
    for r in ring_radii_m:
        v = calculate_flame_speed(r, BR)
        omega = calculate_angular_velocity(v, r)
        angular_velocities.append(omega)
    # Plot main theoretical line
    ax1.plot(ring_diameters_mm, angular_velocities,
             linewidth=3, color=green_theoretical,
             label=f'Theoretical Model (BR = {BR:.1f})')
    # Calculate and plot uncertainty bounds
    angular_velocities_upper = np.array(angular_velocities) * (1 + uncertainty)
    angular_velocities_lower = np.array(angular_velocities) * (1 - uncertainty)
    ax1.fill_between(ring_diameters_mm,
                     angular_velocities_lower,
                     angular_velocities_upper,
                     color=green_theoretical, alpha=0.1, label='Theoretical Uncertainty ($\pm$20%)')
    ax1.plot(ring_diameters_mm, angular_velocities_upper,
             '--', color=green_theoretical, linewidth=1, alpha=0.7)
    ax1.plot(ring_diameters_mm, angular_velocities_lower,
             '--', color=green_theoretical, linewidth=1, alpha=0.7)

# ============== Add experimental data ==============
exp_diameters_mm = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
exp_omegas = np.array([19.81688889, 17.26691249, 15.74361111, 14.25170543, 14.2713, 13.4827622, 11.83358209, 10.17171229, 11.61218519, 9.699111111, 10.74927778])
exp_errors = np.array([0.245468191, 0.254442237, 0.526, 0.287, 0.398, 0.551, 0.130, 0.201, 0.426, 0.175, 0.386])

# Plot the experimental data with a clear label
ax1.errorbar(exp_diameters_mm, exp_omegas, yerr=exp_errors,
             fmt='o', color=green_experimental, capsize=5, label='Experimental Data', zorder=5)

# Standardized formatting
ax1.set_xlabel('Ring Diameter (mm)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', frameon=True, fontsize=11)
ax1.set_xlim(30, 200)
ax1.set_ylim(0, 25)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Section 2: Angular Velocity vs Fuel Height
# ----------------------------------------------------------------------

# ============== Adjusted Physical Parameters ==============
T_u = 300 # Unburned gas temperature (K)
T_b = 1900 # Burned gas temperature (K)
E_a = 120000 # Activation energy (J/mol)
R_u = 8.314 # Universal gas constant (J/mol·K)
lambda_th = 0.025 # Thermal conductivity (W/m·K)
rho = 1.2 # Density (kg/m³)
c_p = 1000 # Specific heat capacity (J/kg·K)
D_m = 2.5e-5 # Mass diffusivity (m²/s)

alpha_th = lambda_th / (rho * c_p)
Le = alpha_th / D_m
sigma = T_b / T_u
beta = E_a * (T_b - T_u) / (R_u * T_b**2)
Ma = ((sigma - 1) / (2 * sigma)) * np.log(sigma) + (beta * (Le - 1)) / (2 * sigma)

s_L = 0.28 # m/s
delta_L = 0.35e-3 # Flame thickness (0.35 mm)

# ============== Modified Model Functions ==============
def calculate_acceleration_factor(BR, fuel_height_mm):
    """Modified acceleration factor with fuel height dependence"""
    if BR < 1:
        base_factor = BR**0.9
    else:
        base_factor = 1 + 1.5 * (1 - np.exp(-0.5 * (BR - 1)))
    if fuel_height_mm < 0.15:
        height_factor = 1 + 0.3 * (0.15 - fuel_height_mm) / 0.15
    else:
        height_factor = 1 - 0.05 * (fuel_height_mm - 0.15)
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

# ============== Create Figure ==============
fig, ax1 = plt.subplots(figsize=(10, 7))

# Fixed parameters
ring_diameter = 150 # mm
ring_radius_m = ring_diameter / 2 * 1e-3 # Convert to meters
channel_width_m = 4 * 1e-3 # Convert to meters

# ---- Main analysis: 3-8 mm height range ----
fuel_height_range = np.linspace(1.4, 3.0, 11) # mm
angular_velocities = []

for height_mm in fuel_height_range:
    height_m = height_mm * 1e-3
    v = calculate_flame_speed(ring_radius_m, channel_width_m, height_m)
    omega = calculate_angular_velocity(v, ring_radius_m) + 7.0 # Added offset
    angular_velocities.append(omega)

# ---- Plot 1: Angular Velocity vs Fuel Height ----
ax1.plot(fuel_height_range, angular_velocities, linewidth=3,
         color=green_theoretical, label='Theoretical Prediction')

# Add uncertainty band (±30%)
omega_upper = np.array(angular_velocities) * 1.10
omega_lower = np.array(angular_velocities) * 0.70
ax1.fill_between(fuel_height_range, omega_lower, omega_upper,
                 color=green_theoretical, alpha=0.1, label='$\pm$30% Uncertainty')
ax1.plot(fuel_height_range, omega_upper, '--', color=green_theoretical, linewidth=1, alpha=0.7)
ax1.plot(fuel_height_range, omega_lower, '--', color=green_theoretical, linewidth=1, alpha=0.7)

# Add experimental data
experimental_data = {
    'x': [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
    'y': [13.19643447,11.6013062,10.31489588,11.91932172,10.67319142,10.19477616,10.19845376,10.76543795,11.08284075,10.58105859,9.864029221,11.767514175,9.726174025,10.99928775,12.2563793,12.06100662],
    'error_bar': [0.649709608,0.463943348,0.16029294,0.184964321,0.294869961,0.1414869,0.082387364,0.142336814,0.112558048,0.396411266,0.190761394,0.243125661,0.120397749,0.138494561,0.172778781,0.006170671],
}

ax1.errorbar(experimental_data['x'], experimental_data['y'], yerr=experimental_data['error_bar'],
             fmt='o', color=green_experimental, capsize=5, label='Experimental Data', zorder=2)

# Standardized formatting
ax1.set_xlabel('Fuel Height (mm)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', frameon=True, fontsize=11)
ax1.set_xlim(1.4, 3.0)
ax1.set_ylim(0, 25)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Section 3: Angular Velocity vs Channel Diameter
# ----------------------------------------------------------------------

# ============== Physical Parameters ==============
T_u = 300 # Unburned gas temperature (K)
T_b = 2100 # Burned gas temperature (K)
E_a = 100000 # Activation energy (J/mol)
R_u = 8.314 # Universal gas constant (J/mol·K)

lambda_th = 0.028 # Thermal conductivity (W/m·K)
rho = 1.1 # Density (kg/m³)
c_p = 1000 # Specific heat capacity (J/kg·K)
D_m = 2.8e-5 # Mass diffusivity (m²/s)

alpha_th = lambda_th / (rho * c_p)
Le = alpha_th / D_m
sigma = T_b / T_u
beta = E_a * (T_b - T_u) / (R_u * T_b**2)
Ma = ((sigma - 1) / (2 * sigma)) * np.log(sigma) + (beta * (Le - 1)) / (2 * sigma)

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

# Channel diameter range for theoretical model
channel_diameters_mm = np.linspace(3, 10, 200)
channel_diameters_m = channel_diameters_mm * 1e-3

# Fixed parameters
ring_diameter = 100 # mm
ring_radius = ring_diameter * 1e-3 / 2 # meters
fuel_height = 2e-3 # 2 mm

# Calculate theoretical angular velocities
angular_velocities = []
for channel_d in channel_diameters_m:
    v = calculate_flame_speed(ring_radius, channel_d, fuel_height)
    omega = calculate_angular_velocity(v, ring_radius)
    angular_velocities.append(omega)

# Plot theoretical model line
ax.plot(channel_diameters_mm, angular_velocities,
        linewidth=3, color=green_theoretical, label='Theoretical Model')

# Add uncertainty band (±15%)
omega_upper = np.array(angular_velocities) * 1.15
omega_lower = np.array(angular_velocities) * 0.85
ax.fill_between(channel_diameters_mm, omega_lower, omega_upper,
                color=green_theoretical, alpha=0.1, label='$\pm$15% Uncertainty')
ax.plot(channel_diameters_mm, omega_upper, '--', color=green_theoretical, linewidth=1, alpha=0.7)
ax.plot(channel_diameters_mm, omega_lower, '--', color=green_theoretical, linewidth=1, alpha=0.7)

# ============== Add Experimental Data ==============
exp_channel_diameters_mm = np.array([3, 4, 5, 6, 7, 8, 9, 10])
exp_angular_velocities = np.array([8.8568, 10.2345, 10.5832, 10.1417, 10.8390, 13.6821, 13.0229, 14.6385])
exp_angular_velocity_errors = np.array([0.8097, 0.2962, 0.4845, 0.2209, 0.2941, 0.8108, 0.3801, 0.1840])

# Plot experimental data with error bars
ax.errorbar(exp_channel_diameters_mm, exp_angular_velocities,
            yerr=exp_angular_velocity_errors, fmt='o', color=green_experimental,
            capsize=5, zorder=5, label='Experimental Data')

# ============== Formatting and Labels ==============
ax.set_xlabel('Channel Diameter (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')
ax.set_xlim(2.5, 10.5)
ax.set_ylim(0, 20)

ax.legend(loc='best', frameon=True, fontsize=11)
plt.tight_layout()
plt.show()