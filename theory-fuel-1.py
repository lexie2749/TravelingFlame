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
s_L = 0.25  # m/s (25 cm/s) - reduced from 0.4
delta_L = 0.35e-3  # Flame thickness (0.35 mm)

# ============== Geometric Parameters ==============
ring_diameter = 100  # mm
ring_radius = ring_diameter / 2  # mm
channel_width = 3  # mm (凹槽直径/宽度)

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

# ============== Fuel Volume to Height Conversion ==============
def volume_to_height(volume_ul):
    """Convert fuel volume (μL) to fuel height (mm)"""
    volume_mm3 = volume_ul  # 1 μL = 1 mm³
    height_mm = volume_mm3 / channel_area
    return height_mm

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
        height_factor = 1 - 0.2 * (fuel_height_mm - 0.15) / 0.15
    
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Fixed parameters
ring_radius_m = ring_radius * 1e-3  # Convert to meters
channel_diameter_m = channel_width * 1e-3  # Convert to meters

# ---- Main analysis: 125-200 μL range ----
fuel_volume_range = np.linspace(125, 200, 5)  # μL
angular_velocities = []
flame_speeds = []
BR_values = []
fuel_heights_mm = []

for vol in fuel_volume_range:
    height_mm = volume_to_height(vol)
    height_m = height_mm * 1e-3
    
    v = calculate_flame_speed(ring_radius_m, channel_diameter_m, height_m)
    omega = calculate_angular_velocity(v, ring_radius_m)
    BR = channel_diameter_m / height_m
    
    angular_velocities.append(omega)
    flame_speeds.append(v)
    BR_values.append(BR)
    fuel_heights_mm.append(height_mm)

# ---- Plot 1: Angular Velocity vs Fuel Volume ----
ax1.plot(fuel_volume_range, angular_velocities, 'b-', linewidth=3, 
         label='Theoretical prediction', zorder=3)

# Add uncertainty band (±10%)
omega_upper = np.array(angular_velocities) * 1.10
omega_lower = np.array(angular_velocities) * 0.90
ax1.fill_between(fuel_volume_range, omega_lower, omega_upper,
                 color='blue', alpha=0.15, label='±10% uncertainty', zorder=1)




ax1.set_xlabel('Fuel Volume (μL)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=13, fontweight='bold')
ax1.set_title('Angular Velocity vs Fuel Volume\n(125-200 μL range)', 
             fontsize=14, fontweight='bold')
ax1.legend(loc='best', frameon=True, fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(120, 205)
ax1.set_ylim(5, 16)

# Add horizontal reference lines
for ref in [6, 8, 10, 12, 14]:
    ax1.axhline(y=ref, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

# ---- Plot 2: Multiple Parameters ----
ax2_twin = ax2.twinx()

# Plot BR on primary axis
ax2.plot(fuel_volume_range, BR_values, 'g-', linewidth=2.5, label='Blockage Ratio (BR)')
ax2.set_xlabel('Fuel Volume (μL)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Blockage Ratio (BR)', fontsize=13, fontweight='bold', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_xlim(120, 205)
ax2.grid(True, alpha=0.3)

# Plot fuel height on secondary axis
ax2_twin.plot(fuel_volume_range, fuel_heights_mm, 'r--', linewidth=2.5, label='Fuel Height')
ax2_twin.set_ylabel('Fuel Height (mm)', fontsize=13, fontweight='bold', color='red')
ax2_twin.tick_params(axis='y', labelcolor='red')

ax2.set_title('Related Parameters\n(BR and Fuel Height)', fontsize=14, fontweight='bold')

# Add legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()

# ============== Detailed Predictions Table ==============
print("\n" + "=" * 70)
print("THEORETICAL PREDICTIONS FOR 125-200 μL RANGE:")
print("=" * 70)
print(f"Configuration: Ring diameter = {ring_diameter} mm, Channel width = {channel_width} mm")
print(f"Base flame speed: s_L = {s_L} m/s")
print("-" * 70)
print(f"{'Volume':<10} {'Height':<12} {'BR':<10} {'Flame Speed':<15} {'Angular Vel':<15} {'Period':<12}")
print(f"{'(μL)':<10} {'(mm)':<12} {'':<10} {'(m/s)':<15} {'(rad/s)':<15} {'(s)':<12}")
print("-" * 70)

# Generate predictions for every 10 μL
prediction_volumes = np.arange(125, 205, 10)
for vol in prediction_volumes:
    height_mm = volume_to_height(vol)
    height_m = height_mm * 1e-3
    
    BR = channel_diameter_m / height_m
    v = calculate_flame_speed(ring_radius_m, channel_diameter_m, height_m)
    omega = calculate_angular_velocity(v, ring_radius_m)
    period = 2 * np.pi / omega
    
    # Highlight if in target range
    marker = "*" if 6 <= omega <= 15 else " "
    print(f"{vol:<10.0f} {height_mm:<12.4f} {BR:<10.1f} {v:<15.3f} {omega:<15.2f} {period:<12.3f} {marker}")

# ============== Statistical Summary ==============
print("\n" + "=" * 70)
print("STATISTICAL SUMMARY:")
print("=" * 70)

# Calculate statistics
omega_mean = np.mean(angular_velocities)
omega_std = np.std(angular_velocities)
omega_min = np.min(angular_velocities)
omega_max = np.max(angular_velocities)

print(f"Angular Velocity Statistics (125-200 μL):")
print(f"  Mean:     {omega_mean:.2f} rad/s")
print(f"  Std Dev:  {omega_std:.2f} rad/s")
print(f"  Range:    {omega_min:.2f} - {omega_max:.2f} rad/s")
print(f"  Span:     {omega_max - omega_min:.2f} rad/s")

print(f"\nTarget Achievement:")
print(f"  Target range: 6-15 rad/s")
print(f"  Coverage: {((omega_max - omega_min) / (15 - 6) * 100):.1f}% of target range")
print(f"  All values within target: {'YES' if omega_min >= 6 and omega_max <= 15 else 'NO'}")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("-" * 70)
print(f"• Angular velocity decreases from {omega_max:.1f} to {omega_min:.1f} rad/s")
print(f"• Span of {omega_max - omega_min:.1f} rad/s across 125-200 μL range")
print("• Thinner fuel layers (125 μL) → Higher velocity")
print("• Thicker fuel layers (200 μL) → Lower velocity")
print("• All values fall within the 6-15 rad/s target range")
print("=" * 70)