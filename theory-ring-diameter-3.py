import numpy as np
import matplotlib.pyplot as plt

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Physical Parameters ==============
# Base parameters for liquid fuel
s_L = 0.1  # Base laminar flame speed (m/s) for liquid fuel
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Define ring diameter range
ring_diameters_mm = np.linspace(30, 200, 200)
ring_radii_m = ring_diameters_mm * 1e-3 / 2

# Define BR values to plot
BR_values = [0.5, 0.8, 1.0, 1.5, 2.0]
colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1', '#96CEB4']

# Uncertainty factor (±20% for theoretical variation)
uncertainty = 0.2

# ---- Plot 1: Angular Velocity vs Ring Diameter with BR effect ----
for i, BR in enumerate(BR_values):
    angular_velocities = []
    angular_velocities_upper = []
    angular_velocities_lower = []
    
    for r in ring_radii_m:
        v = calculate_flame_speed(r, BR)
        omega = calculate_angular_velocity(v, r)
        angular_velocities.append(omega)
        
        # Calculate uncertainty bounds
        omega_upper = omega * (1 + uncertainty)
        omega_lower = omega * (1 - uncertainty)
        angular_velocities_upper.append(omega_upper)
        angular_velocities_lower.append(omega_lower)
    
    # Plot main line
    ax1.plot(ring_diameters_mm, angular_velocities, 
             linewidth=2.5, color=colors[i], 
             label=f'BR = {BR:.1f}', alpha=0.9)
    
    # Plot uncertainty bounds
    ax1.fill_between(ring_diameters_mm, 
                     angular_velocities_lower, 
                     angular_velocities_upper,
                     color=colors[i], alpha=0.15)
    
    # Add dashed lines for bounds (only for BR=1.0 for clarity)
    if BR == 1.0:
        ax1.plot(ring_diameters_mm, angular_velocities_upper, 
                ':', color=colors[i], linewidth=1, alpha=0.5)
        ax1.plot(ring_diameters_mm, angular_velocities_lower, 
                ':', color=colors[i], linewidth=1, alpha=0.5)

ax1.set_xlabel('Ring Diameter (mm)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')
ax1.set_title('Angular Velocity vs Ring Diameter\n(Different Blockage Ratios with ±20% Uncertainty)',
             fontsize=15, fontweight='bold', pad=20)
ax1.legend(loc='upper right', frameon=True, fontsize=11, 
          title='Blockage Ratio', title_fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xlim(30, 200)
ax1.set_ylim(0, 10)

# Add annotations
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax1.text(35, 1.2, '1 rad/s', fontsize=9, color='gray')
ax1.text(35, 5.2, '5 rad/s', fontsize=9, color='gray')

# Add physical insight box
insight_text = ('Physical Insights:\n'
               '• BR < 1: Deceleration\n'
               '• BR = 1: Neutral\n'  
               '• BR > 1: Acceleration\n'
               '• Smaller rings → Higher ω')
ax1.text(0.02, 0.98, insight_text, 
        transform=ax1.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 alpha=0.9, edgecolor='gray'),
        verticalalignment='top', horizontalalignment='left')

# ---- Plot 2: Acceleration Factor Model ----
BR_range = np.linspace(0.3, 3.0, 100)
acc_factors = [calculate_acceleration_factor(br) for br in BR_range]

# Plot theoretical model
ax2.plot(BR_range, acc_factors, 'b-', linewidth=3, 
         label='Theoretical Model', alpha=0.8)

# Add uncertainty band
acc_upper = np.array(acc_factors) * 1.2
acc_lower = np.array(acc_factors) * 0.8
ax2.fill_between(BR_range, acc_lower, acc_upper, 
                 color='blue', alpha=0.1)
ax2.plot(BR_range, acc_upper, 'b:', linewidth=1, alpha=0.5)
ax2.plot(BR_range, acc_lower, 'b:', linewidth=1, alpha=0.5)

# Reference lines
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax2.set_xlabel('Blockage Ratio (BR = D/H)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Acceleration Factor', fontsize=14, fontweight='bold')
ax2.set_title('Acceleration Factor vs Blockage Ratio\n(Theoretical Model)',
             fontsize=15, fontweight='bold', pad=20)
ax2.legend(loc='upper left', frameon=True, fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xlim(0.3, 3.0)
ax2.set_ylim(0.4, 1.8)

# Add regions
ax2.fill_between([0.3, 1.0], [0.4, 0.4], [1.8, 1.8], 
                color='red', alpha=0.05)
ax2.fill_between([1.0, 3.0], [0.4, 0.4], [1.8, 1.8], 
                color='green', alpha=0.05)
ax2.text(0.65, 1.6, 'Deceleration\nRegion', fontsize=10, 
        ha='center', color='red', alpha=0.7)
ax2.text(2.0, 1.6, 'Acceleration\nRegion', fontsize=10, 
        ha='center', color='green', alpha=0.7)

# Add formula box
formula_text = ('Acceleration Model:\n'
               'BR < 1: factor = BR^0.7\n'
               'BR ≥ 1: factor = 1 + 0.6(1-e^{-0.8(BR-1)})')
ax2.text(0.98, 0.02, formula_text, 
        transform=ax2.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                 alpha=0.3, edgecolor='gray'),
        verticalalignment='bottom', horizontalalignment='right')

plt.suptitle('Effect of Ring Diameter and Blockage Ratio on Flame Propagation', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ============== Quantitative Analysis ==============
print("=" * 70)
print("QUANTITATIVE ANALYSIS: Ring Diameter Effect for Different BR Values")
print("=" * 70)
print()

# Select specific ring diameters for comparison
test_diameters = [50, 100, 150]
print(f"{'BR':<6} {'Ring Diameter (mm)':<20} {'ω (rad/s)':<12} {'Period (s)':<12} {'RPM':<10}")
print("-" * 70)

for BR in BR_values:
    for d_mm in test_diameters:
        r_m = d_mm * 1e-3 / 2
        v = calculate_flame_speed(r_m, BR)
        omega = calculate_angular_velocity(v, r_m)
        period = 2 * np.pi / omega
        rpm = omega * 60 / (2 * np.pi)
        
        if d_mm == test_diameters[0]:  # Only print BR once per group
            print(f"{BR:<6.1f} {d_mm:<20} {omega:<12.2f} {period:<12.2f} {rpm:<10.1f}")
        else:
            print(f"{'':^6} {d_mm:<20} {omega:<12.2f} {period:<12.2f} {rpm:<10.1f}")
    print()

print("=" * 70)
print("KEY FINDINGS:")
print("=" * 70)
print("1. Angular velocity shows inverse relationship with ring diameter (ω ∝ 1/R)")
print("2. BR < 1: Flame decelerates compared to base speed")
print("3. BR = 1: Near neutral effect") 
print("4. BR > 1: Flame accelerates, but saturates for large BR")
print("5. Typical range: 1-8 rad/s for practical configurations")
print("6. Uncertainty of ±20% represents theoretical model uncertainty")
print()

# Calculate sensitivity
d_test = 100  # mm
r_test = d_test * 1e-3 / 2
omega_br1 = calculate_angular_velocity(calculate_flame_speed(r_test, 1.0), r_test)
omega_br2 = calculate_angular_velocity(calculate_flame_speed(r_test, 2.0), r_test)
sensitivity = (omega_br2 - omega_br1) / omega_br1 * 100

print(f"Sensitivity Analysis (Ring = 100mm):")
print(f"  BR = 1.0 → ω = {omega_br1:.2f} rad/s")
print(f"  BR = 2.0 → ω = {omega_br2:.2f} rad/s")
print(f"  Relative change: {sensitivity:.1f}%")