import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Physical Parameters ---
# We'll assume some typical values for a hydrocarbon flame.
s_L = 0.4  # Ideal laminar flame speed (m/s)
delta_L = 0.0002  # Characteristic flame thickness (m, e.g., 0.2 mm)

# Define the three cases for the Markstein number based on Lewis number
Ma_negative = -1.5  # Case 1: Ma < 0 (corresponds to Le < 1, e.g., lean H2 flame)
Ma_positive = 1.5   # Case 2: Ma > 0 (corresponds to Le > 1, e.g., lean propane flame)
Ma_zero = 0.0       # Case 3: Ma = 0 (corresponds to Le = 1)

# --- 2. Generate Data ---
# Create a range of values for the Radius of Curvature, R.
# We'll go from 0.5 mm to 5 mm to see the effect at small radii.
R = np.linspace(0.0005, 0.005, 400) # Radius of Curvature in meters

# Calculate the observed flame speed (s_p) for each case using the formula
# s_p = s_L * (1 - Ma * delta_L / R)
s_p_negative_Ma = s_L * (1 - Ma_negative * delta_L / R)
s_p_positive_Ma = s_L * (1 - Ma_positive * delta_L / R)
s_p_zero_Ma = s_L * (1 - Ma_zero * delta_L / R) # This will be a constant line at s_L

# --- 3. Create the Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the ideal, unstretched flame speed as a reference line
ax.axhline(y=s_L, color='black', linestyle='--', label=f'$s_L$ = {s_L} m/s (Unstretched Speed)')

# Plot the results for the three cases
ax.plot(R * 1000, s_p_negative_Ma, color='royalblue', linewidth=2.5, label='$Ma < 0$ (e.g., $Le < 1$) - Stretch Helps')
ax.plot(R * 1000, s_p_positive_Ma, color='firebrick', linewidth=2.5, label='$Ma > 0$ (e.g., $Le > 1$) - Stretch Hurts')
ax.plot(R * 1000, s_p_zero_Ma, color='forestgreen', linewidth=2.5, label='$Ma = 0$ (e.g., $Le = 1$) - Neutral')

# --- 4. Formatting the Plot ---
# Add labels and a title
ax.set_xlabel('Radius of Curvature, R (mm)', fontsize=12)
ax.set_ylabel('Observed Flame Speed, $s_p$ (m/s)', fontsize=12)
ax.set_title('Effect of Flame Curvature on Speed', fontsize=14, fontweight='bold')

# Add a legend
ax.legend(loc='best', fontsize=11)

# Set axis limits to better frame the data
ax.set_xlim(0.5, 5)
ax.set_ylim(0.25, 0.55)

# Add annotations to explain the behavior
ax.annotate('Flame speed increases\nwith more curvature',
            xy=(1, s_p_negative_Ma[np.where(R*1000 > 1)[0][0]]),
            xytext=(2, 0.5),
            arrowprops=dict(facecolor='royalblue', shrink=0.05, alpha=0.7),
            fontsize=10, color='royalblue', ha='center')

ax.annotate('Flame speed decreases\nwith more curvature',
            xy=(1, s_p_positive_Ma[np.where(R*1000 > 1)[0][0]]),
            xytext=(2, 0.3),
            arrowprops=dict(facecolor='firebrick', shrink=0.05, alpha=0.7),
            fontsize=10, color='firebrick', ha='center')


# Save the figure
plt.tight_layout()
plt.grid(False)
plt.savefig('flame_speed_vs_curvature.png', dpi=300)