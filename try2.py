import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- 1. Define Physical and Model Parameters ---

# Base laminar flame speed under ideal conditions (m/s)
s_L0 = 0.4

# Quenching distance (m). The flame extinguishes if the channel width is below this value.
d_quench = 0.002 # 2 mm

# Width effect coefficient (dimensionless). Controls how quickly the speed recovers as width increases.
width_effect_coeff = 500

# Curvature effect coefficient (m). Represents the strength of the radius's influence on speed.
curvature_effect_coeff = 0.005 # 5 mm

# --- 2. Create the Phenomenological Flame Speed Model ---

def calculate_flame_speed(radius, width):
    """
    Calculates the effective flame speed based on the phenomenological model.
    
    Args:
        radius (float or np.ndarray): The radius of the ring (m).
        width (float or np.ndarray): The width of the channel (m).
    
    Returns:
        float or np.ndarray: The effective flame speed (m/s).
    """
    
    # Efficiency factor from channel width (0 to 1)
    width_loss = np.exp(-width_effect_coeff * (width - d_quench))
    width_efficiency = np.where(width > d_quench, 1 - width_loss, 0)
    
    # Efficiency factor from curvature (0 to 1)
    curvature_efficiency = np.where(radius > 0, 1 - curvature_effect_coeff / radius, 0)
    curvature_efficiency = np.maximum(0, curvature_efficiency)

    # Final effective flame speed
    s_effective = s_L0 * width_efficiency * curvature_efficiency
    
    return s_effective

# --- 3. Generate Data for Plotting ---

# Define the range for radius and width with enough points for a smooth surface
num_points = 300
radius_range = np.linspace(0.01, 0.1, num_points)  # 1 cm to 10 cm
width_range = np.linspace(0.001, 0.02, num_points) # 1 mm to 20 mm

# Create a 2D grid
R, W = np.meshgrid(radius_range, width_range)

# Calculate the flame speed at each point on the grid
S = calculate_flame_speed(R, W)

# --- 4. Plot the 3D Surface ---

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(R * 100, W * 100, S, cmap=cm.viridis, linewidth=0, antialiased=True)

# Make the plot cleaner by removing grid and background panes
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Set axis labels and title
# The x-axis label has been updated as requested.
ax.set_xlabel('Ring Radius (cm)', fontsize=12, labelpad=10)
ax.set_ylabel('Channel Width (cm)', fontsize=12, labelpad=10)
ax.set_zlabel('Flame Speed (m/s)', fontsize=12, labelpad=10)
ax.set_title('Model of Flame Speed vs. Geometric Dimensions', fontsize=16)

# Add a color bar
cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Flame Speed (m/s)', fontsize=10)

# Adjust the viewing angle
ax.view_init(elev=30, azim=45)

# Save the figure with high DPI before showing it
plt.savefig('flame_speed_plot_high_res.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()