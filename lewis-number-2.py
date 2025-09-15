import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define a function to calculate the Lewis number
def calculate_lewis_number(lambda_val, rho, cp, Dm):
    """
    Calculates the Lewis number (Le).
    Args:
    lambda_val (float or array): Thermal conductivity (W/m·K)
    rho (float): Density (kg/m^3)
    cp (float): Specific heat capacity at constant pressure (J/kg·K)
    Dm (float or array): Mass diffusivity of the deficient reactant (m^2/s)
    Returns:
    float or array: The dimensionless Lewis number.
    """
    alpha_th = lambda_val / (rho * cp)
    lewis = alpha_th / Dm
    return lewis

# --- 1. Set Baseline and Plotting Ranges ---
baseline_lambda = 0.026  # W/m·K
baseline_rho = 1.2       # kg/m^3
baseline_cp = 1005       # J/kg·K
baseline_Dm = 2.0e-5     # m^2/s

# Create parameter ranges for the axes of the phase diagram
# We will vary thermal conductivity and mass diffusivity
lambda_range = np.linspace(0.01, 0.05, 100)
Dm_range = np.linspace(1.0e-5, 4.0e-5, 100)

# Create a 2D grid of parameters using meshgrid
Lambda_grid, Dm_grid = np.meshgrid(lambda_range, Dm_range)

# --- 2. Calculate Lewis Number over the entire grid ---
Le_grid = calculate_lewis_number(Lambda_grid, baseline_rho, baseline_cp, Dm_grid)

# --- 3. Plot 1: 2D Contour Phase Diagram ---
print("Generating 2D Contour Plot...")
plt.figure(figsize=(10, 8))

# Create a filled contour plot (the color map)
contour_filled = plt.contourf(Lambda_grid, Dm_grid, Le_grid, levels=20, cmap=cm.viridis)

# Add a color bar to show the Lewis number scale
cbar = plt.colorbar(contour_filled)
cbar.set_label('Lewis Number (Le)', fontsize=12)

# Add contour lines for specific Le values, making the Le=1 line prominent
contour_lines = plt.contour(Lambda_grid, Dm_grid, Le_grid, levels=[0.5, 1.0, 1.5, 2.0], colors='white', linewidths=1.5)
plt.clabel(contour_lines, inline=True, fontsize=10, fmt='Le = %.1f')

# Highlight the Le=1 line, which is the critical boundary
contour_le1 = plt.contour(Lambda_grid, Dm_grid, Le_grid, levels=[1.0], colors='red', linewidths=3)
plt.clabel(contour_le1, inline=True, fontsize=12, fmt='Le = %.1f (Critical)')

# Mark the baseline point on the plot
plt.plot(baseline_lambda, baseline_Dm, 'ro', markersize=8, label=f'Baseline (Le={calculate_lewis_number(baseline_lambda, baseline_rho, baseline_cp, baseline_Dm):.2f})')

# Set titles and labels
plt.title('Lewis Number Phase Diagram', fontsize=16)
plt.xlabel(r'Thermal Conductivity, $\lambda$ (W/m·K)', fontsize=12)
plt.ylabel(r'Mass Diffusivity, $D_m$ (m²/s)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


# --- 4. Plot 2: 3D Surface Phase Diagram ---
print("\nGenerating 3D Surface Plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(Lambda_grid, Dm_grid, Le_grid, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
cbar.set_label('Lewis Number (Le)', fontsize=12)

# Set titles and labels for all three axes
ax.set_xlabel(r'Thermal Conductivity, $\lambda$ (W/m·K)', fontsize=12, labelpad=10)
ax.set_ylabel(r'Mass Diffusivity, $D_m$ (m²/s)', fontsize=12, labelpad=10)
ax.set_zlabel('Lewis Number, Le', fontsize=12, labelpad=10)

# Adjust viewing angle
ax.view_init(elev=30, azim=-135)
plt.show()