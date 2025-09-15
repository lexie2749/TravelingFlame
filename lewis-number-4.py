import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def calculate_lewis_number(lambda_val, rho, cp, Dm):
    """
    Calculates the Lewis number (Le) based on thermal and mass diffusivity.
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

# --- 1. Set Physical Parameters and Ranges ---
# Baseline physical properties (assumed constant)
baseline_rho = 1.2       # kg/m^3
baseline_cp = 1005       # J/kg·K

# Define parameter ranges for the new axes
groove_diameter_range = np.linspace(3, 10, 100)  # in mm
fuel_range = np.linspace(125, 200, 100)     # in uL

# Create a 2D grid of the new parameters using meshgrid
Groove_Diameter_grid, Fuel_grid = np.meshgrid(groove_diameter_range, fuel_range)

# --- 2. Map New Axes to Lewis Number Physics ---
# Assumption: A fixed ring diameter of 150mm corresponds to a constant thermal conductivity.
# Using the upper bound of the previous mapping for this fixed value.
lambda_val = 0.040  # W/m·K
lambda_grid = np.full_like(Groove_Diameter_grid, lambda_val)

# Assumption: Map fuel amount to mass diffusivity (Dm)
# Assuming an inverse linear relationship over the new fuel range
Dm_min = 1.5e-5  # m^2/s
Dm_max = 3.0e-5  # m^2/s
Dm_grid = np.interp(Fuel_grid, [125, 200], [Dm_max, Dm_min])

# --- 3. Calculate Lewis Number over the entire grid ---
Le_grid = calculate_lewis_number(lambda_grid, baseline_rho, baseline_cp, Dm_grid)

# --- 4. Plot the Lewis Number Phase Diagram ---
print("Generating 2D Contour Plot...")
plt.figure(figsize=(10, 8))

# Create a filled contour plot (the color map)
# Use the viridis colormap for a blue-green-yellow gradient
contour_filled = plt.contourf(Groove_Diameter_grid, Fuel_grid, Le_grid, levels=20, cmap=cm.viridis)

# Add a color bar to show the Lewis number scale
cbar = plt.colorbar(contour_filled)
cbar.set_label('Lewis Number (Le)', fontsize=12)

# Add contour lines for specific Le values
contour_lines = plt.contour(Groove_Diameter_grid, Fuel_grid, Le_grid, levels=[0.5, 1.5, 2.0], colors='white', linewidths=1.5)
plt.clabel(contour_lines, inline=True, fontsize=10, fmt='Le = %.1f')

# Highlight the Le=1 line, which is the critical boundary
contour_le1 = plt.contour(Groove_Diameter_grid, Fuel_grid, Le_grid, levels=[1.0], colors='red', linewidths=3)
plt.clabel(contour_le1, inline=True, fontsize=12, fmt='Le = %.1f (Critical)')

# Set titles and labels
plt.xlabel('Concave Groove Diameter (mm)', fontsize=12)
plt.ylabel('Fuel Amount ($\mu$L)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
