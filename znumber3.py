import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def calculate_reaction_rate(T, Y, beta, Le, alpha):
    """
    Calculates the non-dimensional chemical reaction rate (Omega).
    
    Args:
        T (float or array): Non-dimensional temperature (0 to 1).
        Y (float or array): Reactant mass fraction (0 to 1).
        beta (float): Zel'dovich number.
        Le (float): Lewis number.
        alpha (float): Heat release parameter.
        
    Returns:
        float or array: The reaction rate Omega.
    """
    epsilon = 1e-9
    denominator = 1 - alpha * (1 - T)
    exponent = - (beta * (1 - T)) / (denominator + epsilon)
    prefactor = (beta**2) / (2 * Le)
    omega = prefactor * Y * np.exp(exponent)
    return omega

# --- 1. Set Constant Physical Parameters ---
beta = 10.0   # Zel'dovich number (high temperature sensitivity)
Le = 0.8      # Lewis number (unstable regime)
alpha = 0.85  # Heat release parameter

# --- 2. Create 2D Grids for T and Y for Plotting ---
# MODIFICATION: Increased the number of points from 100 to 300 for higher resolution.
num_points = 300
T_range = np.linspace(0, 1, num_points)
Y_range = np.linspace(0, 1, num_points)

# Create the meshgrid
T_grid, Y_grid = np.meshgrid(T_range, Y_range)

# --- 3. Calculate the Reaction Rate over the entire grid ---
Omega_grid = calculate_reaction_rate(T_grid, Y_grid, beta, Le, alpha)

# --- 4. Create and Display the 3D Diagram ---
print("Generating High-Resolution 3D Surface Diagram for Reaction Rate (Ω)...")

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface. Using a "hot" colormap is fitting for a reaction rate.
surf = ax.plot_surface(T_grid, Y_grid, Omega_grid, cmap=cm.inferno,
                       linewidth=0, antialiased=False)

# Add a color bar to show the scale of the reaction rate
cbar = fig.colorbar(surf, shrink=0.6, aspect=10)
cbar.set_label(r'Reaction Rate, $\Omega$', fontsize=12)

# Set titles and labels for all three axes
ax.set_title(r'Reaction Rate $\Omega$ vs. Temperature & Reactant Fraction', fontsize=16)
ax.set_xlabel(r'Non-dimensional Temperature, $T$', fontsize=12, labelpad=10)
ax.set_ylabel(r'Reactant Mass Fraction, $Y$', fontsize=12, labelpad=10)
ax.set_zlabel(r'Reaction Rate, $\Omega$', fontsize=12, labelpad=10)

# Adjust viewing angle for better perspective
ax.view_init(elev=25, azim=-140)

plt.tight_layout()
plt.show()