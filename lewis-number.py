import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate the Lewis number
def calculate_lewis_number(lambda_val, rho, cp, Dm):
    """
    Calculates the Lewis number (Le).
    Args:
    lambda_val (float): Thermal conductivity (W/m·K)
    rho (float): Density (kg/m^3)
    cp (float): Specific heat capacity at constant pressure (J/kg·K)
    Dm (float): Mass diffusivity of the deficient reactant (m^2/s)
    Returns:
    float: The dimensionless Lewis number.
    """
    # Thermal diffusivity (alpha_th = lambda / (rho * cp))
    alpha_th = lambda_val / (rho * cp)
    # Lewis number (Le = alpha_th / Dm)
    lewis = alpha_th / Dm
    return lewis

# --- 1. Set Baseline Physical Parameters ---
# These are representative values for a gas mixture like air.
baseline_lambda = 0.026  # W/m·K
baseline_rho = 1.2       # kg/m^3
baseline_cp = 1005       # J/kg·K
baseline_Dm = 2.0e-5     # m^2/s (typical for hydrocarbons in air)

# Calculate the baseline Lewis number
baseline_Le = calculate_lewis_number(baseline_lambda, baseline_rho, baseline_cp, baseline_Dm)
print(f"Baseline Lewis Number (Le): {baseline_Le:.2f}")

# --- 2. Create Parameter Ranges for Plotting ---
# We will vary each parameter by +/- 50% of its baseline value
lambda_range = np.linspace(0.5 * baseline_lambda, 1.5 * baseline_lambda, 200)
rho_range = np.linspace(0.5 * baseline_rho, 1.5 * baseline_rho, 200)
cp_range = np.linspace(0.5 * baseline_cp, 1.5 * baseline_cp, 200)
Dm_range = np.linspace(0.5 * baseline_Dm, 1.5 * baseline_Dm, 200)

# --- 3. Calculate Lewis Numbers for each range ---
Le_vs_lambda = calculate_lewis_number(lambda_range, baseline_rho, baseline_cp, baseline_Dm)
Le_vs_rho = calculate_lewis_number(baseline_lambda, rho_range, baseline_cp, baseline_Dm)
Le_vs_cp = calculate_lewis_number(baseline_lambda, baseline_rho, cp_range, baseline_Dm)
Le_vs_Dm = calculate_lewis_number(baseline_lambda, baseline_rho, baseline_cp, Dm_range)

# --- 4. Create and Show Plots Sequentially ---

# Plot 1: Le vs. Thermal Conductivity (λ)
plt.figure(figsize=(8, 6))
plt.plot(lambda_range, Le_vs_lambda, color='blue', linewidth=2.5)
plt.title(r'Le vs. Thermal Conductivity ($\lambda$)', fontsize=14)
plt.xlabel(r'Thermal Conductivity, $\lambda$ (W/m·K)', fontsize=12)
plt.ylabel('Lewis Number, Le', fontsize=12)
plt.axvline(baseline_lambda, color='red', linestyle='--', 
           label=f'Baseline = {baseline_lambda:.3f} W/m·K')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show() # This will display the first plot and pause execution

# Plot 2: Le vs. Density (ρ)
plt.figure(figsize=(8, 6))
plt.plot(rho_range, Le_vs_rho, color='green', linewidth=2.5)
plt.title(r'Le vs. Density ($\rho$)', fontsize=14)
plt.xlabel(r'Density, $\rho$ (kg/m³)', fontsize=12)
plt.ylabel('Lewis Number, Le', fontsize=12)
plt.axvline(baseline_rho, color='red', linestyle='--', 
           label=f'Baseline = {baseline_rho:.1f} kg/m³')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show() # This will display the second plot after the first is closed

# Plot 3: Le vs. Specific Heat (cp)
plt.figure(figsize=(8, 6))
plt.plot(cp_range, Le_vs_cp, color='purple', linewidth=2.5)
plt.title(r'Le vs. Specific Heat ($c_p$)', fontsize=14)
plt.xlabel(r'Specific Heat, $c_p$ (J/kg·K)', fontsize=12)
plt.ylabel('Lewis Number, Le', fontsize=12)
plt.axvline(baseline_cp, color='red', linestyle='--', 
           label=f'Baseline = {baseline_cp} J/kg·K')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show() # This will display the third plot

# Plot 4: Le vs. Mass Diffusivity (Dm)
plt.figure(figsize=(8, 6))
plt.plot(Dm_range, Le_vs_Dm, color='orange', linewidth=2.5)
plt.title(r'Le vs. Mass Diffusivity ($D_m$)', fontsize=14)
plt.xlabel(r'Mass Diffusivity, $D_m$ (m²/s)', fontsize=12)
plt.ylabel('Lewis Number, Le', fontsize=12)
plt.axvline(baseline_Dm, color='red', linestyle='--', 
           label=f'Baseline = {baseline_Dm:.1e} m²/s')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show() # This will display the fourth plot

# --- 5. Print some analysis ---
# This part will run after you have closed the last plot window
print("\n=== Lewis Number Sensitivity Analysis ===")
print(f"Baseline Lewis Number: {baseline_Le:.3f}")
print(f"Le range with λ variation (±50%): {Le_vs_lambda.min():.3f} - {Le_vs_lambda.max():.3f}")
print(f"Le range with ρ variation (±50%): {Le_vs_rho.min():.3f} - {Le_vs_rho.max():.3f}")
print(f"Le range with cp variation (±50%): {Le_vs_cp.min():.3f} - {Le_vs_cp.max():.3f}")
print(f"Le range with Dm variation (±50%): {Le_vs_Dm.min():.3f} - {Le_vs_Dm.max():.3f}")

# Calculate sensitivity (relative change in Le per relative change in parameter)
print("\n=== Sensitivity Coefficients ===")
print("(Relative change in Le per 1% change in parameter)")
lambda_sensitivity = (Le_vs_lambda.max() - Le_vs_lambda.min()) / baseline_Le / 1.0  # 100% range
rho_sensitivity = (Le_vs_rho.max() - Le_vs_rho.min()) / baseline_Le / 1.0
cp_sensitivity = (Le_vs_cp.max() - Le_vs_cp.min()) / baseline_Le / 1.0
Dm_sensitivity = (Le_vs_Dm.max() - Le_vs_Dm.min()) / baseline_Le / 1.0

print(f"Thermal conductivity (λ): {lambda_sensitivity:.3f}")
print(f"Density (ρ): {rho_sensitivity:.3f}")
print(f"Specific heat (cp): {cp_sensitivity:.3f}")
print(f"Mass diffusivity (Dm): {Dm_sensitivity:.3f}")