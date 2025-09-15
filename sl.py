import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Physical constants
R = 8.314  # Universal gas constant (J/mol·K)
rho_u = 1.2  # Unburned density (kg/m³)
rho_b = 0.2  # Burned density (kg/m³)
c_p = 1200  # Specific heat (J/kg·K)
lambda_b = 0.08  # Thermal conductivity (W/m·K)
B = 1e10  # Pre-exponential factor (1/s)

def calculate_flame_speed(T_b, T_u, E_a, Y_F, Y_O, Le=1.0):
    """
    Calculate laminar flame speed using the simplified equation:
    s_L² ≈ 2(λ_b/ρ_u²c_p²)[ρ_b²Y_F,u·Y_O,u·B·exp(-E_a/RT_b)](RT_b²/E_a)
    
    Parameters:
    -----------
    T_b : float or array - Burned temperature (K)
    T_u : float - Unburned temperature (K)
    E_a : float - Activation energy (J/mol)
    Y_F : float - Fuel mass fraction
    Y_O : float - Oxidizer mass fraction
    Le : float - Lewis number
    
    Returns:
    --------
    s_L : float or array - Laminar flame speed (m/s)
    """
    # Ensure E_a is in J/mol
    if E_a < 1000:  # Assume it's in kJ/mol if small
        E_a = E_a * 1000
    
    # Calculate reaction rate term
    exp_term = np.exp(-E_a / (R * T_b))
    temp_term = (R * T_b**2) / E_a
    
    # Calculate flame speed squared
    s_L_squared = 2 * (lambda_b / (rho_u**2 * c_p**2)) * \
                  (rho_b**2 * Y_F * Y_O * B * exp_term) * temp_term
    
    # Take square root and apply Lewis number correction
    s_L = np.sqrt(np.abs(s_L_squared))
    Le_correction = Le**(-0.5)
    
    return s_L * Le_correction

def calculate_zeldovich(E_a, T_b, T_u):
    """Calculate Zeldovich number β = E_a(T_b - T_u)/(R·T_b²)"""
    if E_a < 1000:
        E_a = E_a * 1000
    return (E_a * (T_b - T_u)) / (R * T_b**2)

def calculate_reaction_rate(T, Y, E_a, T_b):
    """Calculate normalized reaction rate ω̇ for visualization"""
    if E_a < 1000:
        E_a = E_a * 1000
    beta = calculate_zeldovich(E_a, T_b, 300)
    alpha = 0.85  # Heat release parameter
    
    # Normalized temperature (0 at T_u, 1 at T_b)
    T_norm = (T - 300) / (T_b - 300)
    
    # Reaction rate using excitable medium formulation
    omega = (beta**2 / 2) * Y * np.exp(-beta * (1 - T_norm) / (1 - alpha * (1 - T_norm)))
    return omega

# Create comprehensive figure with multiple subplots
def create_flame_analysis():
    # Set default parameters
    T_b_default = 2000  # K
    T_u_default = 300   # K
    E_a_default = 150   # kJ/mol
    Y_F_default = 0.06
    Y_O_default = 0.23
    Le_default = 1.0
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Laminar Flame Speed Analysis: Ring-Shaped Trough Combustion', 
                 fontsize=16, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ============ Plot 1: Flame Speed vs Burned Temperature ============
    ax1 = fig.add_subplot(gs[0, 0])
    T_b_range = np.linspace(1500, 2500, 100)
    s_L_temp = [calculate_flame_speed(T, T_u_default, E_a_default, 
                                      Y_F_default, Y_O_default, Le_default) 
                for T in T_b_range]
    
    ax1.plot(T_b_range, s_L_temp, 'r-', linewidth=2.5, label='s_L(T_b)')
    ax1.axvline(T_b_default, color='k', linestyle='--', alpha=0.5)
    ax1.scatter([T_b_default], 
               [calculate_flame_speed(T_b_default, T_u_default, E_a_default,
                                     Y_F_default, Y_O_default, Le_default)],
               color='red', s=100, zorder=5, label='Current')
    ax1.set_xlabel('Burned Temperature T_b (K)', fontsize=10)
    ax1.set_ylabel('Flame Speed (m/s)', fontsize=10)
    ax1.set_title('Temperature Dependence', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # ============ Plot 2: Flame Speed vs Activation Energy ============
    ax2 = fig.add_subplot(gs[0, 1])
    E_a_range = np.linspace(50, 250, 100)
    s_L_Ea = [calculate_flame_speed(T_b_default, T_u_default, E, 
                                    Y_F_default, Y_O_default, Le_default) 
              for E in E_a_range]
    
    ax2.plot(E_a_range, s_L_Ea, 'b-', linewidth=2.5, label='s_L(E_a)')
    ax2.axvline(E_a_default, color='k', linestyle='--', alpha=0.5)
    ax2.scatter([E_a_default], 
               [calculate_flame_speed(T_b_default, T_u_default, E_a_default,
                                     Y_F_default, Y_O_default, Le_default)],
               color='blue', s=100, zorder=5, label='Current')
    ax2.set_xlabel('Activation Energy E_a (kJ/mol)', fontsize=10)
    ax2.set_ylabel('Flame Speed (m/s)', fontsize=10)
    ax2.set_title('Activation Energy Effect', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # ============ Plot 3: Flame Speed vs Fuel Fraction ============
    ax3 = fig.add_subplot(gs[0, 2])
    Y_F_range = np.linspace(0.01, 0.15, 100)
    s_L_YF = [calculate_flame_speed(T_b_default, T_u_default, E_a_default, 
                                    Y, Y_O_default, Le_default) 
              for Y in Y_F_range]
    
    ax3.plot(Y_F_range, s_L_YF, 'g-', linewidth=2.5, label='s_L(Y_F)')
    ax3.axvline(Y_F_default, color='k', linestyle='--', alpha=0.5)
    ax3.scatter([Y_F_default], 
               [calculate_flame_speed(T_b_default, T_u_default, E_a_default,
                                     Y_F_default, Y_O_default, Le_default)],
               color='green', s=100, zorder=5, label='Current')
    ax3.set_xlabel('Fuel Mass Fraction Y_F', fontsize=10)
    ax3.set_ylabel('Flame Speed (m/s)', fontsize=10)
    ax3.set_title('Fuel Concentration Effect', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # ============ Plot 4: 3D Surface - s_L(T_b, E_a) ============
    ax4 = fig.add_subplot(gs[1, :2], projection='3d')
    T_b_mesh = np.linspace(1500, 2500, 30)
    E_a_mesh = np.linspace(50, 250, 30)
    T_b_grid, E_a_grid = np.meshgrid(T_b_mesh, E_a_mesh)
    
    s_L_grid = np.zeros_like(T_b_grid)
    for i in range(len(E_a_mesh)):
        for j in range(len(T_b_mesh)):
            s_L_grid[i, j] = calculate_flame_speed(T_b_grid[i, j], T_u_default, 
                                                   E_a_grid[i, j], Y_F_default, 
                                                   Y_O_default, Le_default)
    
    surf = ax4.plot_surface(T_b_grid, E_a_grid, s_L_grid, cmap='viridis', 
                            alpha=0.8, edgecolor='none')
    ax4.set_xlabel('T_b (K)', fontsize=10)
    ax4.set_ylabel('E_a (kJ/mol)', fontsize=10)
    ax4.set_zlabel('Flame Speed (m/s)', fontsize=10)
    ax4.set_title('Flame Speed Surface: s_L(T_b, E_a)', fontsize=11, fontweight='bold')
    ax4.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=5)
    
    # ============ Plot 5: Lewis Number Effects ============
    ax5 = fig.add_subplot(gs[1, 2])
    Le_range = np.linspace(0.3, 2.0, 100)
    s_L_Le = [calculate_flame_speed(T_b_default, T_u_default, E_a_default,
                                    Y_F_default, Y_O_default, Le) 
              for Le in Le_range]
    
    ax5.plot(Le_range, s_L_Le, 'm-', linewidth=2.5, label='s_L(Le)')
    ax5.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Le=1')
    ax5.axvspan(0.3, 1.0, alpha=0.2, color='red', label='Unstable (Le<1)')
    ax5.axvspan(1.0, 2.0, alpha=0.2, color='blue', label='Stable (Le>1)')
    ax5.set_xlabel('Lewis Number Le', fontsize=10)
    ax5.set_ylabel('Flame Speed (m/s)', fontsize=10)
    ax5.set_title('Lewis Number Effect', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')
    
    # ============ Plot 6: Reaction Rate Profile ============
    ax6 = fig.add_subplot(gs[2, 0])
    T_profile = np.linspace(T_u_default, T_b_default, 200)
    Y_profile = np.linspace(1, 0, 200)  # Fuel depletes as temperature rises
    omega = calculate_reaction_rate(T_profile, Y_profile, E_a_default, T_b_default)
    
    ax6.plot(T_profile, omega/np.max(omega), 'r-', linewidth=2.5, label='ω̇(T)')
    ax6.fill_between(T_profile, 0, omega/np.max(omega), alpha=0.3, color='red')
    ax6.set_xlabel('Temperature (K)', fontsize=10)
    ax6.set_ylabel('Normalized Reaction Rate', fontsize=10)
    ax6.set_title('Reaction Rate Profile', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # ============ Plot 7: Zeldovich Number Analysis ============
    ax7 = fig.add_subplot(gs[2, 1])
    T_b_zeld = np.linspace(1500, 2500, 100)
    beta_values = [calculate_zeldovich(E_a_default, T, T_u_default) for T in T_b_zeld]
    
    ax7.plot(T_b_zeld, beta_values, 'b-', linewidth=2.5)
    ax7.axhline(10, color='r', linestyle='--', alpha=0.5, label='β=10 (typical)')
    ax7.set_xlabel('Burned Temperature T_b (K)', fontsize=10)
    ax7.set_ylabel('Zeldovich Number β', fontsize=10)
    ax7.set_title('Temperature Sensitivity (Zeldovich Number)', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # ============ Plot 8: Contour Plot - s_L(Y_F, Y_O) ============
    ax8 = fig.add_subplot(gs[2, 2])
    Y_F_cont = np.linspace(0.01, 0.15, 50)
    Y_O_cont = np.linspace(0.15, 0.30, 50)
    Y_F_grid, Y_O_grid = np.meshgrid(Y_F_cont, Y_O_cont)
    
    s_L_YF_YO = np.zeros_like(Y_F_grid)
    for i in range(len(Y_O_cont)):
        for j in range(len(Y_F_cont)):
            s_L_YF_YO[i, j] = calculate_flame_speed(T_b_default, T_u_default, 
                                                    E_a_default, Y_F_grid[i, j], 
                                                    Y_O_grid[i, j], Le_default)
    
    contour = ax8.contourf(Y_F_grid, Y_O_grid, s_L_YF_YO, levels=20, cmap='plasma')
    ax8.scatter([Y_F_default], [Y_O_default], color='white', s=100, 
               marker='*', edgecolor='black', linewidth=2, zorder=5)
    ax8.set_xlabel('Fuel Fraction Y_F', fontsize=10)
    ax8.set_ylabel('Oxidizer Fraction Y_O', fontsize=10)
    ax8.set_title('Flame Speed Map: s_L(Y_F, Y_O)', fontsize=11, fontweight='bold')
    fig.colorbar(contour, ax=ax8, label='s_L (m/s)')
    
    # Add parameter box
    param_text = f"""
    Default Parameters:
    T_b = {T_b_default} K
    T_u = {T_u_default} K
    E_a = {E_a_default} kJ/mol
    Y_F = {Y_F_default}
    Y_O = {Y_O_default}
    Le = {Le_default}
    
    Calculated Values:
    s_L = {calculate_flame_speed(T_b_default, T_u_default, E_a_default, 
                                 Y_F_default, Y_O_default, Le_default):.3f} m/s
    β = {calculate_zeldovich(E_a_default, T_b_default, T_u_default):.2f}
    T_b/T_u = {T_b_default/T_u_default:.2f}
    """
    
    fig.text(0.02, 0.02, param_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Create and display the analysis
fig = create_flame_analysis()
plt.show()

# Additional analysis functions
def parameter_sensitivity_analysis():
    """Analyze sensitivity of flame speed to different parameters"""
    
    # Base parameters
    T_b_base = 2000
    T_u_base = 300
    E_a_base = 150
    Y_F_base = 0.06
    Y_O_base = 0.23
    Le_base = 1.0
    
    # Calculate base flame speed
    s_L_base = calculate_flame_speed(T_b_base, T_u_base, E_a_base, 
                                     Y_F_base, Y_O_base, Le_base)
    
    # Sensitivity analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    
    # Define parameter variations (±20%)
    variations = np.linspace(0.8, 1.2, 50)
    
    parameters = [
        ('T_b', T_b_base, 'Burned Temperature'),
        ('E_a', E_a_base, 'Activation Energy'),
        ('Y_F', Y_F_base, 'Fuel Fraction'),
        ('Y_O', Y_O_base, 'Oxidizer Fraction'),
        ('Le', Le_base, 'Lewis Number'),
        ('T_u', T_u_base, 'Unburned Temperature')
    ]
    
    for idx, (ax, (param_name, param_base, param_label)) in enumerate(zip(axes.flat, parameters)):
        sensitivities = []
        
        for var in variations:
            if param_name == 'T_b':
                s_L = calculate_flame_speed(param_base * var, T_u_base, E_a_base,
                                           Y_F_base, Y_O_base, Le_base)
            elif param_name == 'E_a':
                s_L = calculate_flame_speed(T_b_base, T_u_base, param_base * var,
                                           Y_F_base, Y_O_base, Le_base)
            elif param_name == 'Y_F':
                s_L = calculate_flame_speed(T_b_base, T_u_base, E_a_base,
                                           param_base * var, Y_O_base, Le_base)
            elif param_name == 'Y_O':
                s_L = calculate_flame_speed(T_b_base, T_u_base, E_a_base,
                                           Y_F_base, param_base * var, Le_base)
            elif param_name == 'Le':
                s_L = calculate_flame_speed(T_b_base, T_u_base, E_a_base,
                                           Y_F_base, Y_O_base, param_base * var)
            else:  # T_u
                s_L = calculate_flame_speed(T_b_base, param_base * var, E_a_base,
                                           Y_F_base, Y_O_base, Le_base)
            
            sensitivities.append((s_L - s_L_base) / s_L_base * 100)
        
        ax.plot((variations - 1) * 100, sensitivities, linewidth=2.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel(f'{param_label} Change (%)', fontsize=9)
        ax.set_ylabel('Flame Speed Change (%)', fontsize=9)
        ax.set_title(param_label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Calculate and display sensitivity coefficient
        sensitivity_coeff = np.polyfit((variations - 1) * 100, sensitivities, 1)[0]
        ax.text(0.95, 0.95, f'Sensitivity: {sensitivity_coeff:.2f}%/%',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Run sensitivity analysis
fig_sensitivity = parameter_sensitivity_analysis()
plt.show()

print("\n" + "="*60)
print("LAMINAR FLAME SPEED ANALYSIS COMPLETE")
print("="*60)
print("\nKey Insights:")
print("1. Flame speed increases with burned temperature (T_b)")
print("2. Higher activation energy (E_a) reduces flame speed")
print("3. Lewis number < 1 promotes thermo-diffusive instability")
print("4. Optimal fuel-oxidizer ratio maximizes propagation speed")
print("5. Zeldovich number quantifies temperature sensitivity")
print("\nFor ring-shaped trough combustion:")
print("- Continuous propagation requires balanced heat generation/loss")
print("- Curvature effects can stabilize or destabilize the flame")
print("- Lewis number determines cellular/smooth flame structure")