import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from torch.autograd import grad

class FlameSpeedValidator:
    """
    Comprehensive framework to validate flame speed accuracy
    """
    
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.device = next(model.parameters()).device
        
    def theoretical_flame_speed(self):
        """
        Calculate theoretical flame speed based on governing parameters
        Using simplified Zeldovich-Frank-Kamenetskii (ZFK) theory
        """
        # Extract model parameters
        Da = self.model.Damkohler
        Pe = self.model.Peclet
        Le = self.model.Lewis
        Q = self.model.heat_release
        beta = self.model.activation_energy
        
        # Simplified ZFK scaling for flame speed
        # s_L ∝ sqrt(Da/Pe) for high activation energy
        # With corrections for Lewis number and heat release
        
        # Basic thermal-diffusive speed
        s_thermal = np.sqrt(Da / Pe)
        
        # Lewis number correction (Le < 1 increases speed)
        Le_correction = 1.0 / np.sqrt(Le) if Le < 1 else np.sqrt(Le)
        
        # Heat release correction
        Q_correction = np.sqrt(1 + Q/beta)
        
        # Theoretical estimate
        s_theory = s_thermal * Le_correction * Q_correction
        
        # For annular geometry, add curvature effect
        # Assuming trough radius ~ 1 in normalized units
        curvature_correction = 1.0  # Can be refined based on geometry
        
        s_final = s_theory * curvature_correction
        
        print("\n" + "="*60)
        print("THEORETICAL FLAME SPEED ANALYSIS")
        print("="*60)
        print(f"Basic thermal speed: {s_thermal:.3f} rad/s")
        print(f"Lewis number correction (Le={Le:.2f}): {Le_correction:.3f}")
        print(f"Heat release correction (Q={Q:.1f}): {Q_correction:.3f}")
        print(f"Theoretical speed estimate: {s_final:.3f} rad/s")
        print(f"Prescribed speed in model: {self.model.flame_speed:.3f} rad/s")
        print(f"Ratio (prescribed/theoretical): {self.model.flame_speed/s_final:.3f}")
        
        return s_final
    
    def measure_actual_speed(self, n_samples=5):
        """
        Measure actual propagation speed from the PINN solution
        by tracking the flame front position over time
        """
        self.model.eval()
        
        # Time points to sample
        times = np.linspace(0, 0.08, n_samples)
        flame_positions = []
        
        with torch.no_grad():
            for t_val in times:
                # Sample angular positions
                theta = torch.linspace(0, 2*np.pi, 200).unsqueeze(1).to(self.device)
                t = torch.ones_like(theta) * t_val
                
                inputs = torch.cat([theta, t], dim=1)
                T, Y = self.model(inputs)
                
                # Find flame position (maximum temperature)
                T_np = T.cpu().numpy().flatten()
                theta_np = theta.cpu().numpy().flatten()
                
                # Use temperature gradient to find flame front more accurately
                dT = np.gradient(T_np)
                flame_idx = np.argmax(np.abs(dT))
                flame_pos = theta_np[flame_idx]
                
                # Account for periodicity
                if len(flame_positions) > 0:
                    # Unwrap to handle crossing 2π boundary
                    prev_pos = flame_positions[-1]
                    if flame_pos < prev_pos - np.pi:
                        flame_pos += 2*np.pi
                    elif flame_pos > prev_pos + np.pi:
                        flame_pos -= 2*np.pi
                
                flame_positions.append(flame_pos)
        
        flame_positions = np.array(flame_positions)
        
        # Fit linear trend to get speed
        def linear(t, speed, offset):
            return speed * t + offset
        
        popt, pcov = curve_fit(linear, times, flame_positions)
        measured_speed = popt[0]
        speed_std = np.sqrt(pcov[0, 0])
        
        # Calculate R-squared
        residuals = flame_positions - linear(times, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((flame_positions - np.mean(flame_positions))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print("\n" + "="*60)
        print("MEASURED FLAME SPEED FROM SOLUTION")
        print("="*60)
        print(f"Measured speed: {measured_speed:.3f} ± {speed_std:.3f} rad/s")
        print(f"Prescribed speed: {self.model.flame_speed:.3f} rad/s")
        print(f"Difference: {abs(measured_speed - self.model.flame_speed):.3f} rad/s")
        print(f"R-squared of linear fit: {r_squared:.4f}")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot flame position vs time
        ax1.plot(times, flame_positions, 'bo-', label='Measured positions')
        ax1.plot(times, linear(times, *popt), 'r--', 
                label=f'Linear fit (speed={measured_speed:.3f})')
        ax1.plot(times, self.model.flame_speed * times + flame_positions[0], 'g--', 
                alpha=0.5, label=f'Expected (speed={self.model.flame_speed:.3f})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Flame Position (rad)')
        ax1.set_title('Flame Front Tracking')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot residuals
        ax2.plot(times, residuals, 'ro-')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Residual (rad)')
        ax2.set_title('Fit Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Flame Speed Measurement Validation')
        plt.tight_layout()
        plt.show()
        
        return measured_speed, speed_std
    
    def eigenvalue_analysis(self):
        """
        Linear stability analysis to find the dispersion relation
        and validate flame speed from eigenvalues
        """
        print("\n" + "="*60)
        print("LINEAR STABILITY ANALYSIS")
        print("="*60)
        
        # Parameters
        Pe = self.model.Peclet
        Le = self.model.Lewis
        Da = self.model.Damkohler
        Q = self.model.heat_release
        beta = self.model.activation_energy
        
        # For traveling wave with speed c, linearized system gives
        # characteristic equation for wavenumber k:
        # λ² + λ(c*ik + α₁ + α₂) + (c*ik)² + c*ik*(α₁ + α₂) + α₁*α₂ - ω₀² = 0
        # where α₁ = 1/Pe, α₂ = 1/(Pe*Le), ω₀² = Da*Q
        
        # Simplified analysis for neutral stability (λ = 0)
        k_values = np.linspace(0.1, 10, 100)
        speeds = []
        
        for k in k_values:
            # Dispersion relation for flame speed
            # c² = (α₁ + α₂)*c/k + (α₁*α₂ - ω₀²)/k²
            # This is approximate for high activation energy
            
            alpha1 = 1.0 / Pe
            alpha2 = 1.0 / (Pe * Le)
            omega0_sq = Da * Q * np.exp(-beta/2)  # Linearized reaction rate
            
            # Solve quadratic for c
            a = 1.0
            b = -(alpha1 + alpha2) / k
            c_coef = -(alpha1 * alpha2 - omega0_sq) / (k**2)
            
            discriminant = b**2 - 4*a*c_coef
            if discriminant > 0:
                c_plus = (-b + np.sqrt(discriminant)) / (2*a)
                if c_plus > 0:
                    speeds.append(c_plus)
        
        if speeds:
            # Most unstable mode (typically k ~ 1 for normalized system)
            typical_speed = np.median(speeds)
            
            print(f"Typical flame speed from stability analysis: {typical_speed:.3f} rad/s")
            print(f"Prescribed speed: {self.model.flame_speed:.3f} rad/s")
            print(f"Ratio: {self.model.flame_speed/typical_speed:.3f}")
            
            # Plot dispersion relation
            plt.figure(figsize=(8, 6))
            plt.plot(k_values[:len(speeds)], speeds, 'b-', linewidth=2)
            plt.axhline(y=self.model.flame_speed, color='r', linestyle='--', 
                       label=f'Prescribed speed = {self.model.flame_speed:.2f}')
            plt.xlabel('Wavenumber k')
            plt.ylabel('Phase Speed c')
            plt.title('Dispersion Relation from Linear Stability Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            return typical_speed
        else:
            print("Could not determine speed from stability analysis")
            return None
    
    def physics_residual_check(self):
        """
        Check if the physics equations are satisfied with the prescribed speed
        """
        print("\n" + "="*60)
        print("PHYSICS RESIDUAL ANALYSIS")
        print("="*60)
        
        self.model.eval()
        
        # Sample points
        n_points = 500
        theta = torch.rand(n_points, 1).to(self.device) * 2 * np.pi
        t = torch.rand(n_points, 1).to(self.device) * 0.08
        
        with torch.no_grad():
            # Get residuals
            energy_res, species_res = self.model.physics_loss(theta, t)
            
            energy_res_np = energy_res.cpu().numpy().flatten()
            species_res_np = species_res.cpu().numpy().flatten()
            
            # Statistics
            energy_rms = np.sqrt(np.mean(energy_res_np**2))
            species_rms = np.sqrt(np.mean(species_res_np**2))
            
            print(f"Energy equation RMS residual: {energy_rms:.6f}")
            print(f"Species equation RMS residual: {species_rms:.6f}")
            
            # Get solution magnitudes for scaling
            inputs = torch.cat([theta, t], dim=1)
            T, Y = self.model(inputs)
            T_scale = T.cpu().numpy().std()
            Y_scale = Y.cpu().numpy().std()
            
            print(f"Relative energy residual: {energy_rms/T_scale:.6f}")
            print(f"Relative species residual: {species_rms/Y_scale:.6f}")
            
            # Threshold check
            threshold = 0.01  # 1% relative error
            if energy_rms/T_scale < threshold and species_rms/Y_scale < threshold:
                print("✅ Physics equations are well satisfied!")
            else:
                print("⚠️ Physics residuals are large - speed may be inaccurate")
        
        # Visualize residuals
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(energy_res_np, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax1.set_xlabel('Energy Residual')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Energy Equation Residuals (RMS={energy_rms:.6f})')
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        ax2.hist(species_res_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Species Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Species Equation Residuals (RMS={species_rms:.6f})')
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.suptitle('Physics Residual Distribution')
        plt.tight_layout()
        plt.show()
        
        return energy_rms, species_rms
    
    def parametric_study(self):
        """
        Study how flame speed should vary with parameters
        """
        print("\n" + "="*60)
        print("PARAMETRIC DEPENDENCE OF FLAME SPEED")
        print("="*60)
        
        # Original parameters
        Da_0 = self.model.Damkohler
        Pe_0 = self.model.Peclet
        Le_0 = self.model.Lewis
        
        # Parameter ranges
        Da_range = np.linspace(10, 50, 5)
        Pe_range = np.linspace(50, 200, 5)
        Le_range = np.linspace(0.5, 1.5, 5)
        
        # Theoretical speeds
        speeds_Da = []
        speeds_Pe = []
        speeds_Le = []
        
        # Vary Damköhler number
        for Da in Da_range:
            s = np.sqrt(Da / Pe_0) / np.sqrt(Le_0)
            speeds_Da.append(s)
        
        # Vary Peclet number
        for Pe in Pe_range:
            s = np.sqrt(Da_0 / Pe) / np.sqrt(Le_0)
            speeds_Pe.append(s)
        
        # Vary Lewis number
        for Le in Le_range:
            s = np.sqrt(Da_0 / Pe_0) / np.sqrt(Le) if Le < 1 else np.sqrt(Da_0 / Pe_0) * np.sqrt(Le)
            speeds_Le.append(s)
        
        # Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.plot(Da_range, speeds_Da, 'b-', linewidth=2)
        ax1.axvline(x=Da_0, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=self.model.flame_speed, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Damköhler Number')
        ax1.set_ylabel('Flame Speed (rad/s)')
        ax1.set_title('Speed vs Da (s ∝ √Da)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(Pe_range, speeds_Pe, 'g-', linewidth=2)
        ax2.axvline(x=Pe_0, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.model.flame_speed, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Peclet Number')
        ax2.set_ylabel('Flame Speed (rad/s)')
        ax2.set_title('Speed vs Pe (s ∝ 1/√Pe)')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(Le_range, speeds_Le, 'm-', linewidth=2)
        ax3.axvline(x=Le_0, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=self.model.flame_speed, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Lewis Number')
        ax3.set_ylabel('Flame Speed (rad/s)')
        ax3.set_title('Speed vs Le')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Parametric Dependencies of Flame Speed\n(Red dashed = current model values)')
        plt.tight_layout()
        plt.show()
        
        print(f"\nExpected trends:")
        print(f"• Flame speed ∝ √Da (reaction rate)")
        print(f"• Flame speed ∝ 1/√Pe (diffusion)")
        print(f"• Flame speed affected by Le (thermal-mass diffusion ratio)")
    
    def validate_all(self):
        """
        Run all validation checks
        """
        print("\n" + "="*70)
        print("🔍 COMPREHENSIVE FLAME SPEED VALIDATION")
        print("="*70)
        
        # 1. Theoretical prediction
        s_theory = self.theoretical_flame_speed()
        
        # 2. Measured from solution
        s_measured, s_std = self.measure_actual_speed()
        
        # 3. Stability analysis
        s_stability = self.eigenvalue_analysis()
        
        # 4. Physics residuals
        energy_res, species_res = self.physics_residual_check()
        
        # 5. Parametric study
        self.parametric_study()
        
        # Summary
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)
        print(f"Prescribed speed:    {self.model.flame_speed:.3f} rad/s")
        print(f"Theoretical speed:   {s_theory:.3f} rad/s")
        print(f"Measured speed:      {s_measured:.3f} ± {s_std:.3f} rad/s")
        if s_stability:
            print(f"Stability analysis:  {s_stability:.3f} rad/s")
        
        # Accuracy assessment
        print("\n🎯 ACCURACY ASSESSMENT:")
        
        # Check consistency
        error_theory = abs(self.model.flame_speed - s_theory) / s_theory * 100
        error_measured = abs(self.model.flame_speed - s_measured) / self.model.flame_speed * 100
        
        if error_theory < 20:
            print(f"✅ Good agreement with theory (error: {error_theory:.1f}%)")
        else:
            print(f"⚠️ Large deviation from theory (error: {error_theory:.1f}%)")
        
        if error_measured < 5:
            print(f"✅ Excellent self-consistency (error: {error_measured:.1f}%)")
        elif error_measured < 10:
            print(f"✅ Good self-consistency (error: {error_measured:.1f}%)")
        else:
            print(f"⚠️ Poor self-consistency (error: {error_measured:.1f}%)")
        
        if energy_res < 0.01 and species_res < 0.01:
            print("✅ Physics equations well satisfied")
        else:
            print("⚠️ Large physics residuals")
        
        # Physical interpretation
        print("\n🔬 PHYSICAL INTERPRETATION:")
        if self.model.flame_speed > 0:
            period = 2 * np.pi / self.model.flame_speed
            print(f"• Flame completes one circuit in {period:.3f} seconds")
            print(f"• For a 10 cm diameter trough: {self.model.flame_speed * 5:.1f} cm/s")
            print(f"• For a 20 cm diameter trough: {self.model.flame_speed * 10:.1f} cm/s")
        
        return {
            'prescribed': self.model.flame_speed,
            'theoretical': s_theory,
            'measured': s_measured,
            'measured_std': s_std,
            'stability': s_stability,
            'energy_residual': energy_res,
            'species_residual': species_res
        }

# Example usage with your existing model
def run_validation(model, trainer):
    """
    Run validation on trained model
    """
    validator = FlameSpeedValidator(model, trainer)
    results = validator.validate_all()
    return results

# If you want to find the optimal flame speed automatically
class FlameSpeedOptimizer:
    """
    Find the optimal flame speed that minimizes physics residuals
    """
    
    def __init__(self, model_class, layers, device='cpu'):
        self.model_class = model_class
        self.layers = layers
        self.device = device
        
    def objective(self, speed):
        """
        Objective function: physics residual for given speed
        """
        # Create model with specified speed
        model = self.model_class(self.layers)
        model.flame_speed = speed
        model.to(self.device)
        
        # Generate test points
        n_points = 200
        theta = torch.rand(n_points, 1).to(self.device) * 2 * np.pi
        t = torch.rand(n_points, 1).to(self.device) * 0.05
        
        # Compute physics loss
        with torch.no_grad():
            energy_res, species_res = model.physics_loss(theta, t)
            total_residual = torch.mean(energy_res**2) + torch.mean(species_res**2)
        
        return total_residual.item()
    
    def find_optimal_speed(self, speed_range=(1.0, 10.0), n_trials=20):
        """
        Find optimal speed through grid search
        """
        speeds = np.linspace(speed_range[0], speed_range[1], n_trials)
        residuals = []
        
        print("\n🔍 Searching for optimal flame speed...")
        for speed in speeds:
            res = self.objective(speed)
            residuals.append(res)
            print(f"Speed: {speed:.2f} rad/s, Residual: {res:.6f}")
        
        # Find minimum
        optimal_idx = np.argmin(residuals)
        optimal_speed = speeds[optimal_idx]
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(speeds, residuals, 'bo-')
        plt.axvline(x=optimal_speed, color='r', linestyle='--', 
                   label=f'Optimal: {optimal_speed:.2f} rad/s')
        plt.xlabel('Flame Speed (rad/s)')
        plt.ylabel('Physics Residual')
        plt.title('Optimal Flame Speed Search')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"\n✅ Optimal flame speed: {optimal_speed:.2f} rad/s")
        return optimal_speed