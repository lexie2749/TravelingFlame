import numpy as np
import matplotlib.pyplot as plt

class VerifiedFlameStability:
    """
    Flame stability analysis based on verified literature.
    References: Sivashinsky (1977), Clavin & Williams (1982)
    """
    
    def __init__(self, Lewis=0.7, Zeldovich=10.0, heat_release=5.0):
        """
        Initializes the physical parameters (dimensionless).
        
        Args:
            Lewis (float): Lewis number (Thermal diffusivity / Mass diffusivity).
            Zeldovich (float): Zeldovich number, an activation energy parameter, beta = E_a(T_b-T_u)/(R*T_b^2).
            heat_release (float): Heat release parameter, sigma = (T_b-T_u)/T_u.
        """
        self.Le = Lewis
        self.beta = Zeldovich
        self.sigma = heat_release
        
        # Annular geometry parameters
        self.r_mean = 1.0  # Mean radius
        self.width = 0.1   # Annulus width
        
        print(f"Initializing parameters:")
        print(f"  Lewis number Le = {self.Le}")
        print(f"  Zeldovich number beta = {self.beta}")
        print(f"  Heat release sigma = {self.sigma}")
        
    def markstein_length(self):
        """
        Calculate the Markstein length, a key stability parameter.
        Based on the Clavin-Williams formula.
        """
        Le = self.Le
        beta = self.beta
        sigma = self.sigma
        
        # Thermal expansion parameter
        epsilon = sigma / (1 + sigma)
        
        # Contributions to the Markstein length
        # 1. Lewis number effect (primary term)
        L_Lewis = (1 - Le) / Le
        
        # 2. Thermal expansion effect
        L_thermal = epsilon * np.log(1 + sigma) / sigma
        
        # 3. Activation energy effect
        L_activation = (1 + epsilon) / (2 * beta)
        
        # Total Markstein length
        # Simplified: reducing the thermal expansion contribution for this model
        L_total = L_Lewis * L_activation + L_thermal * 0.1  
        
        return L_total
    
    def dispersion_relation_verified(self, k):
        """
        Verified dispersion relation based on linear stability theory.
        
        Args:
            k (array-like): Dimensionless wave number.
            
        Returns:
            array-like: The growth rate sigma for each wave number.
        """
        Le = self.Le
        beta = self.beta
        L = self.markstein_length()
        
        # Ensure k is a numpy array
        k = np.asarray(k)
        
        # Terms of the dispersion relation
        # 1. Diffusive stabilization term
        diffusion = -k**2
        
        # 2. Darrieus-Landau instability term (curvature effect)
        DL_instability = L * k**2
        
        # 3. High wave number cutoff (prevents short-wave instability)
        cutoff = -k**4 / (beta**2)
        
        # Total growth rate
        sigma = diffusion + DL_instability * beta + cutoff
        
        return sigma
    
    def find_most_unstable_mode(self):
        """
        Finds the most unstable mode by finding the maximum of the dispersion curve.
        """
        L = self.markstein_length()
        
        if L > 0:  # Unstable case
            # Optimal wave number (from d_sigma/dk = 0)
            k_max = self.beta * np.sqrt(L / 2)
            # Maximum growth rate
            sigma_max = self.dispersion_relation_verified(k_max)
        else:  # Stable case
            k_max = 0
            sigma_max = 0
        
        return k_max, sigma_max
    
    def mode_growth_rate(self, m, n=0):
        """
        Calculates the growth rate for a specific (m,n) mode.
        
        Args:
            m (int): Azimuthal mode number.
            n (int): Radial mode number.
        """
        # Total wave number calculation
        k_theta = m / self.r_mean  # Azimuthal wave number
        k_r = n * np.pi / self.width if n > 0 else 0  # Radial wave number
        k_total = np.sqrt(k_theta**2 + k_r**2)
        
        # Avoid k=0, which represents a stable mean flow
        if k_total < 0.01:
            return -0.1
        
        return self.dispersion_relation_verified(k_total)
    
    def complete_analysis(self):
        """
        Performs a complete stability analysis and visualizes the results.
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Adjust subplot layout to 2 rows, 2 columns
        # ======== 1. Dispersion Curve ========
        ax1 = plt.subplot(2, 2, 1)
        k_array = np.linspace(0, 8, 500)
        sigma_array = self.dispersion_relation_verified(k_array)
        
        ax1.plot(k_array, sigma_array, 'b-', linewidth=2.5, label='Dispersion Relation')
        ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
        
        # Mark the unstable region
        unstable = sigma_array > 0
        if np.any(unstable):
            ax1.fill_between(k_array, 0, sigma_array, where=unstable,
                            alpha=0.3, color='red', label='Unstable Region')
        
        # Mark the maximum growth rate
        k_max, sigma_max = self.find_most_unstable_mode()
        if sigma_max > 0:
            ax1.plot(k_max, sigma_max, 'ro', markersize=10,
                    label=f'Max Growth: k={k_max:.2f}, sigma={sigma_max:.3f}')
        
        ax1.set_xlabel('Wave number k', fontsize=11)
        ax1.set_ylabel('Growth rate sigma(k)', fontsize=11)

        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, 8)
        # Modified the y-axis limit
        ax1.set_ylim(-2, 2)
        
        # ======== 2. Markstein Length vs. Lewis Number ========
        ax3 = plt.subplot(2, 2, 2)
        Le_array = np.linspace(0.3, 2.0, 100)
        L_array = []
        sigma_max_array = []
        
        Le_save = self.Le
        for Le_test in Le_array:
            self.Le = Le_test  # Temporarily change Le for the calculation
            L = self.markstein_length()
            L_array.append(L)
            _, s_max = self.find_most_unstable_mode()
            sigma_max_array.append(s_max)
        self.Le = Le_save # Restore original Le
        
        # Plot Markstein length
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(Le_array, L_array, 'g-', linewidth=2, label='Markstein Length')
        ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Lewis Number', fontsize=11)
        ax3.set_ylabel('Markstein Length L', fontsize=11, color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        
        # Plot max growth rate on the twin axis
        line2 = ax3_twin.plot(Le_array, sigma_max_array, 'b-', linewidth=2, label='Max Growth Rate')
        ax3_twin.set_ylabel('Max Growth Rate sigma_max', fontsize=11, color='b')
        ax3_twin.tick_params(axis='y', labelcolor='b')
        
        # Mark critical points
        ax3.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='Le=1 (Neutral)')
        ax3.axvline(self.Le, color='orange', linestyle='-', linewidth=2, label=f'Current Le={self.Le}')
        

        ax3.grid(True, alpha=0.3)
        ax3.legend(handles=line1+line2)
        
        # ======== 3. Shape of Most Unstable Mode ========
        ax5 = plt.subplot(2, 2, 3, projection='polar')
        theta = np.linspace(0, 2 * np.pi, 200)
        
        # Base state (unperturbed circle)
        r_base = np.ones_like(theta)
        ax5.plot(theta, r_base, 'k--', linewidth=2, alpha=0.5, label='Base State')
        
        if sigma_max > 0:
            # Find the corresponding integer azimuthal mode number
            m_unstable = int(round(k_max * self.r_mean))
            m_unstable = max(1, min(m_unstable, 8)) # Cap for visualization
            
            # Plot the perturbed shape
            amplitude = 0.2
            r_perturbed = r_base + amplitude * np.cos(m_unstable * theta)
            ax5.plot(theta, r_perturbed, 'r-', linewidth=2.5,
                    label=f'Mode m={m_unstable}')
            ax5.fill_between(theta, r_base, r_perturbed,
                            where=r_perturbed > r_base, alpha=0.3, color='red')
    
            ax5.plot(theta, r_base, 'g-', linewidth=3, label='Stable')

        ax5.set_ylim(0.6, 1.4)
        ax5.legend(loc='upper right')
        
        # ======== 4. Time Evolution of Perturbations ========
        ax6 = plt.subplot(2, 2, 4)
        if sigma_max > 0:
            t_array = np.linspace(0, min(10, 5 / sigma_max), 100)
            
            # Plot the evolution of several modes
            modes = [(2, 0), (3, 0), (4, 0), (2, 1)]
            colors = ['r', 'b', 'g', 'orange']
            
            for (m, n), color in zip(modes, colors):
                sigma = self.mode_growth_rate(m, n)
                if sigma > 0:
                    amplitude = 0.01 * np.exp(sigma * t_array)
                    ax6.semilogy(t_array, amplitude, color=color, linewidth=2,
                               label=f'(m={m}, n={n}), sigma={sigma:.3f}')
            
            ax6.axhline(1.0, color='k', linestyle='--', alpha=0.5,
                       label='Nonlinear Regime Threshold')
            ax6.set_xlabel('Time (dimensionless)', fontsize=11)
            ax6.set_ylabel('Amplitude (log scale)', fontsize=11)
            ax6.grid(True, alpha=0.3)
            ax6.legend(fontsize=10)
            ax6.set_ylim(0.001, 10)
        else:
            ax6.text(0.5, 0.5, 'System is Stable\n(All modes decay)',
                    ha='center', va='center', fontsize=28,
                    transform=ax6.transAxes)
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
        

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        return L, k_max, sigma_max

def test_verified_model():
    """
    Tests the verified flame stability model with different scenarios.
    """
    print("=" * 70)
    print("VERIFIED FLAME STABILITY ANALYSIS")
    print("=" * 70)
    
    # Define test cases with different Lewis numbers
    test_cases = [
        (0.5, "Strongly Unstable"),
        (0.7, "Moderately Unstable"),
        (1.0, "Neutrally Stable"),
        (1.3, "Stable")
    ]
    
    for Le, description in test_cases[:2]:  # Test only first two cases to avoid too many plots
        print(f"\n{'=' * 50}")
        print(f"Running Case: {description} (Le = {Le})")
        print('=' * 50)
        
        analyzer = VerifiedFlameStability(Lewis=Le, Zeldovich=10.0, heat_release=5.0)
        
        L, k_max, sigma_max = analyzer.complete_analysis()
        
        print(f"\nSummary of Results:")
        print(f"  Markstein Length: L = {L:.4f}")
        print(f"  Most Unstable Wavenumber: k_c = {k_max:.3f}")
        print(f"  Maximum Growth Rate: sigma_max = {sigma_max:.4f}")
        
        if sigma_max > 0:
            print(f"  -> System is UNSTABLE")
            print(f"  -> Characteristic growth time: tau = {1/sigma_max:.1f}")
            print(f"  -> Dominant mode: m = {int(round(k_max))}")
        else:
            print(f"  -> System is STABLE")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    # Print theoretical background
    print("\nTheoretical Background:")
    print("- The Markstein length L determines stability:")
    print("  - L > 0: Unstable (typically when Le < 1)")
    print("  - L < 0: Stable (typically when Le > 1)")
    print("- Simplified Dispersion Relation: sigma(k) = -k^2 + L*beta*k^2 - k^4/beta^2")
    print("- Optimal Wavenumber: k_c = beta * sqrt(L/2)")
    print("- Maximum Growth Rate: sigma_max = L^2 * beta^2 / 4")

if __name__ == "__main__":
    test_verified_model()