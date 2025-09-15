import numpy as np
import matplotlib.pyplot as plt

# Define constants for the model
s_L = 0.4  # m/s, a typical laminar flame speed
sigma_exp = 8.0  # a typical thermal expansion ratio
Ma = 1.5  # a positive Markstein number, indicating a stabilizing effect

# Create a range of wavenumbers (k) from 0 to 200
k = np.linspace(0, 200, 500)

# Calculate the Darrieus-Landau growth rate (sigma_DL)
# This is the destabilizing, linear term
sigma_DL = s_L * k * ((sigma_exp - 1) / (sigma_exp + 1)) * (np.sqrt(1 + sigma_exp) - 1)

# Calculate the stabilizing growth rate from flame stretch
# This is the stabilizing, parabolic term
sigma_stretch = -2 * s_L * (k**2) * Ma

# Calculate the total growth rate, which is the sum of the two terms
sigma_total = sigma_DL + sigma_stretch

# Calculate the critical wavenumber (k_crit) where the total growth rate becomes zero
k_crit = (((sigma_exp - 1) / (sigma_exp + 1)) * (np.sqrt(1 + sigma_exp) - 1)) / (2 * Ma)

# Plot the dispersion relation
plt.figure(figsize=(10, 6))
plt.plot(k, sigma_total, color='r', linewidth=2.5, label=r'$\sigma_{Total} = \sigma_{DL} - 2 s_L k^2 Ma$')
plt.plot(k, sigma_DL, color='b', linestyle='--', linewidth=1.5, label=r'Pure Darrieus-Landau ($\sigma_{DL}$)')
plt.plot(k, sigma_stretch, color='g', linestyle='--', linewidth=1.5, label='Stabilizing Stretch Term')

# Add key features and annotations
plt.axhline(0, color='k', linewidth=0.8, linestyle='-', label='Stability Boundary')
plt.axvline(k_crit, color='purple', linestyle=':', linewidth=1.5, label=r'Critical Wavenumber, $k_{crit}$')
plt.fill_between(k, sigma_total, 0, where=(sigma_total > 0), color='salmon', alpha=0.3, label='Unstable Region')
plt.fill_between(k, sigma_total, 0, where=(sigma_total < 0), color='lightgreen', alpha=0.3, label='Stable Region')

# Set labels and title
plt.title(r'Combined Darrieus-Landau and Stretch Dispersion Relation', fontsize=16)
plt.xlabel(r'Wavenumber, $k$ ($m^{-1}$)', fontsize=14)
plt.ylabel(r'Growth Rate, $\sigma$ ($s^{-1}$)', fontsize=14)
plt.ylim(-15, 30)
plt.xlim(0, 200)
plt.legend(loc='upper right', fontsize=12)
plt.grid(False)

plt.show()